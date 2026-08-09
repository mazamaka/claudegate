"""One live conversation.

A :class:`LiveSession` owns a connected ``ClaudeSDKClient`` and a pump task that
drains it forever, translating what comes out into the small event vocabulary
the HTTP layer speaks. The conversation outlives any single request: that is
what makes both session reuse and suspended tool calls possible.

The turn boundary is the interesting part. A request ends when either

* the run finishes (``ResultMessage``), or
* the model asks for tools this client owns — at which point the conversation
  is left *standing*, blocked inside the tool handlers, and the HTTP response
  closes with ``finish_reason: "tool_calls"``.

Nothing is torn down in the second case, so the tool results that arrive on the
next request resume a conversation that never lost its context.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Transport,
)

from ..config import Settings
from ..openai_api import outbound
from .toolbelt import Toolbelt, ToolCorrelator, ToolInvocation

log = logging.getLogger("claudegate.session")

_warned: set[str] = set()


def _warn_once(key: str) -> bool:
    """True the first time it is asked about ``key``."""
    if key in _warned:
        return False
    _warned.add(key)
    return True


#: How long a tool handler waits for the expectation registry to catch up.
#: The CLI dispatches ``tools/call`` on its own task, so it can beat the
#: assistant message that describes the call into this process.
_REGISTRATION_GRACE_S = 30.0


# ────────────────────────────────────────────────────────────── turn events


@dataclass(slots=True)
class TextDelta:
    text: str


@dataclass(slots=True)
class ReasoningDelta:
    text: str


@dataclass(slots=True)
class ToolCallsRequested:
    calls: list[ToolInvocation]


@dataclass(slots=True)
class TurnFinished:
    stop_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None


@dataclass(slots=True)
class TurnFailed:
    message: str
    status: int = 502


TurnEvent = TextDelta | ReasoningDelta | ToolCallsRequested | TurnFinished | TurnFailed


def sandbox_env() -> dict[str, str]:
    """Environment the CLI needs that the caller usually forgets.

    The CLI refuses permission bypass when it detects it is running as root and
    exits without a word, which surfaces as a server that returns empty replies
    and logs nothing. Setting ``IS_SANDBOX`` is the documented escape hatch;
    doing it automatically turns the single most common way to lose an
    afternoon into a non-event.
    """
    if hasattr(os, "geteuid") and os.geteuid() == 0 and not os.environ.get("IS_SANDBOX"):
        # Once per process, not once per conversation: a line repeated on every
        # spawn stops being a warning and starts being noise people filter out.
        log.log(
            logging.WARNING if _warn_once("root") else logging.DEBUG,
            "running as root: setting IS_SANDBOX=1 so the CLI will accept "
            "permission bypass. This disables a guard the CLI puts there on "
            "purpose \u2014 prefer a dedicated non-root user for a real deployment.",
        )
        return {"IS_SANDBOX": "1"}
    return {}


def build_options(
    settings: Settings,
    *,
    model: str,
    system_prompt: str | None,
    toolbelt: Toolbelt,
    handler: Any,
    reasoning_effort: str | None = None,
) -> ClaudeAgentOptions:
    """Translate settings + one request into SDK options."""
    env = {**sandbox_env(), **settings.claude_env}
    prompt = system_prompt
    if settings.system_prompt_suffix:
        suffix = settings.system_prompt_suffix
        prompt = f"{prompt}\n\n{suffix}" if prompt else suffix

    options: dict[str, Any] = {
        "model": model,
        "permission_mode": settings.permission_mode,
        "cwd": settings.workspace,
        "env": env,
        "include_partial_messages": True,
        "setting_sources": None,
        "max_turns": settings.max_turns,
        "fallback_model": settings.fallback_model,
        "stderr": _stderr_logger(),
    }
    if settings.cli_path:
        options["cli_path"] = settings.cli_path
    if reasoning_effort:
        options["effort"] = reasoning_effort

    if settings.bare_mode:
        # Replace Claude Code's prompt and take away its own tools: what is
        # left behaves like a plain chat model, which is what an OpenAI client
        # is expecting to talk to.
        options["system_prompt"] = prompt or ""
        options["tools"] = toolbelt.allowed_tool_names
    elif prompt:
        options["system_prompt"] = {"type": "preset", "preset": "claude_code", "append": prompt}

    if toolbelt:
        options["mcp_servers"] = {"client": toolbelt.build_server(handler)}
        options["allowed_tools"] = toolbelt.allowed_tool_names
        options["strict_mcp_config"] = True

    return ClaudeAgentOptions(**options)


def _stderr_logger() -> Any:
    def _write(line: str) -> None:
        log.debug("cli: %s", line.rstrip())

    return _write


# ───────────────────────────────────────────────────────────────── session


class LiveSession:
    """A conversation held open across requests."""

    def __init__(
        self,
        *,
        session_id: str,
        identity: str,
        options: ClaudeAgentOptions,
        toolbelt: Toolbelt,
        settings: Settings,
        transport: Transport | None = None,
    ) -> None:
        self.id = session_id
        self.identity = identity
        self.settings = settings
        self.toolbelt = toolbelt
        self.chain: list[str] = []
        self.created_at = time.monotonic()
        self.last_used = time.monotonic()
        self.busy = False
        #: Set when this conversation must not be handed out again. It is not
        #: the same as "torn down" — a retired conversation still has a live
        #: subprocess to close, which is what ``aclose`` is for.
        self.closed = False
        self._shut_down = False
        self.cli_session_id: str | None = None
        self.turns = 0
        #: The last answer this conversation produced. A client that wants to
        #: continue has to hand it back, which is what proves the conversation
        #: is theirs — see ``SessionManager._reuse``.
        self.last_reply = ""

        self._client = ClaudeSDKClient(options, transport=transport)
        self._queue: asyncio.Queue[TurnEvent | None] = asyncio.Queue()
        self._pump: asyncio.Task[None] | None = None
        self._correlator = ToolCorrelator()
        self._results: dict[str, asyncio.Future[str]] = {}
        self._expectations_ready = asyncio.Event()
        self._last_usage: dict[str, Any] = {}
        self._last_stop: str | None = None
        self._reply_buffer: list[str] = []
        self._streamed_this_message = False

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        await self._client.connect()
        self._pump = asyncio.create_task(self._pump_loop(), name=f"claudegate-pump-{self.id}")

    async def aclose(self) -> None:
        if self._shut_down:
            return
        self._shut_down = True
        self.closed = True
        for fut in self._results.values():
            if not fut.done():
                fut.set_result("[claudegate] conversation closed before the result arrived")
        if self._pump:
            self._pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pump
        with contextlib.suppress(Exception):
            await self._client.disconnect()

    @property
    def awaiting_tools(self) -> bool:
        return bool(self._results) and any(not f.done() for f in self._results.values())

    @property
    def pending_call_ids(self) -> set[str]:
        return {cid for cid, fut in self._results.items() if not fut.done()}

    def idle_for(self) -> float:
        return time.monotonic() - self.last_used

    # ── driving a turn ────────────────────────────────────────────────────

    async def send(self, blocks: list[dict[str, Any]]) -> None:
        """Send a user turn built from Anthropic content blocks."""
        self.last_used = time.monotonic()
        self.turns += 1
        self._reply_buffer.clear()
        payload = {
            "type": "user",
            "message": {"role": "user", "content": blocks},
            "parent_tool_use_id": None,
            "session_id": "default",
        }

        async def _once() -> AsyncIterator[dict[str, Any]]:
            yield payload

        await self._client.query(_once())

    async def deliver_tool_results(self, results: dict[str, str]) -> None:
        """Hand back the client's tool results, unblocking the parked handlers."""
        self.last_used = time.monotonic()
        # A continuation is a new turn: the reply buffer must not carry over the
        # narration the model produced *before* it asked for the tool. The client
        # only ever echoes what it was given for this turn, and that is what
        # ``proves_receipt`` compares against.
        self._reply_buffer.clear()
        # Snapshot first. The model may register a *new* round of calls while we
        # are still handing back this one, and those must not be force-released.
        outstanding = list(self._results.items())
        for call_id, text in results.items():
            fut = self._results.get(call_id)
            if fut and not fut.done():
                fut.set_result(text)
        # A client that answers only some of the calls would otherwise leave the
        # conversation wedged until the TTL expires.
        for call_id, fut in outstanding:
            if not fut.done():
                log.warning("session %s: no result for %s; releasing", self.id, call_id)
                fut.set_result("[claudegate] the client returned no result for this tool call")

    async def stream_turn(self, timeout: float | None = None) -> AsyncIterator[TurnEvent]:
        """Yield this turn's events, ending at the turn boundary."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except (TimeoutError, asyncio.TimeoutError):
                # The turn is abandoned but the conversation may still be
                # producing events. Anything it emits from here belongs to a
                # turn nobody is listening to, and would surface inside the
                # *next* one, so the conversation is retired instead of reused.
                self.closed = True
                log.warning("session %s: turn timed out; retiring it", self.id)
                yield TurnFailed("Timed out waiting for the model.", status=504)
                return
            if item is None:
                return
            yield item

    # ── the pump ──────────────────────────────────────────────────────────

    async def _pump_loop(self) -> None:
        try:
            async for message in self._client.receive_messages():
                await self._handle(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("session %s: pump failed: %s", self.id, exc)
            await self._emit(TurnFailed(f"Claude Code CLI error: {exc}"))
            await self._end_turn()
            self.closed = True

    async def _emit(self, event: TurnEvent) -> None:
        await self._queue.put(event)

    async def _end_turn(self) -> None:
        await self._queue.put(None)

    async def _handle(self, message: Any) -> None:
        if isinstance(message, StreamEvent):
            await self._handle_stream_event(message.event)
        elif isinstance(message, SystemMessage):
            if message.subtype == "init":
                self.cli_session_id = message.data.get("session_id") or self.cli_session_id
        elif isinstance(message, AssistantMessage):
            await self._handle_assistant(message)
        elif isinstance(message, ResultMessage):
            await self._handle_result(message)

    async def _handle_stream_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "content_block_delta":
            delta = event.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta" and delta.get("text"):
                self._streamed_this_message = True
                self._reply_buffer.append(delta["text"])
                await self._emit(TextDelta(delta["text"]))
            elif dtype == "thinking_delta" and delta.get("thinking"):
                await self._emit(ReasoningDelta(delta["thinking"]))
        elif kind == "message_start":
            self._streamed_this_message = False
        elif kind == "message_delta":
            self._last_stop = (event.get("delta") or {}).get("stop_reason") or self._last_stop
            if event.get("usage"):
                self._last_usage = event["usage"]

    async def _handle_assistant(self, message: AssistantMessage) -> None:
        if message.error:
            await self._emit(TurnFailed(f"Claude Code reported {message.error}."))
            await self._end_turn()
            return

        invocations: list[ToolInvocation] = []
        for block in message.content:
            if isinstance(block, ToolUseBlock) and self.toolbelt.owns(block.name):
                name = self.toolbelt.openai_name(block.name) or block.name
                invocations.append(
                    ToolInvocation(
                        id=outbound.call_id(block.id),
                        name=name,
                        arguments=dict(block.input or {}),
                    )
                )
            elif isinstance(block, TextBlock):
                # Normally this text has already gone out as deltas. When it
                # has not -- partial messages disabled, or a CLI that skips
                # them -- the answer would otherwise be silently dropped and
                # the client would get `content: null`. Emit it once here.
                if not self._streamed_this_message and block.text:
                    self._reply_buffer.append(block.text)
                    await self._emit(TextDelta(block.text))
            elif isinstance(block, ThinkingBlock):
                continue

        # Whether this message arrived as deltas says nothing about the next
        # one, and a CLI that sends no partial messages sends no
        # ``message_start`` either -- so the flag is cleared here as well.
        self._streamed_this_message = False

        if invocations:
            loop = asyncio.get_running_loop()
            for inv in invocations:
                self._results[inv.id] = loop.create_future()
            self._correlator.expect(invocations)
            # Hand this round's waiters their event and install a fresh one for
            # the next round, so the latch cannot stay set across rounds.
            self._expectations_ready, ready = asyncio.Event(), self._expectations_ready
            ready.set()
            await self._emit(ToolCallsRequested(invocations))
            await self._emit(
                TurnFinished(stop_reason="tool_use", usage=self._last_usage, cost_usd=None)
            )
            await self._end_turn()

    async def _handle_result(self, message: ResultMessage) -> None:
        self.last_reply = "".join(self._reply_buffer)
        self._reply_buffer.clear()
        # Release before clearing. A future dropped while a handler is parked on
        # it leaves that handler waiting out the full tool TTL, holding a
        # subprocess open long after the conversation left the pool.
        for pending in self._results.values():
            if not pending.done():
                pending.set_result("[claudegate] the run ended before this call returned")
        self._results.clear()
        self._correlator.clear()
        # Wake any handler still waiting to be registered: it gets an immediate
        # answer instead of sitting out the registration grace period.
        self._expectations_ready.set()
        if message.is_error:
            detail = "; ".join(message.errors or []) or message.result or message.subtype
            status = 429 if message.api_error_status == 429 else 502
            await self._emit(TurnFailed(f"Claude Code CLI: {detail}", status=status))
        else:
            await self._emit(
                TurnFinished(
                    stop_reason=self._last_stop or "end_turn",
                    usage=message.usage or self._last_usage,
                    cost_usd=message.total_cost_usd,
                )
            )
        self._last_stop = None
        # Usage belongs to the turn that reported it. Carrying it over makes the
        # next turn report numbers it never spent.
        self._last_usage = {}
        await self._end_turn()

    # ── the parked tool handler ───────────────────────────────────────────

    async def tool_handler(self, openai_name: str, args: dict[str, Any]) -> str:
        """Park until the client sends this call's result back.

        The CLI is happy to wait: it is blocked on an MCP response, which is
        exactly the semantics we want for "the caller is running this tool".
        """
        if not self._correlator.pending_ids:
            # Wait on the event as it is *now*: a later round installs a new one,
            # and waking on that would mean claiming another round's call.
            ready = self._expectations_ready
            with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
                await asyncio.wait_for(ready.wait(), timeout=_REGISTRATION_GRACE_S)

        invocation = self._correlator.claim(openai_name, args)
        if invocation is None:
            log.warning("session %s: unmatched tool call %s", self.id, openai_name)
            return "[claudegate] this call could not be matched to a pending request"

        future = self._results.get(invocation.id)
        if future is None:
            return "[claudegate] the call was already resolved"

        try:
            return await asyncio.wait_for(
                asyncio.shield(future), timeout=self.settings.tool_wait_ttl_s
            )
        except (TimeoutError, asyncio.TimeoutError):
            log.warning("session %s: tool %s timed out", self.id, openai_name)
            return "[claudegate] the client did not return a result in time"
