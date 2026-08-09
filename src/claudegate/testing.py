"""A Claude Code CLI that isn't one.

:class:`FakeClaudeCLI` implements the SDK's transport interface and speaks the
same control protocol the real CLI does — including the side of it that matters
most here, where the CLI reaches *back* into this process to invoke an MCP tool.
Swap it in and the entire server can be exercised end to end with no CLI
installed, no token, no network and no cost, in milliseconds.

That is how this project's own test suite runs, and it is public because the
same trick works for anything built on top::

    from claudegate.testing import FakeClaudeCLI, scripted
    from claudegate import create_app

    app = create_app(settings, transport_factory=lambda: FakeClaudeCLI(scripted("hi")))

Scenarios are plain coroutines, so a test can assert on exactly what the model
was sent and script exactly what it does back — including tool calls.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
from collections.abc import Awaitable, Callable
from typing import Any

from claude_agent_sdk import Transport

Scenario = Callable[["Turn"], Awaitable[None]]

DEFAULT_USAGE = {
    "input_tokens": 42,
    "output_tokens": 7,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
}


class Turn:
    """One user turn, from the fake CLI's point of view."""

    def __init__(self, cli: FakeClaudeCLI, payload: dict[str, Any]) -> None:
        self._cli = cli
        self.payload = payload
        self.ended = False

    @property
    def cli(self) -> FakeClaudeCLI:
        """The transport, for scenarios that need to emit raw events."""
        return self._cli

    # -- what the model was given ----------------------------------------

    @property
    def content(self) -> list[dict[str, Any]] | str:
        value = self.payload.get("message", {}).get("content", "")
        return value if isinstance(value, (list, str)) else ""

    @property
    def blocks(self) -> list[dict[str, Any]]:
        content = self.content
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        return content

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.blocks if b.get("type") == "text")

    @property
    def images(self) -> list[dict[str, Any]]:
        return [b for b in self.blocks if b.get("type") == "image"]

    # -- what the model does ---------------------------------------------

    async def think(self, text: str) -> None:
        """Stream a block of reasoning."""
        await self._cli.emit_stream({"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "thinking", "thinking": ""}})
        await self._cli.emit_stream({"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "thinking_delta", "thinking": text}})
        await self._cli.emit_stream({"type": "content_block_stop", "index": 0})

    async def say(self, text: str, *, chunks: int = 2) -> None:
        """Stream an assistant answer, split across ``chunks`` deltas."""
        await self._cli.emit_stream({
            "type": "message_start",
            "message": {
                "id": self._cli.next_message_id(), "model": self._cli.model,
                "type": "message", "role": "assistant", "content": [], "stop_reason": None,
                "usage": {"input_tokens": self._cli.usage["input_tokens"], "output_tokens": 0},
            },
        })
        await self._cli.emit_stream({"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}})
        size = max(1, len(text) // max(1, chunks))
        for start in range(0, len(text), size):
            piece = text[start : start + size]
            await self._cli.emit_stream({
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": piece},
            })
            await asyncio.sleep(0)
        await self._cli.emit_stream({"type": "content_block_stop", "index": 0})
        await self._cli.emit_stream({"type": "message_delta",
                                     "delta": {"stop_reason": "end_turn"},
                                     "usage": self._cli.usage})
        await self._cli.emit_stream({"type": "message_stop"})
        await self._cli.emit_assistant([{"type": "text", "text": text}], stop_reason="end_turn")

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Call one of the client's tools and wait for its result.

        Emits the ``tool_use`` block, then drives the real MCP round trip back
        into the server — which is what parks the turn until the client answers.
        """
        arguments = arguments or {}
        tool_use_id = self._cli.next_tool_id()
        full_name = name if name.startswith("mcp__") else f"mcp__client__{name}"
        await self._cli.emit_assistant(
            [{"type": "tool_use", "id": tool_use_id, "name": full_name, "input": arguments}],
            stop_reason="tool_use",
        )
        return await self._cli.invoke_tool(full_name, arguments)

    async def call_tools(self, calls: list[tuple[str, dict[str, Any]]]) -> list[str]:
        """Ask for several tools in one assistant message, as the model does."""
        blocks = []
        names = []
        for name, arguments in calls:
            full_name = name if name.startswith("mcp__") else f"mcp__client__{name}"
            names.append((full_name, arguments))
            blocks.append({"type": "tool_use", "id": self._cli.next_tool_id(),
                           "name": full_name, "input": arguments})
        await self._cli.emit_assistant(blocks, stop_reason="tool_use")
        # The real CLI dispatches them one at a time, in order.
        return [await self._cli.invoke_tool(n, a) for n, a in names]

    async def fail(self, message: str, *, api_status: int | None = None) -> None:
        await self._cli.emit({"type": "result", "subtype": "error_during_execution",
                              "session_id": self._cli.session_id, "duration_ms": 1,
                              "duration_api_ms": 1, "is_error": True, "num_turns": 1,
                              "result": message, "errors": [message],
                              "api_error_status": api_status, "usage": self._cli.usage,
                              "total_cost_usd": 0.0})
        self.ended = True

    async def end(self, *, result: str = "", cost: float = 0.001) -> None:
        await self._cli.emit({"type": "result", "subtype": "success",
                              "session_id": self._cli.session_id, "duration_ms": 5,
                              "duration_api_ms": 4, "is_error": False, "num_turns": 1,
                              "result": result, "usage": self._cli.usage,
                              "total_cost_usd": cost})
        self.ended = True


def scripted(*replies: str) -> Scenario:
    """A scenario that answers each turn with the next canned reply."""
    remaining = list(replies)

    async def scenario(turn: Turn) -> None:
        reply = remaining.pop(0) if remaining else (replies[-1] if replies else "ok")
        await turn.say(reply)
        await turn.end(result=reply)

    return scenario


def echo(prefix: str = "") -> Scenario:
    """A scenario that echoes what it was sent — handy for asserting on folds."""

    async def scenario(turn: Turn) -> None:
        await turn.say(prefix + turn.text)
        await turn.end()

    return scenario


class FakeClaudeCLI(Transport):
    """A scripted stand-in for the ``claude`` subprocess."""

    def __init__(
        self,
        scenario: Scenario,
        *,
        session_id: str = "fake-session",
        model: str = "claude-fake",
        usage: dict[str, Any] | None = None,
        available_tools: list[str] | None = None,
    ) -> None:
        self.scenario = scenario
        self.session_id = session_id
        self.model = model
        self.usage = dict(usage or DEFAULT_USAGE)
        self.available_tools = available_tools or []
        #: Every user turn this transport received, in order. Assert on it.
        self.turns: list[dict[str, Any]] = []
        self.closed = False

        self._out: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._ready = False
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._ids = itertools.count(1)

    # -- ids --------------------------------------------------------------

    def next_uuid(self) -> str:
        return f"uuid-{next(self._ids)}"

    def next_message_id(self) -> str:
        return f"msg_fake_{next(self._ids)}"

    def next_tool_id(self) -> str:
        return f"toolu_fake_{next(self._ids)}"

    def _next_request_id(self) -> str:
        return f"fake-req-{next(self._ids)}"

    # -- transport --------------------------------------------------------

    async def connect(self) -> None:
        self._ready = True
        await self.emit({
            "type": "system", "subtype": "init", "session_id": self.session_id,
            "cwd": "/tmp", "tools": self.available_tools, "mcp_servers": [],
            "model": self.model, "uuid": self.next_uuid(),
        })

    async def write(self, data: str) -> None:
        for line in data.strip().splitlines():
            if not line.strip():
                continue
            message = json.loads(line)
            kind = message.get("type")
            if kind == "control_request":
                await self._answer_control(message)
            elif kind == "control_response":
                self._resolve_control(message)
            elif kind == "user":
                self.turns.append(message)
                self._spawn(self._run_turn(message))

    async def read_messages(self) -> Any:
        while True:
            item = await self._out.get()
            if item is None:
                return
            yield item

    async def close(self) -> None:
        self.closed = True
        self._ready = False
        for task in list(self._tasks):
            task.cancel()
        await self._out.put(None)

    def is_ready(self) -> bool:
        return self._ready

    async def end_input(self) -> None:
        return None

    # -- internals --------------------------------------------------------

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_turn(self, payload: dict[str, Any]) -> None:
        turn = Turn(self, payload)
        try:
            await self.scenario(turn)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            with contextlib.suppress(Exception):
                await turn.fail(f"scenario raised {type(exc).__name__}: {exc}")
            return
        if not turn.ended:
            await turn.end()

    async def emit(self, message: dict[str, Any]) -> None:
        await self._out.put(message)

    async def emit_assistant(
        self, content: list[dict[str, Any]], *, stop_reason: str = "end_turn"
    ) -> None:
        await self._out.put({
            "type": "assistant", "session_id": self.session_id, "uuid": self.next_uuid(),
            "parent_tool_use_id": None,
            "message": {
                "id": self.next_message_id(), "model": self.model, "role": "assistant",
                "type": "message", "content": content, "stop_reason": stop_reason,
                "usage": self.usage,
            },
        })

    async def emit_stream(self, event: dict[str, Any]) -> None:
        await self._out.put({
            "type": "stream_event", "uuid": self.next_uuid(),
            "session_id": self.session_id, "event": event, "parent_tool_use_id": None,
        })

    async def _answer_control(self, message: dict[str, Any]) -> None:
        request_id = message.get("request_id")
        subtype = (message.get("request") or {}).get("subtype")
        response: dict[str, Any] = {}
        if subtype == "initialize":
            response = {"commands": [], "output_style": "default"}
        await self.emit({
            "type": "control_response",
            "response": {"subtype": "success", "request_id": request_id, "response": response},
        })

    def _resolve_control(self, message: dict[str, Any]) -> None:
        response = message.get("response") or {}
        future = self._pending.pop(response.get("request_id", ""), None)
        if future and not future.done():
            future.set_result(response)

    async def invoke_tool(self, full_name: str, arguments: dict[str, Any]) -> str:
        """Drive a real MCP ``tools/call`` back into the server."""
        request_id = self._next_request_id()
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self.emit({
            "type": "control_request",
            "request_id": request_id,
            "request": {
                "subtype": "mcp_message",
                "server_name": "client",
                "message": {
                    "jsonrpc": "2.0", "id": next(self._ids), "method": "tools/call",
                    "params": {"name": full_name.split("__")[-1], "arguments": arguments},
                },
            },
        })
        response = await future
        if response.get("subtype") == "error":
            return f"[tool error] {response.get('error')}"
        payload = (response.get("response") or {}).get("mcp_response") or {}
        content = ((payload.get("result") or {}).get("content")) or []
        return "".join(part.get("text", "") for part in content if part.get("type") == "text")
