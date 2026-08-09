"""Deciding which conversation a request belongs to.

Three outcomes, in the order they are tried:

1. **Continuation.** The request ends in tool results, and some live
   conversation is parked waiting for exactly those calls. Nothing is sent —
   the results are handed to the waiting handlers and the conversation picks up
   where it stopped.
2. **Reuse.** A live conversation's history is a prefix of this request's.
   Only the messages after the prefix are sent.
3. **Fresh.** Everything else: a new conversation, with the whole history
   rendered into its opening turn.

Case 3 is also the safety net for a continuation whose conversation is gone —
reaped, or lost to a restart. Rebuilding costs a re-read of the history; the
alternative is failing a turn the client cannot retry, because it no longer has
anywhere to send its tool results.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

from claude_agent_sdk import Transport

from ..config import Settings
from ..errors import ConversationExpired, OverloadedError, UpstreamError
from ..openai_api import inbound
from ..openai_api.schema import ChatCompletionRequest, Message
from . import continuity
from .session import LiveSession, build_options
from .toolbelt import Toolbelt, fingerprint

log = logging.getLogger("claudegate.manager")

TransportFactory = Callable[[], Transport]


@dataclass
class Stats:
    created: int = 0
    reused: int = 0
    rebuilt: int = 0
    continued: int = 0
    evicted: int = 0
    expired: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "sessions_created": self.created,
            "sessions_reused": self.reused,
            "sessions_rebuilt": self.rebuilt,
            "turns_continued": self.continued,
            "sessions_evicted": self.evicted,
            "sessions_expired": self.expired,
        }


@dataclass
class Lease:
    """A session checked out for the duration of one request."""

    session: LiveSession
    manager: SessionManager
    blocks: list[dict[str, Any]] = field(default_factory=list)
    tool_results: dict[str, str] | None = None
    mode: str = "fresh"
    model: str = ""

    async def __aenter__(self) -> Lease:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.manager.release(self.session, failed=exc is not None)

    async def dispatch(self) -> None:
        """Feed the session whatever this request carries."""
        if self.tool_results is not None:
            await self.session.deliver_tool_results(self.tool_results)
        if self.blocks:
            await self.session.send(self.blocks)


class SessionManager:
    """Registry of live conversations."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport_factory: TransportFactory | None = None,
    ) -> None:
        self.settings = settings
        self.stats = Stats()
        self._sessions: dict[str, LiveSession] = {}
        self._lock = asyncio.Lock()
        self._gc: asyncio.Task[None] | None = None
        self._transport_factory = transport_factory

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._gc = asyncio.create_task(self._gc_loop(), name="claudegate-gc")

    async def aclose(self) -> None:
        if self._gc:
            self._gc.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._gc
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        await asyncio.gather(*(s.aclose() for s in sessions), return_exceptions=True)

    @property
    def live(self) -> int:
        return len(self._sessions)

    # ── acquiring ─────────────────────────────────────────────────────────

    async def acquire(self, request: ChatCompletionRequest, *, tenant: str = "") -> Lease:
        settings = self.settings
        system_prompt, conversation = inbound.split_system(request.messages)
        tools = [] if request.tools_disabled else (request.tools or [])
        toolbelt = Toolbelt(tools)
        model = settings.resolve_model(request.model)
        identity = continuity.identity_key(
            model=model,
            system_prompt=system_prompt,
            tools_fingerprint=fingerprint(tools),
            bare_mode=settings.bare_mode,
            tenant=tenant,
        )
        incoming_chain = continuity.chain(conversation)
        results = inbound.trailing_tool_results(conversation)

        async with self._lock:
            if results:
                lease = await self._continue(request, identity, incoming_chain, results, model)
                if lease is not None:
                    return lease

            reuse_allowed = settings.reuse_sessions and not (
                settings.reuse_requires_user and not request.user
            )
            if reuse_allowed and not results:
                lease = await self._reuse(request, identity, incoming_chain, conversation, model)
                if lease is not None:
                    return lease

            return await self._fresh(
                request,
                identity=identity,
                chain=incoming_chain,
                conversation=conversation,
                system_prompt=system_prompt,
                toolbelt=toolbelt,
                model=model,
                rebuilt=bool(results),
            )

    async def _continue(
        self,
        request: ChatCompletionRequest,
        identity: str,
        chain: list[str],
        results: list[Message],
        model: str,
    ) -> Lease | None:
        wanted = {m.tool_call_id for m in results if m.tool_call_id}
        for session in self._sessions.values():
            if session.closed or session.busy or session.identity != identity:
                continue
            if wanted and wanted & session.pending_call_ids:
                session.busy = True
                session.chain = chain
                self.stats.continued += 1
                log.debug("session %s: continuing with %d tool result(s)", session.id, len(results))
                return Lease(
                    session=session,
                    manager=self,
                    tool_results={m.tool_call_id: m.text() for m in results if m.tool_call_id},
                    mode="continued",
                    model=model,
                )

        self.stats.expired += 1
        if not self.settings.rebuild_on_expiry:
            raise ConversationExpired(
                "This conversation is no longer held by the server. Send the "
                "conversation again to start a new one."
            )
        log.info("no live conversation for these tool results; rebuilding from history")
        return None

    async def _reuse(
        self,
        request: ChatCompletionRequest,
        identity: str,
        chain: list[str],
        conversation: list[Message],
        model: str,
    ) -> Lease | None:
        best: LiveSession | None = None
        for session in self._sessions.values():
            if session.closed or session.busy or session.identity != identity:
                continue
            if session.awaiting_tools or not session.chain:
                continue
            if not continuity.is_prefix(session.chain, chain):
                continue
            if not continuity.proves_receipt(conversation, session.last_reply):
                continue
            if len(session.chain) == len(chain):
                continue  # nothing new to say; treat as a fresh request
            if best is None or len(session.chain) > len(best.chain):
                best = session

        if best is None:
            return None

        delta = continuity.new_messages(conversation, len(best.chain))
        if not delta:
            return None

        if all(m.role == "user" for m in delta):
            blocks: list[dict[str, Any]] = []
            for msg in delta:
                blocks.extend(
                    inbound.message_blocks(msg, attachments=self.settings.forward_attachments)
                )
        else:
            blocks = inbound.render_history(
                delta, attachments=self.settings.forward_attachments
            )

        best.busy = True
        best.chain = chain
        self.stats.reused += 1
        log.debug("session %s: reused, sending %d new message(s)", best.id, len(delta))
        return Lease(session=best, manager=self, blocks=blocks, mode="reused", model=model)

    async def _fresh(
        self,
        request: ChatCompletionRequest,
        *,
        identity: str,
        chain: list[str],
        conversation: list[Message],
        system_prompt: str | None,
        toolbelt: Toolbelt,
        model: str,
        rebuilt: bool,
    ) -> Lease:
        await self._make_room()

        session_id = uuid.uuid4().hex[:12]
        holder: dict[str, LiveSession] = {}

        async def handler(name: str, args: dict[str, Any]) -> str:
            return await holder["session"].tool_handler(name, args)

        options = build_options(
            self.settings,
            model=model,
            system_prompt=system_prompt,
            toolbelt=toolbelt,
            handler=handler,
            reasoning_effort=request.reasoning_effort,
        )
        session = LiveSession(
            session_id=session_id,
            identity=identity,
            options=options,
            toolbelt=toolbelt,
            settings=self.settings,
            transport=self._transport_factory() if self._transport_factory else None,
        )
        holder["session"] = session

        try:
            await session.start()
        except Exception as exc:
            with contextlib.suppress(Exception):
                await session.aclose()
            raise UpstreamError(f"Could not start the Claude Code CLI: {exc}") from exc

        session.busy = True
        session.chain = chain
        self._sessions[session_id] = session
        self.stats.created += 1
        if rebuilt:
            self.stats.rebuilt += 1

        blocks = inbound.render_history(
            conversation, attachments=self.settings.forward_attachments
        )
        return Lease(
            session=session,
            manager=self,
            blocks=blocks,
            mode="rebuilt" if rebuilt else "fresh",
            model=model,
        )

    # ── releasing and reaping ─────────────────────────────────────────────

    async def release(self, session: LiveSession, *, failed: bool = False) -> None:
        session.busy = False
        session.last_used = time.monotonic()
        if failed or session.closed:
            await self.drop(session)

    async def drop(self, session: LiveSession) -> None:
        async with self._lock:
            self._sessions.pop(session.id, None)
        await session.aclose()

    async def _make_room(self) -> None:
        if len(self._sessions) < self.settings.max_sessions:
            return
        idle = [s for s in self._sessions.values() if not s.busy]
        if not idle:
            raise OverloadedError(
                f"All {self.settings.max_sessions} conversation slots are busy.", retry_after=5
            )
        victim = max(idle, key=lambda s: s.idle_for())
        self._sessions.pop(victim.id, None)
        self.stats.evicted += 1
        log.info("evicting session %s to make room", victim.id)
        asyncio.get_running_loop().create_task(victim.aclose())

    async def _gc_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.gc_interval_s)
            try:
                await self._reap()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("gc pass failed: %s", exc)

    async def _reap(self) -> None:
        now_idle_limit = self.settings.session_idle_ttl_s
        doomed: list[LiveSession] = []
        async with self._lock:
            for session in list(self._sessions.values()):
                if session.busy:
                    continue
                limit = self.settings.tool_wait_ttl_s if session.awaiting_tools else now_idle_limit
                if session.idle_for() > limit or session.closed:
                    self._sessions.pop(session.id, None)
                    doomed.append(session)
        for session in doomed:
            self.stats.expired += 1
            log.info("reaping idle session %s (%.0fs)", session.id, session.idle_for())
            await session.aclose()
