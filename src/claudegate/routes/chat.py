"""``POST /v1/chat/completions``."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..bridge.manager import Lease, SessionManager
from ..bridge.session import (
    ReasoningDelta,
    TextDelta,
    ToolCallsRequested,
    TurnFailed,
    TurnFinished,
)
from ..config import Settings
from ..errors import GatewayError, InvalidRequest, for_status
from ..openai_api import outbound
from ..openai_api.schema import ChatCompletionRequest
from ..security import tenant_id

log = logging.getLogger("claudegate.chat")
router = APIRouter()


def _validate(body: ChatCompletionRequest) -> None:
    if not body.messages:
        raise InvalidRequest("'messages' must contain at least one message.", param="messages")
    if body.tools:
        for t in body.tools:
            if not t.function.name:
                raise InvalidRequest("Every tool needs a function name.", param="tools")


@router.post("/chat/completions")
async def chat_completions(request: Request) -> Any:
    settings: Settings = request.app.state.settings
    manager: SessionManager = request.app.state.manager
    metrics = request.app.state.metrics

    payload = await request.json()
    try:
        body = ChatCompletionRequest.model_validate(payload)
    except Exception as exc:
        raise InvalidRequest(f"Malformed request: {exc}") from exc
    _validate(body)

    reported_model = body.model or settings.default_model
    started = time.monotonic()
    metrics.inc("requests_total")

    lease = await manager.acquire(body, tenant=tenant_id(request, settings, body.user))
    metrics.inc_label("session_mode_total", lease.mode)
    headers = {
        "x-claudegate-session": lease.session.id,
        "x-claudegate-mode": lease.mode,
    }

    if body.stream:
        return StreamingResponse(
            _stream(lease, body, settings, reported_model, metrics, started),
            media_type="text/event-stream",
            headers={**headers, "cache-control": "no-store", "x-accel-buffering": "no"},
        )

    result = await _collect(lease, body, settings, reported_model, metrics, started)
    return JSONResponse(result, headers=headers)


async def _stream(
    lease: Lease,
    body: ChatCompletionRequest,
    settings: Settings,
    model: str,
    metrics: Any,
    started: float,
) -> AsyncIterator[str]:
    cid = outbound.completion_id()
    created = int(time.time())
    calls: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    stop_reason: str | None = None
    first_token_at: float | None = None

    async with lease:
        try:
            await lease.dispatch()
        except GatewayError as exc:
            yield outbound.sse(exc.to_dict())
            yield outbound.DONE
            return

        yield outbound.sse(
            outbound.chunk(cid, model, {"role": "assistant", "content": ""}, created=created)
        )

        async for event in lease.session.stream_turn(timeout=settings.request_timeout_s):
            if isinstance(event, TextDelta):
                if first_token_at is None:
                    first_token_at = time.monotonic()
                yield outbound.sse(
                    outbound.chunk(cid, model, {"content": event.text}, created=created)
                )
            elif isinstance(event, ReasoningDelta):
                if first_token_at is None:
                    first_token_at = time.monotonic()
                yield outbound.sse(
                    outbound.chunk(cid, model, {"reasoning_content": event.text}, created=created)
                )
            elif isinstance(event, ToolCallsRequested):
                for index, call in enumerate(event.calls):
                    calls.append(call.as_openai())
                    yield outbound.sse(
                        outbound.chunk(
                            cid,
                            model,
                            outbound.tool_call_delta(index, call.as_openai()),
                            created=created,
                        )
                    )
            elif isinstance(event, TurnFinished):
                usage = outbound.usage_block(event.usage)
                stop_reason = event.stop_reason
                if event.cost_usd:
                    metrics.inc("cost_usd_total", event.cost_usd)
            elif isinstance(event, TurnFailed):
                metrics.inc("errors_total")
                error = for_status(event.status, event.message)
                yield outbound.sse(error.to_dict())
                yield outbound.DONE
                return

        finish = outbound.finish_reason(stop_reason, had_tool_calls=bool(calls))
        yield outbound.sse(outbound.chunk(cid, model, {}, created=created, finish=finish))
        if body.wants_usage_chunk:
            yield outbound.sse(
                outbound.usage_chunk(cid, model, usage or outbound.usage_block(None))
            )
        yield outbound.DONE

    _record(metrics, started, first_token_at, usage, streamed=True)


async def _collect(
    lease: Lease,
    body: ChatCompletionRequest,
    settings: Settings,
    model: str,
    metrics: Any,
    started: float,
) -> dict[str, Any]:
    text: list[str] = []
    reasoning: list[str] = []
    calls: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    stop_reason: str | None = None

    async with lease:
        await lease.dispatch()
        async for event in lease.session.stream_turn(timeout=settings.request_timeout_s):
            if isinstance(event, TextDelta):
                text.append(event.text)
            elif isinstance(event, ReasoningDelta):
                reasoning.append(event.text)
            elif isinstance(event, ToolCallsRequested):
                calls.extend(c.as_openai() for c in event.calls)
            elif isinstance(event, TurnFinished):
                usage = outbound.usage_block(event.usage)
                stop_reason = event.stop_reason
                if event.cost_usd:
                    metrics.inc("cost_usd_total", event.cost_usd)
            elif isinstance(event, TurnFailed):
                metrics.inc("errors_total")
                raise for_status(event.status, event.message)

    _record(metrics, started, None, usage, streamed=False)
    return outbound.completion(
        outbound.completion_id(),
        model,
        content="".join(text) or None,
        reasoning="".join(reasoning) or None,
        tool_calls=calls or None,
        usage=usage,
        stop_reason=stop_reason,
    )


def _record(
    metrics: Any,
    started: float,
    first_token_at: float | None,
    usage: dict[str, Any] | None,
    *,
    streamed: bool,
) -> None:
    metrics.observe_latency("turn_duration", time.monotonic() - started)
    if first_token_at is not None:
        metrics.observe_latency("time_to_first_token", first_token_at - started)
    if usage:
        metrics.inc("prompt_tokens_total", usage.get("prompt_tokens", 0))
        metrics.inc("completion_tokens_total", usage.get("completion_tokens", 0))
    metrics.inc_label("responses_total", "stream" if streamed else "sync")
