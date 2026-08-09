"""Models, health and metrics.

``/health`` deserves a word. A liveness probe that only proves the HTTP server
is running stays green through the two failures that actually take this service
down — an expired token and a CLI that cannot spawn — so ``?deep=1`` spends one
real completion and reports what came back.
"""

from __future__ import annotations

import shutil
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from ..config import Settings
from ..openai_api.schema import ModelCard, ModelList

router = APIRouter()
#: Health and metrics live at the root as well as under ``/v1``, because that
#: is where probes and scrapers look for them.
ops = APIRouter()

#: Advertised model names. The CLI resolves its own aliases, so anything it
#: accepts works whether or not it is listed here.
CATALOG = ("opus", "sonnet", "haiku")


@router.get("/models")
async def list_models(request: Request) -> ModelList:
    settings: Settings = request.app.state.settings
    created = int(time.time())
    names = list(dict.fromkeys([settings.default_model, *CATALOG, *settings.model_aliases]))
    return ModelList(data=[ModelCard(id=n, created=created) for n in names])


@router.get("/models/{model_id}")
async def get_model(model_id: str) -> ModelCard:
    return ModelCard(id=model_id, created=int(time.time()))


def _health_payload(request: Request) -> dict[str, Any]:
    manager = request.app.state.manager
    settings: Settings = request.app.state.settings
    return {
        "status": "ok",
        "version": request.app.version,
        "sessions": manager.live,
        "max_sessions": settings.max_sessions,
        "bare_mode": settings.bare_mode,
        "cli": settings.cli_path or shutil.which("claude") or "not found on PATH",
    }


@ops.get("/health", include_in_schema=False)
@ops.get("/healthz", include_in_schema=False)
async def health(request: Request, deep: int = 0) -> JSONResponse:
    payload = _health_payload(request)
    if not deep:
        return JSONResponse(payload)

    from ..openai_api.schema import ChatCompletionRequest, Message

    probe = ChatCompletionRequest(
        model=request.app.state.settings.default_model,
        messages=[
            Message(role="system", content="Reply with the single word: ok"),
            Message(role="user", content="ping"),
        ],
    )
    started = time.monotonic()
    try:
        lease = await request.app.state.manager.acquire(probe)
        async with lease:
            await lease.dispatch()
            reply: list[str] = []
            async for event in lease.session.stream_turn(timeout=120):
                text = getattr(event, "text", None)
                if text:
                    reply.append(text)
                if type(event).__name__ == "TurnFailed":
                    raise RuntimeError(getattr(event, "message", "turn failed"))
        payload["probe"] = {
            "ok": True,
            "reply": "".join(reply).strip()[:120],
            "seconds": round(time.monotonic() - started, 2),
        }
        return JSONResponse(payload)
    except Exception as exc:
        payload["status"] = "degraded"
        payload["probe"] = {"ok": False, "error": str(exc)[:400]}
        return JSONResponse(payload, status_code=503)


@ops.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> PlainTextResponse:
    settings: Settings = request.app.state.settings
    if not settings.metrics:
        return PlainTextResponse("metrics are disabled\n", status_code=404)
    manager = request.app.state.manager
    extra = {"sessions_live": manager.live, **manager.stats.as_dict()}
    return PlainTextResponse(
        request.app.state.metrics.prometheus(extra), media_type="text/plain; version=0.0.4"
    )
