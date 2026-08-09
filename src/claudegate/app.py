"""Application factory."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from . import __version__
from .bridge.manager import SessionManager, TransportFactory
from .config import Settings, get_settings
from .errors import (
    GatewayError,
    InvalidRequest,
    gateway_error_handler,
    unhandled_error_handler,
)
from .observability import Metrics, configure_logging
from .routes import chat, meta
from .security import check_request

log = logging.getLogger("claudegate.app")

DESCRIPTION = """
An OpenAI-compatible API in front of the Claude Code CLI.

Point any OpenAI client at this server: streaming, function calling, and image
input all work. Conversations are held open between requests, so a follow-up
turn costs one message rather than the whole history.
"""


def create_app(
    settings: Settings | None = None,
    *,
    transport_factory: TransportFactory | None = None,
) -> FastAPI:
    """Build the ASGI app.

    ``transport_factory`` swaps out the CLI subprocess, which is what lets the
    whole server be tested end to end without a CLI, a token, or a network.
    """
    settings = settings or get_settings()
    configure_logging(settings)

    if settings.auth_required and not settings.api_keys:
        raise SystemExit(
            f"Refusing to start: bound to {settings.host}, which is reachable from "
            "outside, with no CLAUDEGATE_API_KEY set. This server can run code on "
            "this host. Set a key (openssl rand -hex 32), or bind 127.0.0.1, or "
            "set CLAUDEGATE_REQUIRE_AUTH=false if you really mean it."
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        manager = SessionManager(settings, transport_factory=transport_factory)
        await manager.start()
        app.state.settings = settings
        app.state.manager = manager
        app.state.metrics = Metrics()
        log.info(
            "claudegate %s ready on %s:%s (model=%s, bare_mode=%s, auth=%s)",
            __version__,
            settings.host,
            settings.port,
            settings.default_model,
            settings.bare_mode,
            "on" if settings.api_keys else "off",
        )
        try:
            yield
        finally:
            await manager.aclose()

    app = FastAPI(
        title="claudegate",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        docs_url="/docs",
    )
    app.state.settings = settings

    @app.middleware("http")
    async def _gate(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.monotonic()
        try:
            check_request(request, settings)
        except GatewayError as exc:
            return exc.to_response()
        response = await call_next(request)
        if settings.request_log and request.url.path.startswith("/v1"):
            log.info(
                "%s %s → %s in %.2fs",
                request.method,
                request.url.path,
                response.status_code,
                time.monotonic() - started,
            )
        return response

    app.add_exception_handler(GatewayError, gateway_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return InvalidRequest(f"Malformed request: {exc.errors()}").to_response()

    app.include_router(chat.router, prefix="/v1", tags=["chat"])
    app.include_router(meta.router, prefix="/v1", tags=["models"])
    app.include_router(meta.ops)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "name": "claudegate",
            "version": __version__,
            "docs": "/docs",
            "endpoint": "/v1",
        }

    return app
