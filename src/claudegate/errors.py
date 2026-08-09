"""Error envelope.

Clients written against OpenAI parse ``{"error": {...}}`` and branch on
``type``/``code``. Anything else — a bare FastAPI ``{"detail": ...}``, an HTML
500 page — reaches the user as an unhelpful "unknown error" from deep inside
their SDK, so every failure path here is funnelled through :class:`GatewayError`.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


class GatewayError(Exception):
    """An error that is safe to show a client, in OpenAI's shape."""

    status_code = 500
    error_type = "server_error"
    code: str | None = None

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        code: str | None = None,
        status_code: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.param = param
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.headers = headers or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }

    def to_response(self) -> JSONResponse:
        return JSONResponse(self.to_dict(), status_code=self.status_code, headers=self.headers)


class InvalidRequest(GatewayError):
    status_code = 400
    error_type = "invalid_request_error"


class AuthenticationError(GatewayError):
    status_code = 401
    error_type = "invalid_request_error"
    code = "invalid_api_key"

    def __init__(self, message: str = "Invalid or missing API key.") -> None:
        super().__init__(message, headers={"WWW-Authenticate": "Bearer"})


class ModelNotFound(InvalidRequest):
    code = "model_not_found"
    status_code = 404


class OverloadedError(GatewayError):
    """Every session slot is busy; the client should retry."""

    status_code = 429
    error_type = "server_error"
    code = "server_overloaded"

    def __init__(self, message: str, retry_after: int = 5) -> None:
        super().__init__(message, headers={"Retry-After": str(retry_after)})


class UpstreamError(GatewayError):
    """The CLI itself failed: not spawnable, unauthenticated, rate limited."""

    status_code = 502
    error_type = "server_error"
    code = "upstream_error"


class ConversationExpired(GatewayError):
    """Tool results arrived for a conversation the server no longer holds.

    Only ever reaches a client when ``rebuild_on_expiry`` is disabled; the
    default is to silently rebuild the conversation from the request history.
    """

    status_code = 409
    error_type = "invalid_request_error"
    code = "conversation_expired"


class RequestTimeout(GatewayError):
    status_code = 504
    error_type = "server_error"
    code = "timeout"


def for_status(status: int, message: str) -> GatewayError:
    """Pick the error class that carries the right ``code`` for a status.

    Turn failures arrive from the bridge as a status and a message; clients
    branch on ``error.code``, so the specific class matters more than the
    number.
    """
    if status == 504:
        return RequestTimeout(message)
    if status == 429:
        return OverloadedError(message)
    if status == 409:
        return ConversationExpired(message)
    return UpstreamError(message)


async def gateway_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, GatewayError)
    return exc.to_response()


async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    return GatewayError(f"Internal server error: {exc}").to_response()
