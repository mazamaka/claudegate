"""Bearer authentication.

The server drives a CLI with permission bypass, so reaching the port is close
enough to running code on the host. Two consequences are baked in here:

* keys are compared in constant time, and
* binding a non-loopback address without a key is refused at startup rather
  than served insecurely (see :func:`claudegate.app.create_app`).
"""

from __future__ import annotations

import hashlib
import hmac

from fastapi import Request

from .config import Settings
from .errors import AuthenticationError

#: Paths that never require a key: they expose no model access and are what a
#: load balancer or a human debugging a deployment reaches for first.
PUBLIC_PATHS = frozenset({"/health", "/healthz", "/metrics", "/", "/docs", "/openapi.json"})


def extract_key(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if header:
        scheme, _, value = header.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
        if not _ and header.strip():
            return header.strip()
    api_key = request.headers.get("x-api-key")
    return api_key.strip() if api_key else None


def tenant_id(request: Request, settings: Settings, end_user: str | None = None) -> str:
    """A stable identifier for whoever is calling, for conversation isolation.

    Live conversations are matched by hashing history, and two callers can
    legitimately share a prefix — the same system prompt and the same opening
    message is not a coincidence, it is a template. Without this, the second
    caller would be handed the first one's conversation.

    Derived from the presented key (hashed, never logged or returned) and
    OpenAI's ``user`` field, so a single shared key serving many end users is
    still partitioned.
    """
    presented = extract_key(request) or ""
    digest = hashlib.sha256(presented.encode()).hexdigest()[:16] if presented else "anonymous"
    return f"{digest}/{end_user or ''}"


def check_request(request: Request, settings: Settings) -> None:
    """Raise :class:`AuthenticationError` unless the request may proceed."""
    if request.url.path in PUBLIC_PATHS:
        return
    keys = settings.api_keys
    if not keys:
        if settings.auth_required:
            raise AuthenticationError("This server requires an API key, but none is configured.")
        return
    presented = extract_key(request)
    if not presented:
        raise AuthenticationError("Missing bearer token.")
    if not any(hmac.compare_digest(presented, k) for k in keys):
        raise AuthenticationError()
