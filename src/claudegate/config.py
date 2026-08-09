"""Runtime configuration.

Every knob is an environment variable prefixed ``CLAUDEGATE_`` and every knob
has a default that is safe on a laptop, so the zero-config path is::

    claudegate serve

Values are also read from a ``.env`` file in the working directory when present.
"""

from __future__ import annotations

import ipaddress
import os
import tempfile
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]
LogFormat = Literal["text", "json"]

#: Short names accepted in the ``model`` field, mapped to what the CLI expects.
#: The CLI resolves its own aliases too, so unknown values are passed through
#: untouched rather than rejected — a new model works the day it ships.
DEFAULT_MODEL_ALIASES: dict[str, str] = {
    "opus": "opus",
    "sonnet": "sonnet",
    "haiku": "haiku",
    "gpt-4o": "sonnet",
    "gpt-4o-mini": "haiku",
    "gpt-4": "opus",
    "gpt-3.5-turbo": "haiku",
}


class Settings(BaseSettings):
    """Server settings. See ``docs/CONFIGURATION.md`` for the annotated list."""

    model_config = SettingsConfigDict(
        env_prefix="CLAUDEGATE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── bind ──────────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8080

    # ── auth ──────────────────────────────────────────────────────────────
    api_key: str | None = None
    """Bearer token clients must present. Several may be given, comma separated."""

    require_auth: bool | None = None
    """``None`` (default) means: required unless we are bound to loopback."""

    # ── model ─────────────────────────────────────────────────────────────
    default_model: str = "sonnet"
    model_aliases: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_MODEL_ALIASES))
    fallback_model: str | None = None

    # ── agent behaviour ───────────────────────────────────────────────────
    bare_mode: bool = True
    """Present Claude as a plain model: no Claude Code identity, no built-in
    tools, and the request's system message becomes the entire system prompt.
    Turn off to get an autonomous coding agent with file and shell access."""

    workspace: str | None = None

    workspace_is_ephemeral: bool = False
    """Set internally when the workspace was created by us, not configured.

    Only then is it removed on shutdown — a directory the operator named is
    theirs, and deleting it would be a surprise.
    """
    """Working directory for the agent. Defaults to a private temp directory,
    which matters in bare mode where no filesystem access is expected anyway."""

    permission_mode: PermissionMode = "bypassPermissions"
    system_prompt_suffix: str | None = None
    max_turns: int | None = None
    cli_path: str | None = None
    claude_env: dict[str, str] = Field(default_factory=dict)

    # ── lifecycle ─────────────────────────────────────────────────────────
    reuse_sessions: bool = True
    """Keep the conversation alive between requests and send only the new
    messages. Saves the full-history re-send on every turn."""

    reuse_requires_user: bool = False
    """Only reuse a conversation when the request names an end user.

    ``proves_receipt`` asks the caller to hand back the answer they were given,
    which an attacker cannot guess -- unless the answer is predictable (a fixed
    greeting from a templated prompt). If you hand one API key to many end
    users, either set OpenAI's ``user`` field per user (recommended anyway) or
    turn this on, which refuses to reuse anything for requests that omit it.
    """

    rebuild_on_expiry: bool = True
    """If tool results arrive for a conversation we no longer hold, rebuild it
    from the request history instead of failing the turn."""

    forward_attachments: bool = True
    """Forward images and files as native content blocks."""

    session_idle_ttl_s: float = 1800.0
    tool_wait_ttl_s: float = 600.0
    request_timeout_s: float = 900.0
    first_event_timeout_s: float = 90.0
    """How long a turn may produce nothing at all before it is called a failure.

    Distinct from ``request_timeout_s``, which bounds a turn that is streaming.
    A CLI that connects, accepts the message and then stays silent is almost
    always an authentication problem, and 90s of silence is far past what a
    model needs to emit its first token.
    """

    cli_start_timeout_s: float = 120.0
    """How long to wait for a spawned CLI to finish its handshake. Without a
    bound, one wedged process would stall every request behind it."""

    max_request_bytes: int = 32 * 1024 * 1024
    deep_probe_interval_s: float = 30.0
    gc_interval_s: float = 30.0
    max_sessions: int = 64

    # ── observability ─────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: LogFormat = "text"
    metrics: bool = True
    request_log: bool = True

    @field_validator("api_key")
    @classmethod
    def _strip_key(cls, v: str | None) -> str | None:
        return v.strip() if v else None

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _defaults(self) -> Settings:
        # The CLI is spawned with this as its working directory. A path that
        # does not exist makes the spawn fail with an errno the user never
        # sees, so create it here instead of diagnosing it later.
        if self.workspace:
            path = self.workspace
            os.makedirs(path, mode=0o700, exist_ok=True)
        else:
            # Not a fixed name under /tmp. The agent runs with permission bypass,
            # so a predictable shared path lets any other user on the host
            # pre-create it (or symlink it) and own the agent's cwd.
            path = tempfile.mkdtemp(prefix="claudegate-")
            # Ours to create, ours to remove: a service that restarts often
            # would otherwise leave one of these behind every time.
            object.__setattr__(self, "workspace_is_ephemeral", True)
        object.__setattr__(self, "workspace", path)
        return self

    # ── derived ───────────────────────────────────────────────────────────
    @property
    def api_keys(self) -> tuple[str, ...]:
        if not self.api_key:
            return ()
        return tuple(k.strip() for k in self.api_key.split(",") if k.strip())

    @property
    def is_loopback(self) -> bool:
        try:
            return ipaddress.ip_address(self.host).is_loopback
        except ValueError:
            return self.host in {"localhost", ""}

    @property
    def auth_required(self) -> bool:
        """Auth is mandatory as soon as we are reachable from outside.

        The server drives a CLI with permission bypass, so an open port is
        remote code execution. Refusing to start without a key in that case is
        deliberate — see :func:`claudegate.app.create_app`.
        """
        if self.require_auth is not None:
            return self.require_auth
        return not self.is_loopback

    def resolve_model(self, name: str | None) -> str:
        """Map a requested model name onto a CLI model name."""
        if not name:
            return self.model_aliases.get(self.default_model, self.default_model)
        return self.model_aliases.get(name, name)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached, so tests that manipulate the environment must call
    ``get_settings.cache_clear()`` around the change.
    """
    return Settings()
