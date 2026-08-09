"""claudegate — an OpenAI-compatible API server for the Claude Code CLI."""

from __future__ import annotations

__all__ = ["__version__", "create_app"]
__version__ = "0.1.1"


def create_app(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
    """Lazy re-export so ``import claudegate`` stays cheap."""
    from .app import create_app as _factory

    return _factory(*args, **kwargs)  # type: ignore[arg-type]
