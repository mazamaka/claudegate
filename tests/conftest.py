"""Test harness.

Every test in ``tests/unit`` and ``tests/e2e`` runs against
:class:`claudegate.testing.FakeClaudeCLI` — no subprocess, no token, no network.
``tests/live`` is the opt-in counterpart that talks to the real thing.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from claudegate.app import create_app
from claudegate.config import Settings
from claudegate.testing import FakeClaudeCLI, Scenario, scripted

PNG_1PX = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6360000002000100ffff03000006000557bfabd400"
        "00000049454e44ae426082"
    )
).decode()
PNG_DATA_URL = f"data:image/png;base64,{PNG_1PX}"


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 8080,
        "api_key": None,
        "default_model": "sonnet",
        "gc_interval_s": 3600.0,
        "request_timeout_s": 10.0,
        "tool_wait_ttl_s": 5.0,
        "workspace": "/tmp/claudegate-tests",
        "request_log": False,
    }
    base.update(overrides)
    return Settings(**base)


class Harness:
    """An app wired to scripted fake CLIs, plus the transports it created."""

    def __init__(self, scenario: Scenario, **overrides: Any) -> None:
        self.scenario = scenario
        self.settings = make_settings(**overrides)
        self.transports: list[FakeClaudeCLI] = []
        self.app = create_app(self.settings, transport_factory=self._factory)

    def _factory(self) -> FakeClaudeCLI:
        cli = FakeClaudeCLI(self.scenario)
        self.transports.append(cli)
        return cli

    @property
    def cli(self) -> FakeClaudeCLI:
        assert self.transports, "no conversation was started"
        return self.transports[-1]

    @property
    def manager(self) -> Any:
        return self.app.state.manager


@asynccontextmanager
async def gateway(scenario: Scenario | None = None, **overrides: Any) -> AsyncIterator[
    tuple[httpx.AsyncClient, Harness]
]:
    harness = Harness(scenario or scripted("hello"), **overrides)
    async with harness.app.router.lifespan_context(harness.app):
        transport = httpx.ASGITransport(app=harness.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gate", timeout=30) as c:
            yield c, harness


@pytest.fixture
def gateway_factory():  # type: ignore[no-untyped-def]
    return gateway


def chat(**kwargs: Any) -> dict[str, Any]:
    """A minimal valid request body."""
    body: dict[str, Any] = {
        "model": "sonnet",
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(kwargs)
    return body


async def sse_frames(response: httpx.Response) -> list[str]:
    frames: list[str] = []
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            frames.append(line[6:])
    return frames
