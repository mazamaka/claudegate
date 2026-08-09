"""Opt-in checks against the real Claude Code CLI.

    CLAUDEGATE_LIVE_TESTS=1 pytest tests/live

These spend real tokens and take real seconds, so they are skipped by default.
Everything they cover is also covered hermetically in ``tests/e2e``; what they
add is proof that the *contract* those fakes encode still matches the CLI.
"""

from __future__ import annotations

import os
import random

import pytest

from claudegate.app import create_app
from claudegate.smoke import data_url, render_code_png

from ..conftest import chat, make_settings

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("CLAUDEGATE_LIVE_TESTS") != "1",
        reason="set CLAUDEGATE_LIVE_TESTS=1 to run against the real CLI",
    ),
]


@pytest.fixture
async def live_client():  # type: ignore[no-untyped-def]
    import httpx

    settings = make_settings(request_timeout_s=180.0, default_model="sonnet")
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gate", timeout=180) as c:
            yield c


async def test_a_real_turn_completes(live_client) -> None:  # type: ignore[no-untyped-def]
    response = await live_client.post(
        "/v1/chat/completions",
        json=chat(
            messages=[
                {"role": "system", "content": "Reply with exactly: PONG"},
                {"role": "user", "content": "ping"},
            ]
        ),
    )
    assert response.status_code == 200
    assert "PONG" in response.json()["choices"][0]["message"]["content"].upper()


async def test_the_model_really_reads_the_image(live_client) -> None:  # type: ignore[no-untyped-def]
    code = "".join(str(random.randint(0, 9)) for _ in range(4))
    response = await live_client.post(
        "/v1/chat/completions",
        json=chat(
            messages=[
                {"role": "system", "content": "Answer with digits only."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url(render_code_png(code))},
                        },
                        {"type": "text", "text": "What 4-digit number is this? Digits only."},
                    ],
                },
            ]
        ),
    )
    assert response.status_code == 200
    assert code in response.json()["choices"][0]["message"]["content"]


async def test_a_real_tool_loop_is_resumed_not_replayed(live_client) -> None:  # type: ignore[no-untyped-def]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_status",
                "description": "Look up a subsystem's status code.",
                "parameters": {
                    "type": "object",
                    "properties": {"subsystem": {"type": "string"}},
                    "required": ["subsystem"],
                },
            },
        }
    ]
    code = str(random.randint(100, 999))
    messages = [
        {"role": "system", "content": "Use the tools you are given. Be terse."},
        {"role": "user", "content": "What is the status code of the db subsystem?"},
    ]
    first = await live_client.post(
        "/v1/chat/completions", json=chat(messages=messages, tools=tools)
    )
    assert first.status_code == 200
    choice = first.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]

    messages.append(choice["message"])
    messages.append({"role": "tool", "tool_call_id": call["id"], "content": code})
    second = await live_client.post(
        "/v1/chat/completions", json=chat(messages=messages, tools=tools)
    )

    assert second.status_code == 200
    assert second.headers["x-claudegate-mode"] == "continued"
    assert code in second.json()["choices"][0]["message"]["content"]


async def test_a_follow_up_reuses_the_conversation(live_client) -> None:  # type: ignore[no-untyped-def]
    token = str(random.randint(1000, 9999))
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": f"Remember the number {token}. Reply OK."},
    ]
    first = await live_client.post("/v1/chat/completions", json=chat(messages=messages))
    assert first.status_code == 200

    messages.append(
        {"role": "assistant", "content": first.json()["choices"][0]["message"]["content"]}
    )
    messages.append({"role": "user", "content": "What number did I ask you to remember?"})
    second = await live_client.post("/v1/chat/completions", json=chat(messages=messages))

    assert second.status_code == 200
    assert second.headers["x-claudegate-mode"] == "reused"
    assert token in second.json()["choices"][0]["message"]["content"]
