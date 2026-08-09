"""The server, end to end, against a scripted CLI.

These exercise the real routes, the real bridge and the real translation layer.
Only the subprocess is fake — so a passing run means the wire format a client
sees is the wire format asserted here.
"""

from __future__ import annotations

import json

import pytest

from claudegate.testing import Turn, scripted

from ..conftest import PNG_DATA_URL, chat, gateway, sse_frames


async def test_a_plain_completion_has_the_shape_openai_clients_expect() -> None:
    async with gateway(scripted("Paris")) as (client, _harness):
        response = await client.post("/v1/chat/completions", json=chat())
        assert response.status_code == 200
        body = response.json()

    assert body["object"] == "chat.completion"
    assert body["model"] == "sonnet"
    choice = body["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "Paris"}
    assert choice["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 42
    assert body["usage"]["completion_tokens"] == 7
    assert body["usage"]["total_tokens"] == 49


async def test_the_conversation_receives_the_history_and_the_system_prompt() -> None:
    async with gateway(scripted("ok")) as (client, harness):
        await client.post(
            "/v1/chat/completions",
            json=chat(
                messages=[
                    {"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "second"},
                    {"role": "user", "content": "third"},
                ]
            ),
        )
        sent = harness.cli.turns[0]["message"]["content"]

    text = "".join(b.get("text", "") for b in sent)
    assert "first" in text
    assert "second" in text
    assert "third" in text
    # The system message is configuration, not conversation.
    assert "You are terse." not in text


async def test_streaming_emits_a_role_frame_content_frames_and_done() -> None:
    async with (
        gateway(scripted("hello world")) as (client, _),
        client.stream("POST", "/v1/chat/completions", json=chat(stream=True)) as r,
    ):
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        frames = await sse_frames(r)

    assert frames[-1] == "[DONE]"
    payloads = [json.loads(f) for f in frames[:-1]]
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    content = "".join(p["choices"][0]["delta"].get("content", "") for p in payloads)
    assert content == "hello world"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert sum(1 for p in payloads if p["choices"][0]["delta"].get("content")) > 1


async def test_usage_is_only_sent_when_the_client_asks_for_it() -> None:
    async with gateway(scripted("hi")) as (client, _):
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json=chat(stream=True, stream_options={"include_usage": True}),
        ) as r:
            frames = await sse_frames(r)
        with_usage = [json.loads(f) for f in frames[:-1] if "usage" in json.loads(f)]

        async with client.stream("POST", "/v1/chat/completions", json=chat(stream=True)) as r:
            plain = [json.loads(f) for f in (await sse_frames(r))[:-1]]

    assert with_usage
    assert with_usage[-1]["choices"] == []
    assert with_usage[-1]["usage"]["total_tokens"] == 49
    assert not any("usage" in p for p in plain)


async def test_reasoning_is_reported_separately_from_the_answer() -> None:
    async def scenario(turn: Turn) -> None:
        await turn.think("weighing it up")
        await turn.say("42")
        await turn.end()

    async with gateway(scenario) as (client, _):
        body = (await client.post("/v1/chat/completions", json=chat())).json()

    message = body["choices"][0]["message"]
    assert message["content"] == "42"
    assert message["reasoning_content"] == "weighing it up"


async def test_an_image_reaches_the_model_as_a_native_block() -> None:
    async with gateway(scripted("a cat")) as (client, harness):
        response = await client.post(
            "/v1/chat/completions",
            json=chat(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
                            {"type": "text", "text": "what is this?"},
                        ],
                    }
                ]
            ),
        )
        assert response.status_code == 200
        images = harness.cli.turns[0]["message"]["content"]

    image_blocks = [b for b in images if b.get("type") == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["source"]["media_type"] == "image/png"


async def test_an_image_from_an_earlier_turn_is_still_visible_later() -> None:
    async with gateway(scripted("ok", "still there")) as (client, harness):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
                    {"type": "text", "text": "look"},
                ],
            },
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "what was in the image?"},
        ]
        # A fresh conversation each time, so the fold is what has to carry it.
        await client.post("/v1/chat/completions", json=chat(messages=messages, model="opus"))
        sent = harness.cli.turns[0]["message"]["content"]

    assert any(b.get("type") == "image" for b in sent)


async def test_a_failing_turn_becomes_an_openai_error_envelope() -> None:
    async def scenario(turn: Turn) -> None:
        await turn.fail("Claude Code exploded")

    async with gateway(scenario) as (client, _):
        response = await client.post("/v1/chat/completions", json=chat())

    assert response.status_code == 502
    error = response.json()["error"]
    assert "exploded" in error["message"]
    assert error["type"] == "server_error"


async def test_a_failure_mid_stream_is_delivered_as_an_error_frame_then_done() -> None:
    async def scenario(turn: Turn) -> None:
        await turn.say("starting")
        await turn.fail("upstream died")

    async with (
        gateway(scenario) as (client, _),
        client.stream("POST", "/v1/chat/completions", json=chat(stream=True)) as r,
    ):
        frames = await sse_frames(r)

    assert frames[-1] == "[DONE]"
    assert any("error" in json.loads(f) for f in frames[:-1])


async def test_rate_limits_are_reported_as_429() -> None:
    async def scenario(turn: Turn) -> None:
        await turn.fail("rate limit reached", api_status=429)

    async with gateway(scenario) as (client, _):
        response = await client.post("/v1/chat/completions", json=chat())

    assert response.status_code == 429


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({"messages": []}, "at least one message"),
        ({"messages": [{"role": "user", "content": "x"}], "n": 3}, "Malformed request"),
    ],
)
async def test_bad_requests_are_rejected_with_a_reason(payload: dict, fragment: str) -> None:
    async with gateway() as (client, _):
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert fragment in response.json()["error"]["message"]


async def test_an_api_key_is_enforced_when_one_is_configured() -> None:
    async with gateway(scripted("hi"), api_key="s3cret") as (client, _):
        assert (await client.post("/v1/chat/completions", json=chat())).status_code == 401
        bad = {"authorization": "Bearer nope"}
        assert (await client.post("/v1/chat/completions", json=chat(), headers=bad)).status_code == 401
        good = {"authorization": "Bearer s3cret"}
        assert (await client.post("/v1/chat/completions", json=chat(), headers=good)).status_code == 200
        assert (await client.get("/health")).status_code == 200


async def test_models_health_and_metrics_are_served() -> None:
    async with gateway(scripted("hi")) as (client, _):
        models = await client.get("/v1/models")
        assert models.status_code == 200
        assert {m["id"] for m in models.json()["data"]} >= {"opus", "sonnet", "haiku"}

        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        await client.post("/v1/chat/completions", json=chat())
        metrics = await client.get("/metrics")

    assert "claudegate_requests_total 1" in metrics.text
    assert "claudegate_sessions_created 1" in metrics.text
