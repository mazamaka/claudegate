"""Conversation reuse, rebuilds and reaping."""

from __future__ import annotations

import asyncio
from typing import Any

from claudegate.testing import Turn, echo, scripted

from ..conftest import chat, gateway
from .test_tools import TOOLS


async def test_a_follow_up_turn_sends_only_the_new_message() -> None:
    async with gateway(scripted("one", "two")) as (client, harness):
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "first question"},
        ]
        first = await client.post("/v1/chat/completions", json=chat(messages=messages))
        assert first.headers["x-claudegate-mode"] == "fresh"

        messages.append({"role": "assistant", "content": first.json()["choices"][0]["message"]["content"]})
        messages.append({"role": "user", "content": "second question"})
        second = await client.post("/v1/chat/completions", json=chat(messages=messages))

        assert second.headers["x-claudegate-mode"] == "reused"
        assert second.headers["x-claudegate-session"] == first.headers["x-claudegate-session"]
        assert len(harness.transports) == 1

        opening, follow_up = (t["message"]["content"] for t in harness.cli.turns)

    opening_text = "".join(b.get("text", "") for b in opening)
    follow_up_text = "".join(b.get("text", "") for b in follow_up)
    assert "first question" in opening_text
    # The second turn is the new message and nothing else: no transcript, no
    # re-send of the first question, no echo of our own reply.
    assert follow_up_text == "second question"


async def test_editing_the_history_starts_a_new_conversation() -> None:
    async with gateway(scripted("a", "b")) as (client, harness):
        await client.post(
            "/v1/chat/completions", json=chat(messages=[{"role": "user", "content": "original"}])
        )
        response = await client.post(
            "/v1/chat/completions",
            json=chat(
                messages=[
                    {"role": "user", "content": "edited"},
                    {"role": "assistant", "content": "a"},
                    {"role": "user", "content": "next"},
                ]
            ),
        )

        assert response.headers["x-claudegate-mode"] == "fresh"
        assert len(harness.transports) == 2


async def test_changing_the_model_or_the_system_prompt_starts_a_new_conversation() -> None:
    async with gateway(scripted("a", "b", "c")) as (client, harness):
        base: list[dict[str, Any]] = [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hello"},
        ]
        await client.post("/v1/chat/completions", json=chat(messages=base))

        follow_up = [*base, {"role": "assistant", "content": "a"}, {"role": "user", "content": "more"}]
        other_model = await client.post(
            "/v1/chat/completions", json=chat(messages=follow_up, model="opus")
        )
        assert other_model.headers["x-claudegate-mode"] == "fresh"

        changed_prompt = [
            {"role": "system", "content": "Be verbose."},
            *follow_up[1:],
        ]
        other_prompt = await client.post(
            "/v1/chat/completions", json=chat(messages=changed_prompt)
        )
        assert other_prompt.headers["x-claudegate-mode"] == "fresh"
        assert len(harness.transports) == 3


async def test_reuse_can_be_switched_off() -> None:
    async with gateway(scripted("a", "b"), reuse_sessions=False) as (client, harness):
        messages: list[dict[str, Any]] = [{"role": "user", "content": "first"}]
        await client.post("/v1/chat/completions", json=chat(messages=messages))
        messages.append({"role": "assistant", "content": "a"})
        messages.append({"role": "user", "content": "second"})
        response = await client.post("/v1/chat/completions", json=chat(messages=messages))

        assert response.headers["x-claudegate-mode"] == "fresh"
        assert len(harness.transports) == 2


async def test_tool_results_for_a_lost_conversation_are_rebuilt_not_refused() -> None:
    async def scenario(turn: Turn) -> None:
        # A rebuilt conversation is handed the tool result as part of the
        # transcript, so the model can answer straight away.
        await turn.say(f"rebuilt: {'512' in turn.text}")
        await turn.end()

    async with gateway(scenario) as (client, harness):
        messages = [
            {"role": "user", "content": "db status?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_from_a_previous_life",
                        "type": "function",
                        "function": {"name": "lookup_status", "arguments": '{"subsystem":"db"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_from_a_previous_life", "content": "512"},
        ]
        response = await client.post("/v1/chat/completions", json=chat(tools=TOOLS, messages=messages))

        assert response.status_code == 200
        assert response.headers["x-claudegate-mode"] == "rebuilt"
        assert harness.manager.stats.rebuilt == 1
        assert response.json()["choices"][0]["message"]["content"] == "rebuilt: True"


async def test_rebuilding_can_be_turned_off_in_favour_of_a_409() -> None:
    async with gateway(echo(), rebuild_on_expiry=False) as (client, _):
        messages = [
            {"role": "user", "content": "db status?"},
            {"role": "tool", "tool_call_id": "call_gone", "content": "512"},
        ]
        response = await client.post("/v1/chat/completions", json=chat(messages=messages))

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conversation_expired"


async def test_old_conversations_are_evicted_when_the_pool_is_full() -> None:
    async with gateway(scripted("a"), max_sessions=2) as (client, harness):
        for i in range(4):
            response = await client.post(
                "/v1/chat/completions",
                json=chat(messages=[{"role": "user", "content": f"unrelated {i}"}]),
            )
            assert response.status_code == 200

        assert harness.manager.live <= 2
        assert harness.manager.stats.evicted >= 2


async def test_idle_conversations_are_reaped() -> None:
    async with gateway(scripted("a"), session_idle_ttl_s=0.0) as (client, harness):
        await client.post("/v1/chat/completions", json=chat())
        assert harness.manager.live == 1
        await harness.manager._reap()  # the GC loop's body, without the wait
        assert harness.manager.live == 0
        assert harness.manager.stats.expired == 1


async def test_a_timed_out_turn_retires_its_conversation() -> None:
    """Late events from an abandoned turn must not surface inside the next one."""

    async def stalls(turn: Turn) -> None:
        await asyncio.sleep(30)

    async with gateway(stalls, request_timeout_s=0.2) as (client, harness):
        response = await client.post("/v1/chat/completions", json=chat())
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "timeout"
        assert harness.manager.live == 0
        assert harness.cli.closed


async def test_shutdown_closes_every_conversation() -> None:
    async with gateway(scripted("a")) as (client, harness):
        await client.post("/v1/chat/completions", json=chat())
        transport = harness.cli
        assert not transport.closed

    assert transport.closed
