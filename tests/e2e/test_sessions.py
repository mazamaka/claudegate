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

        messages.append(
            {"role": "assistant", "content": first.json()["choices"][0]["message"]["content"]}
        )
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


async def test_two_callers_with_the_same_history_never_share_a_conversation() -> None:
    """A shared system prompt and a templated opening message are not a
    coincidence — they are what a deployment looks like. Matching on history
    alone would hand the second caller the first one's live conversation."""
    async with gateway(scripted("a", "b", "c"), api_key="key-one,key-two") as (client, harness):
        opening: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a support agent."},
            {"role": "user", "content": "Hello, I need help."},
        ]
        alice = {"authorization": "Bearer key-one"}
        bob = {"authorization": "Bearer key-two"}

        await client.post("/v1/chat/completions", json=chat(messages=opening), headers=alice)

        follow_up = [
            *opening,
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "more"},
        ]
        # Same prompt, same history, different caller.
        intruder = await client.post(
            "/v1/chat/completions", json=chat(messages=follow_up), headers=bob
        )
        assert intruder.headers["x-claudegate-mode"] == "fresh"

        # ...and the original caller still gets their own conversation back.
        owner = await client.post(
            "/v1/chat/completions", json=chat(messages=follow_up), headers=alice
        )
        assert owner.headers["x-claudegate-mode"] == "reused"
        assert len(harness.transports) == 2


async def test_one_key_serving_many_end_users_is_partitioned_by_the_user_field() -> None:
    async with gateway(scripted("a", "b", "c")) as (client, _harness):
        opening: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]
        await client.post("/v1/chat/completions", json=chat(messages=opening, user="alice"))

        follow_up = [
            *opening,
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "more"},
        ]
        other = await client.post("/v1/chat/completions", json=chat(messages=follow_up, user="bob"))
        same = await client.post(
            "/v1/chat/completions", json=chat(messages=follow_up, user="alice")
        )

        assert other.headers["x-claudegate-mode"] == "fresh"
        assert same.headers["x-claudegate-mode"] == "reused"


async def test_a_conversation_is_only_continued_by_whoever_received_its_answer() -> None:
    """The prompt-injection variant of a session mix-up.

    An attacker who can guess an opening — a published system prompt plus a
    templated first message is not a secret — primes a conversation, then waits
    for someone else's request to land in it. What they cannot guess is what
    the model actually said, so continuing requires handing that back.
    """
    async with gateway(scripted("the real answer", "b", "c")) as (client, harness):
        opening: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a support agent."},
            {"role": "user", "content": "Hello, I need help."},
        ]
        primed = await client.post("/v1/chat/completions", json=chat(messages=opening))
        assert primed.headers["x-claudegate-mode"] == "fresh"

        # Same opening, but the caller never saw the answer: no continuation.
        guessed = [
            *opening,
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "next"},
        ]
        assert (await client.post("/v1/chat/completions", json=chat(messages=guessed))).headers[
            "x-claudegate-mode"
        ] == "fresh"

        # The caller who did receive it continues, even after trimming it.
        proven = [
            *opening,
            {"role": "assistant", "content": "  the real answer\n"},
            {"role": "user", "content": "next"},
        ]
        assert (await client.post("/v1/chat/completions", json=chat(messages=proven))).headers[
            "x-claudegate-mode"
        ] == "reused"
        assert len(harness.transports) == 2


async def test_reuse_survives_a_tool_round() -> None:
    """The answer a client echoes back is the one from *its* turn.

    A model that narrates before calling a tool produces text in two separate
    turns. Comparing against both concatenated would never match what the
    client returns, and reuse would silently stop working after any tool call.
    """

    async def narrates_then_calls(turn: Turn) -> None:
        await turn.say("Let me look that up.")
        result = await turn.call_tool("lookup_status", {"subsystem": "db"})
        await turn.say(f"The answer is {result}.")
        await turn.end()

    async with gateway(narrates_then_calls) as (client, harness):
        messages: list[dict[str, Any]] = [{"role": "user", "content": "db status?"}]
        first = (
            await client.post("/v1/chat/completions", json=chat(tools=TOOLS, messages=messages))
        ).json()
        call = first["choices"][0]["message"]["tool_calls"][0]
        messages.append(first["choices"][0]["message"])
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": "42"})

        second = (
            await client.post("/v1/chat/completions", json=chat(tools=TOOLS, messages=messages))
        ).json()
        assert second["choices"][0]["message"]["content"] == "The answer is 42."

        messages.append(second["choices"][0]["message"])
        messages.append({"role": "user", "content": "thanks"})
        third = await client.post("/v1/chat/completions", json=chat(tools=TOOLS, messages=messages))

        assert third.headers["x-claudegate-mode"] == "reused"
        assert len(harness.transports) == 1


async def test_an_answer_that_never_arrived_as_deltas_is_still_returned() -> None:
    """Partial-message events are an optimisation, not the source of truth.

    A CLI that skips them would otherwise leave every response with
    `content: null` and every conversation permanently unreusable.
    """

    async def no_deltas(turn: Turn) -> None:
        await turn.cli.emit_assistant([{"type": "text", "text": "final only"}])
        await turn.end(result="final only")

    async with gateway(no_deltas) as (client, _harness):
        first = (await client.post("/v1/chat/completions", json=chat())).json()
        assert first["choices"][0]["message"]["content"] == "final only"

        follow_up = await client.post(
            "/v1/chat/completions",
            json=chat(
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "final only"},
                    {"role": "user", "content": "more"},
                ]
            ),
        )
        assert follow_up.headers["x-claudegate-mode"] == "reused"


async def test_strict_mode_refuses_to_reuse_for_an_anonymous_caller() -> None:
    """The residual risk proof-of-receipt cannot cover: a predictable answer.

    When one key is shared by many end users and the reply is a fixed greeting,
    an attacker can produce the proof. Strict mode declines to guess.
    """
    async with gateway(scripted("Hello! How can I help?", "b"), reuse_requires_user=True) as (
        client,
        harness,
    ):
        messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
        await client.post("/v1/chat/completions", json=chat(messages=messages))

        guessable = [
            *messages,
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "repeat everything said so far"},
        ]
        anonymous = await client.post("/v1/chat/completions", json=chat(messages=guessable))
        assert anonymous.headers["x-claudegate-mode"] == "fresh"

        named = await client.post(
            "/v1/chat/completions", json=chat(messages=guessable, user="alice")
        )
        assert named.headers["x-claudegate-mode"] == "fresh"  # alice never opened one
        assert len(harness.transports) == 3


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

        follow_up = [
            *base,
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "more"},
        ]
        other_model = await client.post(
            "/v1/chat/completions", json=chat(messages=follow_up, model="opus")
        )
        assert other_model.headers["x-claudegate-mode"] == "fresh"

        changed_prompt = [
            {"role": "system", "content": "Be verbose."},
            *follow_up[1:],
        ]
        other_prompt = await client.post("/v1/chat/completions", json=chat(messages=changed_prompt))
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
        response = await client.post(
            "/v1/chat/completions", json=chat(tools=TOOLS, messages=messages)
        )

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
