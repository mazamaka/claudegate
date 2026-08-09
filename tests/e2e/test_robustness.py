"""Failure modes a deterministic, instantaneous fake CLI cannot show you.

Every test here was written against a defect that the rest of the suite was
blind to, because the fake answered immediately and always in the same order.
The fake now takes ``connect_delay`` and ``eager_tools`` for exactly this
reason: the two worst bugs found in review were both timing-dependent.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from claudegate.testing import Turn, scripted

from ..conftest import chat, gateway, sse_frames
from .test_tools import TOOLS


async def test_starting_a_conversation_does_not_block_other_requests() -> None:
    """The registry lock must not be held across a CLI spawn.

    Spawning runs a process and a handshake. Holding the lock through it makes
    one slow start stall every other request, the reaper, and shutdown — a
    single wedged CLI takes the whole server down.
    """
    spawn_s = 0.6

    async with gateway(scripted("hi"), cli_kwargs={"connect_delay": spawn_s}) as (client, _h):
        # Warm one conversation up so the second call has something to reuse.
        first = await client.post("/v1/chat/completions", json=chat(user="alice"))
        reply = first.json()["choices"][0]["message"]["content"]
        reusable = chat(
            user="alice",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": reply},
                {"role": "user", "content": "again"},
            ],
        )

        async def slow_spawn() -> None:
            await client.post("/v1/chat/completions", json=chat(user="bob"))

        spawning = asyncio.create_task(slow_spawn())
        await asyncio.sleep(0.05)  # let it get as far as the spawn

        started = time.monotonic()
        reused = await client.post("/v1/chat/completions", json=reusable)
        waited = time.monotonic() - started
        await spawning

    assert reused.headers["x-claudegate-mode"] == "reused"
    assert waited < spawn_s / 2, f"an unrelated request waited {waited:.2f}s for a spawn"


async def test_a_cli_that_never_becomes_ready_is_given_up_on() -> None:
    async with gateway(
        scripted("hi"), cli_kwargs={"connect_delay": 5.0}, cli_start_timeout_s=0.2
    ) as (client, harness):
        response = await client.post("/v1/chat/completions", json=chat())

        assert response.status_code == 502
        assert "did not become ready" in response.json()["error"]["message"]
        assert harness.manager.live == 0


async def test_a_second_round_of_tool_calls_still_gets_its_results() -> None:
    """The readiness latch used to stay set after the first round.

    A round-two handler that arrived before its assistant message then skipped
    the grace wait, found an empty registry, and the model was handed
    "could not be matched" in place of the client's actual result.
    """

    async def two_rounds(turn: Turn) -> None:
        one = await turn.call_tool("lookup_status", {"subsystem": "db"})
        two = await turn.call_tool("lookup_status", {"subsystem": "cache"})
        await turn.say(f"db={one} cache={two}")
        await turn.end()

    async with gateway(two_rounds, cli_kwargs={"eager_tools": True}) as (client, _h):
        messages: list[dict[str, Any]] = [{"role": "user", "content": "both statuses?"}]
        for value in ("111", "222"):
            response = await client.post(
                "/v1/chat/completions", json=chat(tools=TOOLS, messages=messages)
            )
            assert response.status_code == 200
            choice = response.json()["choices"][0]
            if choice["finish_reason"] != "tool_calls":
                break
            call = choice["message"]["tool_calls"][0]
            messages.append(choice["message"])
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": value})

        final = await client.post("/v1/chat/completions", json=chat(tools=TOOLS, messages=messages))
        answer = final.json()["choices"][0]["message"]["content"] or ""

    assert answer == "db=111 cache=222", f"a tool result was lost: {answer!r}"


async def test_a_failed_stream_retires_its_conversation() -> None:
    """Returning early from inside the lease looked like success to __aexit__,
    so a conversation that had just failed went back into the pool."""

    async def fails(turn: Turn) -> None:
        await turn.say("starting")
        await turn.fail("upstream died")

    async with gateway(fails) as (client, harness):
        async with client.stream("POST", "/v1/chat/completions", json=chat(stream=True)) as r:
            frames = await sse_frames(r)

        assert frames[-1] == "[DONE]"
        assert harness.manager.live == 0, "a broken conversation stayed available for reuse"


async def test_a_finished_run_releases_a_handler_still_waiting_on_it() -> None:
    """A future dropped while its handler is parked leaves that handler waiting
    out the full tool TTL, holding a subprocess long after the conversation is
    gone."""

    async def calls_then_finishes_anyway(turn: Turn) -> None:
        pending = asyncio.ensure_future(turn.call_tool("lookup_status", {"subsystem": "db"}))
        await asyncio.sleep(0.05)
        # The run ends without the tool ever being answered.
        await turn.end(result="never mind")
        turn.cli.late_result = await pending  # type: ignore[attr-defined]

    async with gateway(calls_then_finishes_anyway, tool_wait_ttl_s=30.0) as (client, harness):
        await client.post("/v1/chat/completions", json=chat(tools=TOOLS))
        # The handler must be released promptly, not after tool_wait_ttl_s.
        for _ in range(50):
            if getattr(harness.cli, "late_result", None) is not None:
                break
            await asyncio.sleep(0.02)

        released = getattr(harness.cli, "late_result", None)

    assert released is not None, "the parked handler was orphaned"
    assert "the run ended" in released


async def test_a_silent_cli_is_reported_as_credentials_not_as_slowness() -> None:
    """The failure mode a dead token actually produces.

    The CLI starts, accepts the message, and then emits nothing at all. Calling
    that "timed out waiting for the model" sends people to look at their
    network; it is nearly always authentication.
    """

    async def says_nothing(turn: Turn) -> None:
        await asyncio.sleep(30)  # far longer than the deadline under test

    async with gateway(says_nothing, first_event_timeout_s=0.3, request_timeout_s=30.0) as (
        client,
        harness,
    ):
        response = await client.post("/v1/chat/completions", json=chat())

    assert response.status_code == 502
    message = response.json()["error"]["message"]
    assert "authentication" in message
    assert "doctor" in message
    assert harness.manager.live == 0, "a conversation that never spoke was kept for reuse"


async def test_a_turn_that_started_streaming_is_not_blamed_on_credentials() -> None:
    """Once output is flowing, silence is slowness, and the deadline that
    applies is the request timeout."""

    async def starts_then_stalls(turn: Turn) -> None:
        await turn.say("thinking")
        await asyncio.sleep(30)

    async with gateway(starts_then_stalls, first_event_timeout_s=0.3, request_timeout_s=1.0) as (
        client,
        _harness,
    ):
        response = await client.post("/v1/chat/completions", json=chat())

    assert response.status_code == 504
    assert "Timed out" in response.json()["error"]["message"]


async def test_a_non_ascii_api_key_is_rejected_not_crashed() -> None:
    """Constant-time comparison on str raises on non-ASCII, which turned any
    client with an umlaut in its key into an unauthenticated 500."""
    async with gateway(scripted("hi"), api_key="s3cret") as (client, _h):
        response = await client.post(
            "/v1/chat/completions",
            json=chat(),
            # Sent as latin-1 bytes, which is what a real client puts on the
            # wire; Starlette hands the server a non-ASCII str.
            headers={"authorization": "Bearer pässword".encode("latin-1")},
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


async def test_the_deep_probe_is_not_free_to_strangers() -> None:
    """`/health` is public because it is cheap. `?deep=1` spends a completion
    and takes a session slot, so it is not."""
    async with gateway(scripted("ok"), api_key="s3cret") as (client, harness):
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/health?deep=1")).status_code == 401
        assert harness.manager.live == 0

        authorised = await client.get("/health?deep=1", headers={"authorization": "Bearer s3cret"})
        assert authorised.status_code == 200
        assert authorised.json()["probe"]["ok"] is True

        # And it will not be used as an amplifier by whoever holds the key.
        throttled = await client.get("/health?deep=1", headers={"authorization": "Bearer s3cret"})
        assert throttled.json()["probe"]["skipped"] == "throttled"


async def test_a_malformed_body_is_a_400_that_says_why() -> None:
    async with gateway() as (client, _h):
        broken = await client.post(
            "/v1/chat/completions",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
        wrong_shape = await client.post(
            "/v1/chat/completions",
            content=b"[1, 2, 3]",
            headers={"content-type": "application/json"},
        )
        bad_field = await client.post("/v1/chat/completions", json=chat(n=3))

    assert broken.status_code == 400
    assert "not valid JSON" in broken.json()["error"]["message"]
    assert wrong_shape.status_code == 400
    assert bad_field.status_code == 400
    # A client's error handler should not be shown pydantic's internals.
    assert "pydantic" not in bad_field.text
    assert "n:" in bad_field.json()["error"]["message"]


async def test_an_oversized_body_is_refused_before_it_is_read() -> None:
    async with gateway(max_request_bytes=2048) as (client, _h):
        response = await client.post(
            "/v1/chat/completions", json=chat(messages=[{"role": "user", "content": "x" * 4096}])
        )

    assert response.status_code == 400
    assert "larger than" in response.json()["error"]["message"]


async def test_usage_is_not_carried_over_from_the_previous_turn() -> None:
    async def rich_then_silent(turn: Turn) -> None:
        if not getattr(turn.cli, "seen", False):
            turn.cli.seen = True  # type: ignore[attr-defined]
            turn.cli.usage = {"input_tokens": 1000, "output_tokens": 500}
            await turn.say("first")
            await turn.end()
            return
        # A turn that reports no usage at all: if the previous turn's numbers
        # were still buffered, they would be billed to this one.
        turn.cli.usage = {}
        await turn.cli.emit_assistant([{"type": "text", "text": "second"}])
        await turn.end(result="second")

    async with gateway(rich_then_silent) as (client, _h):
        first = (await client.post("/v1/chat/completions", json=chat(user="a"))).json()
        second = (
            await client.post(
                "/v1/chat/completions",
                json=chat(
                    user="a",
                    messages=[
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "first"},
                        {"role": "user", "content": "again"},
                    ],
                ),
            )
        ).json()

    assert first["usage"]["total_tokens"] == 1500
    assert second["usage"]["total_tokens"] == 0, "reported tokens the turn never spent"


async def test_a_reused_conversation_is_not_handed_a_transcript_preamble() -> None:
    """The conversation already holds everything before the delta. Re-announcing
    "the conversation so far is transcribed below" mid-conversation is noise."""

    async def scenario(turn: Turn) -> None:
        await turn.say("ok")
        await turn.end()

    async with gateway(scenario) as (client, harness):
        messages: list[dict[str, Any]] = [{"role": "user", "content": "first"}]
        await client.post("/v1/chat/completions", json=chat(messages=messages, user="a"))
        messages.append({"role": "assistant", "content": "ok"})
        messages.append({"role": "tool", "tool_call_id": "call_x", "content": "42"})
        messages.append({"role": "user", "content": "second"})
        follow_up = await client.post(
            "/v1/chat/completions", json=chat(messages=messages, user="a")
        )

        sent = harness.cli.turns[-1]["message"]["content"]

    assert follow_up.headers["x-claudegate-mode"] in {"reused", "fresh"}
    text = "".join(b.get("text", "") for b in sent if b.get("type") == "text")
    if follow_up.headers["x-claudegate-mode"] == "reused":
        assert "transcribed below" not in text
