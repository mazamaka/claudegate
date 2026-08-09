"""Function calling, including the part that makes this server different:
the conversation is parked mid-call rather than replayed afterwards.
"""

from __future__ import annotations

import json
from typing import Any

from claudegate.testing import Turn

from ..conftest import chat, gateway, sse_frames

TOOLS: list[dict[str, Any]] = [
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


async def one_call(turn: Turn) -> None:
    result = await turn.call_tool("lookup_status", {"subsystem": "db"})
    await turn.say(f"The db status code is {result}.")
    await turn.end()


async def test_a_tool_call_is_returned_to_the_client() -> None:
    async with gateway(one_call) as (client, _):
        response = await client.post(
            "/v1/chat/completions",
            json=chat(tools=TOOLS, messages=[{"role": "user", "content": "db status?"}]),
        )
        body = response.json()

    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    call = choice["message"]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "lookup_status"
    assert json.loads(call["function"]["arguments"]) == {"subsystem": "db"}
    assert call["id"].startswith("call_")


async def test_the_result_resumes_the_same_conversation_without_replaying_it() -> None:
    """The point of the whole design.

    The second request carries the full history, as every OpenAI client does.
    None of it is sent to the model: the conversation was never unwound, so the
    CLI still sees exactly one user turn.
    """
    async with gateway(one_call) as (client, harness):
        messages: list[dict[str, Any]] = [{"role": "user", "content": "db status?"}]
        first = (
            await client.post("/v1/chat/completions", json=chat(tools=TOOLS, messages=messages))
        ).json()
        call = first["choices"][0]["message"]["tool_calls"][0]

        messages.append(first["choices"][0]["message"])
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": "512"})
        response = await client.post(
            "/v1/chat/completions", json=chat(tools=TOOLS, messages=messages)
        )
        body = response.json()

        assert response.headers["x-claudegate-mode"] == "continued"
        assert len(harness.transports) == 1, "a second conversation was started"
        assert len(harness.cli.turns) == 1, "the history was replayed to the model"

    assert body["choices"][0]["message"]["content"] == "The db status code is 512."
    assert body["choices"][0]["finish_reason"] == "stop"


async def test_tool_calls_stream_as_deltas_and_finish_as_tool_calls() -> None:
    async with (
        gateway(one_call) as (client, _),
        client.stream(
            "POST",
            "/v1/chat/completions",
            json=chat(stream=True, tools=TOOLS, messages=[{"role": "user", "content": "db?"}]),
        ) as r,
    ):
        frames = await sse_frames(r)

    payloads = [json.loads(f) for f in frames[:-1]]
    deltas = [p for p in payloads if p["choices"] and "tool_calls" in p["choices"][0]["delta"]]
    assert len(deltas) == 1
    call = deltas[0]["choices"][0]["delta"]["tool_calls"][0]
    assert call["index"] == 0
    assert call["function"]["name"] == "lookup_status"
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"


async def test_several_tools_in_one_turn_are_all_returned_and_all_answered() -> None:
    async def three_calls(turn: Turn) -> None:
        results = await turn.call_tools(
            [
                ("lookup_status", {"subsystem": "db"}),
                ("lookup_status", {"subsystem": "cache"}),
                ("lookup_status", {"subsystem": "queue"}),
            ]
        )
        await turn.say("codes: " + ", ".join(results))
        await turn.end()

    async with gateway(three_calls) as (client, harness):
        messages: list[dict[str, Any]] = [{"role": "user", "content": "all statuses?"}]
        first = (
            await client.post("/v1/chat/completions", json=chat(tools=TOOLS, messages=messages))
        ).json()
        calls = first["choices"][0]["message"]["tool_calls"]
        assert len(calls) == 3
        assert [json.loads(c["function"]["arguments"])["subsystem"] for c in calls] == [
            "db",
            "cache",
            "queue",
        ]

        messages.append(first["choices"][0]["message"])
        for call, value in zip(calls, ["1", "2", "3"], strict=True):
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": value})
        body = (
            await client.post("/v1/chat/completions", json=chat(tools=TOOLS, messages=messages))
        ).json()

        assert len(harness.cli.turns) == 1

    # Each result reached the call it belongs to, in order.
    assert body["choices"][0]["message"]["content"] == "codes: 1, 2, 3"


async def test_a_client_that_answers_only_some_calls_does_not_wedge_the_server() -> None:
    async def two_calls(turn: Turn) -> None:
        results = await turn.call_tools(
            [("lookup_status", {"subsystem": "db"}), ("lookup_status", {"subsystem": "cache"})]
        )
        await turn.say(" | ".join(results))
        await turn.end()

    async with gateway(two_calls) as (client, _):
        messages: list[dict[str, Any]] = [{"role": "user", "content": "statuses?"}]
        first = (
            await client.post("/v1/chat/completions", json=chat(tools=TOOLS, messages=messages))
        ).json()
        calls = first["choices"][0]["message"]["tool_calls"]

        messages.append(first["choices"][0]["message"])
        messages.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": "only-one"})
        body = (
            await client.post("/v1/chat/completions", json=chat(tools=TOOLS, messages=messages))
        ).json()

    content = body["choices"][0]["message"]["content"]
    assert content.startswith("only-one | ")
    assert "returned no result" in content


async def test_tool_choice_none_hides_the_tools_entirely() -> None:
    async def scenario(turn: Turn) -> None:
        await turn.say("no tools for me")
        await turn.end()

    async with gateway(scenario) as (client, harness):
        response = await client.post(
            "/v1/chat/completions", json=chat(tools=TOOLS, tool_choice="none")
        )

    assert response.status_code == 200
    assert harness.manager.stats.created == 1
    assert response.json()["choices"][0]["finish_reason"] == "stop"


async def test_tools_are_rejected_when_they_have_no_name() -> None:
    async with gateway() as (client, _):
        response = await client.post(
            "/v1/chat/completions",
            json=chat(tools=[{"type": "function", "function": {"name": ""}}]),
        )

    assert response.status_code == 400
    assert "function name" in response.json()["error"]["message"]
