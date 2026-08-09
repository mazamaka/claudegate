"""The exact bytes a client will receive."""

from __future__ import annotations

import json

import pytest

from claudegate.openai_api import outbound


@pytest.mark.parametrize(
    ("stop", "expected"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "tool_calls"),
        ("refusal", "content_filter"),
        (None, "stop"),
        ("something_new", "stop"),
    ],
)
def test_finish_reason_mapping(stop: str | None, expected: str) -> None:
    assert outbound.finish_reason(stop) == expected


def test_tool_calls_win_over_the_reported_stop_reason() -> None:
    assert outbound.finish_reason("end_turn", had_tool_calls=True) == "tool_calls"


def test_usage_counts_cache_reads_as_prompt_tokens_but_reports_them_separately() -> None:
    usage = outbound.usage_block(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 30,
        }
    )
    assert usage["prompt_tokens"] == 1030
    assert usage["completion_tokens"] == 20
    assert usage["total_tokens"] == 1050
    assert usage["prompt_tokens_details"]["cached_tokens"] == 900


def test_usage_block_survives_a_missing_usage_report() -> None:
    assert outbound.usage_block(None)["total_tokens"] == 0


def test_sse_frame_is_one_json_object_terminated_by_a_blank_line() -> None:
    frame = outbound.sse({"a": 1})
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    assert json.loads(frame[6:].strip()) == {"a": 1}


def test_sse_passes_sentinels_through_verbatim() -> None:
    assert outbound.sse("[DONE]") == "data: [DONE]\n\n"


def test_chunk_shape() -> None:
    payload = outbound.chunk("id1", "sonnet", {"content": "hi"}, created=7, finish=None)
    assert payload["object"] == "chat.completion.chunk"
    assert payload["created"] == 7
    assert payload["choices"][0] == {
        "index": 0,
        "delta": {"content": "hi"},
        "finish_reason": None,
        "logprobs": None,
    }


def test_usage_chunk_has_no_choices() -> None:
    payload = outbound.usage_chunk("id1", "sonnet", {"total_tokens": 3})
    assert payload["choices"] == []
    assert payload["usage"] == {"total_tokens": 3}


def test_tool_call_delta_serialises_arguments_as_a_json_string() -> None:
    delta = outbound.tool_call_delta(0, {"id": "call_1", "name": "f", "arguments": {"x": 1}})
    call = delta["tool_calls"][0]
    assert call["index"] == 0
    assert call["type"] == "function"
    assert json.loads(call["function"]["arguments"]) == {"x": 1}


def test_completion_with_tool_calls_has_null_content_and_the_right_finish_reason() -> None:
    payload = outbound.completion(
        "id1",
        "sonnet",
        content=None,
        tool_calls=[{"id": "call_1", "name": "f", "arguments": {}}],
        stop_reason="tool_use",
    )
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "f"


def test_completion_carries_reasoning_when_there_is_any() -> None:
    payload = outbound.completion("id", "m", content="a", reasoning="because")
    assert payload["choices"][0]["message"]["reasoning_content"] == "because"
    assert "reasoning_content" not in outbound.completion("id", "m", content="a")["choices"][0]["message"]


def test_call_id_is_derived_from_the_anthropic_id_and_is_stable() -> None:
    first = outbound.call_id("toolu_01ABCDEF")
    assert first.startswith("call_")
    assert outbound.call_id(first) == first  # already converted: left alone
