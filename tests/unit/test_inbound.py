"""Translation of OpenAI requests into Claude turns."""

from __future__ import annotations

import base64

import pytest

from claudegate.openai_api import inbound
from claudegate.openai_api.schema import Message

from ..conftest import PNG_1PX, PNG_DATA_URL


def msg(role: str, content: object = None, **kw: object) -> Message:
    return Message.model_validate({"role": role, "content": content, **kw})


# ── data URLs ────────────────────────────────────────────────────────────


def test_decode_data_url_roundtrip() -> None:
    assert inbound.decode_data_url(PNG_DATA_URL) == ("image/png", PNG_1PX)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/cat.png",  # not a data URL
        "data:image/png,%89PNG",  # percent-encoded, not base64
        "data:image/png;base64,not!valid!base64",
        "",
    ],
)
def test_decode_data_url_rejects(url: str) -> None:
    assert inbound.decode_data_url(url) is None


def test_image_block_from_data_url() -> None:
    block = inbound.image_block(PNG_DATA_URL)
    assert block == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": PNG_1PX},
    }


def test_image_block_from_remote_url_is_passed_through() -> None:
    block = inbound.image_block("https://example.com/cat.jpg")
    assert block == {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/cat.jpg"},
    }


def test_image_block_rejects_unsupported_media_type() -> None:
    assert inbound.image_block("data:image/tiff;base64,AAAA") is None


def test_file_block_pdf_becomes_a_document() -> None:
    part = inbound.ContentPart.model_validate(
        {
            "type": "file",
            "file": {"file_data": "data:application/pdf;base64,QQ==", "filename": "a.pdf"},
        }
    )
    assert inbound.file_block(part) == {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": "QQ=="},
    }


def test_file_block_text_is_inlined_with_its_name() -> None:
    data = base64.b64encode(b"hello file").decode()
    part = inbound.ContentPart.model_validate(
        {
            "type": "file",
            "file": {"file_data": f"data:text/plain;base64,{data}", "filename": "n.txt"},
        }
    )
    block = inbound.file_block(part)
    assert block is not None
    assert "hello file" in block["text"]
    assert "n.txt" in block["text"]


# ── message → blocks ─────────────────────────────────────────────────────


def test_string_content_becomes_one_text_block() -> None:
    assert inbound.message_blocks(msg("user", "hi")) == [{"type": "text", "text": "hi"}]


def test_mixed_content_keeps_order() -> None:
    message = msg(
        "user",
        [
            {"type": "text", "text": "before"},
            {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
            {"type": "text", "text": "after"},
        ],
    )
    kinds = [b["type"] for b in inbound.message_blocks(message)]
    assert kinds == ["text", "image", "text"]


def test_attachments_can_be_disabled_without_losing_the_turn() -> None:
    message = msg("user", [{"type": "image_url", "image_url": {"url": PNG_DATA_URL}}])
    blocks = inbound.message_blocks(message, attachments=False)
    assert blocks == [{"type": "text", "text": "[image omitted]"}]


# ── system prompt ────────────────────────────────────────────────────────


def test_split_system_collects_every_system_message_in_order() -> None:
    system, rest = inbound.split_system(
        [
            msg("system", "first"),
            msg("user", "a"),
            msg("developer", "second"),
            msg("assistant", "b"),
        ]
    )
    assert system == "first\n\nsecond"
    assert [m.role for m in rest] == ["user", "assistant"]


def test_split_system_returns_none_when_there_is_no_system_message() -> None:
    system, rest = inbound.split_system([msg("user", "a")])
    assert system is None
    assert len(rest) == 1


# ── history fold ─────────────────────────────────────────────────────────


def test_history_is_labelled_and_ends_by_handing_over_to_the_assistant() -> None:
    blocks = inbound.render_history([msg("user", "a"), msg("assistant", "b"), msg("user", "c")])
    text = "".join(b["text"] for b in blocks if b["type"] == "text")
    assert text.count("[user]") == 2
    assert "[assistant]" in text
    assert text.rstrip().endswith("[assistant]")


def test_history_keeps_images_from_earlier_turns() -> None:
    """The regression that makes a model answer confidently about a picture it
    can no longer see: an image is only kept when it is on the last message."""
    blocks = inbound.render_history(
        [
            msg("user", [{"type": "image_url", "image_url": {"url": PNG_DATA_URL}}]),
            msg("assistant", "I see it"),
            msg("user", "what was in the image?"),
        ]
    )
    images = [b for b in blocks if b["type"] == "image"]
    assert len(images) == 1
    assert images[0]["source"]["data"] == PNG_1PX


def test_history_renders_tool_calls_and_their_results() -> None:
    blocks = inbound.render_history(
        [
            msg("user", "weather?"),
            msg(
                "assistant",
                None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Prague"}'},
                    }
                ],
            ),
            msg("tool", "17C", tool_call_id="call_1", name="get_weather"),
            msg("user", "and tomorrow?"),
        ]
    )
    text = "".join(b["text"] for b in blocks if b["type"] == "text")
    assert 'get_weather({"city": "Prague"})' in text
    assert "[tool result: get_weather]" in text
    assert "17C" in text


def test_history_tolerates_unparseable_tool_arguments() -> None:
    blocks = inbound.render_history(
        [
            msg(
                "assistant",
                None,
                tool_calls=[
                    {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{oops"}}
                ],
            )
        ]
    )
    text = "".join(b["text"] for b in blocks if b["type"] == "text")
    assert "f({oops)" in text


# ── turn assembly ────────────────────────────────────────────────────────


def test_build_turn_without_history_sends_only_the_message() -> None:
    turn = inbound.build_turn([msg("system", "sys"), msg("user", "hello")], history=False)
    assert turn.system_prompt == "sys"
    assert turn.blocks == [{"type": "text", "text": "hello"}]
    assert not turn.is_empty


def test_build_turn_with_history_wraps_in_a_transcript() -> None:
    turn = inbound.build_turn([msg("user", "hello")])
    assert "[user]" in turn.text()


def test_trailing_tool_results_are_the_unbroken_run_at_the_end() -> None:
    messages = [
        msg("user", "go"),
        msg("tool", "old", tool_call_id="call_0"),
        msg("assistant", "mid"),
        msg("tool", "a", tool_call_id="call_1"),
        msg("tool", "b", tool_call_id="call_2"),
    ]
    assert [m.tool_call_id for m in inbound.trailing_tool_results(messages)] == ["call_1", "call_2"]


def test_no_trailing_tool_results_when_the_last_message_is_from_the_user() -> None:
    assert inbound.trailing_tool_results([msg("tool", "a"), msg("user", "b")]) == []
