"""Recognising a conversation we already hold."""

from __future__ import annotations

from claudegate.bridge import continuity
from claudegate.openai_api.schema import Message


def msg(role: str, content: object = None, **kw: object) -> Message:
    return Message.model_validate({"role": role, "content": content, **kw})


def test_growing_a_conversation_extends_its_chain() -> None:
    first = [msg("user", "a")]
    second = [msg("user", "a"), msg("assistant", "b"), msg("user", "c")]
    assert continuity.is_prefix(continuity.chain(first), continuity.chain(second))
    assert len(continuity.chain(second)) == 2


def test_editing_an_earlier_message_breaks_the_chain() -> None:
    original = continuity.chain([msg("user", "a"), msg("user", "b")])
    edited = continuity.chain([msg("user", "a!"), msg("user", "b")])
    assert not continuity.is_prefix(original, edited)


def test_assistant_turns_are_ignored_so_a_reformatted_reply_still_matches() -> None:
    """Clients hand back their own rendering of what we said. Hashing it would
    lose a conversation that is really the same one."""
    ours = [msg("user", "a"), msg("assistant", "verbatim reply"), msg("user", "b")]
    theirs = [msg("user", "a"), msg("assistant", "  reply, trimmed  "), msg("user", "b")]
    assert continuity.chain(ours) == continuity.chain(theirs)


def test_tool_results_are_part_of_the_identity() -> None:
    without = continuity.chain([msg("user", "a")])
    with_result = continuity.chain([msg("user", "a"), msg("tool", "42", tool_call_id="c1")])
    assert continuity.is_prefix(without, with_result)
    assert len(with_result) == 2


def test_the_empty_chain_is_a_prefix_of_everything() -> None:
    assert continuity.is_prefix([], continuity.chain([msg("user", "a")]))


def test_identity_changes_with_model_prompt_and_tools() -> None:
    base = {
        "model": "sonnet",
        "system_prompt": "be terse",
        "tools_fingerprint": "[]",
        "bare_mode": True,
    }
    key = continuity.identity_key(**base)  # type: ignore[arg-type]
    assert key == continuity.identity_key(**base)  # type: ignore[arg-type]
    assert key != continuity.identity_key(**{**base, "model": "opus"})  # type: ignore[arg-type]
    assert key != continuity.identity_key(**{**base, "system_prompt": "be verbose"})  # type: ignore[arg-type]
    assert key != continuity.identity_key(**{**base, "tools_fingerprint": "[x]"})  # type: ignore[arg-type]
    assert key != continuity.identity_key(**{**base, "bare_mode": False})  # type: ignore[arg-type]


def test_new_messages_returns_the_tail_after_the_synced_prefix() -> None:
    """Our own replies are not sent back to us.

    The conversation still holds everything it said, so an assistant turn that
    sits between the last synced user message and the new one is already in
    context. Echoing it would make the model read its own answer twice.
    """
    conversation = [
        msg("user", "a"),
        msg("assistant", "b"),
        msg("user", "c"),
        msg("assistant", "d"),  # our reply to "c" — the CLI already has it
        msg("user", "e"),
    ]
    tail = continuity.new_messages(conversation, already_synced=2)
    assert [m.text() for m in tail] == ["e"]


def test_new_messages_skips_system_turns() -> None:
    conversation = [msg("system", "s"), msg("user", "a"), msg("user", "b")]
    assert [m.text() for m in continuity.new_messages(conversation, 1)] == ["b"]


def test_nothing_is_new_when_the_whole_history_is_synced() -> None:
    conversation = [msg("user", "a"), msg("user", "b")]
    assert continuity.new_messages(conversation, 2) == []
