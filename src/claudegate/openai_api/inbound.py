"""OpenAI request → Claude turn.

Pure functions only: no I/O, no SDK objects. That keeps the format-mapping
rules — the part most likely to be wrong — unit testable without a CLI, a
network, or a token.

Two shapes come out of here:

``system_prompt``
    The system/developer messages, joined. In bare mode this *replaces*
    Claude Code's own prompt.

``content blocks``
    The Anthropic content blocks for the user turn. For a live conversation
    that is just the new messages; for a fresh one it is the whole history
    rendered as a transcript, with every image kept in place.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .schema import ContentPart, Message

Block = dict[str, Any]

_DATA_URL = re.compile(r"^data:(?P<media>[\w.+-]+/[\w.+-]+)?(?P<b64>;base64)?,(?P<data>.*)$", re.S)

#: Anthropic accepts these image types. Anything else is described in text
#: rather than dropped silently, so the model knows something was attached.
_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

_TRANSCRIPT_HEADER = (
    "The conversation so far is transcribed below. Continue it as the assistant: "
    "reply to the final user message only, and do not repeat or re-announce "
    "earlier turns.\n"
)


@dataclass(slots=True)
class Turn:
    """Everything needed to drive one turn of the conversation."""

    blocks: list[Block] = field(default_factory=list)
    system_prompt: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.blocks

    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.blocks if b.get("type") == "text")


# ─────────────────────────────────────────────────────────────── attachments


def decode_data_url(url: str) -> tuple[str, str] | None:
    """``data:image/png;base64,AAA`` → ``("image/png", "AAA")``.

    Returns ``None`` for anything that is not a decodable data URL, including
    percent-encoded (non-base64) payloads, which Anthropic cannot take as-is.
    """
    m = _DATA_URL.match(url.strip())
    if not m or not m.group("b64"):
        return None
    media = m.group("media") or "application/octet-stream"
    data = m.group("data").strip()
    try:
        base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None
    return media, data


def image_block(url: str) -> Block | None:
    """Build a native image block from a data URL or a remote URL."""
    decoded = decode_data_url(url)
    if decoded:
        media, data = decoded
        if media not in _IMAGE_TYPES:
            return None
        return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
    if url.startswith(("http://", "https://")):
        return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def file_block(part: ContentPart) -> Block | None:
    """Build a document block for a ``{"type": "file"}`` part (PDFs, text)."""
    f = part.file
    if not f or not f.file_data:
        return None
    decoded = decode_data_url(f.file_data)
    if decoded:
        media, data = decoded
    else:  # bare base64 with the type carried by the filename
        media = "application/pdf" if (f.filename or "").lower().endswith(".pdf") else "text/plain"
        data = f.file_data
    if media == "application/pdf":
        return {"type": "document", "source": {"type": "base64", "media_type": media, "data": data}}
    try:
        text = base64.b64decode(data, validate=True).decode("utf-8", "replace")
    except (binascii.Error, ValueError):
        return None
    name = f.filename or "attachment"
    return {"type": "text", "text": f"<file name={name!r}>\n{text}\n</file>"}


def message_blocks(msg: Message, *, attachments: bool = True) -> list[Block]:
    """Content blocks for a single message, images and files included."""
    if msg.content is None:
        return []
    if isinstance(msg.content, str):
        return [{"type": "text", "text": msg.content}] if msg.content else []

    out: list[Block] = []
    for part in msg.content:
        if part.type == "text" and part.text:
            out.append({"type": "text", "text": part.text})
        elif part.type == "image_url" and part.image_url:
            block = image_block(part.image_url.url) if attachments else None
            out.append(block or {"type": "text", "text": "[image omitted]"})
        elif part.type == "file":
            block = file_block(part) if attachments else None
            out.append(block or {"type": "text", "text": "[file omitted]"})
        elif part.type == "input_audio":
            out.append({"type": "text", "text": "[audio omitted: unsupported by this backend]"})
    return out


# ─────────────────────────────────────────────────────────────────── prompts


def split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """Pull ``system``/``developer`` messages out of the list.

    They are joined in order, wherever they appeared: some clients send a
    system message mid-conversation, and dropping it would quietly change the
    model's instructions.
    """
    system: list[str] = []
    rest: list[Message] = []
    for m in messages:
        if m.role in ("system", "developer"):
            text = m.text()
            if text:
                system.append(text)
        else:
            rest.append(m)
    return ("\n\n".join(system) or None), rest


def _tool_call_line(call: Any) -> str:
    args = call.function.arguments or "{}"
    # Pretty-print when it is valid JSON, keep it raw when it is not.
    with contextlib.suppress(TypeError, ValueError):
        args = json.dumps(json.loads(args), ensure_ascii=False)
    return f"{call.function.name}({args})"


def render_history(
    messages: list[Message], *, attachments: bool = True, header: bool = True
) -> list[Block]:
    """Render a full conversation as one user turn, preserving every image.

    Used when there is no live conversation to continue: on the first request,
    and when rebuilding one that expired. Images from *earlier* turns are kept
    inline — a transcript that silently drops them makes the model answer
    questions about a picture it can no longer see, confidently and wrongly.
    """
    out: list[Block] = [{"type": "text", "text": _TRANSCRIPT_HEADER}] if header else []

    def add_text(text: str) -> None:
        if out and out[-1]["type"] == "text":
            out[-1]["text"] += text
        else:
            out.append({"type": "text", "text": text})

    for msg in messages:
        if msg.role == "tool":
            name = msg.name or msg.tool_call_id or "tool"
            add_text(f"\n[tool result: {name}]\n{msg.text()}\n")
            continue

        label = "user" if msg.role == "user" else "assistant"
        add_text(f"\n[{label}]\n")
        for block in message_blocks(msg, attachments=attachments):
            if block["type"] == "text":
                add_text(block["text"])
            else:
                out.append(block)
        if msg.tool_calls:
            for call in msg.tool_calls:
                add_text(f"\n[called tool] {_tool_call_line(call)}\n")

    add_text("\n[assistant]\n")
    return out


def build_turn(
    messages: list[Message],
    *,
    attachments: bool = True,
    history: bool = True,
) -> Turn:
    """Build the turn to send.

    ``history=False`` renders only the trailing user messages, for a live
    conversation that already holds everything before them.
    """
    system, rest = split_system(messages)
    if history:
        blocks = render_history(rest, attachments=attachments) if rest else []
    else:
        blocks = []
        for msg in rest:
            blocks.extend(message_blocks(msg, attachments=attachments))
    return Turn(blocks=blocks, system_prompt=system)


def trailing_tool_results(messages: list[Message]) -> list[Message]:
    """The unbroken run of ``role: tool`` messages at the end of the list.

    Their presence is what makes a request a *continuation*: the client is
    handing back results for tool calls we asked it to run.
    """
    out: list[Message] = []
    for msg in reversed(messages):
        if msg.role != "tool":
            break
        out.append(msg)
    out.reverse()
    return out
