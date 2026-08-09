"""Claude events → OpenAI responses.

Pure builders, so the exact wire shape a client will receive can be asserted in
a unit test. Nothing here knows about the SDK or HTTP.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

Chunk = dict[str, Any]

#: Anthropic stop reason → OpenAI finish reason.
FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "content_filter",
    "pause_turn": "stop",
}


def completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def call_id(seed: str | None = None) -> str:
    """OpenAI-shaped tool call id, derived from Anthropic's when we have one."""
    if seed:
        return f"call_{seed[-24:]}" if not seed.startswith("call_") else seed
    return f"call_{uuid.uuid4().hex[:24]}"


def finish_reason(stop_reason: str | None, *, had_tool_calls: bool = False) -> str:
    if had_tool_calls:
        return "tool_calls"
    if stop_reason is None:
        return "stop"
    return FINISH_REASONS.get(stop_reason, "stop")


def usage_block(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Anthropic usage → OpenAI usage.

    Cache reads are counted as prompt tokens (they *are* prompt tokens, just
    cheap ones) and also reported in ``prompt_tokens_details.cached_tokens`` so
    a client can tell the difference.
    """
    raw = raw or {}
    cache_read = int(raw.get("cache_read_input_tokens") or 0)
    cache_write = int(raw.get("cache_creation_input_tokens") or 0)
    prompt = int(raw.get("input_tokens") or 0) + cache_read + cache_write
    completion = int(raw.get("output_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cache_read},
    }


def sse(payload: Chunk | str) -> str:
    """One Server-Sent Event frame."""
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


DONE = "data: [DONE]\n\n"


def _envelope(cid: str, model: str, created: int, obj: str) -> Chunk:
    return {"id": cid, "object": obj, "created": created, "model": model}


def chunk(
    cid: str,
    model: str,
    delta: dict[str, Any],
    *,
    created: int | None = None,
    finish: str | None = None,
) -> Chunk:
    """One ``chat.completion.chunk``."""
    payload = _envelope(cid, model, created or int(time.time()), "chat.completion.chunk")
    payload["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}]
    return payload


def usage_chunk(
    cid: str, model: str, usage: dict[str, Any], *, created: int | None = None
) -> Chunk:
    """The final, choice-less chunk sent when ``stream_options.include_usage``."""
    payload = _envelope(cid, model, created or int(time.time()), "chat.completion.chunk")
    payload["choices"] = []
    payload["usage"] = usage
    return payload


def tool_call_delta(index: int, call: dict[str, Any]) -> dict[str, Any]:
    """A tool call as a streaming delta.

    Arguments are sent whole rather than trickled: they are only known once the
    model closes the block, and a single delta is what every client's
    accumulator expects to end up with anyway.
    """
    return {
        "tool_calls": [
            {
                "index": index,
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
                },
            }
        ]
    }


def completion(
    cid: str,
    model: str,
    *,
    content: str | None,
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    reasoning: str | None = None,
    stop_reason: str | None = None,
    created: int | None = None,
) -> Chunk:
    """A complete, non-streamed ``chat.completion``."""
    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": c["id"],
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": json.dumps(c.get("arguments", {}), ensure_ascii=False),
                },
            }
            for c in tool_calls
        ]
        message.setdefault("content", None)

    payload = _envelope(cid, model, created or int(time.time()), "chat.completion")
    payload["choices"] = [
        {
            "index": 0,
            "message": message,
            "finish_reason": finish_reason(stop_reason, had_tool_calls=bool(tool_calls)),
            "logprobs": None,
        }
    ]
    payload["usage"] = usage or usage_block(None)
    return payload
