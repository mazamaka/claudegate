"""Recognising a conversation the server has already seen.

An OpenAI client is stateless: it re-sends the entire history on every request.
The CLI is not — it holds the conversation. Bridging the two naively means
re-sending everything every turn and paying to re-read it, which is what makes
the obvious implementation of this kind of server slow and expensive.

So instead each live conversation remembers a hash chain of the messages it has
already been given. A new request is matched against those chains; if one is a
prefix of the incoming history, that conversation *is* this conversation and
only the new messages are sent.

Only ``user`` and ``tool`` messages go into the chain. Assistant turns are
excluded on purpose: clients routinely hand back a lightly edited copy of what
we produced — trimmed whitespace, dropped reasoning, re-serialised tool call
arguments — and a chain over those would miss a match that is really there.
The user side is echoed verbatim, so it is the reliable half.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..openai_api.schema import Message


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def message_key(msg: Message) -> str:
    """A content hash of one message, stable across re-serialisation."""
    if isinstance(msg.content, str):
        body = msg.content
    elif msg.content is None:
        body = ""
    else:
        body = _stable([p.model_dump(exclude_none=True) for p in msg.content])
    payload = _stable([msg.role, msg.name or "", msg.tool_call_id or "", body])
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def chain(messages: list[Message]) -> list[str]:
    """Rolling hash chain over the messages that identify a conversation."""
    out: list[str] = []
    previous = ""
    for msg in messages:
        if msg.role not in ("user", "tool"):
            continue
        previous = hashlib.sha256((previous + message_key(msg)).encode()).hexdigest()[:32]
        out.append(previous)
    return out


def is_prefix(candidate: list[str], full: list[str]) -> bool:
    """Is ``candidate`` a prefix of ``full``? (Empty is a prefix of anything.)"""
    return len(candidate) <= len(full) and full[: len(candidate)] == candidate


def _normalise(text: str) -> str:
    return " ".join(text.split())


def last_assistant_text(messages: list[Message]) -> str | None:
    """The most recent assistant answer in the request, if there is one."""
    for msg in reversed(messages):
        if msg.role == "assistant":
            return msg.text()
    return None


def proves_receipt(messages: list[Message], last_reply: str) -> bool:
    """Did this request come from whoever received ``last_reply``?

    Identity alone is not enough to hand back a live conversation. Callers
    sharing one API key share a tenant, and an attacker who can guess an
    opening — a published system prompt and a templated first message is not a
    secret — could otherwise prime a conversation and have someone else's next
    request land inside it.

    What an attacker cannot guess is what the model actually said. Continuing a
    conversation therefore requires handing our own last answer back, which
    every OpenAI client does anyway because it is how the format works.
    Whitespace is normalised, since clients trim. A client that rewrites our
    answers more heavily than that simply gets a fresh conversation: slower,
    never wrong.
    """
    if not last_reply.strip():
        return False
    echoed = last_assistant_text(messages)
    if echoed is None:
        return False
    return _normalise(echoed) == _normalise(last_reply)


def identity_key(
    *,
    model: str,
    system_prompt: str | None,
    tools_fingerprint: str,
    bare_mode: bool,
    tenant: str = "",
) -> str:
    """What must match for two requests to be able to share a conversation.

    The model, the system prompt and the tool set are all fixed when the CLI is
    spawned. A request that changes any of them needs a new conversation, so
    they are folded into the key rather than checked case by case later.

    ``tenant`` is the caller's identity. It is part of the key because matching
    on history alone is not safe across callers: a shared system prompt and a
    templated opening message give two unrelated clients the same prefix, and
    the second one would inherit the first one's conversation.
    """
    payload = _stable([model, system_prompt or "", tools_fingerprint, bare_mode, tenant])
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def new_messages(messages: list[Message], already_synced: int) -> list[Message]:
    """The messages after the synced prefix, system turns excluded.

    ``already_synced`` counts chain entries (user/tool messages), so the
    matching walk has to skip assistant turns the same way :func:`chain` does.
    """
    seen = 0
    out: list[Message] = []
    for msg in messages:
        if msg.role in ("system", "developer"):
            continue
        if msg.role in ("user", "tool"):
            seen += 1
            if seen <= already_synced:
                continue
        if seen > already_synced:
            out.append(msg)
    return out
