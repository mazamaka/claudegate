"""End-to-end checks against a running server.

Unit tests prove the code is right. These prove the *deployment* is right: the
CLI is on PATH, its token is valid, subprocesses can spawn under whatever
supervises this process, streaming survives the reverse proxy in front of it,
images get through, and a tool loop can be resumed.

Standard library only, so it runs anywhere the package is installed —
including inside a container with nothing else in it.
"""

from __future__ import annotations

import json
import random
import struct
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

# ─────────────────────────────────────────────────────────── test image

# A 5x7 bitmap font: enough to draw a code the model has never seen, so a
# correct answer cannot be a lucky guess about what is usually in the picture.
_FONT = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
}


def render_code_png(code: str, scale: int = 8, margin: int = 12) -> bytes:
    """Draw ``code`` as a black-on-white PNG. No third-party imaging needed."""
    glyph_w, glyph_h, gap = 5, 7, 2
    width = margin * 2 + len(code) * (glyph_w + gap) * scale
    height = margin * 2 + glyph_h * scale
    rows = [bytearray(b"\xff" * width) for _ in range(height)]

    for index, char in enumerate(code):
        glyph = _FONT.get(char)
        if not glyph:
            continue
        ox = margin + index * (glyph_w + gap) * scale
        for gy, line in enumerate(glyph):
            for gx, bit in enumerate(line):
                if bit != "1":
                    continue
                for dy in range(scale):
                    row = rows[margin + gy * scale + dy]
                    start = ox + gx * scale
                    row[start : start + scale] = b"\x00" * scale

    raw = b"".join(b"\x00" + bytes(row) for row in rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def data_url(png: bytes) -> str:
    import base64

    return "data:image/png;base64," + base64.b64encode(png).decode()


# ─────────────────────────────────────────────────────────────── client


class Client:
    """Minimal HTTP client for the checks."""

    def __init__(self, base: str, key: str | None = None, timeout: float = 180.0) -> None:
        self.base = base.rstrip("/")
        self.key = key
        self.timeout = timeout

    def _request(self, path: str, payload: Any | None = None) -> urllib.request.Request:
        headers = {"content-type": "application/json"}
        if self.key:
            headers["authorization"] = f"Bearer {self.key}"
        data = json.dumps(payload).encode() if payload is not None else None
        return urllib.request.Request(f"{self.base}{path}", data=data, headers=headers)

    def get(self, path: str) -> tuple[int, Any, dict[str, str]]:
        try:
            with urllib.request.urlopen(self._request(path), timeout=self.timeout) as r:
                return r.status, json.loads(r.read() or b"{}"), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, _maybe_json(e.read()), dict(e.headers)

    def post(self, path: str, payload: Any) -> tuple[int, Any, dict[str, str]]:
        try:
            with urllib.request.urlopen(self._request(path, payload), timeout=self.timeout) as r:
                return r.status, json.loads(r.read() or b"{}"), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, _maybe_json(e.read()), dict(e.headers)

    def stream(self, path: str, payload: Any) -> Iterable[tuple[float, str]]:
        with urllib.request.urlopen(self._request(path, payload), timeout=self.timeout) as r:
            for line in r:
                text = line.decode("utf-8", "replace").strip()
                if text.startswith("data: "):
                    yield time.monotonic(), text[6:]


def _maybe_json(body: bytes) -> Any:
    try:
        return json.loads(body)
    except ValueError:
        return {"raw": body.decode("utf-8", "replace")[:400]}


# ──────────────────────────────────────────────────────────────── checks


@dataclass
class Result:
    name: str
    ok: bool
    detail: str
    seconds: float = 0.0


@dataclass
class Suite:
    client: Client
    model: str = "sonnet"
    expect_expired: int = 200
    checks: list[tuple[str, Callable[[], str]]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.checks = [
            ("health", self.check_health),
            ("models", self.check_models),
            ("text", self.check_text),
            ("stream", self.check_stream),
            ("tools", self.check_tools),
            ("vision", self.check_vision),
            ("session-reuse", self.check_reuse),
            ("expired", self.check_expired),
        ]

    # -- individual checks; each returns a one-line detail or raises ------

    def check_health(self) -> str:
        status, body, _ = self.client.get("/health")
        assert status == 200, f"status={status}"
        return f"status={status} sessions={body.get('sessions')}"

    def check_models(self) -> str:
        status, body, _ = self.client.get("/v1/models")
        assert status == 200, f"status={status}"
        data = body.get("data") or []
        assert data, "empty model list"
        return f"{len(data)} models, first={data[0]['id']}"

    def check_text(self) -> str:
        status, body, _ = self.client.post(
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Reply with exactly: PONG"},
                    {"role": "user", "content": "ping"},
                ],
            },
        )
        assert status == 200, f"status={status} body={json.dumps(body)[:200]}"
        content = body["choices"][0]["message"]["content"] or ""
        assert "PONG" in content.upper(), f"unexpected reply {content!r}"
        return f"{content.strip()[:20]!r} prompt_tokens={body['usage']['prompt_tokens']}"

    def check_stream(self) -> str:
        frames: list[tuple[float, str]] = []
        for ts, payload in self.client.stream(
            "/v1/chat/completions",
            {
                "model": self.model,
                "stream": True,
                "messages": [
                    {"role": "user", "content": "Count from 1 to 10, separated by spaces."}
                ],
            },
        ):
            frames.append((ts, payload))
        assert frames, "no frames received"
        assert frames[-1][1] == "[DONE]", "stream did not terminate with [DONE]"
        content = []
        for _, payload in frames[:-1]:
            chunk = json.loads(payload)
            assert "error" not in chunk, f"error frame: {payload[:200]}"
            for choice in chunk.get("choices", []):
                piece = choice.get("delta", {}).get("content")
                if piece:
                    content.append(piece)
        # A single frame carrying the whole answer means something buffered it.
        spread = frames[-1][0] - frames[0][0]
        assert len(content) > 1, "response arrived in one frame — is a proxy buffering?"
        return f"{len(frames)} frames, {len(''.join(content))} chars over {spread:.1f}s"

    def check_tools(self) -> str:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_status",
                    "description": "Look up the numeric status code of an internal subsystem.",
                    "parameters": {
                        "type": "object",
                        "properties": {"subsystem": {"type": "string"}},
                        "required": ["subsystem"],
                    },
                },
            }
        ]
        code = random.randint(100, 999)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Use the tools you are given. Be terse."},
            {"role": "user", "content": "What is the status code for the db subsystem?"},
        ]
        status, body, _ = self.client.post(
            "/v1/chat/completions", {"model": self.model, "messages": messages, "tools": tools}
        )
        assert status == 200, f"status={status} body={json.dumps(body)[:200]}"
        choice = body["choices"][0]
        assert choice["finish_reason"] == "tool_calls", f"finish={choice['finish_reason']}"
        call = choice["message"]["tool_calls"][0]
        assert call["function"]["name"] == "lookup_status", call

        messages.append(choice["message"])
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(code)})
        status, body, headers = self.client.post(
            "/v1/chat/completions", {"model": self.model, "messages": messages, "tools": tools}
        )
        assert status == 200, f"resume status={status}"
        answer = body["choices"][0]["message"]["content"] or ""
        assert str(code) in answer, f"result {code} not used: {answer[:120]!r}"
        return f"called {call['function']['name']}, resumed via {headers.get('x-claudegate-mode')}"

    def check_vision(self) -> str:
        code = "".join(str(random.randint(0, 9)) for _ in range(4))
        url = data_url(render_code_png(code))
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Answer with digits only."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": "What 4-digit number is in this image? Digits only."},
                ],
            },
        ]
        status, body, _ = self.client.post(
            "/v1/chat/completions", {"model": self.model, "messages": messages}
        )
        assert status == 200, f"status={status}"
        answer = (body["choices"][0]["message"]["content"] or "").strip()
        assert code in answer, f"expected {code}, model said {answer[:60]!r}"

        # And again, one turn later: an image the model saw earlier has to stay
        # visible, or a follow-up question gets a confident wrong answer.
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": "Repeat that number, digits only."})
        status, body, headers = self.client.post(
            "/v1/chat/completions", {"model": self.model, "messages": messages}
        )
        assert status == 200, f"follow-up status={status}"
        again = (body["choices"][0]["message"]["content"] or "").strip()
        assert code in again, f"lost the image: expected {code}, got {again[:60]!r}"
        return f"read {code} and recalled it ({headers.get('x-claudegate-mode')})"

    def check_reuse(self) -> str:
        token = random.randint(1000, 9999)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": f"Remember the number {token}. Reply OK."},
        ]
        status, body, headers = self.client.post(
            "/v1/chat/completions", {"model": self.model, "messages": messages}
        )
        assert status == 200, f"status={status}"
        first_tokens = body["usage"]["prompt_tokens"]
        messages.append({"role": "assistant", "content": body["choices"][0]["message"]["content"]})
        messages.append({"role": "user", "content": "What number did I ask you to remember?"})
        status, body, headers = self.client.post(
            "/v1/chat/completions", {"model": self.model, "messages": messages}
        )
        assert status == 200, f"status={status}"
        answer = body["choices"][0]["message"]["content"] or ""
        assert str(token) in answer, f"lost context: {answer[:80]!r}"
        mode = headers.get("x-claudegate-mode", "?")
        assert mode == "reused", f"expected a reused conversation, got {mode!r}"
        return f"mode={mode}, prompt_tokens {first_tokens} → {body['usage']['prompt_tokens']}"

    def check_expired(self) -> str:
        """Tool results for a conversation the server does not have."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_status",
                    "description": "Look up a status code.",
                    "parameters": {
                        "type": "object",
                        "properties": {"subsystem": {"type": "string"}},
                    },
                },
            }
        ]
        messages = [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "What is the status code for the db subsystem?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_ghost_conversation",
                        "type": "function",
                        "function": {
                            "name": "lookup_status",
                            "arguments": '{"subsystem": "db"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_ghost_conversation", "content": "512"},
        ]
        status, body, headers = self.client.post(
            "/v1/chat/completions", {"model": self.model, "messages": messages, "tools": tools}
        )
        assert status == self.expect_expired, f"status={status}, expected {self.expect_expired}"
        if status != 200:
            return f"status={status} ({body.get('error', {}).get('code')})"
        answer = body["choices"][0]["message"]["content"] or ""
        assert "512" in answer, f"rebuilt but lost the result: {answer[:120]!r}"
        return f"{headers.get('x-claudegate-mode')} and answered with the tool result"

    # -- runner ----------------------------------------------------------

    def run(self, only: set[str] | None = None) -> list[Result]:
        results: list[Result] = []
        for name, fn in self.checks:
            if only and name not in only:
                continue
            started = time.monotonic()
            try:
                detail = fn()
                results.append(Result(name, True, detail, time.monotonic() - started))
            except Exception as exc:
                results.append(
                    Result(name, False, f"{type(exc).__name__}: {exc}", time.monotonic() - started)
                )
        return results
