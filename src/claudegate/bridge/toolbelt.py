"""The client's tools, exposed to Claude in-process.

An OpenAI client sends tool *definitions* and expects to be asked to run them.
The CLI, on the other hand, only knows how to call tools it can reach — so each
function the client declares is published as a tool on an in-process MCP server
the SDK hosts inside this process. No sockets, no bridge process, no
per-conversation config file: the handler is a Python coroutine.

What that coroutine does is the whole trick. It does not run anything; it
*waits*. The server answers the HTTP request with ``finish_reason:
"tool_calls"``, the client goes off and runs the function, and the result
arrives on a later request and resolves the future the handler is parked on.
The conversation never unwinds, so nothing has to be replayed to resume it.

One wrinkle: MCP hands the handler its arguments but not the ``tool_use`` id
Anthropic assigned to the call, and that id is what the client answers with.
:class:`ToolCorrelator` re-attaches it by matching each invocation against the
tool_use blocks seen in the assistant message that triggered it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import McpSdkServerConfig, create_sdk_mcp_server, tool

from ..openai_api.schema import ToolDef

#: Name of the in-process MCP server. Claude sees tools as ``mcp__<server>__<tool>``.
SERVER_NAME = "client"
TOOL_PREFIX = f"mcp__{SERVER_NAME}__"

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_-]")

ToolHandler = Callable[[str, dict[str, Any]], Awaitable[str]]


@dataclass(slots=True)
class ToolInvocation:
    """A tool call the model made, in OpenAI terms."""

    id: str
    name: str
    arguments: dict[str, Any]

    def as_openai(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


def sanitize(name: str) -> str:
    """MCP tool names allow ``[a-zA-Z0-9_-]``; OpenAI is not quite that strict."""
    cleaned = _SAFE_NAME.sub("_", name).strip("_") or "tool"
    return cleaned[:60]


def fingerprint(tools: list[ToolDef] | None) -> str:
    """Stable identity of a tool set.

    A live conversation is bound to the tools it was started with — the CLI is
    told about them once, at spawn. A request with a different tool set
    therefore cannot reuse it, and this is what the session key compares.
    """
    if not tools:
        return "-"
    items = [
        {
            "name": t.function.name,
            "description": t.function.description or "",
            "parameters": t.function.parameters or {},
        }
        for t in tools
    ]
    items.sort(key=lambda i: str(i["name"]))
    return json.dumps(items, sort_keys=True, separators=(",", ":"))


@dataclass
class Toolbelt:
    """OpenAI tool definitions published as an in-process MCP server."""

    tools: list[ToolDef] = field(default_factory=list)
    _by_mcp_name: dict[str, str] = field(default_factory=dict, init=False)
    _by_openai_name: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for t in self.tools:
            safe = sanitize(t.function.name)
            # Keep names unique after sanitising.
            candidate, n = safe, 1
            taken = self._by_mcp_name
            while candidate in taken and taken[candidate] != t.function.name:
                n += 1
                candidate = f"{safe}_{n}"
            self._by_mcp_name[candidate] = t.function.name
            self._by_openai_name[t.function.name] = candidate

    def __bool__(self) -> bool:
        return bool(self.tools)

    @property
    def allowed_tool_names(self) -> list[str]:
        return [f"{TOOL_PREFIX}{n}" for n in self._by_mcp_name]

    def openai_name(self, mcp_name: str) -> str | None:
        """``mcp__client__get_weather`` → ``get_weather`` as the client named it."""
        short = mcp_name[len(TOOL_PREFIX):] if mcp_name.startswith(TOOL_PREFIX) else mcp_name
        return self._by_mcp_name.get(short)

    def owns(self, mcp_name: str) -> bool:
        return self.openai_name(mcp_name) is not None

    def build_server(self, handler: ToolHandler) -> McpSdkServerConfig:
        """Create the MCP server; ``handler(openai_name, args)`` awaits the result."""
        sdk_tools = []
        for t in self.tools:
            mcp_name = self._by_openai_name[t.function.name]
            schema = t.function.parameters or {"type": "object", "properties": {}}
            sdk_tools.append(
                tool(mcp_name, t.function.description or t.function.name, schema)(
                    _make_handler(t.function.name, handler)
                )
            )
        return create_sdk_mcp_server(SERVER_NAME, "1.0.0", sdk_tools)


def _make_handler(openai_name: str, handler: ToolHandler):  # type: ignore[no-untyped-def]
    async def _run(args: dict[str, Any]) -> dict[str, Any]:
        text = await handler(openai_name, args)
        return {"content": [{"type": "text", "text": text}]}

    return _run


class ToolCorrelator:
    """Matches handler invocations back to the tool_use blocks that caused them.

    The CLI calls handlers one at a time, in the order the model emitted the
    blocks, so position alone would usually do. "Usually" is not good enough
    when the wrong id means a client's result is attributed to the wrong call,
    so arguments are matched first and position is only the tie-breaker.
    """

    def __init__(self) -> None:
        self._pending: list[ToolInvocation] = []

    def expect(self, invocations: list[ToolInvocation]) -> None:
        self._pending.extend(invocations)

    @property
    def pending_ids(self) -> list[str]:
        return [i.id for i in self._pending]

    def claim(self, openai_name: str, args: dict[str, Any]) -> ToolInvocation | None:
        """Take the invocation this handler call belongs to."""
        for i, inv in enumerate(self._pending):
            if inv.name == openai_name and inv.arguments == args:
                return self._pending.pop(i)
        for i, inv in enumerate(self._pending):
            if inv.name == openai_name:
                return self._pending.pop(i)
        return self._pending.pop(0) if self._pending else None

    def clear(self) -> None:
        self._pending.clear()
