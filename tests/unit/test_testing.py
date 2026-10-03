"""Keep the fake CLI honest at its boundary with the real SDK/MCP server."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, create_sdk_mcp_server, tool

from claudegate.testing import FakeClaudeCLI, scripted


async def test_concurrent_calls_initialize_a_real_mcp_session_and_return_results() -> None:
    @tool("echo__value", "Echo a value", {"value": str})
    async def echo_value(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": args["value"]}]}

    fake = FakeClaudeCLI(scripted("unused"))
    options = ClaudeAgentOptions(
        mcp_servers={"client": create_sdk_mcp_server(name="client", tools=[echo_value])}
    )
    async with ClaudeSDKClient(options=options, transport=fake):
        results = await asyncio.wait_for(
            asyncio.gather(
                fake.invoke_tool("mcp__client__echo__value", {"value": "one"}),
                fake.invoke_tool("mcp__client__echo__value", {"value": "two"}),
            ),
            timeout=5,
        )
        assert results == ["one", "two"]
        assert await fake.invoke_tool("mcp__client__echo__value", {"value": "three"}) == "three"


@pytest.mark.parametrize("error_layer", ["control", "jsonrpc"])
async def test_protocol_errors_are_not_silently_returned_as_empty_text(error_layer: str) -> None:
    fake = FakeClaudeCLI(scripted("unused"))
    await fake.connect()

    async def reject_request() -> None:
        async for message in fake.read_messages():
            if message.get("type") != "control_request":
                continue
            response: dict[str, Any] = {
                "request_id": message["request_id"],
                "subtype": "success",
            }
            if error_layer == "control":
                response.update(subtype="error", error="SDK server unavailable")
            else:
                response["response"] = {
                    "mcp_response": {
                        "jsonrpc": "2.0",
                        "id": message["request"]["message"]["id"],
                        "error": {"code": -32602, "message": "Invalid request parameters"},
                    }
                }
            await fake.write(json.dumps({"type": "control_response", "response": response}))
            return

    responder = asyncio.create_task(reject_request())
    try:
        with pytest.raises(RuntimeError, match="MCP initialize:"):
            await asyncio.wait_for(fake.invoke_tool("mcp__client__echo", {}), timeout=5)
        await responder
    finally:
        responder.cancel()
        await fake.close()
