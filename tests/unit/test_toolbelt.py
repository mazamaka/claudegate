"""Publishing the client's tools, and matching the calls back to them."""

from __future__ import annotations

from claudegate.bridge.toolbelt import (
    Toolbelt,
    ToolCorrelator,
    ToolInvocation,
    fingerprint,
    sanitize,
)
from claudegate.openai_api.schema import ToolDef


def tool(name: str, description: str = "", params: dict | None = None) -> ToolDef:
    return ToolDef.model_validate(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": params or {"type": "object", "properties": {}},
            },
        }
    )


def test_sanitize_keeps_legal_names_and_repairs_illegal_ones() -> None:
    assert sanitize("get_weather") == "get_weather"
    assert sanitize("my.tool:v2") == "my_tool_v2"
    assert sanitize("!!!") == "tool"
    assert len(sanitize("x" * 200)) <= 60


def test_fingerprint_ignores_declaration_order() -> None:
    assert fingerprint([tool("a"), tool("b")]) == fingerprint([tool("b"), tool("a")])


def test_fingerprint_notices_a_changed_schema() -> None:
    a = fingerprint([tool("a", params={"type": "object", "properties": {"x": {"type": "string"}}})])
    b = fingerprint([tool("a", params={"type": "object", "properties": {"y": {"type": "string"}}})])
    assert a != b


def test_no_tools_has_a_stable_empty_fingerprint() -> None:
    assert fingerprint(None) == fingerprint([]) == "-"


def test_toolbelt_maps_names_in_both_directions() -> None:
    belt = Toolbelt([tool("get.weather")])
    assert belt.allowed_tool_names == ["mcp__client__get_weather"]
    assert belt.openai_name("mcp__client__get_weather") == "get.weather"
    assert belt.owns("mcp__client__get_weather")
    assert not belt.owns("Bash")


def test_toolbelt_keeps_colliding_names_apart() -> None:
    belt = Toolbelt([tool("a.b"), tool("a:b")])
    assert len(set(belt.allowed_tool_names)) == 2
    assert belt.openai_name(belt.allowed_tool_names[0]) == "a.b"
    assert belt.openai_name(belt.allowed_tool_names[1]) == "a:b"


def test_an_empty_toolbelt_is_falsey() -> None:
    assert not Toolbelt([])
    assert Toolbelt([tool("a")])


# ── correlator ───────────────────────────────────────────────────────────


def test_correlator_prefers_an_exact_argument_match() -> None:
    correlator = ToolCorrelator()
    correlator.expect(
        [
            ToolInvocation("call_1", "get_temp", {"city": "Prague"}),
            ToolInvocation("call_2", "get_temp", {"city": "Kyiv"}),
        ]
    )
    # The CLI happens to dispatch the second one first.
    assert correlator.claim("get_temp", {"city": "Kyiv"}).id == "call_2"
    assert correlator.claim("get_temp", {"city": "Prague"}).id == "call_1"


def test_correlator_falls_back_to_the_name_when_arguments_were_rewritten() -> None:
    correlator = ToolCorrelator()
    correlator.expect([ToolInvocation("call_1", "f", {"a": 1})])
    claimed = correlator.claim("f", {"a": 1, "b": None})
    assert claimed is not None
    assert claimed.id == "call_1"


def test_identical_calls_are_handed_out_in_order() -> None:
    correlator = ToolCorrelator()
    correlator.expect(
        [ToolInvocation("call_1", "f", {"x": 1}), ToolInvocation("call_2", "f", {"x": 1})]
    )
    assert correlator.claim("f", {"x": 1}).id == "call_1"
    assert correlator.claim("f", {"x": 1}).id == "call_2"


def test_claiming_from_an_empty_correlator_returns_nothing() -> None:
    assert ToolCorrelator().claim("f", {}) is None


def test_pending_ids_track_what_is_outstanding() -> None:
    correlator = ToolCorrelator()
    correlator.expect([ToolInvocation("call_1", "f", {})])
    assert correlator.pending_ids == ["call_1"]
    correlator.claim("f", {})
    assert correlator.pending_ids == []
