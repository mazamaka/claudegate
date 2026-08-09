"""Settings and auth."""

from __future__ import annotations

import os

import pytest
from fastapi import Request

from claudegate.config import Settings
from claudegate.errors import AuthenticationError
from claudegate.security import check_request, extract_key


def settings(**kw: object) -> Settings:
    return Settings(**{"workspace": "/tmp/claudegate-tests", **kw})  # type: ignore[arg-type]


def test_aliases_resolve_and_unknown_models_pass_through() -> None:
    s = settings(default_model="sonnet")
    assert s.resolve_model("gpt-4o") == "sonnet"
    assert s.resolve_model("opus") == "opus"
    assert s.resolve_model("claude-sonnet-4-5-20250929") == "claude-sonnet-4-5-20250929"
    assert s.resolve_model(None) == "sonnet"


def test_a_workspace_we_created_is_ephemeral_and_one_we_were_given_is_not(
    tmp_path: object,
) -> None:
    """Restarting a service should not leave a directory behind every time,
    and it should never delete a directory the operator named."""
    ours = Settings()
    assert ours.workspace_is_ephemeral is True
    assert os.path.isdir(str(ours.workspace))

    theirs = Settings(workspace=str(tmp_path))
    assert theirs.workspace_is_ephemeral is False


def test_the_workspace_is_created_so_the_cli_can_be_spawned_in_it(tmp_path: object) -> None:
    """A configured workspace that does not exist makes the spawn fail with an
    error no one can act on. Creating it is cheaper than diagnosing it."""
    target = f"{tmp_path}/nested/workspace"
    s = Settings(workspace=target)  # type: ignore[arg-type]
    assert s.workspace == target
    assert os.path.isdir(target)


def test_a_workspace_is_chosen_when_none_is_configured() -> None:
    s = Settings()
    assert s.workspace is not None
    assert os.path.isdir(s.workspace)


def test_several_keys_may_be_configured_at_once() -> None:
    assert settings(api_key="a, b ,c").api_keys == ("a", "b", "c")
    assert settings(api_key=None).api_keys == ()


def test_auth_is_required_by_default_only_when_reachable_from_outside() -> None:
    assert settings(host="127.0.0.1").auth_required is False
    assert settings(host="localhost").auth_required is False
    assert settings(host="0.0.0.0").auth_required is True
    assert settings(host="0.0.0.0", require_auth=False).auth_required is False
    assert settings(host="127.0.0.1", require_auth=True).auth_required is True


def request_for(path: str = "/v1/chat/completions", **headers: str) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "root_path": "",
        "query_string": b"",
        "headers": [(k.replace("_", "-").encode(), v.encode()) for k, v in headers.items()],
    }
    return Request(scope)


def test_key_is_read_from_either_header_style() -> None:
    assert extract_key(request_for(authorization="Bearer abc")) == "abc"
    assert extract_key(request_for(x_api_key="abc")) == "abc"
    assert extract_key(request_for()) is None


def test_health_and_metrics_never_need_a_key() -> None:
    s = settings(api_key="secret")
    for path in ("/health", "/healthz", "/metrics"):
        check_request(request_for(path), s)


def test_a_missing_or_wrong_key_is_rejected() -> None:
    s = settings(api_key="secret")
    with pytest.raises(AuthenticationError):
        check_request(request_for(), s)
    with pytest.raises(AuthenticationError):
        check_request(request_for(authorization="Bearer wrong"), s)
    check_request(request_for(authorization="Bearer secret"), s)


def test_without_a_configured_key_loopback_is_open_and_exposed_is_not() -> None:
    check_request(request_for(), settings(host="127.0.0.1"))
    with pytest.raises(AuthenticationError):
        check_request(request_for(), settings(host="0.0.0.0"))
