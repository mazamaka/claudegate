"""The command line, including the parts that differ per platform.

None of the suite imported ``claudegate.cli`` before this file existed, which
is how a syntax error in it survived a full green run. Everything here is
cheap: no server, no CLI, no network.
"""

from __future__ import annotations

import os
import sys

import pytest

from claudegate import cli


def test_the_installed_version_matches_the_package() -> None:
    """Two copies of a version number drift; this catches it at the seam."""
    from importlib.metadata import PackageNotFoundError, version

    import claudegate

    try:
        installed = version("claude-code-openai")
    except PackageNotFoundError:  # running from a source tree
        pytest.skip("not installed")
    assert installed == claudegate.__version__


def test_every_subcommand_parses() -> None:
    parser = cli.build_parser()
    for argv in (
        ["serve"],
        ["serve", "--port", "9000", "--no-bare"],
        ["doctor", "--offline"],
        ["smoke", "--base", "http://127.0.0.1:8080", "--only", "text"],
        ["install-service", "--user", "svc"],
    ):
        args = parser.parse_args(argv)
        assert callable(args.func)


def test_no_command_prints_help_rather_than_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 1
    assert "claudegate" in capsys.readouterr().out


def test_the_offline_doctor_runs_without_a_cli_or_a_token() -> None:
    """It reports failures; it must never raise."""
    assert cli.cmd_doctor(cli.build_parser().parse_args(["doctor", "--offline"])) in (0, 1)


def test_colour_is_off_when_it_would_be_mojibake(monkeypatch: pytest.MonkeyPatch) -> None:
    """A legacy Windows console prints the escape codes literally."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.delenv("WT_SESSION", raising=False)
    monkeypatch.delenv("ANSICON", raising=False)
    assert cli._colour("x", cli.GREEN) == "x"

    monkeypatch.setenv("WT_SESSION", "1")
    assert cli._colour("x", cli.GREEN) != "x"


def test_no_color_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setenv("NO_COLOR", "1")
    assert cli._colour("x", cli.GREEN) == "x"


def test_a_windows_batch_shim_is_reported_as_unusable(monkeypatch: pytest.MonkeyPatch) -> None:
    """npm installs ``claude.cmd``; the SDK refuses to execute it.

    A check that only asked "is it on PATH?" would pass on a host where every
    single turn then fails.
    """
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(cli.shutil, "which", lambda name: r"C:\npm\claude.cmd")
    monkeypatch.setattr(cli, "get_settings", lambda: None, raising=False)

    check = cli._check_cli()

    assert check.ok is False
    assert "batch shim" in check.detail
    assert "claude.exe" in check.fix


def test_install_service_refuses_where_there_is_no_systemd(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = cli.build_parser().parse_args(["install-service"])

    monkeypatch.setattr(sys, "platform", "darwin")
    assert cli.cmd_install_service(args) == 2
    assert "launchd" in capsys.readouterr().err

    args.force = True
    assert cli.cmd_install_service(args) == 0
    assert "[Unit]" in capsys.readouterr().out


def test_install_service_renders_a_unit_on_linux(capsys: pytest.CaptureFixture[str]) -> None:
    args = cli.build_parser().parse_args(["install-service", "--user", "svc", "--force"])
    assert cli.cmd_install_service(args) == 0
    unit = capsys.readouterr().out
    assert "User=svc" in unit
    # Restart limits are ignored by systemd when they sit in [Service]. Match on
    # the section header itself, not the word — it also appears in a comment.
    assert unit.index("StartLimitBurst") < unit.index("\n[Service]\n")
