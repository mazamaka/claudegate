"""``claudegate`` — serve it, check it, prove it works.

claudegate serve            start the server
claudegate doctor           diagnose the host before you blame the server
claudegate smoke            run the end-to-end suite against a running one
claudegate install-service  render a systemd unit with the details right
"""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import __version__

if TYPE_CHECKING:
    from .config import Settings

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def _supports_colour() -> bool:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return False
    if os.name != "nt":
        return True
    # Windows Terminal and ConEmu speak ANSI; the legacy console host only does
    # when something has enabled VT for it, which we cannot assume.
    return bool(os.environ.get("WT_SESSION") or os.environ.get("ANSICON"))


def _colour(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if _supports_colour() else text


# ────────────────────────────────────────────────────────────────── serve


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .config import get_settings

    overrides = {
        k: v
        for k, v in {
            "host": args.host,
            "port": args.port,
            "default_model": args.model,
            "log_level": args.log_level,
        }.items()
        if v is not None
    }
    settings = get_settings().model_copy(update=overrides) if overrides else get_settings()
    if args.no_bare:
        settings = settings.model_copy(update={"bare_mode": False})

    from .app import create_app

    app = create_app(settings)
    _print_banner(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=False,
        timeout_graceful_shutdown=args.grace,
    )
    return 0


def _print_banner(settings: Settings) -> None:
    key = "set" if settings.api_keys else _colour("none", YELLOW)
    mode = "bare model" if settings.bare_mode else "coding agent"
    print(f"claudegate {__version__}")
    print(f"  endpoint  http://{settings.host}:{settings.port}/v1")
    print(f"  model     {settings.default_model}")
    print(f"  api key   {key}")
    print(f"  mode      {mode}")
    print()


# ───────────────────────────────────────────────────────────────── doctor


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""


def _install_hint() -> str:
    if os.name == "nt":
        return "install the native Windows build (the npm claude.cmd shim will not work)"
    return "npm install -g @anthropic-ai/claude-code"


def _check_cli() -> Check:
    from .config import get_settings

    path = get_settings().cli_path or shutil.which("claude")
    if not path and os.name == "nt":
        path = shutil.which("claude.exe")
    if not path:
        return Check("claude CLI", False, "not found on PATH", _install_hint())

    # On Windows the SDK refuses to spawn a .bat/.cmd, because arguments would
    # go through cmd.exe. npm installs exactly such a shim, so a check that only
    # asked "is it on PATH?" would pass on a host where every turn fails.
    if os.name == "nt" and os.path.splitext(path)[1].lower() in {".cmd", ".bat"}:
        return Check(
            "claude CLI",
            False,
            f"{path} is a batch shim",
            "the SDK will not execute .cmd/.bat — install the native build, "
            "or point CLAUDEGATE_CLI_PATH at claude.exe",
        )
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30)
        ok = out.returncode == 0
        return Check("claude CLI", ok, f"{path} ({out.stdout.strip()})")
    except Exception as exc:
        return Check("claude CLI", False, f"{path}: {exc}")


def _check_node() -> Check:
    node = shutil.which("node")
    if not node:
        return Check("node", False, "not found on PATH", "install Node.js 18+")
    try:
        out = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=15)
        version = out.stdout.strip()
        major = int(version.lstrip("v").split(".")[0])
        ok = major >= 18
        return Check("node", ok, f"{version} ({node})", "" if ok else "need Node.js 18 or newer")
    except Exception as exc:
        return Check("node", False, str(exc))


def _check_auth() -> Check:
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return Check("auth", True, "CLAUDE_CODE_OAUTH_TOKEN is set (long-lived token)")
    if os.environ.get("ANTHROPIC_API_KEY"):
        return Check("auth", True, "ANTHROPIC_API_KEY is set (billed per token)")
    creds = os.path.expanduser("~/.claude/.credentials.json")
    if os.path.exists(creds) and os.path.getsize(creds) > 2:
        return Check(
            "auth",
            True,
            f"{creds} (session credentials — they rotate)",
            "for a service, prefer: claude setup-token → CLAUDE_CODE_OAUTH_TOKEN",
        )
    return Check(
        "auth",
        False,
        "no token found",
        "claude setup-token   # ~1 year, then export CLAUDE_CODE_OAUTH_TOKEN=…",
    )


def _check_platform() -> Check:
    """Windows only: the event loop has to be able to spawn a subprocess.

    ``SelectorEventLoop`` cannot, and a server started under one fails every
    turn with a bare ``NotImplementedError``. ``claudegate serve`` picks the
    right loop; this check is for anyone embedding the app elsewhere.
    """
    detail = f"{platform.system()} {platform.release()} (python {platform.python_version()})"
    if os.name != "nt":
        return Check("platform", True, detail)
    return Check(
        "platform",
        True,
        detail,
        "on Windows the server needs a ProactorEventLoop to spawn the CLI; "
        "`claudegate serve` selects it, other ASGI runners may not",
    )


def _check_root() -> Check:
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return Check("privileges", True, "running as a normal user")
    if os.environ.get("IS_SANDBOX"):
        return Check("privileges", True, "root with IS_SANDBOX set")
    return Check(
        "privileges",
        True,
        "root — claudegate will set IS_SANDBOX=1 for the CLI automatically",
        "the CLI refuses permission bypass as root; without this it exits silently",
    )


def _check_port(host: str, port: int) -> Check:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        busy = sock.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port)) == 0
    if busy:
        return Check("port", True, f"{host}:{port} already has something listening")
    return Check("port", True, f"{host}:{port} is free")


def _check_bind_security() -> Check:
    from .config import get_settings

    settings = get_settings()
    if settings.is_loopback:
        return Check("exposure", True, f"bound to {settings.host} (loopback only)")
    if settings.api_keys:
        return Check("exposure", True, f"bound to {settings.host} with an API key set")
    return Check(
        "exposure",
        False,
        f"bound to {settings.host} with no API key",
        "set CLAUDEGATE_API_KEY=$(openssl rand -hex 32)",
    )


async def _probe() -> Check:
    """Actually complete a turn. This is the check that catches expired auth."""
    from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, ResultMessage, TextBlock

    from .bridge.session import build_options, sandbox_env
    from .bridge.toolbelt import Toolbelt
    from .config import get_settings

    settings = get_settings()
    options = build_options(
        settings,
        model=settings.resolve_model(None),
        system_prompt="Reply with exactly: ok",
        toolbelt=Toolbelt([]),
        handler=None,
    )
    reply: list[str] = []

    async def turn() -> Check | None:
        async with ClaudeSDKClient(options) as client:
            await client.query("ping")
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    reply.extend(b.text for b in message.content if isinstance(b, TextBlock))
                elif isinstance(message, ResultMessage) and message.is_error:
                    # `errors` is often empty, and "CLI error" on its own has
                    # never helped anyone. Say what the CLI actually reported.
                    detail = "; ".join(message.errors) if message.errors else message.subtype
                    return Check(
                        "live turn",
                        False,
                        f"the CLI reported {detail}",
                        "no tokens were billed, which usually means the credentials "
                        "were rejected before the request was made",
                    )
        return None

    try:
        failure = await asyncio.wait_for(turn(), timeout=settings.first_event_timeout_s)
        if failure is not None:
            return failure
    except (TimeoutError, asyncio.TimeoutError):
        return Check(
            "live turn",
            False,
            f"no reply within {settings.first_event_timeout_s:.0f}s",
            "a CLI that connects and then goes quiet is nearly always auth: "
            "check CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_API_KEY",
        )
    except Exception as exc:
        hint = ""
        if "root" in str(exc).lower():
            hint = "export IS_SANDBOX=1"
        return Check("live turn", False, str(exc)[:200], hint)
    text = "".join(reply).strip()
    env_note = " (IS_SANDBOX injected)" if sandbox_env() else ""
    return Check("live turn", bool(text), f"model replied {text[:40]!r}{env_note}")


def cmd_doctor(args: argparse.Namespace) -> int:
    from .config import get_settings

    settings = get_settings()
    checks = [
        Check("python", sys.version_info >= (3, 10), sys.version.split()[0]),
        _check_platform(),
        _check_node(),
        _check_cli(),
        _check_auth(),
        _check_root(),
        _check_bind_security(),
        _check_port(settings.host, settings.port),
    ]
    if not args.offline:
        checks.append(asyncio.run(_probe()))

    print(f"claudegate doctor {_colour(f'({__version__})', DIM)}\n")
    failed = 0
    for check in checks:
        mark = _colour("✓", GREEN) if check.ok else _colour("✗", RED)
        print(f"  {mark} {check.name:<12} {check.detail}")
        if check.fix:
            print(f"    {_colour(check.fix, DIM if check.ok else YELLOW)}")
        failed += not check.ok
    print()
    if failed:
        print(_colour(f"{failed} check(s) failed", RED))
    else:
        print(_colour("all checks passed", GREEN))
    return 1 if failed else 0


# ────────────────────────────────────────────────────────────────── smoke


def cmd_smoke(args: argparse.Namespace) -> int:
    from .smoke import Client, Suite

    key = args.key or os.environ.get("CLAUDEGATE_API_KEY")
    suite = Suite(
        Client(args.base, key, timeout=args.timeout),
        model=args.model,
        expect_expired=args.expect_expired,
    )
    only = set(args.only.split(",")) if args.only else None
    print(f"smoke → {args.base}  model={args.model}\n")
    results = suite.run(only)
    for result in results:
        mark = _colour("✓", GREEN) if result.ok else _colour("✗", RED)
        print(f"  {mark} {result.name:<15} {result.seconds:5.1f}s  {result.detail}")
    passed = sum(r.ok for r in results)
    print()
    verdict = f"{passed}/{len(results)} passed"
    if passed == len(results):
        print(_colour(f"{verdict} — deployment looks healthy", GREEN))
        return 0
    print(_colour(verdict, RED))
    return 1


# ──────────────────────────────────────────────────────── install-service


UNIT = """\
[Unit]
Description=claudegate — OpenAI-compatible API for the Claude Code CLI
Documentation=https://github.com/mazamaka/claudegate
After=network-online.target
Wants=network-online.target
# Restart limits belong in [Unit]; systemd ignores them in [Service] and the
# unit then restarts forever through a failure it will never recover from.
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=exec
User={user}
WorkingDirectory={workdir}
EnvironmentFile={env_file}
# The CLI refuses permission bypass as root and exits silently without this.
Environment=IS_SANDBOX=1
ExecStart={executable} serve
Restart=on-failure
RestartSec=5
# Stop the parent politely and let in-flight turns finish rather than
# SIGKILLing the whole cgroup, which drops every parked tool call.
KillMode=mixed
TimeoutStopSec=30
NoNewPrivileges=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
"""


def cmd_install_service(args: argparse.Namespace) -> int:
    if sys.platform != "linux" and not args.force:
        # Writing a systemd unit on a host with no systemd produces a file that
        # looks right and does nothing. Say so instead.
        def say(line: str) -> None:
            print(line, file=sys.stderr)

        say(f"install-service renders a systemd unit; this host is {platform.system()}.")
        if sys.platform == "darwin":
            say("On macOS use a launchd plist, or run `claudegate serve` under a supervisor.")
        elif os.name == "nt":
            say("On Windows use NSSM, a Scheduled Task, or run `claudegate serve` directly.")
        say("Pass --force to render the unit anyway (e.g. when targeting a Linux host).")
        return 2
    executable = args.executable or shutil.which("claudegate") or f"{sys.executable} -m claudegate"
    unit = UNIT.format(
        user=args.user,
        workdir=args.workdir,
        env_file=args.env_file,
        executable=executable,
    )
    if args.output == "-":
        print(unit)
        return 0
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(unit)
    print(f"wrote {args.output}")
    print("next:")
    print(f"  systemd-analyze verify {args.output}")
    print("  systemctl daemon-reload && systemctl enable --now claudegate")
    return 0


# ──────────────────────────────────────────────────────────────── parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="claudegate", description=__doc__)
    parser.add_argument("--version", action="version", version=f"claudegate {__version__}")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="start the server")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--model")
    serve.add_argument("--log-level")
    serve.add_argument("--grace", type=int, default=20, help="graceful shutdown seconds")
    serve.add_argument(
        "--no-bare",
        action="store_true",
        help="expose Claude Code as an autonomous agent (file and shell tools), not a plain model",
    )
    serve.set_defaults(func=cmd_serve)

    doctor = sub.add_parser("doctor", help="check this host can run the server")
    doctor.add_argument("--offline", action="store_true", help="skip the live model turn")
    doctor.set_defaults(func=cmd_doctor)

    smoke = sub.add_parser("smoke", help="end-to-end checks against a running server")
    smoke.add_argument("--base", default="http://127.0.0.1:8080")
    smoke.add_argument("--model", default="sonnet")
    smoke.add_argument("--key")
    smoke.add_argument("--only", help="comma-separated check names")
    smoke.add_argument("--timeout", type=float, default=180.0)
    smoke.add_argument(
        "--expect-expired",
        type=int,
        default=200,
        help="expected status when tool results arrive for a lost conversation",
    )
    smoke.set_defaults(func=cmd_smoke)

    unit = sub.add_parser("install-service", help="render a systemd unit")
    unit.add_argument("--user", default="claudegate")
    unit.add_argument("--workdir", default="/opt/claudegate")
    unit.add_argument("--env-file", default="/opt/claudegate/.env")
    unit.add_argument("--executable")
    unit.add_argument("--output", default="-", help="path to write, or - for stdout")
    unit.add_argument(
        "--force",
        action="store_true",
        help="render the unit even when this host does not run systemd",
    )
    unit.set_defaults(func=cmd_install_service)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
