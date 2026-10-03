# 🧰 Integration & operations

## The interesting bit: conversations stay open

An OpenAI client is stateless — it re-sends the whole history every turn. The
CLI is not; it *holds* the conversation. The obvious way to bridge that is to
replay the entire history into a fresh process on every request, which is
correct, slow, and expensive.

`claudegate` keeps the conversation alive instead, and recognises it again on
the next request by hashing the messages it has already seen. A follow-up turn
sends one message:

```
  turn 1   ████████████████████  240 prompt tokens   (fresh conversation)
  turn 2   █                      17 prompt tokens   (same conversation, one new message)
```

The same machinery makes tool calls cheap. When the model calls one of *your*
functions, the conversation is not unwound and replayed later — it is left
standing, parked inside the tool handler, while the HTTP response returns
`finish_reason: "tool_calls"`. Your result arrives on the next request and
resolves it:

```
   client                        claudegate                         claude
     │  POST (tools: [...])          │                                 │
     │ ─────────────────────────────>│  in-process MCP server ────────>│
     │                               │                    tool call <──│
     │  <── finish_reason:tool_calls │  ← conversation parked, alive    ┊
     │                               │                                 ┊
     │  POST (role: tool, result)    │                                 ┊
     │ ─────────────────────────────>│  handler resolves ─────────────>│
     │  <── the answer               │                                 │
```

A resumed turn avoids resending its full history to a fresh CLI process.

If the conversation really is gone — reaped after an idle timeout, or lost to a
restart — the tool results are not refused. The history in the request is enough
to rebuild it, which costs one re-read instead of failing a turn the client
cannot retry.


## Testing your integration, without a CLI

The fake CLI this project tests itself with is part of the package. It speaks
the same control protocol — including the side where the CLI reaches back to
invoke your tools — so you can test an integration end to end with no CLI, no
token, no network, and no cost:

```python
from claudegate import create_app
from claudegate.config import Settings
from claudegate.testing import FakeClaudeCLI, Turn

async def scenario(turn: Turn) -> None:
    assert "weather" in turn.text                     # assert on what the model got
    result = await turn.call_tool("get_weather", {"city": "Prague"})
    await turn.say(f"It is {result}.")
    await turn.end()

app = create_app(Settings(), transport_factory=lambda: FakeClaudeCLI(scenario))
```

The default test suite uses this fake CLI, including in CI on Linux, macOS and Windows.

## Verifying a deployment

The smoke suite exercises the running deployment — the
CLI is on `PATH`, its token is valid, subprocesses can spawn under your service
manager, your reverse proxy isn't buffering the stream, images get through, and
a tool loop can be resumed:

```console
$ claudegate smoke --base http://127.0.0.1:8080
smoke → http://127.0.0.1:8080  model=sonnet

  ✓ health            0.0s  status=200 sessions=0
  ✓ models            0.0s  7 models, first=sonnet
  ✓ text              2.3s  'PONG' prompt_tokens=233
  ✓ stream            2.7s  8 frames, 20 chars over 2.2s
  ✓ tools             3.3s  called lookup_status, resumed via continued
  ✓ vision            3.9s  read 2437 and recalled it (reused)
  ✓ session-reuse     4.1s  mode=reused, prompt_tokens 240 → 257
  ✓ expired           2.2s  rebuilt and answered with the tool result

8/8 passed — deployment looks healthy
```

The vision check draws a **freshly randomised number** into the image it sends,
so a correct answer cannot be a lucky guess about what is usually in the picture.

## Deployment

```bash
claudegate install-service --user claudegate --output /etc/systemd/system/claudegate.service
systemd-analyze verify /etc/systemd/system/claudegate.service
systemctl enable --now claudegate
```

The rendered unit gets the details that are easy to get wrong: restart limits in
`[Unit]` where systemd actually reads them, `KillMode=mixed` so a stop lets
in-flight turns finish instead of `SIGKILL`ing the cgroup, and `IS_SANDBOX=1`
(see below). There is a `Dockerfile` and a `docker-compose.yml` too.

Behind nginx, turn buffering off or streaming arrives in one lump at the end:

```nginx
location /v1/ {
    proxy_pass http://127.0.0.1:8080;
    proxy_buffering off;
    proxy_read_timeout 900s;
}
```

## Gotchas this handles for you

- **Running as root.** The CLI refuses permission bypass as root and exits
  without a word, which looks like a server that returns empty replies and logs
  nothing. `claudegate` sets `IS_SANDBOX=1` for the CLI automatically.
- **Auth that expires weeks later.** Copying `~/.claude/.credentials.json` into a
  service account works until the CLI rotates it. `claudegate doctor` says so,
  and points at `claude setup-token` for a long-lived token instead.
- **A green `/health` on a broken server.** A liveness probe that never spawns
  the CLI stays green through expired auth. `/health?deep=1` spends one real
  completion and reports what came back.


## Platforms

| Platform | Status |
|---|---|
| Linux | Supported, and what the live suite runs on. |
| macOS | Supported. The test suite runs on the macOS runner in CI. |
| Windows | Supported, with one caveat below. The suite runs on the Windows runner in CI. |

The suite is hermetic — it fakes the CLI — so a green Windows job proves the
server, the wire format and the bridge are portable. It does not exercise a
real `claude.exe`; that part is verified on Linux.

**Windows caveat.** `npm install -g @anthropic-ai/claude-code` installs a
`claude.cmd` shim, and the SDK refuses to execute `.bat`/`.cmd` (arguments
would pass through `cmd.exe`). Use the native build, or point
`CLAUDEGATE_CLI_PATH` at `claude.exe`. `claudegate doctor` checks for exactly
this rather than reporting a shim as usable. The server also needs a
`ProactorEventLoop` to spawn the CLI — `claudegate serve` selects one; if you
mount the ASGI app in another runner on Windows, make sure it does too.

`install-service` renders a systemd unit, so it refuses to run off Linux unless
you pass `--force` (useful when generating a unit for a remote host). On macOS
use launchd; on Windows use NSSM or a Scheduled Task.
