<h1 align="center">claudegate</h1>

<p align="center">
  <b>An OpenAI-compatible API in front of the Claude Code CLI.</b><br>
  Streaming, function calling, image input — and conversations that stay open between requests.
</p>

<p align="center">
  <a href="https://github.com/mazamaka/claudegate/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/mazamaka/claudegate/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.10%20%E2%80%93%203.13-3776AB">
  <img alt="typing" src="https://img.shields.io/badge/mypy-strict-2A6DB2">
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>

---

```bash
pip install git+https://github.com/mazamaka/claudegate
claudegate doctor     # is this host ready?
claudegate serve      # http://127.0.0.1:8080/v1
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
print(client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "Explain a monad in one sentence."}],
).choices[0].message.content)
```

That is the whole setup. Anything that speaks the OpenAI chat API works: the
official SDKs, LangChain, LlamaIndex, Open WebUI, Aider, Cursor, your own curl.

---

## Why

The Claude Code CLI is an excellent model runner that you can only talk to as a
human at a terminal. Your code, your tools and your agents speak the OpenAI chat
API. `claudegate` is the piece in between — a real service, not a script: bearer
auth, backpressure, graceful shutdown, Prometheus metrics, and a test suite that
runs without a CLI installed.

## What works

| | |
|---|---|
| **Chat completions** | streaming and non-streaming, with correct `finish_reason` and `usage` |
| **Function calling** | parallel calls, `tool_choice`, results resumed on the *same* conversation |
| **Vision** | `image_url` parts as native image blocks — data URLs or remote URLs |
| **Files** | PDFs as document blocks, text files inlined |
| **Reasoning** | streamed separately as `reasoning_content`, never mixed into the answer |
| **Models** | `opus` / `sonnet` / `haiku`, plus `gpt-4o`-style aliases so unmodified clients work |
| **Ops** | `/health` (with a real probe behind `?deep=1`), `/metrics`, structured JSON logs |

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

Nothing is replayed, so nothing can be replayed *wrong* — and a 40-message
conversation costs the same to resume as a 2-message one.

If the conversation really is gone — reaped after an idle timeout, or lost to a
restart — the tool results are not refused. The history in the request is enough
to rebuild it, which costs one re-read instead of failing a turn the client
cannot retry.

## Configuration

Everything is an environment variable prefixed `CLAUDEGATE_`, and everything has
a working default. A `.env` file in the working directory is read too.

```bash
CLAUDEGATE_HOST=127.0.0.1          # bind
CLAUDEGATE_PORT=8080
CLAUDEGATE_API_KEY=                # bearer token; required for non-loopback binds
CLAUDEGATE_DEFAULT_MODEL=sonnet
CLAUDEGATE_BARE_MODE=true          # plain model; false = autonomous coding agent
CLAUDEGATE_REUSE_SESSIONS=true     # keep conversations open between requests
CLAUDEGATE_MAX_SESSIONS=64         # concurrency ceiling; over it, 429 with Retry-After
CLAUDEGATE_SESSION_IDLE_TTL_S=1800
CLAUDEGATE_TOOL_WAIT_TTL_S=600     # how long a parked tool call waits for your result
CLAUDEGATE_LOG_FORMAT=text         # or json
```

The full list, annotated, is in [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

**Bare mode** is on by default: Claude Code's own identity and built-in tools are
removed, and your `system` message becomes the entire system prompt — so the
model behaves like the plain chat model your client expects. Turn it off
(`claudegate serve --no-bare`) to expose the real coding agent, with file and
shell access, through the same API.

## Security

The CLI is driven with permission bypass, so **anyone who can reach the port can
run code as the user running this server**. Two things follow, both enforced:

- binding anything other than loopback without `CLAUDEGATE_API_KEY` set is
  refused at startup, with an explanation rather than a stack trace;
- keys are compared in constant time, and `/health` and `/metrics` are the only
  endpoints that never need one.

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

That is how the 102 tests in this repo run in under a second.

## Verifying a deployment

Unit tests prove the code is right; these prove the *deployment* is right — the
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

## Compatibility notes

The CLI does not expose sampling controls, so `temperature`, `top_p`, `stop`,
`seed` and the penalties are **accepted and ignored** rather than rejected —
a client that always sends them keeps working. `n > 1` is rejected, because
silently returning one choice would be worse. `reasoning_effort` is mapped onto
the agent's thinking budget. Details in
[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

## Requirements

- Python 3.10+
- Node.js 18+ and the [`claude`][cli] CLI on `PATH`, logged in
- Linux or macOS

## Development

```bash
pip install -e ".[dev]"
pytest                                    # 102 tests, no CLI needed, < 1s
CLAUDEGATE_LIVE_TESTS=1 pytest tests/live # the real thing
ruff check src tests && mypy
```

Built on Anthropic's official [Claude Agent SDK][sdk]. Architecture notes are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT — see [LICENSE](LICENSE). Not affiliated with Anthropic.

[cli]: https://docs.anthropic.com/en/docs/claude-code
[sdk]: https://github.com/anthropics/claude-agent-sdk-python
