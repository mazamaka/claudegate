# Configuration

Every setting is an environment variable prefixed `CLAUDEGATE_`, and a `.env`
file in the working directory is read as well. All of them have defaults that
work on a laptop.

## Bind

| Variable | Default | Notes |
|---|---|---|
| `CLAUDEGATE_HOST` | `127.0.0.1` | Anything else makes an API key mandatory. |
| `CLAUDEGATE_PORT` | `8080` | |

## Auth

| Variable | Default | Notes |
|---|---|---|
| `CLAUDEGATE_API_KEY` | — | Bearer token. Several may be given, comma separated. |
| `CLAUDEGATE_REQUIRE_AUTH` | auto | Auto = required unless bound to loopback. |

Startup is refused when the server is reachable from outside without a key. It
drives the CLI with permission bypass, so an open port is remote code execution;
`CLAUDEGATE_REQUIRE_AUTH=false` is there if you have your own perimeter.

## Model

| Variable | Default | Notes |
|---|---|---|
| `CLAUDEGATE_DEFAULT_MODEL` | `sonnet` | Used when a request names no model. |
| `CLAUDEGATE_FALLBACK_MODEL` | — | Passed to the CLI as its fallback. |
| `CLAUDEGATE_MODEL_ALIASES` | `opus`/`sonnet`/`haiku` + `gpt-4o`-style | JSON object. Unknown names pass through, so a new model works the day it ships. |

## Agent behaviour

| Variable | Default | Notes |
|---|---|---|
| `CLAUDEGATE_BARE_MODE` | `true` | Plain chat model. `false` exposes the real coding agent, with file and shell access. |
| `CLAUDEGATE_WORKSPACE` | temp dir | The CLI's working directory. Created if missing. |
| `CLAUDEGATE_PERMISSION_MODE` | `bypassPermissions` | |
| `CLAUDEGATE_SYSTEM_PROMPT_SUFFIX` | — | Appended to every request's system prompt. |
| `CLAUDEGATE_MAX_TURNS` | — | Ceiling on agent turns per request. |
| `CLAUDEGATE_CLI_PATH` | auto | Explicit path to the `claude` binary. |
| `CLAUDEGATE_CLAUDE_ENV` | `{}` | JSON object of extra environment for the CLI. |

In bare mode the request's `system` message *replaces* Claude Code's prompt, and
its built-in tools are removed. Only the tools your request declares are
available — which is what an OpenAI client expects.

## Lifecycle

| Variable | Default | Notes |
|---|---|---|
| `CLAUDEGATE_REUSE_SESSIONS` | `true` | Keep conversations open and send only new messages. |
| `CLAUDEGATE_REUSE_REQUIRES_USER` | `false` | Refuse to reuse a conversation for a request that names no `user`. Worth turning on when one API key is shared by many end users. |
| `CLAUDEGATE_REBUILD_ON_EXPIRY` | `true` | Rebuild from history when tool results arrive for a conversation we no longer hold. `false` returns `409`. |
| `CLAUDEGATE_FORWARD_ATTACHMENTS` | `true` | Images and files as native blocks. |
| `CLAUDEGATE_SESSION_IDLE_TTL_S` | `1800` | |
| `CLAUDEGATE_TOOL_WAIT_TTL_S` | `600` | How long a parked tool call waits for your result. Raise it if your tools are slow or need human approval. |
| `CLAUDEGATE_REQUEST_TIMEOUT_S` | `900` | |
| `CLAUDEGATE_MAX_SESSIONS` | `64` | Each live conversation is a CLI process (~200 MB). Concurrency is bounded by RAM, not CPU. |
| `CLAUDEGATE_GC_INTERVAL_S` | `30` | |

## Observability

| Variable | Default | Notes |
|---|---|---|
| `CLAUDEGATE_LOG_LEVEL` | `INFO` | |
| `CLAUDEGATE_LOG_FORMAT` | `text` | `json` for structured logs. |
| `CLAUDEGATE_METRICS` | `true` | Prometheus text at `/metrics`. |
| `CLAUDEGATE_REQUEST_LOG` | `true` | One line per `/v1` request. |

## The CLI's own credentials

`claudegate` does not manage them; the CLI does. For a service, prefer a
long-lived token over copied session credentials:

```bash
claude setup-token        # ~1 year, no refresh
export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
```

`CLAUDE_CODE_OAUTH_TOKEN` takes precedence over file credentials, so the account
running the server never needs an interactive login of its own. Copying
`~/.claude/.credentials.json` instead appears to work and then fails days later,
because the CLI rotates those and a copy it does not own goes stale.
`claudegate doctor` tells you which one you are using.
