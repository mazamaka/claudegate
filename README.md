# 🔌 Claudegate

**Use the Claude Code CLI through an OpenAI-compatible chat API.**

Connect **Hermes Agent**, **OpenClaw** and other OpenAI-compatible clients to Claude Code, with streaming responses, tool calling and conversations that stay open between requests.

## ⚡ What it does

- 💬 **Chat & streaming** — chat completions, usage reporting and separate reasoning output.
- 🧰 **Tool calling** — return a tool result and continue the same conversation.
- 🖼️ **Image & file input** — image URLs, PDFs and text attachments.
- 🔁 **Session reuse** — match follow-up requests to live conversations; rebuild from request history when a session is gone.
- 📡 **Service endpoints** — model discovery, health probes and Prometheus metrics.

## 🔧 Built for integration

- **FastAPI + Claude Agent SDK**, with an OpenAI-compatible request/response layer.
- **Bearer authentication**, session limits, structured logs and graceful shutdown.
- **A fake CLI for tests**, so the default suite needs no Claude credentials.
- **Deployment smoke checks** for streaming, tools, images and session reuse.

## 🚀 Quick start

Requires **Python 3.10+** and an installed, authenticated **Claude Code CLI** on `PATH`.

```bash
pip install claude-code-openai
claudegate doctor
claudegate serve
```

The PyPI distribution is **`claude-code-openai`**; the command and Python package are **`claudegate`**.

Try the default local endpoint:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"sonnet","messages":[{"role":"user","content":"Say hello."}]}'
```

**[Configuration →](docs/CONFIGURATION.md)** · **[Integration & deployment →](docs/OPERATIONS.md)**

## 🤝 Hermes Agent & OpenClaw

- **Hermes Agent** — use Claudegate as a custom provider with `transport: chat_completions`.
- **OpenClaw** — add a custom model provider with `api: "openai-completions"`.
- **Both** — base URL `http://127.0.0.1:8080/v1`, model `sonnet`, `opus` or `haiku`. Keep bare mode enabled so the client owns tool execution.

**[Setup examples & verification →](docs/AGENT_CLIENTS.md)**

## 🔐 Access & compatibility

- **Keep access restricted.** The CLI uses permission bypass by default; coding-agent mode can execute code as the server user. Non-loopback binds require `CLAUDEGATE_API_KEY`.
- **Separate end users.** Set OpenAI's `user` field for session isolation, or require it with `CLAUDEGATE_REUSE_REQUIRES_USER=true`. See the shared-key caveat in **[Security](docs/SECURITY.md)**.
- **Compatibility has limits.** Some sampling parameters are accepted but ignored; `n > 1` is rejected. See **[Compatibility](docs/COMPATIBILITY.md)** and the Windows CLI notes in **[Operations](docs/OPERATIONS.md#platforms)**.

## 🧪 Development

From a source checkout:

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
ruff format --check src tests
mypy
```

To check a running deployment, use `claudegate smoke --base http://127.0.0.1:8080`. This makes real model requests.

---

**Python · FastAPI · Claude Agent SDK · MCP · Prometheus**

**[MIT license](LICENSE)** · Not affiliated with Anthropic.
