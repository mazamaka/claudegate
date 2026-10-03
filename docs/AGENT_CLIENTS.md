# 🤝 Hermes Agent & OpenClaw

Claudegate exposes Claude Code through **Chat Completions**. The agent client sends its tools, executes requested calls and returns the results; Claudegate keeps the underlying Claude conversation alive.

## 🔌 Start the gateway

Use an installed, authenticated Claude Code CLI. Start Claudegate on the same machine as the client:

```bash
claudegate doctor
claudegate serve
```

Keep **`CLAUDEGATE_BARE_MODE=true`** (the default), so Hermes or OpenClaw owns tool execution. Available model aliases include **`sonnet`**, **`opus`** and **`haiku`**; model availability depends on the Claude account.

For authentication, set `CLAUDEGATE_API_KEY` in the gateway environment and the same value in the client environment. A key is optional for the default loopback bind; it is required for a non-loopback bind. See [security and user isolation](SECURITY.md) before sharing a gateway.

## 🪽 Hermes Agent

Merge into `~/.hermes/config.yaml`:

```yaml
providers:
  claudegate:
    api: http://127.0.0.1:8080/v1
    transport: chat_completions
    # key_env: CLAUDEGATE_API_KEY  # enable when the gateway requires a key

model:
  provider: custom:claudegate
  default: sonnet
```

For an authenticated gateway, enable `key_env` and put the key in Hermes's environment or `~/.hermes/.env`. A keyless local endpoint needs neither setting.

You can also select the provider with `hermes model`, or switch in a session with `/model custom:claudegate:sonnet`.

Reference: [Hermes custom-provider configuration](https://hermes-agent.nousresearch.com/docs/integrations/providers/).

## 🦞 OpenClaw

Merge into `~/.openclaw/openclaw.json`:

```json5
{
  agents: {
    defaults: { model: { primary: "claudegate/sonnet" } },
  },
  models: {
    mode: "merge",
    providers: {
      claudegate: {
        baseUrl: "http://127.0.0.1:8080/v1",
        api: "openai-completions",
        apiKey: "local-no-auth",
        models: [
          { id: "sonnet", name: "Claude Sonnet via Claudegate", input: ["text", "image"] },
        ],
      },
    },
  },
}
```

`local-no-auth` is a placeholder for a keyless local gateway. If authentication is enabled, replace it with `"${CLAUDEGATE_API_KEY}"` and provide the real key through OpenClaw's environment. Add `opus` or `haiku` to the model list if needed.

Reference: [OpenClaw custom providers](https://docs.openclaw.ai/concepts/model-providers/custom-providers).

## 🧪 Verify your deployment

```bash
claudegate smoke --base http://127.0.0.1:8080
```

This spends real model tokens. Then ask your agent a short question, exercise one harmless client tool and send a follow-up. This checks the client settings as well as the gateway.

- **Use Chat Completions.** Responses and Anthropic Messages endpoints are not implemented.
- **Check the network address.** Inside a container, `127.0.0.1` refers to that container. Use a reachable, protected gateway address instead.
- **Know the coverage.** The offline suite exercises the gateway and real SDK/MCP dispatch with a fake CLI. The opt-in live suite checks text, vision, tool continuation and session reuse against Claude Code. Neither suite launches Hermes or OpenClaw itself; these examples follow their documented custom-provider interfaces.
- **Check API limits.** Some generation parameters are accepted but ignored; see [compatibility](COMPATIBILITY.md).
- **Use the declared SDK version.** Claudegate requires `claude-agent-sdk>=0.2.163`, verified with MCP 2. Older SDK releases can install alongside MCP 2 but fail when creating the tool server.
