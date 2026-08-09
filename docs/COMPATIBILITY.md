# Compatibility

What a client can send, and what actually happens to it.

## Endpoints

| Endpoint | Status |
|---|---|
| `POST /v1/chat/completions` | Full: streaming, tools, vision, usage |
| `GET /v1/models`, `GET /v1/models/{id}` | Yes |
| `GET /health`, `/health?deep=1`, `/metrics` | Yes (no auth) |
| `/v1/completions` (legacy), `/v1/embeddings`, audio, images | Not implemented — the CLI has no equivalent |

## Request fields

| Field | Behaviour |
|---|---|
| `messages` | `system`, `developer`, `user`, `assistant`, `tool` |
| `model` | Aliases resolved; unknown names passed to the CLI unchanged |
| `stream` | Yes |
| `stream_options.include_usage` | Yes — a final choice-less chunk carrying usage |
| `tools`, `tool_choice` | Yes, including parallel calls. `tool_choice: "none"` hides them entirely |
| `reasoning_effort` | Mapped onto the agent's thinking budget |
| `n` | **Rejected** unless `1` |
| `temperature`, `top_p`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `max_tokens` | **Accepted and ignored** |
| `logprobs`, `response_format`, `logit_bias` | Ignored |

The CLI does not expose sampling controls. Rejecting requests that carry them
would break every client that sets `temperature=0.7` by habit, so they are
accepted and ignored — and documented here rather than hidden. `n > 1` is the
exception: quietly returning one choice would corrupt the caller's result, so it
is an error.

## Response fields

Standard, with two additions:

- `message.reasoning_content` / `delta.reasoning_content` — extended thinking,
  kept out of `content` so it never contaminates a parsed answer. The field name
  follows the convention used by DeepSeek and OpenRouter.
- `usage.prompt_tokens_details.cached_tokens` — cache reads are counted in
  `prompt_tokens` (they are prompt tokens, just cheap ones) and reported
  separately here.

Two response headers help when debugging: `x-claudegate-session` and
`x-claudegate-mode` (`fresh`, `reused`, `continued` or `rebuilt`).

## Finish reasons

| Anthropic | OpenAI |
|---|---|
| `end_turn`, `stop_sequence`, `pause_turn` | `stop` |
| `tool_use` | `tool_calls` |
| `max_tokens` | `length` |
| `refusal` | `content_filter` |

## Errors

Always the OpenAI envelope — `{"error": {"message", "type", "param", "code"}}` —
including inside a stream, where an error is delivered as a final `data:` frame
followed by `[DONE]`.

| Status | When |
|---|---|
| `400` | Malformed request, empty `messages`, `n > 1`, a tool without a name |
| `401` | Missing or wrong API key |
| `409` | Tool results for a lost conversation, *only* when `REBUILD_ON_EXPIRY=false` |
| `429` | Every conversation slot busy, or the upstream rate limited us |
| `502` | The CLI failed to start or reported an error |
| `504` | The model did not answer within `REQUEST_TIMEOUT_S` |

## Images

`image_url` accepts base64 data URLs (`image/jpeg`, `png`, `gif`, `webp`) and
remote `http(s)` URLs, which are passed through as URL sources. `detail` is
ignored. Unsupported media types become a text placeholder rather than
disappearing, so the model knows something was attached.

`file` parts: PDFs become document blocks, other text files are inlined with
their filename. `input_audio` is not supported.
