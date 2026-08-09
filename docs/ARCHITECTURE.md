# Architecture

## The shape of the problem

An OpenAI client is stateless. It re-sends the entire conversation on every
request and expects a self-contained answer back. The Claude Code CLI is the
opposite: it is a long-lived process that holds the conversation and streams
events about it.

Bridging the two naively means folding the whole history into one prompt and
spawning a fresh process per request. That works, and it is what makes this
kind of server slow: every turn re-sends and re-reads everything that came
before it, and a tool call costs a full replay because the process that asked
for the tool is long gone by the time the answer arrives.

`claudegate` keeps the process. Everything below follows from that decision.

## Layers

```
routes/          HTTP: parse, authenticate, stream SSE, shape errors
  chat.py        POST /v1/chat/completions
  meta.py        models, health (+ deep probe), metrics

bridge/          conversations
  manager.py     continuation / reuse / fresh, eviction, reaping
  session.py     one live conversation: the pump, the turn boundary, tool parking
  toolbelt.py    client tools → in-process MCP server; call correlation
  continuity.py  hash chains that recognise a conversation we already hold

openai_api/      pure format translation, no I/O
  inbound.py     OpenAI messages → Anthropic content blocks
  outbound.py    Claude events → OpenAI chunks and completions
  schema.py      the request/response models

testing.py       a fake CLI that speaks the real control protocol
smoke.py         end-to-end checks against a running deployment
```

The translation layer is pure functions on purpose. Format mapping is where the
bugs that reach users live — an image dropped from a fold, a `finish_reason`
that says `stop` when the model wanted a tool — and pure functions make those
assertable without a process, a token or a network.

## Why the official SDK

The transport is Anthropic's [`claude-agent-sdk`][sdk], not a hand-rolled
subprocess and stream-json parser. Three things fall out of that:

1. **In-process MCP.** Client tools are published on an SDK MCP server that runs
   in this process. No bridge process, no socket, no per-conversation config
   file — the tool handler is a coroutine.
2. **A `Transport` seam.** The SDK takes an injectable transport, so the entire
   server can be tested against a scripted fake (`claudegate.testing`).
3. **Protocol drift is someone else's problem.** When the CLI's wire format
   moves, the SDK moves with it.

## The turn boundary

A `LiveSession` owns a connected client and a pump task that drains it forever,
translating what comes out into five events: `TextDelta`, `ReasoningDelta`,
`ToolCallsRequested`, `TurnFinished`, `TurnFailed`. A `None` on the queue marks
the end of an HTTP turn.

A turn ends when either:

* the run finishes (`ResultMessage`), or
* the model asks for tools the client owns.

The second case is the interesting one. The tool handler does not run anything —
it awaits a future. The HTTP response closes with `finish_reason: "tool_calls"`,
the CLI stays blocked on the MCP response, and the conversation is left standing
with all of its context. When the client posts the results, the futures resolve
and the same conversation continues.

### Correlating a tool call

MCP hands the handler its arguments but not the `tool_use` id Anthropic assigned
to the call — and that id is what the client answers with. `ToolCorrelator`
re-attaches it by matching each invocation against the `tool_use` blocks seen in
the assistant message that triggered it: exact arguments first, then name, then
position. The CLI dispatches handlers in order, so position alone would usually
work; "usually" is not good enough when the cost of being wrong is a result
attributed to the wrong call.

There is also a race to handle. The CLI dispatches `tools/call` on its own task,
so a handler can be entered before the pump has processed the assistant message
that describes it. Handlers wait on an event for the registry to catch up.

## Recognising a conversation

Each live conversation remembers a hash chain of the messages it has been given.
An incoming request is matched against those chains; a chain that is a *prefix*
of the incoming history identifies the conversation, and only the messages after
the prefix are sent.

Only `user` and `tool` messages go into the chain. Clients routinely hand back a
lightly edited copy of what we produced — trimmed whitespace, dropped reasoning,
re-serialised tool arguments — and hashing that would miss matches that are
really there. The user side is echoed verbatim, so it is the reliable half.

Model, system prompt and tool set are fixed when the CLI is spawned, so they are
folded into an identity key rather than checked case by case.

### Why matching on history is not enough

Two callers can legitimately share a prefix. A published system prompt and a
templated opening message is not a secret — it is what a deployment looks like —
so history alone would let one caller's request land in another's live
conversation. Worse in the other direction: an attacker can *prime* a
conversation with an injected instruction and wait for a victim whose opening
matches to arrive in it.

Two independent checks close that:

1. **The caller's identity is part of the session key** — the presented API key,
   hashed, plus OpenAI's `user` field, so one shared key serving many end users
   is still partitioned.
2. **Continuing requires proof of receipt.** The request must hand back the
   answer the conversation actually produced (whitespace-normalised). An
   attacker can guess an opening; they cannot guess what the model said. Every
   OpenAI client echoes it anyway, because that is how the format works — and a
   client that rewrites our answers more heavily than whitespace simply gets a
   fresh conversation: slower, never wrong.

Editing history mid-conversation (a regenerate, a branch) simply fails to match,
and starts a new conversation. That is the correct answer, not a fallback.

## Failure modes, on purpose

* **Expired conversation.** Tool results arrive for a conversation we no longer
  hold. The request contains the whole history, so it is rebuilt rather than
  refused — a `409` here fails a turn the client has no way to retry.
* **A client that answers only some calls.** The unanswered handlers are
  released with an explanatory result instead of hanging until the TTL.
* **Pool exhaustion.** The idlest conversation is evicted; if everything is
  busy, `429` with `Retry-After` rather than an unbounded queue.
* **Shutdown.** The lifespan closes every conversation, so a restart never
  leaves orphaned CLI processes behind.

[sdk]: https://github.com/anthropics/claude-agent-sdk-python
