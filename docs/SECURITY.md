# 🔐 Security & conversation isolation

## Security

The CLI is driven with permission bypass, so **anyone who can reach the port can
run code as the user running this server**. Two things follow, both enforced:

- binding anything other than loopback without `CLAUDEGATE_API_KEY` set is
  refused at startup, with an explanation rather than a stack trace;
- keys are compared in constant time, and `/health` and `/metrics` are the only
  endpoints that never need one.

### Conversation isolation

Reuse matches a request against a live conversation's history, so two callers
must not be able to collide. Two things prevent that:

1. Conversations are partitioned by caller — the presented API key (hashed,
   never logged) plus OpenAI's `user` field.
2. Continuing one requires handing back the answer it actually produced. An
   attacker can guess an opening — a published system prompt and a templated
   first message is not a secret — but not what the model said.

**The residual risk, stated plainly:** if one API key is shared by many end
users *and* the reply to the opening turn is predictable (a fixed greeting), a
caller who guesses both could land in someone else's conversation. Set `user`
per end user — the OpenAI convention anyway — and the partition is exact. Or set
`CLAUDEGATE_REUSE_REQUIRES_USER=true`, which declines to reuse anything for a
request that omits it, or `CLAUDEGATE_REUSE_SESSIONS=false` to turn the whole
optimisation off. Reuse is only ever a saving; a fresh conversation is always
correct.
