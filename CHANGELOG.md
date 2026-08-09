# Changelog

## 0.1.0

First release.

- OpenAI-compatible `/v1/chat/completions`: streaming and non-streaming, tool
  calling (including parallel calls), image and file input, usage accounting.
- Conversations are held open between requests and recognised again by hashing
  the history, so a follow-up turn sends one message instead of the transcript.
- Tool calls park the conversation instead of unwinding it: the result arrives
  on a later request and resumes the same context, with nothing replayed.
- Tool results for a conversation the server no longer holds rebuild it from the
  request history rather than failing a turn the client cannot retry.
- `claudegate doctor`, `claudegate smoke` and `claudegate install-service`.
- Linux, macOS and Windows. The hermetic suite runs on all three in CI, and the
  built wheel is installed and executed on all three before release.
- `claudegate.testing`: a fake CLI speaking the real control protocol, so
  integrations can be tested without a CLI, a token or a network. It can also
  be told to be slow (`connect_delay`) or to race (`eager_tools`), because the
  two worst defects found while reviewing this release were both invisible to a
  fake that answered instantly and always in the same order.
