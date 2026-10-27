# sovergate

Python SDK for [Sovergate](https://sovergate.com): EU AI Act Article 12 audit
logging for LLM calls, with PII scrubbed inside your own infrastructure.

> **Early access.** The SDK is published as an alpha and the hosted ingest
> service is not generally available yet. APIs may change before 1.0.

## How it works

Your app keeps calling OpenAI or Anthropic directly. Sovergate does not proxy
or intercept the request. After a call completes, the SDK takes a copy,
scrubs personal data **locally**, and sends the scrubbed copy to Sovergate
from a background thread.

```
Your app ─────────────────────────────► OpenAI / Anthropic
              │
              └──► scrub PII locally ──► Sovergate (background, fire and forget)
```

## Install

```bash
pip install --pre sovergate
```

`--pre` is needed while the SDK is in alpha. Requires Python 3.10+. No runtime
dependencies.

## Quick start

```python
import openai
import sovergate

sovergate.init(api_key="sov_prod_xxxx")
sovergate.instrument(openai)

# Your existing code is unchanged
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
# The response is returned immediately; logging happens in the background
```

You can also instrument a client instance, sync or async:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()
sovergate.instrument(client)
```

### Anthropic

Pass a client instance (the `anthropic` module itself has no client to patch):

```python
import anthropic
import sovergate

client = anthropic.Anthropic()        # or anthropic.AsyncAnthropic()
sovergate.init(api_key="sov_prod_xxxx")
sovergate.instrument(client)
```

### Streaming

Streamed calls are logged too: `create(stream=True)` on both providers, plus
OpenAI's `chat.completions.stream()` and Anthropic's `messages.stream()`. Your
app receives every chunk unchanged and as soon as it arrives; the SDK keeps a
copy of the text and logs it once the stream ends. If your app stops reading
part-way, what it had received up to that point is logged.

OpenAI only reports token counts for a stream when you ask for them:

```python
client.chat.completions.create(..., stream=True, stream_options={"include_usage": True})
```

Without that option the record is still logged, with 0 tokens. The SDK never
adds the option itself, because it changes the chunks your app receives.

## Stable PII tokens (recommended)

By default, detected PII is replaced with positional placeholders
(`[EMAIL_REDACTED_1]`), which are not correlatable between logs. Pass a
`pii_key` to replace values with deterministic, irreversible HMAC-SHA256 tokens
instead. The same value always maps to the same token, so Sovergate can count
unique people, IBANs and so on without ever receiving the raw values.

```python
sovergate.init(api_key="sov_prod_xxxx", pii_key="<your-secret>")
```

The `pii_key` is your secret. Keep it stable, and never send it to us: it stays
on your infrastructure, and without it the tokens cannot be reversed.

## Options

```python
sovergate.init(
    api_key="sov_prod_xxxx",           # required
    base_url="https://...",            # default: https://api.sovergate.com
    pii_key="<your-secret>",           # optional, see above
    debug=True,                        # verbose output from the "sovergate" logger only
)
```

## What is sent to Sovergate

For each completed call, one JSON record:

| Field | Content |
|---|---|
| `model` | Model name returned by the provider |
| `messages` | Your prompt messages (and Anthropic `system` prompt), **after** PII scrubbing |
| `response` | The model's text output, **after** PII scrubbing |
| `prompt_tokens`, `completion_tokens` | Token counts from the provider |
| `latency_ms`, `timestamp` | Call duration and start time |
| `finish_reason` | Why the model stopped |
| `pii_detections` | Category and placeholder of each scrubbed item, never the original value |

Never sent: the raw prompt or response, your `pii_key`, your LLM provider API
key, or request headers.

### What the scrubber detects

Regex-based, applied to every text part of the prompt and response: email
addresses, IP addresses, IBANs (including the spaced format), card numbers,
phone numbers, dates of birth, ID numbers, and names that follow a title (Mr,
Mrs, Ms, Dr, Prof). Other personal data, such as names without a title, is
**not** detected yet, so keep data minimisation in mind when you build prompts.
The code is in [`sovergate/scrubber.py`](sovergate/scrubber.py).

## Guarantees

- Never blocks your call: logging runs in a background daemon thread
- Never raises: if setup fails, the SDK logs a warning on the `sovergate`
  logger and stays out of the way
- Returns the provider's response unchanged, and passes stream chunks through
  unchanged as they arrive; `with_raw_response` keeps working
- Never changes your app's logging configuration, even with `debug=True`
- 2 second timeout on every HTTP call to Sovergate
- If Sovergate is unreachable, records are queued in memory (up to 10,000) and
  retried with exponential backoff
- A record Sovergate permanently rejects (for example after a key is revoked) is
  dropped with a warning, so it never blocks the records behind it

## Current limitations

- Only the text of a response is logged, streamed or not; tool calls are not.
- Calls made through `with_raw_response` are not logged.
- The retry queue is in memory, so queued records are lost if the process exits.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Security

Please report vulnerabilities privately. See [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)
