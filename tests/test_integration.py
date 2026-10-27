"""The patched create() must behave exactly like the original, using the real provider SDKs."""

import asyncio
import json
import logging
import time

import anthropic
import httpx
import httpx2  # anthropic>=2 uses httpx2 instead of httpx
import openai
import pytest

import sovergate.client as client
from sovergate.patcher import patch_anthropic, patch_openai

OPENAI_BODY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o",
    "choices": [
        {"index": 0, "finish_reason": "stop",
         "message": {"role": "assistant", "content": "Reply to jane@example.com"}},
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
}

ANTHROPIC_BODY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": "Reply to jane@example.com"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 5},
}

PROMPT = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Email john@example.com"}]}
CLAUDE_PROMPT = {**PROMPT, "model": "claude-sonnet-5-5", "max_tokens": 100}


def transport(body, status=200, lib=httpx):
    return lib.MockTransport(lambda request: lib.Response(status, json=body))


def wait_for(items, count=1, timeout=2.0):
    deadline = time.monotonic() + timeout
    while len(items) < count and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)  # catch any unexpected extra records
    return items


def openai_client(body=OPENAI_BODY, status=200):
    return openai.OpenAI(api_key="sk-test", base_url="http://llm.test/v1", max_retries=0,
                         http_client=httpx.Client(transport=transport(body, status)))


def anthropic_client(body=ANTHROPIC_BODY):
    return anthropic.Anthropic(api_key="sk-ant-test", base_url="http://llm.test", max_retries=0,
                               http_client=httpx2.Client(transport=transport(body, lib=httpx2)))


def assert_scrubbed(payload):
    assert "@example.com" not in json.dumps(payload)
    assert payload["prompt_tokens"] == 12
    assert payload["completion_tokens"] == 5
    # john@ in the prompt and jane@ in the response: both detected, distinct placeholders
    assert [d["placeholder"] for d in payload["pii_detections"]] == ["[EMAIL_REDACTED_1]", "[EMAIL_REDACTED_2]"]


# --- OpenAI ---------------------------------------------------------------

def test_openai_sync_returns_same_data_and_logs_scrubbed_copy():
    c, logged = openai_client(), []
    patch_openai(c, logged.append)

    res = c.chat.completions.create(**PROMPT)

    assert res.choices[0].message.content == "Reply to jane@example.com"
    assert len(wait_for(logged)) == 1
    assert_scrubbed(logged[0])


def test_openai_async_client_is_logged():
    logged = []
    c = openai.AsyncOpenAI(api_key="sk-test", base_url="http://llm.test/v1", max_retries=0,
                           http_client=httpx.AsyncClient(transport=transport(OPENAI_BODY)))
    patch_openai(c, logged.append)

    res = asyncio.run(c.chat.completions.create(**PROMPT))

    assert res.choices[0].message.content == "Reply to jane@example.com"
    assert len(wait_for(logged)) == 1
    assert_scrubbed(logged[0])


def test_openai_module_level_usage_is_logged():
    logged = []
    openai.api_key = "sk-test"
    openai.base_url = "http://llm.test/v1"
    openai.max_retries = 0
    openai.http_client = httpx.Client(transport=transport(OPENAI_BODY))
    patch_openai(openai, logged.append)

    res = openai.chat.completions.create(**PROMPT)

    assert res.id == "chatcmpl-1"
    assert len(wait_for(logged)) == 1


def test_openai_with_raw_response_still_works():
    c, logged = openai_client(), []
    patch_openai(c, logged.append)

    raw = c.chat.completions.with_raw_response.create(**PROMPT)

    assert raw.http_response.status_code == 200
    assert raw.parse().id == "chatcmpl-1"


def test_openai_errors_reach_the_app_and_nothing_is_logged():
    c, logged = openai_client({"error": {"message": "bad request"}}, status=400), []
    patch_openai(c, logged.append)

    with pytest.raises(openai.BadRequestError):
        c.chat.completions.create(**PROMPT)
    assert wait_for(logged) == []


def test_patching_twice_logs_once():
    c, logged = openai_client(), []
    patch_openai(c, logged.append)
    patch_openai(c, logged.append)

    c.chat.completions.create(**PROMPT)
    assert len(wait_for(logged, count=2, timeout=0.5)) == 1


# --- Streaming -------------------------------------------------------------

def sse(events, lib=httpx):
    """A mock transport answering with server-sent events: (event name or None, data)."""
    body = "".join(
        (f"event: {name}\n" if name else "") + f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n"
        for name, data in events
    )
    return lib.MockTransport(
        lambda request: lib.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())
    )


def openai_chunk(content=None, finish_reason=None, usage=None, choices=True):
    return {
        "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"content": content} if content else {}, "finish_reason": finish_reason}]
        if choices else [],
        "usage": usage,
    }


OPENAI_STREAM = [
    (None, openai_chunk("Reply to ")),
    (None, openai_chunk("jane@example.com")),
    (None, openai_chunk(finish_reason="stop")),
    # Only sent with stream_options={"include_usage": True}
    (None, openai_chunk(choices=False, usage={"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17})),
    (None, "[DONE]"),
]


def openai_stream_client(events=OPENAI_STREAM):
    return openai.OpenAI(api_key="sk-test", base_url="http://llm.test/v1", max_retries=0,
                         http_client=httpx.Client(transport=sse(events)))


def test_openai_stream_reaches_the_app_unchanged_and_is_logged_once_it_ends():
    c, logged = openai_stream_client(), []
    patch_openai(c, logged.append)

    stream = c.chat.completions.create(**PROMPT, stream=True, stream_options={"include_usage": True})
    text = "".join(ch.choices[0].delta.content or "" for ch in stream if ch.choices)

    assert text == "Reply to jane@example.com"
    assert stream.response.status_code == 200  # the real stream's attributes are still there
    assert len(wait_for(logged)) == 1
    assert_scrubbed(logged[0])
    assert logged[0]["response"] == "Reply to [EMAIL_REDACTED_2]"
    assert logged[0]["finish_reason"] == "stop"


def test_openai_stream_without_usage_is_logged_with_zero_tokens():
    c, logged = openai_stream_client(OPENAI_STREAM[:3] + [(None, "[DONE]")]), []
    patch_openai(c, logged.append)

    for _ in c.chat.completions.create(**PROMPT, stream=True):
        pass

    assert len(wait_for(logged)) == 1
    assert logged[0]["prompt_tokens"] == 0 and logged[0]["completion_tokens"] == 0
    assert logged[0]["response"] == "Reply to [EMAIL_REDACTED_2]"


def test_openai_stream_abandoned_part_way_logs_what_the_app_received():
    c, logged = openai_stream_client(), []
    patch_openai(c, logged.append)

    with c.chat.completions.create(**PROMPT, stream=True) as stream:
        first = next(iter(stream))
    assert first.choices[0].delta.content == "Reply to "

    assert len(wait_for(logged, count=2, timeout=0.5)) == 1
    assert logged[0]["response"] == "Reply to "
    assert logged[0]["finish_reason"] is None


def test_openai_async_stream_is_logged():
    logged = []
    c = openai.AsyncOpenAI(api_key="sk-test", base_url="http://llm.test/v1", max_retries=0,
                           http_client=httpx.AsyncClient(transport=sse(OPENAI_STREAM)))
    patch_openai(c, logged.append)

    async def run():
        stream = await c.chat.completions.create(**PROMPT, stream=True)
        return "".join([ch.choices[0].delta.content or "" async for ch in stream if ch.choices])

    assert asyncio.run(run()) == "Reply to jane@example.com"
    assert len(wait_for(logged)) == 1
    assert_scrubbed(logged[0])


def test_openai_stream_helper_is_logged():
    c, logged = openai_stream_client(), []
    patch_openai(c, logged.append)

    with c.chat.completions.stream(**PROMPT, stream_options={"include_usage": True}) as stream:
        final = stream.get_final_completion()

    assert final.choices[0].message.content == "Reply to jane@example.com"
    assert len(wait_for(logged)) == 1
    assert_scrubbed(logged[0])


def test_a_stream_that_is_never_read_is_not_logged():
    c, logged = openai_stream_client(), []
    patch_openai(c, logged.append)

    c.chat.completions.create(**PROMPT, stream=True).close()
    assert wait_for(logged, timeout=0.2) == []


def test_unexpected_chunks_still_reach_the_app():
    chunks = [object(), {"not": "a chunk"}, None]

    class Completions:
        def create(self, **kwargs):
            return iter(chunks)

    class Fake:
        class chat:
            completions = Completions()

    logged = []
    patch_openai(Fake, logged.append)

    assert list(Fake.chat.completions.create(**PROMPT, stream=True)) == chunks
    assert len(wait_for(logged)) == 1  # logged with what could be read: nothing
    assert logged[0]["response"] == ""


# --- Anthropic ------------------------------------------------------------

def test_anthropic_sync_returns_same_data_and_logs_scrubbed_copy():
    c, logged = anthropic_client(), []
    patch_anthropic(c, logged.append)

    res = c.messages.create(**CLAUDE_PROMPT)

    assert res.content[0].text == "Reply to jane@example.com"
    assert len(wait_for(logged)) == 1
    assert_scrubbed(logged[0])
    assert logged[0]["finish_reason"] == "end_turn"


def test_anthropic_async_client_is_logged():
    logged = []
    c = anthropic.AsyncAnthropic(api_key="sk-ant-test", base_url="http://llm.test", max_retries=0,
                                 http_client=httpx2.AsyncClient(transport=transport(ANTHROPIC_BODY, lib=httpx2)))
    patch_anthropic(c, logged.append)

    res = asyncio.run(c.messages.create(**CLAUDE_PROMPT))

    assert res.id == "msg_1"
    assert len(wait_for(logged)) == 1


ANTHROPIC_STREAM = [
    ("message_start", {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 12, "output_tokens": 1}}}),
    ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Reply to "}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0,
                             "delta": {"type": "text_delta", "text": "jane@example.com"}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                       "usage": {"output_tokens": 5}}),
    ("message_stop", {"type": "message_stop"}),
]


def anthropic_stream_client(cls=anthropic.Anthropic, http=httpx2.Client):
    return cls(api_key="sk-ant-test", base_url="http://llm.test", max_retries=0,
               http_client=http(transport=sse(ANTHROPIC_STREAM, lib=httpx2)))


def test_anthropic_create_stream_is_logged():
    c, logged = anthropic_stream_client(), []
    patch_anthropic(c, logged.append)

    events = list(c.messages.create(**CLAUDE_PROMPT, stream=True))

    assert [e.type for e in events][-1] == "message_stop"
    assert len(wait_for(logged)) == 1
    assert_scrubbed(logged[0])
    assert logged[0]["finish_reason"] == "end_turn"


def test_anthropic_stream_helper_is_logged():
    c, logged = anthropic_stream_client(), []
    patch_anthropic(c, logged.append)

    with c.messages.stream(**CLAUDE_PROMPT) as stream:
        text = "".join(stream.text_stream)

    assert text == "Reply to jane@example.com"
    assert len(wait_for(logged)) == 1
    assert_scrubbed(logged[0])
    assert logged[0]["finish_reason"] == "end_turn"


def test_anthropic_async_stream_helper_is_logged():
    c, logged = anthropic_stream_client(anthropic.AsyncAnthropic, httpx2.AsyncClient), []
    patch_anthropic(c, logged.append)

    async def run():
        async with c.messages.stream(**CLAUDE_PROMPT) as stream:
            return (await stream.get_final_message()).content[0].text

    assert asyncio.run(run()) == "Reply to jane@example.com"
    assert len(wait_for(logged)) == 1
    assert_scrubbed(logged[0])


def test_instrument_with_anthropic_module_explains_what_to_pass(caplog):
    client._state.update(queue=None)
    client.init(api_key="sov_test_x", base_url="http://127.0.0.1:9")
    try:
        with caplog.at_level(logging.WARNING, logger="sovergate"):
            client.instrument(anthropic)
        assert "Anthropic client instance" in caplog.text
    finally:
        client.shutdown()
        client._state.update(queue=None)
