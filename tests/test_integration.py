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


def test_streams_are_returned_untouched_and_not_logged():
    sentinel = object()

    class Completions:
        def create(self, **kwargs):
            return sentinel

    class Chat:
        completions = Completions()

    class Fake:
        chat = Chat()

    logged = []
    patch_openai(Fake(), logged.append)

    assert Fake.chat.completions.create(**PROMPT, stream=True) is sentinel
    assert wait_for(logged, timeout=0.2) == []


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
