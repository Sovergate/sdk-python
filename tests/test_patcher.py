"""Tests that the patcher intercepts calls, logs asynchronously, and never blocks."""

import time
import threading
from types import SimpleNamespace
import pytest
from sovergate.patcher import patch_openai


def _make_fake_openai(response_delay: float = 0.0):
    """Create a minimal fake openai module."""

    def fake_create(*args, **kwargs):
        if response_delay:
            time.sleep(response_delay)
        return SimpleNamespace(
            model=kwargs.get("model", "gpt-4o"),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Hello!"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )

    openai = SimpleNamespace()
    openai.chat = SimpleNamespace()
    openai.chat.completions = SimpleNamespace()
    openai.chat.completions.create = fake_create
    return openai


def test_patcher_intercepts_and_enqueues():
    received = []

    def enqueue(payload):
        received.append(payload)

    openai = _make_fake_openai()
    patch_openai(openai, enqueue)

    result = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
    )

    # response returned immediately
    assert result.choices[0].message.content == "Hello!"

    # background thread should enqueue shortly after
    time.sleep(0.2)
    assert len(received) == 1
    assert received[0]["model"] == "gpt-4o"
    assert received[0]["prompt_tokens"] == 10
    assert received[0]["completion_tokens"] == 5


def test_patcher_scrubs_pii_before_enqueue():
    received = []

    def enqueue(payload):
        received.append(payload)

    openai = _make_fake_openai()
    patch_openai(openai, enqueue)

    openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Email me at secret@example.com"}],
    )

    time.sleep(0.2)
    assert len(received) == 1
    msg_content = received[0]["messages"][0]["content"]
    assert "secret@example.com" not in msg_content
    assert "EMAIL_REDACTED" in msg_content


def test_patcher_does_not_block_response():
    """Response must arrive before background logging completes."""
    log_times = []
    response_time = None

    original_enqueue_ref = []

    def enqueue(payload):
        time.sleep(0.1)  # simulate slow logging
        log_times.append(time.monotonic())

    openai = _make_fake_openai()
    patch_openai(openai, enqueue)

    t0 = time.monotonic()
    openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Fast?"}],
    )
    response_received_at = time.monotonic() - t0

    time.sleep(0.3)  # let background thread finish

    # response should arrive in < 50ms (no 100ms sleep on main thread)
    assert response_received_at < 0.05, f"Response took {response_received_at:.3f}s — patcher is blocking!"
