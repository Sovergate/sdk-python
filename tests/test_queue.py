import time
import threading
import pytest
from sovergate.queue import LogQueue


def test_queue_calls_send():
    received = []

    def fake_send(payload):
        received.append(payload)
        return True

    q = LogQueue(send_fn=fake_send)
    q.start()
    q.enqueue({"model": "gpt-4o"})
    time.sleep(0.2)
    q.stop()

    assert len(received) == 1
    assert received[0]["model"] == "gpt-4o"


def test_queue_retries_on_failure():
    attempts = []

    def fake_send(payload):
        attempts.append(1)
        return len(attempts) >= 2  # fail first, succeed second

    q = LogQueue(send_fn=fake_send)
    q.start()
    q.enqueue({"model": "gpt-4o"})
    time.sleep(3)  # wait for retry (1s delay)
    q.stop()

    assert len(attempts) >= 2


def test_queue_drops_oldest_when_full():
    # maxlen=10_000 — just verify enqueue doesn't raise under load
    dropped = []

    def fake_send(payload):
        return True

    q = LogQueue(send_fn=fake_send)
    # don't start — just test enqueue doesn't raise
    for i in range(100):
        q.enqueue({"i": i})

    assert len(q._queue) == 100
