"""Local queue with retry/backoff — survives transient API outages without blocking the app."""

import threading
import time
import logging
from collections import deque
from typing import Any, Callable

logger = logging.getLogger("sovergate.queue")

_MAX_QUEUE = 10_000
_RETRY_DELAYS = [1, 2, 4, 8, 16, 30, 60]  # seconds between retries


class LogQueue:
    def __init__(self, send_fn: Callable[[dict[str, Any]], bool]) -> None:
        self._send = send_fn
        self._queue: deque[dict[str, Any]] = deque(maxlen=_MAX_QUEUE)
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name="sovergate-queue")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._event.set()

    def enqueue(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._queue.append(payload)
        self._event.set()

    def _worker(self) -> None:
        retry_index = 0
        while self._running:
            with self._lock:
                has_item = bool(self._queue)

            if not has_item:
                # Nothing to send — block until something is enqueued or stop() is called
                self._event.wait()
                self._event.clear()
                continue

            with self._lock:
                payload = self._queue[0]

            success = False
            try:
                success = self._send(payload)
            except Exception as e:
                logger.debug("sovergate queue send error: %s", e)

            if success:
                with self._lock:
                    self._queue.popleft()
                retry_index = 0
            else:
                delay = _RETRY_DELAYS[min(retry_index, len(_RETRY_DELAYS) - 1)]
                retry_index += 1
                logger.debug("sovergate retrying in %ds", delay)
                time.sleep(delay)
                # loop continues — item stays in queue, we try again
