"""Patches OpenAI / Anthropic clients so completed calls are logged after the fact."""

import inspect
import logging
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Optional

from .scrubber import ScrubContext, scrub, scrub_messages

logger = logging.getLogger("sovergate.patcher")

_PATCHED = "__sovergate_patched__"

Enqueue = Callable[[dict], None]
BuildPayload = Callable[[dict, Any, str, int], dict]
Done = Callable[[Any], None]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_in_background(build: BuildPayload, enqueue: Enqueue, kwargs: dict, response: Any,
                       request_at: str, latency_ms: int) -> None:
    def run() -> None:
        try:
            enqueue(build(kwargs, response, request_at, latency_ms))
        except Exception as e:
            logger.debug("sovergate background log error: %s", e)

    try:
        threading.Thread(target=run, daemon=True, name="sovergate-log").start()
    except Exception as e:
        logger.debug("sovergate could not start log thread: %s", e)


def _safe(fn: Callable[..., Any], *args: Any) -> Any:
    try:
        return fn(*args)
    except Exception as e:
        logger.debug("sovergate stream bookkeeping error: %s", e)
        return None


class _LoggedStream:
    """
    Stands in for a provider's stream. Every chunk reaches the app unchanged and in
    real time; a copy of the text is collected on the side and logged in the
    background once the stream ends, is closed, or is abandoned part-way. Anything
    else (``response``, ``close``, ...) is passed through to the real stream.
    """

    def __init__(self, stream: Any, collector: Any, done: Done) -> None:
        object.__setattr__(self, "_sg_stream", stream)
        object.__setattr__(self, "_sg_collector", collector)
        object.__setattr__(self, "_sg_done", done)
        object.__setattr__(self, "_sg_gen", None)
        object.__setattr__(self, "_sg_finished", False)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_sg_"):
            raise AttributeError(name)
        return getattr(self._sg_stream, name)

    def _sg_finish(self) -> None:
        if self._sg_finished:
            return
        object.__setattr__(self, "_sg_finished", True)
        response = _safe(self._sg_collector.response)
        if response is not None:
            _safe(self._sg_done, response)

    # Like the providers' streams, one underlying iterator serves for-loops and next().
    def _sg_chunks(self) -> Any:
        if self._sg_gen is None:
            object.__setattr__(self, "_sg_gen", self._sg_generate())
        return self._sg_gen

    def _sg_generate(self) -> Any:
        try:
            for chunk in self._sg_stream:
                _safe(self._sg_collector.add, chunk)
                yield chunk
        finally:
            self._sg_finish()

    def __iter__(self) -> Any:
        return self._sg_chunks()

    def __next__(self) -> Any:
        return next(self._sg_chunks())

    def __enter__(self) -> "_LoggedStream":
        self._sg_stream.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        try:
            return self._sg_stream.__exit__(*exc)
        finally:
            self._sg_finish()

    def close(self) -> Any:
        try:
            return self._sg_stream.close()
        finally:
            self._sg_finish()


class _AsyncLoggedStream(_LoggedStream):
    """The async-client version of ``_LoggedStream``."""

    async def _sg_generate(self) -> Any:  # type: ignore[override]
        try:
            async for chunk in self._sg_stream:
                _safe(self._sg_collector.add, chunk)
                yield chunk
        finally:
            self._sg_finish()

    def __aiter__(self) -> Any:
        return self._sg_chunks()

    async def __anext__(self) -> Any:
        return await self._sg_chunks().__anext__()

    async def __aenter__(self) -> "_AsyncLoggedStream":
        await self._sg_stream.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        try:
            return await self._sg_stream.__aexit__(*exc)
        finally:
            self._sg_finish()

    async def close(self) -> Any:  # type: ignore[override]
        try:
            return await self._sg_stream.close()
        finally:
            self._sg_finish()


def _when_done(build: BuildPayload, enqueue: Enqueue, kwargs: dict, request_at: str, t0: float) -> Done:
    def done(response: Any) -> None:
        latency_ms = int((time.monotonic() - t0) * 1000)
        _log_in_background(build, enqueue, kwargs, response, request_at, latency_ms)

    return done


def _wrap(target: Any, build: BuildPayload, enqueue: Enqueue, collector: Callable[[], Any]) -> None:
    """
    Replace target.create with a wrapper that returns the original result unchanged
    and logs a scrubbed copy in a daemon thread. Works for sync and async clients:
    async SDK methods are sync wrappers returning a coroutine, so awaitability is
    checked on the result rather than on the function.
    """
    original = target.create
    if getattr(original, _PATCHED, False):
        return

    def patched_create(*args: Any, **kwargs: Any) -> Any:
        request_at = _utcnow()
        t0 = time.monotonic()

        result = original(*args, **kwargs)
        done = _when_done(build, enqueue, kwargs, request_at, t0)

        if kwargs.get("stream") is True:
            # Logged when the stream ends; latency then covers the whole response.
            if inspect.isawaitable(result):
                async def await_stream() -> Any:
                    return _AsyncLoggedStream(await result, collector(), done)

                return await_stream()
            return _LoggedStream(result, collector(), done)

        if inspect.isawaitable(result):
            async def await_and_log() -> Any:
                response = await result
                done(response)
                return response

            return await_and_log()

        done(result)
        return result  # returned immediately — logging happens in the background

    setattr(patched_create, _PATCHED, True)
    patched_create.__wrapped__ = original  # type: ignore[attr-defined]
    target.create = patched_create


class _OpenAIStreamCollector:
    """Rebuilds the parts of a ChatCompletion that are logged, from its chunks."""

    def __init__(self) -> None:
        self.seen = False
        self.model: Optional[str] = None
        self.parts: list = []
        self.finish_reason: Optional[str] = None
        self.usage: Any = None

    def add(self, chunk: Any) -> None:
        self.seen = True
        self.model = getattr(chunk, "model", None) or self.model
        # Only sent when the app asks for it with stream_options={"include_usage": True}.
        self.usage = getattr(chunk, "usage", None) or self.usage
        for choice in getattr(chunk, "choices", None) or []:
            if getattr(choice, "index", 0) != 0:
                continue
            text = getattr(getattr(choice, "delta", None), "content", None)
            if text:
                self.parts.append(text)
            self.finish_reason = getattr(choice, "finish_reason", None) or self.finish_reason

    def response(self) -> Any:
        if not self.seen:
            return None
        choice = SimpleNamespace(message=SimpleNamespace(content="".join(self.parts)), finish_reason=self.finish_reason)
        return SimpleNamespace(model=self.model, choices=[choice], usage=self.usage)


class _AnthropicStreamCollector:
    """Rebuilds the parts of a Message that are logged, from its stream events."""

    def __init__(self) -> None:
        self.seen = False
        self.model: Optional[str] = None
        self.parts: list = []
        self.stop_reason: Optional[str] = None
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, event: Any) -> None:
        self.seen = True
        kind = getattr(event, "type", None)
        if kind == "message_start":
            message = event.message
            self.model = message.model
            self.input_tokens = getattr(message.usage, "input_tokens", 0) or 0
            self.output_tokens = getattr(message.usage, "output_tokens", 0) or 0
        elif kind == "content_block_delta" and getattr(event.delta, "type", None) == "text_delta":
            self.parts.append(event.delta.text)
        elif kind == "message_delta":
            self.stop_reason = getattr(event.delta, "stop_reason", None) or self.stop_reason
            usage = getattr(event, "usage", None)
            self.output_tokens = getattr(usage, "output_tokens", None) or self.output_tokens
            self.input_tokens = getattr(usage, "input_tokens", None) or self.input_tokens

    def response(self) -> Any:
        if not self.seen:
            return None
        return SimpleNamespace(
            model=self.model,
            content=[SimpleNamespace(type="text", text="".join(self.parts))],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=self.input_tokens, output_tokens=self.output_tokens),
        )


class _LoggedStreamManager:
    """
    Wraps the manager returned by Anthropic's ``messages.stream()``. The app gets the
    real MessageStream; on exit, the message it accumulated is logged.
    """

    def __init__(self, manager: Any, done: Done) -> None:
        self._sg_manager = manager
        self._sg_done = done
        self._sg_stream: Any = None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_sg_"):
            raise AttributeError(name)
        return getattr(self._sg_manager, name)

    def _sg_finish(self) -> None:
        stream, self._sg_stream = self._sg_stream, None
        if stream is None:
            return
        # Raises if no event arrived; then there is nothing to log.
        message = _safe(lambda: stream.current_message_snapshot)
        if message is not None:
            _safe(self._sg_done, message)

    def __enter__(self) -> Any:
        self._sg_stream = self._sg_manager.__enter__()
        return self._sg_stream

    def __exit__(self, *exc: Any) -> Any:
        try:
            return self._sg_manager.__exit__(*exc)
        finally:
            self._sg_finish()

    async def __aenter__(self) -> Any:
        self._sg_stream = await self._sg_manager.__aenter__()
        return self._sg_stream

    async def __aexit__(self, *exc: Any) -> Any:
        try:
            return await self._sg_manager.__aexit__(*exc)
        finally:
            self._sg_finish()


def _wrap_stream_helper(target: Any, build: BuildPayload, enqueue: Enqueue) -> None:
    """Anthropic's messages.stream() sends its own request, without going through create()."""
    original = getattr(target, "stream", None)
    if original is None or getattr(original, _PATCHED, False):
        return

    def patched_stream(*args: Any, **kwargs: Any) -> Any:
        request_at = _utcnow()
        t0 = time.monotonic()
        manager = original(*args, **kwargs)
        return _LoggedStreamManager(manager, _when_done(build, enqueue, kwargs, request_at, t0))

    setattr(patched_stream, _PATCHED, True)
    patched_stream.__wrapped__ = original  # type: ignore[attr-defined]
    target.stream = patched_stream


def patch_openai(openai_module: Any, enqueue: Enqueue, pii_key: Optional[bytes] = None) -> None:
    """Wraps chat.completions.create on the openai module or an OpenAI / AsyncOpenAI client."""

    def build(kwargs: dict, response: Any, request_at: str, latency_ms: int) -> dict:
        context = ScrubContext()
        scrubbed_messages, detections = scrub_messages(kwargs.get("messages", []), pii_key, context)

        choice = response.choices[0] if response.choices else None
        response_text = choice.message.content if choice else ""
        scrubbed_response = ""
        if response_text:
            result = scrub(response_text, pii_key, context)
            scrubbed_response = result.text
            detections = detections + result.detections

        usage = response.usage
        return {
            "model": response.model or kwargs.get("model", ""),
            "messages": scrubbed_messages,
            "response": scrubbed_response,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "latency_ms": latency_ms,
            "timestamp": request_at,
            "finish_reason": choice.finish_reason if choice else None,
            "pii_detections": [
                {"category": d.category, "placeholder": d.placeholder} for d in detections
            ],
        }

    # chat.completions.stream() calls create(stream=True), so this covers it too.
    _wrap(openai_module.chat.completions, build, enqueue, _OpenAIStreamCollector)
    logger.debug("sovergate: patched chat.completions.create")


def patch_anthropic(anthropic_client: Any, enqueue: Enqueue, pii_key: Optional[bytes] = None) -> None:
    """Wraps messages.create on an Anthropic / AsyncAnthropic client."""

    def build(kwargs: dict, response: Any, request_at: str, latency_ms: int) -> dict:
        all_messages = []
        system = kwargs.get("system")
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(kwargs.get("messages", []))

        context = ScrubContext()
        scrubbed_messages, detections = scrub_messages(all_messages, pii_key, context)

        response_text = "".join(
            block.text for block in (response.content or []) if getattr(block, "type", None) == "text"
        )
        scrubbed_response = ""
        if response_text:
            result = scrub(response_text, pii_key, context)
            scrubbed_response = result.text
            detections = detections + result.detections

        usage = response.usage
        return {
            "model": response.model or kwargs.get("model", ""),
            "messages": scrubbed_messages,
            "response": scrubbed_response,
            "prompt_tokens": usage.input_tokens if usage else 0,
            "completion_tokens": usage.output_tokens if usage else 0,
            "latency_ms": latency_ms,
            "timestamp": request_at,
            "finish_reason": response.stop_reason,
            "pii_detections": [
                {"category": d.category, "placeholder": d.placeholder} for d in detections
            ],
        }

    _wrap(anthropic_client.messages, build, enqueue, _AnthropicStreamCollector)
    _wrap_stream_helper(anthropic_client.messages, build, enqueue)
    logger.debug("sovergate: patched messages.create and messages.stream")
