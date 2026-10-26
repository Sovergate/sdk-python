"""Patches OpenAI / Anthropic clients so completed calls are logged after the fact."""

import inspect
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .scrubber import ScrubContext, scrub, scrub_messages

logger = logging.getLogger("sovergate.patcher")

_PATCHED = "__sovergate_patched__"

Enqueue = Callable[[dict], None]
BuildPayload = Callable[[dict, Any, str, int], dict]


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


def _wrap(target: Any, build: BuildPayload, enqueue: Enqueue) -> None:
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

        # Streams can only be consumed once and belong to the app; not logged yet.
        if kwargs.get("stream") is True:
            return result

        def log(response: Any) -> None:
            latency_ms = int((time.monotonic() - t0) * 1000)
            _log_in_background(build, enqueue, kwargs, response, request_at, latency_ms)

        if inspect.isawaitable(result):
            async def await_and_log() -> Any:
                response = await result
                log(response)
                return response

            return await_and_log()

        log(result)
        return result  # returned immediately — logging happens in the background

    setattr(patched_create, _PATCHED, True)
    patched_create.__wrapped__ = original  # type: ignore[attr-defined]
    target.create = patched_create


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

    _wrap(openai_module.chat.completions, build, enqueue)
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

    _wrap(anthropic_client.messages, build, enqueue)
    logger.debug("sovergate: patched messages.create")
