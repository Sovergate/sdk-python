"""Public entry point — init() and instrument()."""

import logging
from typing import Any, Optional

from .sender import send
from .queue import LogQueue

logger = logging.getLogger("sovergate")

_state: dict = {
    "api_key": None,
    "base_url": "https://api.sovergate.com",
    "pii_key": None,
    "queue": None,
}


def init(
    api_key: str,
    base_url: str = "https://api.sovergate.com",
    pii_key: Optional[str] = None,
    debug: bool = False,
) -> None:
    """
    Call once at app startup before any LLM calls.

    Args:
        api_key:  Your sovergate SDK key (starts with sov_).
        base_url: Override API endpoint (useful for self-hosted or local dev).
        pii_key:  Secret used to derive deterministic, irreversible PII tokens.
                  The same value always maps to the same token, so the platform
                  can count unique entities without ever holding raw PII. Keep it
                  secret and stable — it NEVER leaves your infrastructure. If
                  omitted, tokens are positional and not correlatable across logs.
        debug:    Enable verbose logging from the SDK.
    """
    try:
        if debug:
            _enable_debug_logging()

        _state["api_key"] = api_key
        _state["base_url"] = base_url.rstrip("/")
        _state["pii_key"] = pii_key.encode("utf-8") if pii_key else None
        if _state["pii_key"] is None:
            logger.debug(
                "sovergate: no pii_key set — PII tokens will be positional and not "
                "correlatable across logs (unique-entity metrics unavailable)."
            )

        def _send(payload: dict) -> bool:
            return send(api_key, payload, base_url=_state["base_url"])

        q = LogQueue(send_fn=_send)
        q.start()
        _state["queue"] = q

        logger.debug("sovergate initialised (base_url=%s)", _state["base_url"])
    except Exception as e:
        # Never let SDK setup break the host app; logging just stays disabled.
        _state["queue"] = None
        logger.warning("sovergate: init failed, logging disabled: %s", e)


def _enable_debug_logging() -> None:
    """Verbose output for the SDK's own loggers only — never the app's root logger."""
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
        logger.addHandler(handler)


def instrument(library: Any) -> None:
    """
    Patch an LLM client library so all calls are automatically logged.

    Supported:
        - the openai module, or an OpenAI / AsyncOpenAI client   (chat.completions.create)
        - an Anthropic / AsyncAnthropic client                     (messages.create)

    Usage:
        import openai
        sovergate.instrument(openai)

        import anthropic
        client = anthropic.Anthropic()
        sovergate.instrument(client)

    The original library and its responses are completely unchanged.
    Never raises: if instrumentation isn't possible, a warning is logged and
    the library is left untouched.
    """
    if _state["queue"] is None:
        logger.warning(
            "sovergate: instrument() called before init(api_key=...) — LLM calls will not be logged."
        )
        return

    try:
        enqueue = _state["queue"].enqueue
        pii_key = _state["pii_key"]
        module_name = getattr(library, "__name__", "") or getattr(library, "__class__", type(library)).__name__

        if module_name == "openai" or hasattr(library, "chat"):
            from .patcher import patch_openai
            patch_openai(library, enqueue, pii_key)
            return

        if module_name == "anthropic" and not hasattr(library, "messages"):
            logger.warning(
                "sovergate: pass an Anthropic client instance, not the anthropic module — "
                "e.g. sovergate.instrument(anthropic.Anthropic()). LLM calls will not be logged."
            )
            return

        if hasattr(library, "messages"):
            from .patcher import patch_anthropic
            patch_anthropic(library, enqueue, pii_key)
            return

        logger.warning(
            "sovergate: unsupported library '%s' — LLM calls will not be logged. "
            "Supported: openai, anthropic. Open an issue at "
            "https://github.com/sovergate/sdk-python/issues",
            module_name,
        )
    except Exception as e:
        logger.warning("sovergate: instrument() failed, LLM calls will not be logged: %s", e)


def shutdown() -> None:
    """Stop the background sender thread. Records still queued at that point are discarded."""
    try:
        if _state["queue"]:
            _state["queue"].stop()
    except Exception as e:
        logger.debug("sovergate: shutdown error: %s", e)
