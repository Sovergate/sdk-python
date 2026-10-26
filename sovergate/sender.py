"""Async HTTP sender — fire-and-forget, 2s timeout, never raises."""

import json
import logging
import urllib.request
import urllib.error
from typing import Any

logger = logging.getLogger("sovergate.sender")

_BASE_URL = "https://api.sovergate.com"
_TIMEOUT = 2  # seconds — never exceed this

# Outcomes of one send attempt.
SENT = "sent"    # accepted by the server
RETRY = "retry"  # temporary failure: keep the record and try again later
DROP = "drop"    # permanently rejected: retrying the same record can never succeed

_RETRYABLE_STATUS = {408, 425, 429}


def send(api_key: str, payload: dict[str, Any], base_url: str = _BASE_URL) -> str:
    """
    POST payload to /ingest and return SENT, RETRY or DROP.
    Never raises — the caller's app must not be affected by our failures.
    """
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=f"{base_url}/ingest",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return SENT if resp.status < 400 else RETRY
    except urllib.error.HTTPError as e:
        if e.code in _RETRYABLE_STATUS or e.code >= 500:
            logger.debug("sovergate ingest HTTP %s, will retry: %s", e.code, e.reason)
            return RETRY
        # 400 (invalid record), 401/403 (bad or revoked key), 413 (too large)...
        # Retrying would block every later record behind this one.
        logger.warning("sovergate: ingest rejected a record (HTTP %s %s); dropping it", e.code, e.reason)
        return DROP
    except Exception as e:
        logger.debug("sovergate ingest error, will retry: %s", e)
        return RETRY
