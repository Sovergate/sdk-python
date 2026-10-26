"""Async HTTP sender — fire-and-forget, 2s timeout, never raises."""

import json
import logging
import urllib.request
import urllib.error
from typing import Any

logger = logging.getLogger("sovergate.sender")

_BASE_URL = "https://api.sovergate.com"
_TIMEOUT = 2  # seconds — never exceed this


def send(api_key: str, payload: dict[str, Any], base_url: str = _BASE_URL) -> bool:
    """
    POST payload to /ingest. Returns True on success, False on any failure.
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
            return resp.status < 400
    except urllib.error.HTTPError as e:
        logger.debug("sovergate ingest HTTP %s: %s", e.code, e.reason)
        return False
    except Exception as e:
        logger.debug("sovergate ingest error: %s", e)
        return False
