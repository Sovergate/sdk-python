"""PII detection and scrubbing — runs locally, before any data leaves the customer's infra."""

import re
import hmac
import hashlib
from dataclasses import dataclass
from typing import Optional

# Hex chars of the HMAC digest shown in a token. 12 hex = 48 bits — enough to
# keep entity collisions negligible at realistic volumes while staying readable.
_TOKEN_HEX_LEN = 12


def _normalize(value: str) -> str:
    """Fold trivial formatting differences so the same logical value (e.g. a
    spaced vs. unspaced IBAN, or differently-cased email) yields one token."""
    return re.sub(r"[\s\-]", "", value).lower()


def _hmac_token(key: bytes, base_placeholder: str, value: str) -> str:
    digest = hmac.new(key, _normalize(value).encode("utf-8"), hashlib.sha256).hexdigest()
    return f"[{base_placeholder}_{digest[:_TOKEN_HEX_LEN]}]"

# Each rule: (category, compiled regex, placeholder template)
_RULES: list[tuple[str, re.Pattern, str]] = [
    ("email",     re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'), "EMAIL_REDACTED"),
    # IP before phone — phone regex would otherwise swallow dotted quads
    ("ip",        re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'), "IP_REDACTED"),
    # IBAN: tolerate the conventional 4-char grouping (e.g. "DE89 3704 0044 ...")
    # so a spaced IBAN can't leak its country/check digits to the phone rule.
    ("iban",      re.compile(r'\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b'), "IBAN_REDACTED"),
    # Credit card before phone — captures the FULL grouped PAN (13–20 digits in
    # groups of 4) so the phone rule's length cap can't leave a trailing group.
    # Emitted as id_number until a first-class credit_card category exists in the
    # ingest DTO + DB CHECK constraint. See follow-up.
    ("id_number", re.compile(r'\b\d{4}(?:[ -]?\d{4}){2,4}\b'), "CARD_REDACTED"),
    # Phone: require at least 7 digits (not just any digit-like chars)
    ("phone",     re.compile(r'\b(?:\+?\d[\d\s\-().]{5,14}\d)\b'), "PHONE_REDACTED"),
    ("dob",       re.compile(r'\b(?:\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b'), "DOB_REDACTED"),
    ("id_number", re.compile(r'\b[A-Z]{1,3}\d{6,9}\b'), "ID_REDACTED"),
    # Names last — broad heuristic, apply only inside common name patterns
    ("name",      re.compile(r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?: [A-Z][a-z]+)*\b'), "NAME_REDACTED"),
]


@dataclass
class Detection:
    category: str
    placeholder: str


@dataclass
class ScrubResult:
    text: str
    detections: list[Detection]


def scrub(text: str, key: Optional[bytes] = None) -> ScrubResult:
    """Replace PII in text with placeholders. Returns scrubbed text + detection list.

    If ``key`` is given, each value is replaced with a deterministic, irreversible
    HMAC token — the same value yields the same token across every log, which lets
    the platform count unique entities without ever holding the raw PII. The key
    is the customer's secret and is never sent to our servers. Without a key, we
    fall back to per-text positional numbering (tokens are not correlatable).
    """
    if not text:
        return ScrubResult(text=text, detections=[])

    detections: list[Detection] = []
    counters: dict[str, int] = {}

    for category, pattern, base_placeholder in _RULES:
        def replace(m: re.Match, cat: str = category, ph: str = base_placeholder) -> str:
            if key is not None:
                placeholder = _hmac_token(key, ph, m.group(0))
            else:
                counters[cat] = counters.get(cat, 0) + 1
                placeholder = f"[{ph}_{counters[cat]}]"
            detections.append(Detection(category=cat, placeholder=placeholder))
            return placeholder

        text = pattern.sub(replace, text)

    return ScrubResult(text=text, detections=detections)


def scrub_messages(
    messages: list[dict], key: Optional[bytes] = None
) -> tuple[list[dict], list[Detection]]:
    """Scrub a list of OpenAI-style message dicts. Returns scrubbed messages + all detections."""
    all_detections: list[Detection] = []
    scrubbed: list[dict] = []

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            result = scrub(content, key)
            scrubbed.append({**msg, "content": result.text})
            all_detections.extend(result.detections)
        elif isinstance(content, list):
            # multi-modal content blocks
            new_blocks = []
            for block in content:
                if block.get("type") == "text":
                    result = scrub(block.get("text", ""), key)
                    new_blocks.append({**block, "text": result.text})
                    all_detections.extend(result.detections)
                else:
                    new_blocks.append(block)
            scrubbed.append({**msg, "content": new_blocks})
        else:
            scrubbed.append(msg)

    return scrubbed, all_detections
