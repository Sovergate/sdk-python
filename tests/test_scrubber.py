import pytest
from sovergate.scrubber import scrub, scrub_messages


def test_email_redacted():
    result = scrub("Contact me at alice@example.com for details")
    assert "alice@example.com" not in result.text
    assert "[EMAIL_REDACTED_1]" in result.text
    assert any(d.category == "email" for d in result.detections)


def test_iban_redacted():
    result = scrub("Please transfer to DE89370400440532013000")
    assert "DE89370400440532013000" not in result.text
    assert any(d.category == "iban" for d in result.detections)


def test_ip_redacted():
    result = scrub("Request came from 192.168.1.100")
    assert "192.168.1.100" not in result.text
    assert any(d.category == "ip" for d in result.detections)


def test_multiple_emails_numbered():
    result = scrub("From alice@x.com to bob@y.com")
    assert "[EMAIL_REDACTED_1]" in result.text
    assert "[EMAIL_REDACTED_2]" in result.text
    assert len([d for d in result.detections if d.category == "email"]) == 2


def test_spaced_iban_fully_redacted():
    # Regression: a space-grouped IBAN must not leak its country/check digits.
    result = scrub("Pay to DE89 3704 0044 0532 0130 00 today")
    for fragment in ("DE89", "3704", "0044", "0130"):
        assert fragment not in result.text
    assert "[IBAN_REDACTED_1]" in result.text
    assert any(d.category == "iban" for d in result.detections)


def test_spaced_card_fully_redacted():
    # Regression: a space-grouped 16-digit card must not leak its trailing group.
    result = scrub("Charge 4111 1111 1111 1111 now")
    assert "4111" not in result.text
    assert "1111" not in result.text
    assert "[CARD_REDACTED_1]" in result.text


def test_contiguous_card_redacted():
    result = scrub("Card 4111111111111111 end")
    assert "4111111111111111" not in result.text
    assert "[CARD_REDACTED_1]" in result.text


def test_card_category_is_valid_enum():
    # Emitted category must be one the ingest DTO / DB CHECK accepts, else the
    # whole log entry is rejected. credit_card is not yet a first-class category.
    valid = {"email", "name", "phone", "id_number", "iban", "ip", "dob"}
    result = scrub("Charge 4111 1111 1111 1111 now")
    assert all(d.category in valid for d in result.detections)


def test_short_numbers_not_treated_as_card():
    # An order number / quantity must not be redacted as a card.
    result = scrub("Order 12345 shipped, qty 4")
    assert result.text == "Order 12345 shipped, qty 4"
    assert len(result.detections) == 0


def test_clean_text_unchanged():
    result = scrub("What is the capital of France?")
    assert result.text == "What is the capital of France?"
    assert len(result.detections) == 0


def test_empty_string():
    result = scrub("")
    assert result.text == ""
    assert result.detections == []


def test_scrub_messages():
    messages = [
        {"role": "user", "content": "My email is test@test.com"},
        {"role": "assistant", "content": "Got it"},
    ]
    scrubbed, detections = scrub_messages(messages)
    assert "test@test.com" not in scrubbed[0]["content"]
    assert scrubbed[1]["content"] == "Got it"
    assert any(d.category == "email" for d in detections)


def test_scrub_messages_preserves_role():
    messages = [{"role": "system", "content": "You are a helpful assistant"}]
    scrubbed, _ = scrub_messages(messages)
    assert scrubbed[0]["role"] == "system"


# --- keyed HMAC tokens -------------------------------------------------------

KEY = b"customer-secret-key"


def test_keyed_token_is_deterministic_across_calls():
    # Same value -> same token in separate scrubs, so unique entities can be counted.
    a = scrub("Reach alice@example.com", KEY)
    b = scrub("Also alice@example.com here", KEY)
    assert a.detections[0].placeholder == b.detections[0].placeholder
    assert "alice@example.com" not in a.text


def test_keyed_token_differs_per_value():
    result = scrub("From alice@x.com to bob@y.com", KEY)
    tokens = {d.placeholder for d in result.detections}
    assert len(tokens) == 2  # two distinct people -> two distinct tokens


def test_keyed_token_normalizes_formatting():
    # A spaced and unspaced IBAN are the same entity -> same token.
    spaced = scrub("IBAN DE89 3704 0044 0532 0130 00", KEY).detections[0].placeholder
    flat = scrub("IBAN DE89370400440532013000", KEY).detections[0].placeholder
    assert spaced == flat


def test_keyed_token_changes_with_key():
    # The token is derived from the secret — a different key yields a different token.
    t1 = scrub("alice@example.com", b"key-one").detections[0].placeholder
    t2 = scrub("alice@example.com", b"key-two").detections[0].placeholder
    assert t1 != t2


def test_keyed_token_is_not_the_raw_value():
    # Irreversible: the token must not embed the plaintext.
    result = scrub("alice@example.com", KEY)
    assert "alice" not in result.text.lower().replace("alice@example.com", "")
    assert "alice@example.com" not in result.text


def test_no_key_falls_back_to_positional():
    result = scrub("From alice@x.com to bob@y.com")
    assert "[EMAIL_REDACTED_1]" in result.text
    assert "[EMAIL_REDACTED_2]" in result.text
