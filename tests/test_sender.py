"""Tests for the HTTP sender against a local mock server."""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import pytest
from sovergate.sender import DROP, RETRY, SENT, send


class _Handler(BaseHTTPRequestHandler):
    received: list = []
    status_to_return: int = 200

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _Handler.received.append({
            "path": self.path,
            "auth": self.headers.get("Authorization", ""),
            "body": json.loads(body),
        })
        self.send_response(_Handler.status_to_return)
        self.end_headers()

    def log_message(self, *args):
        pass  # suppress server log noise in tests


def _start_server(port: int = 19876):
    server = HTTPServer(("127.0.0.1", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def test_send_posts_to_ingest(tmp_path):
    _Handler.received.clear()
    _Handler.status_to_return = 200
    server = _start_server(19876)

    result = send(
        api_key="sov_test_key",
        payload={"model": "gpt-4o", "prompt_tokens": 10},
        base_url="http://127.0.0.1:19876",
    )

    server.shutdown()
    assert result == SENT
    assert len(_Handler.received) == 1
    req = _Handler.received[0]
    assert req["path"] == "/ingest"
    assert req["auth"] == "Bearer sov_test_key"
    assert req["body"]["model"] == "gpt-4o"


def test_send_retries_on_server_error():
    _Handler.received.clear()
    _Handler.status_to_return = 500
    server = _start_server(19877)

    result = send(
        api_key="sov_test_key",
        payload={"model": "gpt-4o"},
        base_url="http://127.0.0.1:19877",
    )

    server.shutdown()
    assert result == RETRY


def test_send_retries_when_server_unreachable():
    result = send(
        api_key="sov_test_key",
        payload={"model": "gpt-4o"},
        base_url="http://127.0.0.1:19999",  # nothing listening here
    )
    assert result == RETRY


@pytest.mark.parametrize("status,expected", [
    (400, DROP),   # invalid record: retrying can never succeed
    (401, DROP),   # unknown or revoked key
    (413, DROP),   # record too large
    (429, RETRY),  # rate limited: temporary
    (503, RETRY),  # server unavailable: temporary
])
def test_send_classifies_http_errors(status, expected):
    _Handler.status_to_return = status
    port = 19880 + status % 100
    server = _start_server(port)

    result = send(api_key="sov_test_key", payload={}, base_url=f"http://127.0.0.1:{port}")

    server.shutdown()
    server.server_close()
    assert result == expected


def test_send_never_raises():
    # Shouldn't raise even with a garbage URL
    try:
        send(api_key="k", payload={}, base_url="http://not-a-real-host-xyz.invalid")
    except Exception as e:
        pytest.fail(f"send() raised an exception: {e}")
