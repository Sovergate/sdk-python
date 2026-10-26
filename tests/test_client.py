"""init() / instrument() must never raise into the host application."""

import logging
from types import SimpleNamespace

import pytest

import sovergate.client as client


@pytest.fixture(autouse=True)
def reset_state():
    client.shutdown()
    client._state.update(api_key=None, pii_key=None, queue=None)
    yield
    client.shutdown()
    client._state.update(api_key=None, pii_key=None, queue=None)


def test_instrument_before_init_warns_instead_of_raising(caplog):
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: None)))
    original = fake.chat.completions.create

    with caplog.at_level(logging.WARNING, logger="sovergate"):
        client.instrument(fake)

    assert fake.chat.completions.create is original
    assert "before init" in caplog.text


def test_instrument_unsupported_library_warns_instead_of_raising(caplog):
    client.init(api_key="sov_test_x", base_url="http://127.0.0.1:9")

    with caplog.at_level(logging.WARNING, logger="sovergate"):
        client.instrument(object())

    assert "unsupported library" in caplog.text


def test_instrument_broken_client_warns_instead_of_raising(caplog):
    client.init(api_key="sov_test_x", base_url="http://127.0.0.1:9")
    broken = SimpleNamespace(chat=SimpleNamespace())  # no .completions

    with caplog.at_level(logging.WARNING, logger="sovergate"):
        client.instrument(broken)

    assert "instrument() failed" in caplog.text


def test_init_with_bad_arguments_does_not_raise(caplog):
    with caplog.at_level(logging.WARNING, logger="sovergate"):
        client.init(api_key="sov_test_x", pii_key=12345)  # type: ignore[arg-type]

    assert client._state["queue"] is None
    assert "init failed" in caplog.text


def test_debug_does_not_touch_the_apps_root_logger():
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)

    client.init(api_key="sov_test_x", base_url="http://127.0.0.1:9", debug=True)

    assert root.level == level
    assert root.handlers == handlers
