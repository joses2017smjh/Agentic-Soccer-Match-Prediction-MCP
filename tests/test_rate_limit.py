"""Tests: gateway rate limiting on the prediction endpoints."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def strict_client(tmp_path) -> TestClient:
    import os

    os.environ.pop("GATEWAY_API_KEY", None)
    os.environ["PREDICT_RATE_LIMIT"] = "2/minute"
    os.environ["MEMORY_PATH"] = str(tmp_path / "predictions.jsonl")
    import gateway.app as app_module

    importlib.reload(app_module)
    return TestClient(app_module.app)


def test_predict_rate_limited_after_threshold(strict_client: TestClient) -> None:
    body = {"text": "Predict Arsenal vs Man City"}
    assert strict_client.post("/predict", json=body).status_code == 200
    assert strict_client.post("/predict", json=body).status_code == 200
    third = strict_client.post("/predict", json=body)
    assert third.status_code == 429
    assert "2 per 1 minute" in third.text


def test_health_and_calibration_not_rate_limited(strict_client: TestClient) -> None:
    for _ in range(10):
        assert strict_client.get("/health").status_code == 200
        assert strict_client.get("/calibration").status_code == 200


# ---------------------------------------------------------------- client key
#
# Every browser request reaches the gateway through the UI's server-side
# proxy, so the caller's network address is the proxy's. Keyed on that, the
# limit is one bucket shared by all users. The UI names the real client in
# X-Client-Address; the gateway may use it only on requests that prove they
# come from the proxy by carrying the API key.

BODY = {"text": "Predict Arsenal vs Man City"}
PROXY = {"X-API-Key": "proxy-key"}


def _as(client: str, **extra: str) -> dict[str, str]:
    return {**PROXY, "X-Client-Address": client, **extra}


@pytest.fixture()
def proxied_client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("GATEWAY_API_KEY", PROXY["X-API-Key"])
    monkeypatch.setenv("PREDICT_RATE_LIMIT", "2/minute")
    monkeypatch.setenv("MEMORY_PATH", str(tmp_path / "predictions.jsonl"))
    monkeypatch.setenv("TRACE_PATH", str(tmp_path / "runs.jsonl"))
    import gateway.app as app_module

    importlib.reload(app_module)
    return TestClient(app_module.app)


def test_each_named_client_gets_its_own_bucket(proxied_client: TestClient) -> None:
    ana, ben = "203.0.113.7", "203.0.113.8"
    assert proxied_client.post("/predict", json=BODY, headers=_as(ana)).status_code == 200
    assert proxied_client.post("/predict", json=BODY, headers=_as(ana)).status_code == 200
    assert proxied_client.post("/predict", json=BODY, headers=_as(ana)).status_code == 429
    # one user exhausting their budget must not lock out the next
    assert proxied_client.post("/predict", json=BODY, headers=_as(ben)).status_code == 200


def test_client_header_is_ignored_when_no_key_is_configured(
    strict_client: TestClient,
) -> None:
    """With auth off anyone can send the header, so honouring it would let a
    caller mint a fresh bucket per request."""
    for spoofed in ("198.51.100.1", "198.51.100.2"):
        r = strict_client.post("/predict", json=BODY, headers={"X-Client-Address": spoofed})
        assert r.status_code == 200
    r = strict_client.post("/predict", json=BODY, headers={"X-Client-Address": "198.51.100.3"})
    assert r.status_code == 429


def test_forwarded_for_alone_names_nobody(proxied_client: TestClient) -> None:
    """X-Forwarded-For can be extended by anything on the path, so the
    gateway does not read it: only the proxy's own header counts."""
    for spoofed in ("198.51.100.1", "198.51.100.2"):
        r = proxied_client.post(
            "/predict", json=BODY, headers={**PROXY, "X-Forwarded-For": spoofed}
        )
        assert r.status_code == 200
    r = proxied_client.post(
        "/predict", json=BODY, headers={**PROXY, "X-Forwarded-For": "198.51.100.3"}
    )
    assert r.status_code == 429


def test_unauthenticated_requests_cannot_spend_a_clients_budget(
    proxied_client: TestClient,
) -> None:
    victim = "203.0.113.20"
    for _ in range(5):
        r = proxied_client.post(
            "/predict", json=BODY, headers={"X-API-Key": "wrong", "X-Client-Address": victim}
        )
        assert r.status_code == 401
    assert proxied_client.post("/predict", json=BODY, headers=_as(victim)).status_code == 200


def test_authenticated_caller_without_client_header_falls_back_to_its_address(
    proxied_client: TestClient,
) -> None:
    assert proxied_client.post("/predict", json=BODY, headers=PROXY).status_code == 200
    assert proxied_client.post("/predict", json=BODY, headers=PROXY).status_code == 200
    assert proxied_client.post("/predict", json=BODY, headers=PROXY).status_code == 429


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_client_header_falls_back_to_the_caller_address(
    proxied_client: TestClient, blank: str
) -> None:
    for _ in range(2):
        assert proxied_client.post("/predict", json=BODY, headers=_as(blank)).status_code == 200
    assert proxied_client.post("/predict", json=BODY, headers=PROXY).status_code == 429


def test_oversized_client_header_is_bounded(proxied_client: TestClient) -> None:
    """Two ids that differ only past the length bound share a bucket, so a
    caller cannot grow the limiter's key space without limit."""
    long_a, long_b = "9" * 64 + "a" * 500, "9" * 64 + "b" * 500
    assert proxied_client.post("/predict", json=BODY, headers=_as(long_a)).status_code == 200
    assert proxied_client.post("/predict", json=BODY, headers=_as(long_b)).status_code == 200
    assert proxied_client.post("/predict", json=BODY, headers=_as(long_a)).status_code == 429


@pytest.mark.parametrize("bad_key", ["cl\u00e9", "\u00ff\u00fe", "proxy-key\u00a0"])
def test_non_ascii_key_is_a_401_not_a_500(proxied_client: TestClient, bad_key: str) -> None:
    """hmac.compare_digest raises on non-ASCII str; a malformed key must
    still be an ordinary rejection, in the auth check and in the limiter."""
    r = proxied_client.post(
        "/predict", json=BODY,
        headers={"X-API-Key": bad_key.encode("latin-1"), "X-Client-Address": "203.0.113.40"},
    )
    assert r.status_code == 401
