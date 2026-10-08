"""Mobile contract, provenance, and authentication across the HTTP boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.mobile import FIXTURE_PATH, MobileSettings, create_app

ROOT = Path(__file__).parents[1]
PROMPT = "Predict Arsenal vs Man City"
LIVE_KEY = {"X-API-Key": "phone-test-key"}


def live_client(handler, **kwargs) -> TestClient:
    return TestClient(
        create_app(
            MobileSettings(
                gateway_url="https://soccer.internal/gateway",
                gateway_api_key="server-only-key",
                mobile_api_key="phone-test-key",
                **kwargs,
            ),
            transport=httpx.MockTransport(handler),
        )
    )


def test_keyless_demo_has_actual_workflow_provenance() -> None:
    client = TestClient(create_app(MobileSettings()))
    bundle = client.get("/mobile/demo").json()
    assert bundle["model_card"]["training_window"] == "synthetic-demo"
    assert len(bundle["samples"]) == 3
    for filename, digest in bundle["source_sha256"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest
    for sample in bundle["samples"]:
        response = client.post("/mobile/predict", json=sample["input"])
        assert response.status_code == 200
        body = response.json()
        assert body["mode"] == "demo"
        assert body["provenance"]["kind"] == "synthetic-fixture"
        if body["status"] == "pending_approval":
            # Match the original gateway's gate: internal prediction/state
            # stays withheld and the approval request remains unresolved.
            assert body["prediction"] is None
            assert body["tool_calls"] == []
            assert body["approval_request"]
            continue
        assert body["prediction"]["model_version"] == "v0-mobile-demo"
        probabilities = body["prediction"]["match_outcome"]
        assert sum(probabilities[key] for key in ["home", "draw", "away"]) == pytest.approx(1)
        assert all(0 <= probabilities[key] <= 1 for key in ["home", "draw", "away"])
        # Actual workflow collected data/news before model inference, rather
        # than returning hand-authored score probabilities.
        assert {call["server"] for call in body["tool_calls"]} >= {
            "sports-data",
            "news-sentiment",
            "ml-inference",
        }
        assert all("latency_ms" not in call for call in body["tool_calls"])


def test_frozen_response_digest_matches_and_has_no_fake_latency() -> None:
    bundle = json.loads(FIXTURE_PATH.read_bytes())
    for sample in bundle["samples"]:
        response = sample["response"]
        digest = response["provenance"].pop("fixture_sha256")
        raw = json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(raw).hexdigest() == digest


def test_demo_rejects_unrepresented_match_instead_of_relabeling_a_fixture() -> None:
    client = TestClient(create_app(MobileSettings()))
    response = client.post("/mobile/predict", json={"text": "PSG vs Bayern", "mode": "demo"})
    assert response.status_code == 422
    assert "demo_prompts" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"text": " "},
        {"text": "x" * 1001},
        {"text": PROMPT, "mode": "swarm"},
        {"text": PROMPT, "mode": "live", "gateway_url": "https://evil.example"},
    ],
)
def test_invalid_requests_are_refused(body) -> None:
    assert (
        TestClient(create_app(MobileSettings())).post("/mobile/predict", json=body).status_code
        == 422
    )


def test_health_reports_configuration_not_upstream_reachability() -> None:
    response = TestClient(create_app(MobileSettings())).get("/mobile/health").json()
    assert response["ok"] and response["demo_available"]
    assert response["live_configured"] is False
    assert response["upstream_reachability"] == "not_probed"
    assert response["fixture_sha256"] == hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()


def test_live_disabled_without_operator_configured_token() -> None:
    client = TestClient(create_app(MobileSettings(gateway_url="https://soccer.internal")))
    assert client.post("/mobile/predict", json={"text": PROMPT, "mode": "live"}).status_code == 503


def test_live_auth_happens_before_any_request_and_server_key_is_not_phone_key() -> None:
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url == "https://soccer.internal/gateway/predict"
        assert request.headers["X-API-Key"] == "server-only-key"
        assert json.loads(request.content) == {"text": PROMPT, "mode": "workflow"}
        return httpx.Response(
            200,
            json={
                "status": "complete",
                "thread_id": "real-thread",
                "answer": "Actual server answer",
                "prediction": {"model_version": "configured-v1"},
                "degraded": ["No live odds"],
                "tool_calls": [{"server": "ml-inference", "ok": True}],
            },
        )

    client = live_client(handler)
    for headers in [{}, {"X-API-Key": "wrong"}]:
        assert (
            client.post(
                "/mobile/predict", json={"text": PROMPT, "mode": "live"}, headers=headers
            ).status_code
            == 401
        )
    assert requests == []
    response = client.post(
        "/mobile/predict", json={"text": PROMPT, "mode": "live"}, headers=LIVE_KEY
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "live" and body["provenance"]["kind"] == "gateway"
    assert body["answer"] == "Actual server answer"
    assert body["degraded"] == ["No live odds"]
    assert body["provenance"]["model_version"] == "configured-v1"
    assert "server-only-key" not in response.text
    assert "phone-test-key" not in response.text


def test_live_pending_approval_is_not_resumed() -> None:
    def handler(request):
        assert request.url.path.endswith("/predict")
        return httpx.Response(
            200,
            json={
                "status": "pending_approval",
                "thread_id": "pending-thread",
                "approval_request": {"suggestions": [{"selection": "home"}]},
            },
        )

    client = live_client(handler)
    body = client.post(
        "/mobile/predict", json={"text": PROMPT, "mode": "live"}, headers=LIVE_KEY
    ).json()
    assert body["status"] == "pending_approval" and body["prediction"] is None
    assert body["approval_request"]["suggestions"]
    assert client.post("/mobile/approve", json={"thread_id": "pending-thread"}).status_code == 404


def test_pending_approval_withholds_accidental_upstream_internal_state() -> None:
    def handler(request):
        return httpx.Response(
            200,
            json={
                "status": "pending_approval",
                "thread_id": "pending-thread",
                "answer": "Internal unapproved draft",
                "prediction": {"model_version": "internal", "unapproved": True},
                "tool_calls": [{"server": "internal-state", "result": "draft"}],
                "approval_request": {"suggestions": [{"selection": "home"}]},
            },
        )

    response = live_client(handler).post(
        "/mobile/predict", json={"text": PROMPT, "mode": "live"}, headers=LIVE_KEY
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending_approval"
    assert body["prediction"] is None and body["tool_calls"] == [] and body["answer"] == ""
    assert body["approval_request"]["suggestions"]
    assert "Internal unapproved draft" not in response.text


@pytest.mark.parametrize("status", [401, 422, 429, 503])
def test_upstream_failures_preserve_status_without_leaking_body(status) -> None:
    client = live_client(
        lambda request: httpx.Response(
            status,
            json={"detail": "private traceback server-only-key"},
            headers={"Retry-After": "9"},
        )
    )
    response = client.post(
        "/mobile/predict", json={"text": PROMPT, "mode": "live"}, headers=LIVE_KEY
    )
    assert response.status_code == status
    assert response.headers["Retry-After"] == "9"
    assert "private" not in response.text and "server-only-key" not in response.text


@pytest.mark.parametrize(
    "error, status",
    [
        (httpx.ReadTimeout("upstream timed out"), 504),
        (httpx.ConnectError("upstream unreachable"), 502),
    ],
)
def test_live_transport_errors_are_visible(error, status) -> None:
    def handler(request):
        raise error

    response = live_client(handler).post(
        "/mobile/predict", json={"text": PROMPT, "mode": "live"}, headers=LIVE_KEY
    )
    assert response.status_code == status
    assert response.json()["detail"]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(302, headers={"Location": "https://evil.example/steal"}),
        httpx.Response(200, text="not JSON"),
        httpx.Response(200, json={"status": "complete"}),
        httpx.Response(200, json={"status": "new-unknown-status", "thread_id": "x"}),
    ],
)
def test_redirects_and_invalid_envelopes_fail_closed(response) -> None:
    assert (
        live_client(lambda request: response)
        .post("/mobile/predict", json={"text": PROMPT, "mode": "live"}, headers=LIVE_KEY)
        .status_code
        == 502
    )


def test_browser_preview_cors_allows_only_configured_origin() -> None:
    client = TestClient(create_app(MobileSettings(allowed_origins=("http://localhost:8081",))))
    headers = {
        "Origin": "http://localhost:8081",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type,X-API-Key",
    }
    allowed = client.options("/mobile/predict", headers=headers)
    assert allowed.status_code == 200
    assert allowed.headers["Access-Control-Allow-Origin"] == "http://localhost:8081"
    rejected = client.options(
        "/mobile/predict", headers={**headers, "Origin": "https://evil.example"}
    )
    assert rejected.status_code == 400
    assert "Access-Control-Allow-Origin" not in rejected.headers


@pytest.mark.parametrize(
    "kwargs",
    [
        {"allowed_origins": ("*",)},
        {"gateway_url": "https://key:secret@example.com"},
        {"gateway_url": "file:///tmp/predict"},
        {"gateway_url": "https://example.com?token=private"},
        {"timeout_seconds": 0},
        {"timeout_seconds": 121},
    ],
)
def test_operator_settings_reject_unsafe_or_unbounded_values(kwargs) -> None:
    with pytest.raises(ValueError):
        MobileSettings(**kwargs)
