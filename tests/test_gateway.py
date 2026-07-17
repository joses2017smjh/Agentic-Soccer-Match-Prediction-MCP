"""Tests: FastAPI gateway — auth, predict, HITL approve flow, reflection."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    import os

    os.environ["GATEWAY_API_KEY"] = "test-key"
    os.environ["EV_THRESHOLD"] = "-1.0"  # flag everything → approval flow fires
    os.environ["MEMORY_PATH"] = str(
        tmp_path_factory.mktemp("mem") / "predictions.jsonl"
    )
    import gateway.app as app_module

    importlib.reload(app_module)
    return TestClient(app_module.app)


KEY = {"X-API-Key": "test-key"}


def test_health_needs_no_key(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] and r.json()["model_version"]


def test_predict_requires_api_key(client: TestClient) -> None:
    assert client.post("/predict", json={"text": "Arsenal vs Man City"}).status_code == 401


def test_predict_complete_without_stakes(client: TestClient) -> None:
    r = client.post("/predict", json={"text": "Predict Arsenal vs Man City"},
                    headers=KEY)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "complete"
    assert body["prediction"]["match_outcome"]
    assert body["answer"]
    assert body["tool_calls"]  # ledger surfaced, results stripped


def test_predict_then_approve_flow(client: TestClient) -> None:
    r = client.post("/predict",
                    json={"text": "Arsenal vs Man City — any value bets?"},
                    headers=KEY)
    body = r.json()
    assert body["status"] == "pending_approval"
    assert body["approval_request"]["suggestions"]
    thread = body["thread_id"]

    r2 = client.post("/approve",
                     json={"thread_id": thread, "action": "approve"},
                     headers=KEY)
    done = r2.json()
    assert done["status"] == "complete"
    assert "Approved value suggestions" in done["answer"]


def test_unparseable_request_is_422(client: TestClient) -> None:
    r = client.post("/predict", json={"text": "hello"}, headers=KEY)
    assert r.status_code == 422


def test_reflect_and_calibration(client: TestClient) -> None:
    client.post("/predict", json={"text": "Predict Arsenal vs Man City"},
                headers=KEY)
    r = client.post("/reflect",
                    json={"match_id": "ARS-MCI-2026-07-18", "actual": "home"},
                    headers=KEY)
    assert r.status_code == 200
    assert r.json()["match_id"] == "ARS-MCI-2026-07-18"

    cal = client.get("/calibration", headers=KEY).json()
    assert cal["settled"] >= 1


def test_reflect_unknown_match_404(client: TestClient) -> None:
    r = client.post("/reflect",
                    json={"match_id": "AAA-BBB-2020-01-01", "actual": "home"},
                    headers=KEY)
    assert r.status_code == 404
