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
