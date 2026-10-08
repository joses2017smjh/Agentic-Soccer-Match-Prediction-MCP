"""Mobile companion API: immutable demo fixtures and the existing agent gateway.

Run independently with ``uvicorn gateway.mobile:app --port 8011``. No model,
MCP client, provider key, or network is needed for demo mode. Live requests
delegate to the original /predict workflow rather than bypassing it.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mobile_demo.json"


@dataclass(frozen=True)
class MobileSettings:
    gateway_url: str = ""
    gateway_api_key: str = ""
    mobile_api_key: str = ""
    allowed_origins: tuple[str, ...] = ()
    timeout_seconds: float = 45.0

    def __post_init__(self) -> None:
        if self.gateway_url:
            url = urlsplit(self.gateway_url)
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError(
                    "SOCCER_GATEWAY_URL must be an HTTP(S) base URL without credentials"
                )
        for origin in self.allowed_origins:
            url = urlsplit(origin)
            if (
                url.scheme not in {"http", "https"}
                or not url.hostname
                or url.path not in {"", "/"}
                or url.query
                or url.fragment
                or url.username
                or url.password
            ):
                raise ValueError("MOBILE_ALLOWED_ORIGINS must list explicit HTTP(S) origins")
        if not 1 <= self.timeout_seconds <= 120:
            raise ValueError("MOBILE_TIMEOUT_SECONDS must be between 1 and 120")

    @classmethod
    def from_environment(cls) -> MobileSettings:
        return cls(
            gateway_url=os.environ.get("SOCCER_GATEWAY_URL", "").rstrip("/"),
            gateway_api_key=os.environ.get("SOCCER_GATEWAY_API_KEY", ""),
            mobile_api_key=os.environ.get("MOBILE_API_KEY", ""),
            allowed_origins=tuple(
                value.strip().rstrip("/")
                for value in os.environ.get("MOBILE_ALLOWED_ORIGINS", "").split(",")
                if value.strip()
            ),
            timeout_seconds=float(os.environ.get("MOBILE_TIMEOUT_SECONDS", "45")),
        )


class MobilePredictIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1000)
    mode: Literal["demo", "live"] = "demo"

    @field_validator("text")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text cannot be blank")
        return value


class MobileProvenance(BaseModel):
    kind: Literal["synthetic-fixture", "gateway"]
    model_version: str | None = None
    data_backend: str | None = None
    fixture_id: str | None = None
    fixture_sha256: str | None = None
    generated_at_utc: str | None = None
    notes: list[str] = Field(default_factory=list)


class MobilePrediction(BaseModel):
    service: Literal["soccer"] = "soccer"
    mode: Literal["demo", "live"]
    status: Literal["complete", "pending_approval"]
    thread_id: str
    answer: str = ""
    prediction: dict[str, Any] | None = None
    degraded: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    approval_request: dict[str, Any] | None = None
    provenance: MobileProvenance


def _load_fixture_bundle() -> dict[str, Any]:
    bundle = json.loads(FIXTURE_PATH.read_bytes())
    for sample in bundle["samples"]:
        MobilePrediction.model_validate(sample["response"])
    return bundle


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def create_app(
    settings: MobileSettings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    settings = settings or MobileSettings.from_environment()
    app = FastAPI(title="Soccer mobile companion API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    @app.get("/mobile/health")
    def health() -> dict[str, Any]:
        raw = FIXTURE_PATH.read_bytes()
        return {
            "ok": True,
            "service": "soccer",
            "schema_version": "1.0",
            "demo_available": True,
            "live_configured": bool(settings.gateway_url and settings.mobile_api_key),
            "upstream_reachability": "not_probed",
            "fixture_sha256": hashlib.sha256(raw).hexdigest(),
            "capabilities": ["prediction", "probabilities", "scorelines", "agent_trace"],
        }

    @app.get("/mobile/demo")
    def demo() -> dict[str, Any]:
        return _load_fixture_bundle()

    @app.post("/mobile/predict", response_model=MobilePrediction)
    async def predict(
        body: MobilePredictIn,
        x_api_key: str | None = Header(default=None),
    ) -> MobilePrediction:
        if body.mode == "demo":
            bundle = _load_fixture_bundle()
            for sample in bundle["samples"]:
                inputs = [sample["input"]["text"], *sample.get("aliases", [])]
                if _normalise(body.text) in {_normalise(value) for value in inputs}:
                    return MobilePrediction.model_validate(copy.deepcopy(sample["response"]))
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "No offline fixture matches this request. Choose a demo prompt or use live mode.",
                    "demo_prompts": [sample["input"]["text"] for sample in bundle["samples"]],
                },
            )

        # A provider/gateway key never comes from the phone. Live access has a
        # separate operator-configured token, checked before any upstream call.
        if not settings.gateway_url or not settings.mobile_api_key:
            raise HTTPException(
                status_code=503, detail="Live mode is not configured on this server"
            )
        if not x_api_key or not hmac.compare_digest(
            x_api_key.encode("utf-8"), settings.mobile_api_key.encode("utf-8")
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing mobile API key")

        headers = {"X-API-Key": settings.gateway_api_key} if settings.gateway_api_key else {}
        try:
            async with httpx.AsyncClient(
                transport=transport,
                timeout=settings.timeout_seconds,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                response = await client.post(
                    settings.gateway_url.rstrip("/") + "/predict",
                    json={"text": body.text, "mode": "workflow"},
                    headers=headers,
                )
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail="Soccer agent request timed out") from exc
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail="Soccer agent is unavailable") from exc

        if response.status_code >= 400:
            # Preserve rate-limit, auth, validation, and service status without
            # exposing arbitrary upstream tracebacks or credential-bearing URLs.
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Soccer agent returned HTTP {response.status_code}",
                headers={"Retry-After": response.headers["Retry-After"]}
                if "Retry-After" in response.headers
                else None,
            )
        if 300 <= response.status_code < 400:
            raise HTTPException(
                status_code=502, detail="Soccer agent returned an unexpected redirect"
            )
        try:
            raw = response.json()
            pending = raw["status"] == "pending_approval"
            # Preserve the gateway's public approval boundary even if a future
            # upstream revision includes extra internal state in its payload.
            prediction = None if pending else raw.get("prediction")
            provenance = MobileProvenance(
                kind="gateway",
                model_version=(prediction or {}).get("model_version"),
                notes=[
                    "Delegated to the existing agent workflow; provider/model freshness is not independently verified.",
                    "Live means a server request. The upstream can still use synthetic data or a demo model.",
                    "Pending approvals remain pending; this adapter exposes no approval or wagering action.",
                ],
            )
            return MobilePrediction.model_validate(
                {
                    "mode": "live",
                    "status": raw["status"],
                    "thread_id": raw["thread_id"],
                    "answer": "" if pending else raw.get("answer", ""),
                    "prediction": prediction,
                    "degraded": raw.get("degraded", []),
                    "tool_calls": [] if pending else raw.get("tool_calls", []),
                    "approval_request": raw.get("approval_request"),
                    "provenance": provenance,
                }
            )
        except (ValueError, TypeError, KeyError, AttributeError, ValidationError) as exc:
            raise HTTPException(
                status_code=502, detail="Soccer agent returned an invalid response"
            ) from exc

    return app


app = create_app()
