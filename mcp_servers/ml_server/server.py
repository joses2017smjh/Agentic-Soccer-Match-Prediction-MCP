"""MCP Server 3 — ML Inference Engine.

Loads one versioned ArtifactBundle at startup and exposes:
- predict_match       full layered prediction JSON (Phase A composer)
- explain_prediction  per-feature contributions for the last prediction of a
                      match, via XGBoost's TreeSHAP (pred_contribs)
- get_model_card      version, training window, features, eval metrics

Schema discipline: a match_context missing trained features is REFUSED with
the exact list of what is missing — never silently imputed. The orchestrator
is expected to gather those fields from Servers 1 and 2 first.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from mcp.server.fastmcp import FastMCP

from mcp_servers.common import run_server, with_as_of
from src.models.artifact import ArtifactBundle
from src.models.gbm import OUTCOME_CLASSES
from src.models.predict import compose_prediction
from src.models.suggestions import MarketQuote

server = FastMCP("ml-inference")

_ARTIFACT_ROOT = Path(
    os.environ.get(
        "ARTIFACT_ROOT",
        Path(__file__).resolve().parent.parent.parent / "data" / "artifacts",
    )
)
_bundle: ArtifactBundle | None = None
_prediction_store: dict[str, dict[str, Any]] = {}  # match_id -> {row, result}


def get_bundle() -> ArtifactBundle:
    global _bundle
    if _bundle is None:
        _bundle = ArtifactBundle.load(_ARTIFACT_ROOT, os.environ.get("MODEL_VERSION"))
    return _bundle


def _feature_row(match_context: dict[str, Any]) -> pd.DataFrame:
    features = get_bundle().card["feature_names"]
    missing = [f for f in features if f not in match_context]
    if missing:
        raise ValueError(
            "REFUSED: match_context is missing trained features "
            f"{missing}. Gather them via the data/news servers and retry; "
            "this server never imputes."
        )
    return pd.DataFrame([{f: float(match_context[f]) for f in features}])


def _quotes(match_context: dict[str, Any]) -> list[MarketQuote] | None:
    raw = match_context.get("market_quotes")
    if not raw:
        return None
    return [MarketQuote(**q) for q in raw]


def run_predict(match_id: str, match_context: dict[str, Any]) -> dict[str, Any]:
    bundle = get_bundle()
    row = _feature_row(match_context)
    players = {
        side: pd.DataFrame(match_context[f"players_{side}"])
        if match_context.get(f"players_{side}") else None
        for side in ("home", "away")
    }
    result = compose_prediction(
        bundle, row,
        home_players=players["home"], away_players=players["away"],
        quotes=_quotes(match_context),
        knockout=bool(match_context.get("knockout", False)),
        ev_threshold=float(match_context.get("ev_threshold", 0.03)),
    )
    result["match_id"] = match_id
    _prediction_store[match_id] = {"row": row, "result": result}
    return with_as_of(result)


def run_explain(match_id: str, top_k: int = 5) -> dict[str, Any]:
    stored = _prediction_store.get(match_id)
    if stored is None:
        raise ValueError(
            f"no prediction stored for {match_id!r}; call predict_match first"
        )
    bundle = get_bundle()
    booster = bundle.outcome._model.get_booster()  # noqa: SLF001
    row: pd.DataFrame = stored["row"]
    contribs = booster.predict(
        xgb.DMatrix(row, feature_names=list(row.columns)), pred_contribs=True
    )  # multiclass: (1, n_classes, n_features + 1); last col is bias
    probs = stored["result"]["match_outcome"]
    pred_class = int(np.argmax([probs["home"], probs["draw"], probs["away"]]))
    values = contribs[0, pred_class, :-1]
    order = np.argsort(-np.abs(values))[:top_k]
    return with_as_of({
        "match_id": match_id,
        "explained_class": OUTCOME_CLASSES[pred_class],
        "top_features": [
            {
                "feature": row.columns[i],
                "value": float(row.iloc[0, i]),
                "contribution": float(values[i]),
            }
            for i in order
        ],
        "note": "TreeSHAP log-odds contributions toward the predicted class "
                "of the raw (pre-calibration) outcome model.",
    })


@server.tool()
def predict_match(match_id: str, match_context: dict) -> dict[str, Any]:
    """Run the full prediction stack for one match: calibrated 1X2 with a
    conformal uncertainty set, expected goals, top scorelines + O/U + BTTS,
    first-scorer and goal-timing bands, optional player props and market
    value suggestions. `match_context` MUST contain every feature listed by
    get_model_card (gather from the data and news servers); optionally
    `players_home`/`players_away` (xg_share, xa_share, exp_minutes,
    availability per player), `market_quotes` (market, selection, model-free
    market_prob, decimal_odds), and `knockout`. Refuses with the missing
    field list on schema mismatch."""
    return run_predict(match_id, match_context)


@server.tool()
def explain_prediction(match_id: str, top_k: int = 5) -> dict[str, Any]:
    """Top feature contributions (TreeSHAP) for this match's most recent
    predict_match call — use to write evidence-grounded rationales, e.g.
    'availability_away (-0.31) was the largest driver'. Requires
    predict_match to have been called for match_id in this session. The
    prediction store is per-process: in a multi-worker deployment, route
    both calls to the same worker (sticky session) or back the store with
    Redis before scaling out."""
    return run_explain(match_id, top_k)


@server.tool()
def get_model_card() -> dict[str, Any]:
    """Model card of the loaded artifact bundle: version, training window,
    the exact feature names predict_match requires, Dixon-Coles rho,
    conformal alpha, and offline eval metrics. Call this before assembling
    a match_context."""
    bundle = get_bundle()
    return with_as_of(dict(bundle.card))


if __name__ == "__main__":
    run_server(server)
