"""FastAPI gateway — the thin public edge of the system.

It authenticates, validates, invokes the agent, and streams. It never calls
models directly: predictions only exist behind the ML inference MCP server,
reached through the orchestrator graph.

Endpoints:
    GET  /health            liveness + loaded model version
    POST /predict           run the workflow; may return pending_approval
    POST /approve           resume a HITL-interrupted thread
    POST /predict/stream    NDJSON stream of node updates then the result
    POST /reflect           settle a finished match, write the lesson
    GET  /calibration       rolling deployed-system calibration
    POST /parlay/price      correlated-parlay pricing from the Dixon-Coles grid
    POST /chat              LLM conversational layer (parlay pricing, routing,
                            prediction requests, site navigation)

Auth: set GATEWAY_API_KEY to require X-API-Key on every non-health route.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agent.graph import build_graph
from agent.memory import PredictionMemory
from agent.state import AgentState, ParsedRequest
from agent.tooling import InProcessRunner
from agent.tracing import record_trace

app = FastAPI(title="soccer-prediction-gateway", version="0.1.0")

# rate limiting: strict on the prediction endpoints (each request drives a
# full agent run). In-memory storage for local dev; set REDIS_URL for a
# shared counter across gateway replicas.
PREDICT_RATE_LIMIT = os.environ.get("PREDICT_RATE_LIMIT", "5/minute")
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=os.environ.get("REDIS_URL", "memory://"),
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def _make_runner():
    """AGENT_RUNNER=mcp → real MCP client (Compose); default is in-process."""
    if os.environ.get("AGENT_RUNNER", "inprocess") == "mcp":
        from agent.tooling import MCPRunner

        return MCPRunner()
    return InProcessRunner()


_runner = _make_runner()
_graph = build_graph(
    _runner,
    ev_threshold=float(os.environ.get("EV_THRESHOLD", "0.03")),
)
# the cognitive-swarm mode shares the same MCP tooling; toggled per request
# or globally with AGENT_MODE. Built lazily so a workflow-only deploy pays
# nothing for it.
_swarm = None


def _swarm_graph():
    global _swarm
    if _swarm is None:
        from agent.swarm.supervisor import build_swarm

        _swarm = build_swarm(_runner)
    return _swarm


DEFAULT_MODE = os.environ.get("AGENT_MODE", "workflow")
_memory = PredictionMemory(
    Path(os.environ.get("MEMORY_PATH", "data/memory/predictions.jsonl"))
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("GATEWAY_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


class PredictIn(BaseModel):
    text: str = Field(..., examples=["Predict Arsenal vs Man City, any value bets?"])
    thread_id: str | None = None
    mode: str | None = Field(
        default=None, pattern="^(workflow|swarm)$",
        description="workflow (fixed graph, HITL) or swarm (cognitive swarm: "
                    "DAG planner + parallel executors + adversarial critic). "
                    "Defaults to AGENT_MODE.",
    )


class ApproveIn(BaseModel):
    thread_id: str
    action: str = Field(..., pattern="^(approve|reject|edit)$")
    suggestions: list[dict[str, Any]] | None = None


class ReflectIn(BaseModel):
    match_id: str
    actual: str = Field(..., pattern="^(home|draw|away)$")


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _payload(result: dict[str, Any], thread_id: str) -> dict[str, Any]:
    if "__interrupt__" in result:
        intr = result["__interrupt__"][0]
        return {"status": "pending_approval", "thread_id": thread_id,
                "approval_request": intr.value}
    state = AgentState.model_validate(result)
    if state.prediction:
        _memory.record_prediction(
            thread_id=thread_id, match_id=state.request.match_id,
            probs={k: state.prediction["match_outcome"][k]
                   for k in ("home", "draw", "away")},
            degraded=state.degraded,
            model_version=state.prediction["model_version"],
        )
    return {
        "status": "complete", "thread_id": thread_id,
        "answer": state.answer, "prediction": state.prediction,
        "degraded": state.degraded,
        "tool_calls": [c.model_dump(exclude={"result"}) for c in state.ledger],
    }


@app.get("/")
def root() -> dict[str, Any]:
    """Signpost: this is the API, not the website. The browser UI runs
    separately (Next.js, default port 3000)."""
    return {
        "service": "soccer-prediction-gateway",
        "note": "This is the backend API. The website (UI) runs separately "
                "on port 3000. Hitting this port in a browser is expected to "
                "show JSON, not a page.",
        "endpoints": ["/health", "/predict", "/approve", "/predict/stream",
                      "/reflect", "/calibration", "/parlay/price", "/chat",
                      "/bracket", "/leagues",
                      "/leagues/{id}", "/leagues/{id}/predict"],
        "interactive_api_docs": "/docs",
    }


@app.get("/leagues")
def leagues() -> dict[str, Any]:
    """Directory of leagues (grouped by region) and tournament features. Fast:
    catalog only — per-league stats load lazily via /leagues/{id}."""
    from src.data.leagues import CATALOG

    groups: dict[str, list[dict[str, str]]] = {}
    for lg in CATALOG:
        groups.setdefault(lg.region, []).append(
            {"id": lg.id, "name": lg.name, "country": lg.country})
    return {
        "regions": [{"region": r, "leagues": lgs} for r, lgs in groups.items()],
        "tournaments": [
            {"id": "wwc", "name": "Women's World Cup (projection)",
             "type": "bracket", "endpoint": "/bracket"},
            {"id": "wc26", "name": "World Cup 2026 (played)", "type": "report"},
        ],
    }


_league_cache: dict[str, Any] = {}


def _league_data(league_id: str) -> dict[str, Any]:
    if league_id not in _league_cache:
        from src.data.leagues import build_ratings, get_league, load_results

        lg = get_league(league_id)          # raises KeyError on unknown id
        results = load_results(lg)
        elo, rho = build_ratings(results)
        _league_cache[league_id] = {"league": lg, "results": results,
                                    "elo": elo, "rho": rho}
    return _league_cache[league_id]


@app.get("/leagues/{league_id}")
def league_detail(league_id: str) -> dict[str, Any]:
    """One league: standings table, in-league Elo, recent results, upcoming
    fixtures (European where available), and the team list for matchups."""
    from src.data.leagues import (
        fixtures_for, latest_season, recent_results, standings,
    )

    try:
        d = _league_data(league_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — offline / data unavailable
        raise HTTPException(status_code=503, detail=f"league data unavailable: {exc}") from exc

    lg, results, elo = d["league"], d["results"], d["elo"]
    season = latest_season(results)
    season_teams = (set(results[results["season"] == season]["home_team"])
                    | set(results[results["season"] == season]["away_team"]))
    elos = {t: round(elo.ratings.get(t, 1500.0), 1) for t in season_teams}
    fixtures, fixtures_source = fixtures_for(lg)
    return {
        "id": lg.id, "name": lg.name, "region": lg.region, "country": lg.country,
        "season": season,
        "standings": standings(results, season),
        "elo": dict(sorted(elos.items(), key=lambda kv: -kv[1])),
        "recent_results": recent_results(results, 12),
        "upcoming_fixtures": fixtures,
        "fixtures_source": fixtures_source,
        "teams": sorted(season_teams),
        "note": ("Standings and results are real and current (days old for "
                 "live seasons). Forward fixtures come from The Odds API when "
                 "ODDS_API_KEY is set (all leagues, with live odds), else the "
                 "European fixtures feed; without either, pick two teams to "
                 "project a matchup."),
    }


@app.get("/leagues/{league_id}/predict")
def league_predict(league_id: str, home: str, away: str) -> dict[str, Any]:
    """Model prediction for any two teams in the league (Elo engine): outcome,
    scorelines, advancement, timing, role-level scorers/assists, evidence."""
    from src.data.leagues import predict_matchup

    try:
        d = _league_data(league_id)
        return predict_matchup(d["elo"], d["rho"], home, away)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


_bracket_cache: dict[str, Any] | None = None


@app.get("/bracket")
def bracket(refresh: bool = False) -> dict[str, Any]:
    """Women's World Cup bracket projection: seeded field (Elo), every tie's
    outcome/advancement/scoreline/timing, role-level scorers+assists, and a
    per-match `evidence` block to drill into what drove each decision.
    Cached after first build (Elo over ~11k matches takes a few seconds)."""
    global _bracket_cache
    if _bracket_cache is None or refresh:
        try:
            from src.data.womens_international import fetch_results
            from src.models.bracket import simulate_bracket

            _bracket_cache = simulate_bracket(fetch_results())
        except Exception as exc:  # noqa: BLE001 — offline / data unavailable
            raise HTTPException(
                status_code=503, detail=f"bracket data unavailable: {exc}"
            ) from exc
    return _bracket_cache


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        from mcp_servers.ml_server.server import get_bundle

        version = get_bundle().version
    except Exception:  # noqa: BLE001 — remote-runner gateways hold no artifacts
        version = "remote (ml-inference server)"
    return {"ok": True, "model_version": version}


def _traced(
    result: dict[str, Any], thread_id: str, elapsed_ms: float, mode: str = "workflow"
) -> dict[str, Any]:
    payload = _payload(result, thread_id)
    payload["mode"] = mode
    record_trace(
        thread_id=thread_id, mode=mode,
        state=AgentState.model_validate(
            {k: v for k, v in result.items() if k != "__interrupt__"}
        ),
        elapsed_ms=elapsed_ms, outcome=payload["status"],
    )
    return payload


@app.post("/predict", dependencies=[Depends(require_api_key)])
@limiter.limit(PREDICT_RATE_LIMIT)
def predict(request: Request, body: PredictIn) -> dict[str, Any]:
    thread_id = body.thread_id or str(uuid.uuid4())
    mode = body.mode or DEFAULT_MODE
    start = time.monotonic()
    try:
        if mode == "swarm":
            from agent.swarm.state import SwarmState

            result = _swarm_graph().invoke(
                SwarmState(request=ParsedRequest(raw_text=body.text)),
                config=_config(thread_id),
            )
        else:
            result = _graph.invoke(
                AgentState(request=ParsedRequest(raw_text=body.text)),
                config=_config(thread_id),
            )
    except ValueError as exc:  # unparseable request
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _traced(result, thread_id, (time.monotonic() - start) * 1000, mode)


@app.post("/approve", dependencies=[Depends(require_api_key)])
def approve(body: ApproveIn) -> dict[str, Any]:
    resume: dict[str, Any] = {"action": body.action}
    if body.suggestions is not None:
        resume["suggestions"] = body.suggestions
    start = time.monotonic()
    result = _graph.invoke(Command(resume=resume), config=_config(body.thread_id))
    return _traced(result, body.thread_id, (time.monotonic() - start) * 1000)


@app.post("/predict/stream", dependencies=[Depends(require_api_key)])
@limiter.limit(PREDICT_RATE_LIMIT)
def predict_stream(request: Request, body: PredictIn) -> StreamingResponse:
    thread_id = body.thread_id or str(uuid.uuid4())

    # sync generator: Starlette runs it in a threadpool, which keeps the
    # MCPRunner (anyio.run inside) usable here as well
    def gen() -> Iterator[str]:
        start = time.monotonic()
        state = AgentState(request=ParsedRequest(raw_text=body.text))
        for update in _graph.stream(
            state, config=_config(thread_id), stream_mode="updates"
        ):
            for node in update:
                yield json.dumps({"event": "node", "node": node,
                                  "thread_id": thread_id}) + "\n"
        final = _graph.get_state(_config(thread_id))
        result = (dict(final.values) if not final.next
                  else {**dict(final.values),
                        "__interrupt__": final.tasks[0].interrupts})
        yield json.dumps(
            {"event": "result",
             **_traced(result, thread_id, (time.monotonic() - start) * 1000)}
        ) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/reflect", dependencies=[Depends(require_api_key)])
def reflect(body: ReflectIn) -> dict[str, Any]:
    try:
        return _memory.reflect_on_outcome(body.match_id, body.actual)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/calibration", dependencies=[Depends(require_api_key)])
def calibration() -> dict[str, Any]:
    return _memory.rolling_calibration()


# ---- correlated-parlay pricing ----


class ParlayLegIn(BaseModel):
    match_id: str
    leg_type: str = Field(
        ..., pattern="^(outcome|over|under|btts|exact_score|"
                     "home_over|home_under|away_over|away_under|anytime_scorer)$",
    )
    selection: str
    line: float = 0.0
    decimal_odds: float = 0.0
    team_side: str = ""
    xg_share: float = 0.0


class ParlayIn(BaseModel):
    legs: list[ParlayLegIn] = Field(..., min_length=1)
    kelly_fraction: float = 0.25


@app.post("/parlay/price", dependencies=[Depends(require_api_key)])
@limiter.limit(PREDICT_RATE_LIMIT)
def parlay_price(request: Request, body: ParlayIn) -> dict[str, Any]:
    """Price a correlated parlay using the Dixon-Coles grid.

    Each match referenced in the legs needs team xG estimates.  The endpoint
    resolves them from the league Elo engine (fast, no artifact required) or
    the loaded ML model when available.  Legs within the same match are priced
    jointly through the grid; cross-match legs are independent.
    """
    from src.models.parlay import ParlayLeg, parlay_result_to_dict, price_parlay
    from src.models.score_grid import fit_rho

    # resolve per-match xG from the Elo engine (fast path, always available)
    match_ids = {leg.match_id for leg in body.legs}
    mus: dict[str, tuple[float, float]] = {}
    rho = -0.05  # default

    for mid in match_ids:
        # try league engine first (match_id format: HOME-AWAY or HOME-AWAY-DATE)
        parts = mid.split("-")
        home_team = parts[0] if len(parts) >= 2 else mid
        away_team = parts[1] if len(parts) >= 2 else mid

        resolved = False
        # try the ML model bundle for trained xG estimates
        try:
            from mcp_servers.ml_server.server import get_bundle

            bundle = get_bundle()
            rho = bundle.rho
            # use form-based xG priors as a fast fallback
            mus[mid] = (1.5, 1.2)  # will be overridden below if league data found
            resolved = True
        except Exception:  # noqa: BLE001
            pass

        # try league-level Elo for better estimates
        try:
            from src.data.leagues import CATALOG

            for lg in CATALOG:
                try:
                    d = _league_data(lg.id)
                    elo = d["elo"]
                    if home_team in elo.ratings and away_team in elo.ratings:
                        from src.data.leagues import predict_matchup

                        matchup = predict_matchup(elo, d["rho"], home_team, away_team)
                        mus[mid] = (
                            matchup["expected_goals"]["home"],
                            matchup["expected_goals"]["away"],
                        )
                        rho = d["rho"]
                        resolved = True
                        break
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            pass

        if not resolved:
            mus[mid] = (1.35, 1.35)

    legs = [
        ParlayLeg(
            match_id=leg.match_id,
            leg_type=leg.leg_type,
            selection=leg.selection,
            line=leg.line,
            decimal_odds=leg.decimal_odds,
            team_side=leg.team_side,
            xg_share=leg.xg_share,
        )
        for leg in body.legs
    ]

    result = price_parlay(legs, mus, rho=rho, kelly_frac=body.kelly_fraction)
    return parlay_result_to_dict(result)


# ---- LLM conversational layer ----


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatIn(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)


def _parse_chat_intent(text: str) -> dict[str, Any]:
    """Deterministic intent parser (fallback when no LLM available).

    Detects: parlay, predict, navigate, explain, help.
    """
    lower = text.lower().strip()

    # parlay intent
    parlay_kws = ["parlay", "combo", "accumulator", "acca", "multi", "same game"]
    if any(kw in lower for kw in parlay_kws):
        return {"intent": "parlay", "raw": text}

    # predict intent
    if " vs " in lower or " v " in lower or "predict" in lower or "forecast" in lower:
        return {"intent": "predict", "raw": text}

    # navigate intent
    nav_map = {
        "standings": "/leagues", "table": "/leagues", "league": "/leagues",
        "bracket": "/bracket", "world cup": "/bracket", "wwc": "/bracket",
        "parlay builder": "/parlay", "builder": "/parlay",
    }
    for kw, route in nav_map.items():
        if kw in lower:
            return {"intent": "navigate", "route": route, "raw": text}

    # explain intent
    if any(kw in lower for kw in ["how", "explain", "model", "what is", "calibrat"]):
        return {"intent": "explain", "raw": text}

    # help
    if any(kw in lower for kw in ["help", "what can", "features", "capability"]):
        return {"intent": "help", "raw": text}

    return {"intent": "general", "raw": text}


def _chat_respond(intent: dict[str, Any]) -> dict[str, Any]:
    """Generate a response for the parsed intent."""
    kind = intent["intent"]

    if kind == "parlay":
        return {
            "reply": (
                "I can price correlated parlays using the Dixon-Coles grid. "
                "Head to the **Parlay Builder** to add legs interactively, or "
                "tell me your legs like: 'Arsenal win + Over 2.5 + Saka anytime scorer'.\n\n"
                "The key insight: sportsbooks multiply independent odds, but legs "
                "within a match are correlated through the scoreline distribution. "
                "Home win + Over 2.5 is *positively* correlated (winning means more goals), "
                "so the true probability is higher than the sportsbook assumes."
            ),
            "action": {"type": "navigate", "route": "/parlay"},
            "intent": kind,
        }

    if kind == "predict":
        return {
            "reply": (
                "I'll route this to the prediction agent. Head to **Ask the Agent** "
                "and enter your matchup for a full prediction with calibrated "
                "probabilities, conformal uncertainty, and market edge analysis."
            ),
            "action": {"type": "navigate", "route": "/predict"},
            "suggested_input": intent.get("raw", ""),
            "intent": kind,
        }

    if kind == "navigate":
        route = intent.get("route", "/")
        labels = {
            "/leagues": "Leagues hub", "/bracket": "Women's World Cup Bracket",
            "/predict": "Ask the Agent", "/parlay": "Parlay Builder",
        }
        return {
            "reply": f"Taking you to **{labels.get(route, route)}**.",
            "action": {"type": "navigate", "route": route},
            "intent": kind,
        }

    if kind == "explain":
        try:
            from mcp_servers.ml_server.server import get_bundle

            card = dict(get_bundle().card)
            return {
                "reply": (
                    f"**Model: {card.get('version', 'unknown')}**\n\n"
                    f"- Training window: {card.get('training_window', 'N/A')}\n"
                    f"- Features: {', '.join(card.get('feature_names', []))}\n"
                    f"- Dixon-Coles rho: {card.get('dixon_coles_rho', 'N/A')}\n"
                    f"- Conformal alpha: {card.get('conformal_alpha', 'N/A')}\n\n"
                    "The system uses XGBoost for outcome/xG, Dixon-Coles bivariate "
                    "Poisson for the scoreline grid, isotonic calibration + split-conformal "
                    "uncertainty sets, and a Poisson allocation model for player props. "
                    "Every market reads from one grid -- no contradictions."
                ),
                "intent": kind,
            }
        except Exception:  # noqa: BLE001
            return {
                "reply": (
                    "The prediction system uses:\n"
                    "- **XGBoost** for match outcome and expected goals\n"
                    "- **Dixon-Coles** bivariate Poisson for the scoreline grid\n"
                    "- **Isotonic calibration** + split-conformal prediction sets\n"
                    "- **Poisson allocation** for player props (anytime scorer)\n"
                    "- **Correlated parlay pricer** using the same grid\n\n"
                    "Every derived market reads from one grid, so nothing contradicts."
                ),
                "intent": kind,
            }

    if kind == "help":
        return {
            "reply": (
                "Here's what I can help with:\n\n"
                "- **Predict a match**: 'Predict Arsenal vs Man City'\n"
                "- **Price a parlay**: 'Home win + Over 2.5 + BTTS yes'\n"
                "- **Browse leagues**: 'Show me La Liga standings'\n"
                "- **View the bracket**: 'Women's World Cup bracket'\n"
                "- **Explain the model**: 'How does the prediction work?'\n"
                "- **Navigate**: 'Take me to the parlay builder'\n\n"
                "I use the Dixon-Coles grid to price correlated parlays -- "
                "the same math that powers single-match predictions."
            ),
            "intent": kind,
        }

    return {
        "reply": (
            "I can help with match predictions, correlated parlay pricing, "
            "league standings, and model explanations. Try asking me to "
            "'predict Arsenal vs Man City' or 'price a parlay with home win + over 2.5'."
        ),
        "intent": kind,
    }


@app.post("/chat", dependencies=[Depends(require_api_key)])
@limiter.limit(PREDICT_RATE_LIMIT)
def chat(request: Request, body: ChatIn) -> dict[str, Any]:
    """LLM conversational layer: parses intent, prices parlays, routes users,
    and answers questions.  Falls back to deterministic parsing when no LLM
    is available."""
    intent = _parse_chat_intent(body.message)
    response = _chat_respond(intent)
    return response
