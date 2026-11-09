"""Tests: the staged data pipeline (src/pipeline).

Three properties carry the design, so they get the weight here:
  1. nothing is dropped silently -- every reject is counted under a reason;
  2. a run with nothing changed is a no-op, and a change re-runs exactly the
     stages downstream of it;
  3. a run that rejects too much fails and leaves the last good outputs alone.

Raw inputs are synthetic season files written to a temp directory, drawn from
a seeded generator in the house style (random draws, exact assertions).
"""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.football_data_uk import closing_odds_frame, season_code, team_match_frame
from src.data.interfaces import TEAM_MATCH_COLUMNS, validate_frame
from src.pipeline.quality import (
    PipelineError,
    RejectLog,
    gate_odds,
    gate_team_matches,
    read_raw_csv,
)
from src.pipeline.run import PipelineConfig, main, run_pipeline
from src.pipeline.stages import ingest_season
from src.pipeline.storage import read_frame, sha256_file, write_frame

TEAMS = ["Alpha", "Bravo", "Cedar", "Delta", "Echo", "Fjord"]
N_MATCHES = len(TEAMS) * (len(TEAMS) - 1)   # 30 per synthetic season


def season_rows(start_year: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    first = datetime(start_year, 8, 10)
    rows = []
    for i, (home, away) in enumerate(itertools.permutations(TEAMS, 2)):
        gh, ga = int(rng.poisson(1.5)), int(rng.poisson(1.1))
        hst, ast = gh + int(rng.integers(0, 4)), ga + int(rng.integers(0, 4))
        row = {
            "Div": "E0", "Date": (first + timedelta(days=3 * i)).strftime("%d/%m/%Y"),
            "Time": "15:00", "HomeTeam": home, "AwayTeam": away, "FTHG": gh, "FTAG": ga,
            "HS": hst + int(rng.integers(2, 9)), "AS": ast + int(rng.integers(2, 9)),
            "HST": hst, "AST": ast,
        }
        for suffix in ("", "C"):   # pre-match and closing prices, 3% overround
            p = rng.dirichlet([4.0, 2.5, 3.0])
            for outcome, prob in zip("HDA", p):
                row[f"PS{suffix}{outcome}"] = round(1.0 / (prob * 1.03), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def write_season(raw_dir: Path, start_year: int, seed: int = 0, rows: pd.DataFrame | None = None) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"E0_{season_code(start_year)}.csv"
    (rows if rows is not None else season_rows(start_year, seed)).to_csv(path, index=False)
    return path


@pytest.fixture()
def cfg(tmp_path: Path) -> PipelineConfig:
    raw = tmp_path / "raw"
    for i, year in enumerate((2020, 2021)):
        write_season(raw, year, seed=i)
    return PipelineConfig(seasons=(2020, 2021), raw_dir=raw, out_root=tmp_path / "out")


def output_hashes(cfg: PipelineConfig) -> dict[str, str]:
    files = sorted(cfg.interim_dir.glob("*.csv")) + sorted(cfg.processed_dir.glob("*"))
    return {f.name: sha256_file(f) for f in files}


def statuses(manifest: dict) -> dict[str, str]:
    return {name: stage["status"] for name, stage in manifest["stages"].items()}


# ---------------------------------------------------------------- happy path

def test_run_writes_every_layer_and_accounts_for_every_row(cfg: PipelineConfig) -> None:
    manifest = run_pipeline(cfg)
    assert statuses(manifest) == {"ingest": "ran", "features": "ran"}

    team = read_frame(cfg.interim_dir / "team_matches.csv")
    validate_frame(team, TEAM_MATCH_COLUMNS, "TEAM_MATCH")      # contract holds after storage
    assert team["match_id"].nunique() == 2 * N_MATCHES and len(team) == 4 * N_MATCHES

    ingest = manifest["stages"]["ingest"]
    assert ingest["rejects"] == {"checked": 2 * N_MATCHES, "rejected": 0, "rate": 0.0, "reasons": {}}
    assert [i["rows"] for i in ingest["inputs"]] == [N_MATCHES, N_MATCHES]
    assert all(len(i["sha256"]) == 64 for i in ingest["inputs"] + ingest["outputs"])

    features = read_frame(cfg.processed_dir / "match_features.csv")
    assert manifest["stages"]["features"]["coverage"] == {
        "matches_in": 2 * N_MATCHES, "rows_out": len(features), "without_both_odds": 0,
    }
    assert features["match_id"].is_unique
    assert {"odds_imp_home", "close_imp_home", "goals_for", "home_team"} <= set(features.columns)
    assert features["kickoff_utc"].is_monotonic_increasing


def test_features_stage_consumes_exactly_what_ingest_wrote(cfg: PipelineConfig) -> None:
    """Lineage: the features stage's inputs are the ingest outputs, by hash."""
    stages = run_pipeline(cfg)["stages"]
    assert stages["features"]["inputs"] == stages["ingest"]["outputs"]


# ----------------------------------------------------------------- idempotency

def test_second_run_is_a_noop(cfg: PipelineConfig) -> None:
    run_pipeline(cfg)
    before = output_hashes(cfg)
    mtimes = {f: f.stat().st_mtime_ns for f in [*cfg.interim_dir.iterdir(), *cfg.processed_dir.iterdir()]}

    again = run_pipeline(cfg)
    assert statuses(again) == {"ingest": "skipped", "features": "skipped"}
    assert output_hashes(cfg) == before
    assert {f: f.stat().st_mtime_ns for f in mtimes} == mtimes     # not even rewritten


def test_changed_raw_input_reruns_both_stages(cfg: PipelineConfig) -> None:
    first = run_pipeline(cfg)
    rows = season_rows(2021, seed=1)
    rows.loc[0, "FTHG"] += 1                                       # one corrected result
    write_season(cfg.raw_dir, 2021, rows=rows)

    second = run_pipeline(cfg)
    assert statuses(second) == {"ingest": "ran", "features": "ran"}
    for stage in ("ingest", "features"):
        assert second["stages"][stage]["fingerprint"] != first["stages"][stage]["fingerprint"]


def test_changed_feature_parameter_reruns_only_features(cfg: PipelineConfig, tmp_path: Path) -> None:
    first = run_pipeline(cfg)
    tuned = PipelineConfig(seasons=cfg.seasons, raw_dir=cfg.raw_dir, out_root=cfg.out_root,
                           form_window=4)
    second = run_pipeline(tuned)
    assert statuses(second) == {"ingest": "skipped", "features": "ran"}
    assert second["stages"]["ingest"]["outputs"] == first["stages"]["ingest"]["outputs"]


@pytest.mark.parametrize("victim", ["interim/team_matches.csv", "processed/match_features.csv"])
def test_missing_or_altered_output_is_rebuilt(cfg: PipelineConfig, victim: str) -> None:
    run_pipeline(cfg)
    good = output_hashes(cfg)
    (cfg.out_root / victim).write_text("tampered\n")
    rebuilt = run_pipeline(cfg)
    assert rebuilt["stages"][victim.split("/")[0] == "interim" and "ingest" or "features"]["status"] == "ran"
    assert output_hashes(cfg)[Path(victim).name] == good[Path(victim).name]


def test_force_reruns_everything_and_reproduces_the_same_bytes(cfg: PipelineConfig) -> None:
    run_pipeline(cfg)
    before = {k: v for k, v in output_hashes(cfg).items() if k != "manifest.json"}
    forced = run_pipeline(cfg, force=True)
    assert statuses(forced) == {"ingest": "ran", "features": "ran"}
    assert {k: v for k, v in output_hashes(cfg).items() if k != "manifest.json"} == before


@pytest.mark.parametrize("seed", [3, 11, 42])
def test_same_input_gives_identical_bytes_in_a_fresh_location(tmp_path: Path, seed: int) -> None:
    hashes = []
    for run in ("a", "b"):
        raw = tmp_path / run / "raw"
        write_season(raw, 2022, seed=seed)
        c = PipelineConfig(seasons=(2022,), raw_dir=raw, out_root=tmp_path / run / "out")
        run_pipeline(c)
        hashes.append({k: v for k, v in output_hashes(c).items() if k != "manifest.json"})
    assert hashes[0] == hashes[1]


# ------------------------------------------------------------------ rejection

def test_bad_rows_are_counted_by_reason_not_hidden(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    rows = season_rows(2020, seed=5)
    rows.loc[3, "FTHG"] = np.nan                    # result missing
    rows["Date"] = rows["Date"].astype(object)
    rows.loc[7, "Date"] = "not a date"              # kickoff unparseable
    path = write_season(raw, 2020, rows=rows)
    with path.open("a") as f:                       # a line with too many fields
        f.write("E0,10/08/2020,15:00,Alpha,Bravo,1,0,5,5,2,2,2.1,3.3,3.6,2.0,3.4,3.7,EXTRA,EXTRA\n")

    frames = ingest_season(path, "EPL", 2020)
    assert frames.rejects.to_dict() == {
        "checked": N_MATCHES + 1, "rejected": 3, "rate": round(3 / (N_MATCHES + 1), 6),
        "reasons": {"malformed_line": 1, "missing_result_or_team": 1, "unparseable_kickoff": 1},
    }
    assert frames.team_matches["match_id"].nunique() == N_MATCHES - 2

    manifest = run_pipeline(PipelineConfig(seasons=(2020,), raw_dir=raw, out_root=tmp_path / "out",
                                           max_reject_rate=0.2))
    assert manifest["stages"]["ingest"]["rejects"]["reasons"] == frames.rejects.reasons
    assert manifest["stages"]["ingest"]["by_season"]["2020-2021"]["rejected"] == 3


def test_too_many_rejects_fails_and_leaves_the_last_good_outputs(cfg: PipelineConfig) -> None:
    run_pipeline(cfg)
    good = output_hashes(cfg)

    rows = season_rows(2021, seed=1)
    rows.loc[:9, "FTAG"] = np.nan                   # a third of one season unusable
    write_season(cfg.raw_dir, 2021, rows=rows)
    with pytest.raises(PipelineError, match="over the 2.0% limit"):
        run_pipeline(cfg)
    assert output_hashes(cfg) == good               # nothing overwritten, manifest included


def test_missing_raw_file_is_reported_before_anything_runs(cfg: PipelineConfig) -> None:
    cfg.raw_path(2021).unlink()
    with pytest.raises(PipelineError, match="raw input missing"):
        run_pipeline(cfg)
    assert not cfg.interim_dir.exists()


def test_raw_file_without_required_columns_is_refused(tmp_path: Path) -> None:
    path = write_season(tmp_path, 2020, rows=season_rows(2020, 0).drop(columns=["FTAG"]))
    with pytest.raises(PipelineError, match="FTAG"):
        ingest_season(path, "EPL", 2020)


# ----------------------------------------------------------------------- gates

@pytest.fixture()
def canonical() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = season_rows(2020, seed=9)
    return team_match_frame(raw, "EPL", "2020-2021"), closing_odds_frame(raw, "pre_close")


def _break(team: pd.DataFrame, how: str) -> pd.DataFrame:
    team = team.copy()
    first = team["match_id"].iloc[0]
    rows = team.index[team["match_id"] == first]
    if how == "duplicate_match":
        return pd.concat([team, team.loc[rows]], ignore_index=True)
    if how == "missing_side":
        return team.drop(index=rows[0]).reset_index(drop=True)
    if how == "invalid_goals":
        team.loc[rows[0], "goals_for"] = -1.0
    if how == "negative_shots":
        team.loc[rows[0], "shots_for"] = -3.0
    if how == "kickoff_outside_season":
        team.loc[rows, "kickoff_utc"] = pd.Timestamp("2031-01-01", tz="UTC")
    return team


@pytest.mark.parametrize("reason", ["duplicate_match", "missing_side", "invalid_goals",
                                    "negative_shots", "kickoff_outside_season"])
def test_team_match_gate_rejects_the_whole_match_under_one_reason(canonical, reason: str) -> None:
    team, _ = canonical
    clean, log = gate_team_matches(_break(team, reason), start_year=2020)
    assert log.reasons == {reason: 1}
    assert log.checked == N_MATCHES and clean["match_id"].nunique() == N_MATCHES - 1
    assert (clean.groupby("match_id").size() == 2).all()


def test_team_match_gate_passes_clean_data_untouched(canonical) -> None:
    team, _ = canonical
    clean, log = gate_team_matches(team, start_year=2020)
    assert log.rejected == 0 and len(clean) == len(team)


def test_odds_gate_reasons(canonical) -> None:
    team, odds = canonical
    valid = set(team["match_id"])
    bad = odds.copy()
    bad.loc[0, "match_id"] = "Nobody-Nowhere-2020-01-01"          # orphan
    bad.loc[1, "odds_home"] = 1.0                                 # unpriced
    bad.loc[2, "odds_imp_draw"] += 0.05                           # does not sum to 1
    bad = pd.concat([bad, bad.loc[[5]]], ignore_index=True)       # duplicate
    clean, log = gate_odds(bad, valid)
    assert log.reasons == {"orphan_match": 1, "price_not_above_one": 1,
                           "implied_not_normalized": 1, "duplicate_match": 1}
    assert len(clean) == len(odds) - 3 and clean["match_id"].is_unique


def test_reject_log_arithmetic() -> None:
    log = RejectLog(checked=200)
    log.add("a", 3); log.add("a", 1); log.add("b", 0)
    assert (log.rejected, log.rate, log.reasons) == (4, 0.02, {"a": 4})
    log.enforce("stage", max_rate=0.02)                          # at the limit is allowed
    with pytest.raises(PipelineError):
        log.enforce("stage", max_rate=0.019)
    assert RejectLog().rate == 0.0                               # nothing checked, nothing rejected


# --------------------------------------------------------------------- storage

@pytest.mark.parametrize("seed", range(5))
def test_frames_round_trip_through_storage(tmp_path: Path, seed: int) -> None:
    team = team_match_frame(season_rows(2020, seed), "EPL", "2020-2021")
    entry = write_frame(team, tmp_path / "t.csv")
    back = read_frame(tmp_path / "t.csv")
    validate_frame(back, TEAM_MATCH_COLUMNS, "TEAM_MATCH")
    assert entry["rows"] == len(back) == len(team)
    assert (back["kickoff_utc"] == team["kickoff_utc"].reset_index(drop=True)).all()
    numeric = team.select_dtypes("number").columns
    np.testing.assert_allclose(back[numeric].to_numpy(float), team[numeric].to_numpy(float),
                               rtol=1e-11, equal_nan=True)
    assert write_frame(back, tmp_path / "again.csv")["sha256"] == entry["sha256"]   # stable bytes


def test_raw_reader_counts_what_the_default_reader_skips(tmp_path: Path) -> None:
    path = write_season(tmp_path, 2020, seed=1)
    with path.open("a") as f:
        f.write("a,b,c\n" + ",".join(["x"] * 40) + "\n")
    frame, malformed = read_raw_csv(path)
    assert malformed == 1
    assert len(frame) == len(pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip"))


# ------------------------------------------------------------------------- CLI

def test_cli_runs_and_reports(cfg: PipelineConfig, monkeypatch, capsys) -> None:
    argv = ["run", "--seasons", "2020", "2021", "--raw-dir", str(cfg.raw_dir),
            "--out-root", str(cfg.out_root)]
    monkeypatch.setattr("sys.argv", argv)
    assert main() == 0
    assert "ingest    ran" in capsys.readouterr().out
    assert main() == 0
    assert "ingest    skipped" in capsys.readouterr().out


def test_cli_exits_nonzero_on_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["run", "--seasons", "2020", "--raw-dir", str(tmp_path),
                                     "--out-root", str(tmp_path / "out")])
    assert main() == 1
    assert "pipeline failed" in capsys.readouterr().out


# ------------------------------------------------------------------- real data

REAL_RAW = Path("data/raw/football_data_uk")
REAL_SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)


@pytest.mark.skipif(
    not all((REAL_RAW / f"E0_{season_code(y)}.csv").exists() for y in REAL_SEASONS),
    reason="cached football-data.co.uk seasons not present (data/raw is not in git)",
)
def test_real_cached_seasons_pass_every_gate(tmp_path: Path) -> None:
    manifest = run_pipeline(PipelineConfig(seasons=REAL_SEASONS, raw_dir=REAL_RAW,
                                           out_root=tmp_path))
    ingest = manifest["stages"]["ingest"]
    assert ingest["rejects"]["rejected"] == 0 and ingest["rejects"]["checked"] == 6 * 380
    assert manifest["stages"]["features"]["coverage"]["rows_out"] == 6 * 380
