"""Run the staged pipeline and keep its manifest.

    python -m src.pipeline.run --division E0 --seasons 2019 2020 2021

A stage runs only when its fingerprint changed or one of its outputs is
missing or altered. The fingerprint covers the hashes of the stage's inputs
and every parameter that affects its output, so "did anything change?" is a
comparison of two strings, and a second run with nothing changed writes
nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.football_data_uk import fetch_season_csv, season_code
from src.pipeline.quality import PipelineError, RejectLog
from src.pipeline.stages import assemble_match_dataset, check_features, ingest_season
from src.pipeline.storage import describe, read_frame, sha256_file, write_frame

# Bump when a stage's logic changes in a way that should invalidate old outputs.
PIPELINE_VERSION = "1"

INTERIM_FILES = ("team_matches.csv", "odds_pre_close.csv", "odds_close.csv")
FEATURES_FILE = "match_features.csv"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class PipelineConfig:
    seasons: tuple[int, ...]
    division: str = "E0"
    competition: str = "EPL"
    raw_dir: Path = Path("data/raw/football_data_uk")
    out_root: Path = Path("data")
    form_window: int = 10
    form_half_life: float = 5.0
    cutoff_minutes: int = 60
    max_reject_rate: float = 0.02

    @property
    def interim_dir(self) -> Path:
        return self.out_root / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.out_root / "processed"

    @property
    def manifest_path(self) -> Path:
        return self.processed_dir / MANIFEST_FILE

    def raw_path(self, start_year: int) -> Path:
        return self.raw_dir / f"{self.division}_{season_code(start_year)}.csv"


def _fingerprint(stage: str, inputs: list[dict[str, Any]], params: dict[str, Any]) -> str:
    payload = {
        "version": PIPELINE_VERSION, "stage": stage, "params": params,
        "inputs": sorted((Path(i["path"]).name, i["sha256"]) for i in inputs),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _outputs_intact(previous: dict[str, Any] | None) -> bool:
    if not previous:
        return False
    for out in previous.get("outputs", []):
        path = Path(out["path"])
        if not path.exists() or sha256_file(path) != out["sha256"]:
            return False
    return True


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _sorted(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return df.sort_values(keys, kind="mergesort").reset_index(drop=True) if not df.empty else df


def _run_ingest(cfg: PipelineConfig) -> dict[str, Any]:
    team, pre, close = [], [], []
    rejects, odds_rejects = RejectLog(), RejectLog()
    per_season: dict[str, Any] = {}
    for year in cfg.seasons:
        frames = ingest_season(cfg.raw_path(year), cfg.competition, year)
        team.append(frames.team_matches)
        pre.append(frames.odds_pre_close)
        close.append(frames.odds_close)
        rejects.merge(frames.rejects)
        odds_rejects.merge(frames.odds_rejects)
        per_season[f"{year}-{year + 1}"] = frames.rejects.to_dict()
    # gate BEFORE writing: a failed run must leave the previous outputs alone
    rejects.enforce("ingest", cfg.max_reject_rate)
    odds_rejects.enforce("ingest (odds)", cfg.max_reject_rate)

    frames_out = {
        "team_matches.csv": _sorted(pd.concat(team, ignore_index=True), ["kickoff_utc", "match_id", "is_home"]),
        "odds_pre_close.csv": _sorted(pd.concat(pre, ignore_index=True), ["match_id"]),
        "odds_close.csv": _sorted(pd.concat(close, ignore_index=True), ["match_id"]),
    }
    outputs = [write_frame(df, cfg.interim_dir / name) for name, df in frames_out.items()]
    return {"outputs": outputs, "rejects": rejects.to_dict(),
            "odds_rejects": odds_rejects.to_dict(), "by_season": per_season}


def _run_features(cfg: PipelineConfig) -> dict[str, Any]:
    team = read_frame(cfg.interim_dir / "team_matches.csv")
    pre = read_frame(cfg.interim_dir / "odds_pre_close.csv")
    close = read_frame(cfg.interim_dir / "odds_close.csv")
    features = assemble_match_dataset(
        team, pre, close, form_window=cfg.form_window,
        form_half_life=cfg.form_half_life, cutoff_minutes=cfg.cutoff_minutes,
    )
    coverage = check_features(features, matches_in=int(team["match_id"].nunique()))
    features = _sorted(features, ["kickoff_utc", "match_id"])
    return {"outputs": [write_frame(features, cfg.processed_dir / FEATURES_FILE)],
            "coverage": coverage, "columns": len(features.columns)}


def run_pipeline(cfg: PipelineConfig, *, force: bool = False) -> dict[str, Any]:
    """Run every stage that needs running; returns the manifest."""
    missing = [str(cfg.raw_path(y)) for y in cfg.seasons if not cfg.raw_path(y).exists()]
    if missing:
        raise PipelineError(f"raw input missing (run with --fetch to download): {missing}")

    previous = _load_manifest(cfg.manifest_path).get("stages", {})
    stages: dict[str, Any] = {}

    def run_stage(name: str, inputs: list[dict[str, Any]], params: dict[str, Any], fn) -> None:
        fingerprint = _fingerprint(name, inputs, params)
        old = previous.get(name)
        if not force and old and old.get("fingerprint") == fingerprint and _outputs_intact(old):
            stages[name] = {**old, "status": "skipped"}
            return
        started = time.monotonic()
        result = fn(cfg)
        stages[name] = {"status": "ran", "fingerprint": fingerprint, "inputs": inputs,
                        "params": params, "seconds": round(time.monotonic() - started, 3), **result}

    raw_inputs = []
    for year in cfg.seasons:
        path = cfg.raw_path(year)
        with path.open("rb") as f:
            raw_inputs.append(describe(path, rows=max(sum(1 for _ in f) - 1, 0)))
    run_stage("ingest", raw_inputs,
              {"division": cfg.division, "competition": cfg.competition,
               "seasons": list(cfg.seasons), "max_reject_rate": cfg.max_reject_rate},
              _run_ingest)
    run_stage("features", stages["ingest"]["outputs"],
              {"form_window": cfg.form_window, "form_half_life": cfg.form_half_life,
               "cutoff_minutes": cfg.cutoff_minutes},
              _run_features)

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config": {k: (str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v)
                   for k, v in asdict(cfg).items()},
        "stages": stages,
    }
    if any(s["status"] == "ran" for s in stages.values()):
        cfg.processed_dir.mkdir(parents=True, exist_ok=True)
        tmp = cfg.manifest_path.with_name(MANIFEST_FILE + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, cfg.manifest_path)
    return manifest


def _summary(manifest: dict[str, Any]) -> str:
    lines = []
    for name, stage in manifest["stages"].items():
        lines.append(f"{name:<9} {stage['status']:<8} fingerprint {stage['fingerprint'][:12]}")
        verb = "wrote" if stage["status"] == "ran" else "kept "
        for out in stage["outputs"]:
            lines.append(f"            {verb} {out['rows']:>6} rows  {out['path']}")
        if "rejects" in stage:
            r = stage["rejects"]
            lines.append(f"            rejected {r['rejected']} of {r['checked']} raw rows {r['reasons'] or ''}")
        if "coverage" in stage:
            lines.append(f"            coverage {stage['coverage']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--division", default="E0")
    parser.add_argument("--competition", default="EPL")
    parser.add_argument("--seasons", type=int, nargs="+", required=True,
                        help="season start years, e.g. 2019 2020")
    parser.add_argument("--raw-dir", type=Path, default=PipelineConfig.raw_dir)
    parser.add_argument("--out-root", type=Path, default=PipelineConfig.out_root)
    parser.add_argument("--max-reject-rate", type=float, default=PipelineConfig.max_reject_rate)
    parser.add_argument("--fetch", action="store_true", help="download raw files that are not cached")
    parser.add_argument("--force", action="store_true", help="re-run every stage")
    args = parser.parse_args()

    cfg = PipelineConfig(seasons=tuple(args.seasons), division=args.division,
                         competition=args.competition, raw_dir=args.raw_dir,
                         out_root=args.out_root, max_reject_rate=args.max_reject_rate)
    if args.fetch:
        for year in cfg.seasons:
            fetch_season_csv(cfg.division, year, cfg.raw_dir)
    try:
        manifest = run_pipeline(cfg, force=args.force)
    except PipelineError as exc:
        print(f"pipeline failed: {exc}")
        return 1
    print(_summary(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
