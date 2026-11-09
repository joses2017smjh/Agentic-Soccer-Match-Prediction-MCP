"""Staged, re-runnable data pipeline: raw files -> canonical frames -> features.

    raw       data/raw/football_data_uk/<division>_<season>.csv   (cached downloads)
    interim   data/interim/{team_matches,odds_pre_close,odds_close}.csv
    processed data/processed/match_features.csv + manifest.json

What it adds on top of the loaders and feature code it reuses:

- accounting: every row a stage drops is counted under a named reason, and a
  stage whose reject rate exceeds the configured limit fails instead of
  writing a thinner dataset;
- idempotency: each stage has a fingerprint over its inputs and parameters;
  a re-run with nothing changed touches no file;
- lineage: the manifest records the hash and row count of everything a run
  read and wrote.

Run:  python -m src.pipeline.run --division E0 --seasons 2019 2020 2021
"""

from src.pipeline.quality import PipelineError, RejectLog

# run.py is the `python -m` entry point, so it is imported from there
# (src.pipeline.run.PipelineConfig, run_pipeline) rather than re-exported here.
__all__ = ["PipelineError", "RejectLog"]
