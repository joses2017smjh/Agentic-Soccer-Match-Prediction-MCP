"""Prepare and validate real historical CSVs required by the full test suite.

Reuses the official football-data.co.uk provider and its ignored local cache.
No records are invented and raw provider data is not committed to the repository.
Run before pytest: ``python -m scripts.prepare_test_data``.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from envs.real_market import load_season
from src.data.football_data_uk import BASE_URL, fetch_season_csv, season_code

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "football_data_uk"
START_YEARS = (2019, 2020, 2021, 2022, 2023, 2024)
REQUIRED_COLUMNS = {
    "Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA",
    "B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA",
}


def prepare() -> dict:
    records = []
    for year in START_YEARS:
        code = season_code(year)
        raw = fetch_season_csv("E0", year, cache_dir=CACHE)
        missing = REQUIRED_COLUMNS - set(raw.columns)
        if missing or len(raw) != 380 or not raw["Div"].eq("E0").all():
            raise ValueError(f"Invalid EPL {code} dataset: {len(raw)} rows; missing columns {sorted(missing)}")
        path = CACHE / f"E0_{code}.csv"
        fixtures = load_season(path)
        if len(fixtures) != 380 or any(
            not fixture.home or not fixture.away or fixture.ftr not in {"H", "D", "A"}
            or any(not math.isfinite(odd) or odd <= 1 for odd in (
                fixture.odds_h, fixture.odds_d, fixture.odds_a,
                fixture.close_h, fixture.close_d, fixture.close_a,
            )) for fixture in fixtures
        ):
            raise ValueError(f"EPL {code} dataset cannot replay all 380 fixtures with valid results and odds")
        data = path.read_bytes()
        record = {"file": path.name, "source_url": f"{BASE_URL}/{code}/E0.csv",
                  "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                  "rows": len(raw), "replayed_fixtures": len(fixtures)}
        records.append(record)
        print(f"Prepared {path.name}: {len(fixtures)} real fixtures; SHA256 {record['sha256']}")
    result = {"schema_version": 1, "source": "https://www.football-data.co.uk/englandm.php",
              "notice": "Historical provider downloads for the full test suite; raw data remains gitignored.",
              "files": records}
    output = ROOT / "evals" / "out" / "test_data_sources.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    prepare()
