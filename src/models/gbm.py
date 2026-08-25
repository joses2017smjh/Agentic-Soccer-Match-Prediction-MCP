"""Base statistical model: gradient-boosted trees over the tabular features.

Two heads share the feature frame from src/features/build_features.py (plus
availability and sentiment columns from src/news/):

- ``OutcomeGBM``   — multiclass H/D/A probabilities (raw; calibrated and
  conformal-wrapped downstream in src/models/calibration.py).
- ``TeamXGGBM``    — one regressor per side predicting team expected goals,
  which parameterize the Dixon–Coles grid (src/models/score_grid.py) so every
  scoreline-derived market is consistent with the same xG estimates.

Both persist with metadata (feature list, training window, seed) so the
Phase B inference server can refuse to score on a feature-schema mismatch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

OUTCOME_CLASSES: tuple[str, str, str] = ("home", "draw", "away")


class FeatureSchemaError(RuntimeError):
    """Inference frame does not match the schema the model was trained on."""


def _check_schema(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise FeatureSchemaError(f"missing features at inference: {missing}")
    return df[feature_names]


@dataclass
class OutcomeGBM:
    """XGBoost multiclass model for Home/Draw/Away."""

    params: dict = field(default_factory=dict)
    seed: int = 42
    feature_names: list[str] = field(default_factory=list)
    _model: xgb.XGBClassifier | None = None

    def fit(
        self, x: pd.DataFrame, y: pd.Series,
        base_margin: np.ndarray | None = None,
    ) -> "OutcomeGBM":
        """y contains labels from OUTCOME_CLASSES.

        base_margin: optional (n, 3) log-probability offset.  When supplied
        the GBM learns residuals on top of the market prior; with zero trees
        the predictions reproduce softmax(base_margin) exactly.
        """
        self.feature_names = list(x.columns)
        self._uses_base_margin = base_margin is not None
        defaults = dict(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
            eval_metric="mlogloss", random_state=self.seed,
        )
        if base_margin is not None:
            defaults["base_score"] = 0
        self._model = xgb.XGBClassifier(**{**defaults, **self.params})
        codes = pd.Categorical(y, categories=list(OUTCOME_CLASSES)).codes
        if (codes < 0).any():
            raise ValueError(f"labels must be one of {OUTCOME_CLASSES}")
        fit_kw: dict = {}
        if base_margin is not None:
            fit_kw["base_margin"] = base_margin.reshape(-1, len(OUTCOME_CLASSES))
        self._model.fit(x, codes, **fit_kw)
        return self

    def predict_proba(
        self, x: pd.DataFrame, base_margin: np.ndarray | None = None,
    ) -> np.ndarray:
        """(n, 3) array ordered home/draw/away.

        If the model was trained with base_margin, pass the corresponding
        market log-probabilities here too.
        """
        assert self._model is not None, "call fit() or load() first"
        feats = _check_schema(x, self.feature_names)
        if base_margin is not None:
            margin = base_margin.reshape(-1, len(OUTCOME_CLASSES))
            dmat = xgb.DMatrix(feats, base_margin=margin)
            preds = self._model.get_booster().predict(dmat)
            return preds.reshape(-1, len(OUTCOME_CLASSES))
        return self._model.predict_proba(feats)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        assert self._model is not None
        self._model.save_model(path / "outcome.ubj")
        (path / "outcome_meta.json").write_text(json.dumps(
            {"feature_names": self.feature_names, "seed": self.seed,
             "classes": OUTCOME_CLASSES,
             "uses_base_margin": getattr(self, "_uses_base_margin", False)}
        ))

    @classmethod
    def load(cls, path: Path) -> "OutcomeGBM":
        meta = json.loads((path / "outcome_meta.json").read_text())
        obj = cls(seed=meta["seed"], feature_names=meta["feature_names"])
        obj._uses_base_margin = meta.get("uses_base_margin", False)
        obj._model = xgb.XGBClassifier()
        obj._model.load_model(path / "outcome.ubj")
        return obj


@dataclass
class TeamXGGBM:
    """Two XGBoost regressors predicting home and away expected goals."""

    params: dict = field(default_factory=dict)
    seed: int = 42
    feature_names: list[str] = field(default_factory=list)
    _home: xgb.XGBRegressor | None = None
    _away: xgb.XGBRegressor | None = None

    def _new(self) -> xgb.XGBRegressor:
        defaults = dict(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="count:poisson", random_state=self.seed,
        )
        return xgb.XGBRegressor(**{**defaults, **self.params})

    def fit(self, x: pd.DataFrame, xg_home: pd.Series, xg_away: pd.Series) -> "TeamXGGBM":
        self.feature_names = list(x.columns)
        self._home, self._away = self._new(), self._new()
        self._home.fit(x, xg_home)
        self._away.fit(x, xg_away)
        return self

    def predict(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(mu_home, mu_away), clipped to a sane positive range."""
        assert self._home is not None and self._away is not None
        feats = _check_schema(x, self.feature_names)
        clip = lambda a: np.clip(a, 0.05, 6.0)  # noqa: E731
        return clip(self._home.predict(feats)), clip(self._away.predict(feats))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        assert self._home is not None and self._away is not None
        self._home.save_model(path / "xg_home.ubj")
        self._away.save_model(path / "xg_away.ubj")
        (path / "xg_meta.json").write_text(json.dumps(
            {"feature_names": self.feature_names, "seed": self.seed}
        ))

    @classmethod
    def load(cls, path: Path) -> "TeamXGGBM":
        meta = json.loads((path / "xg_meta.json").read_text())
        obj = cls(seed=meta["seed"], feature_names=meta["feature_names"])
        obj._home, obj._away = xgb.XGBRegressor(), xgb.XGBRegressor()
        obj._home.load_model(path / "xg_home.ubj")
        obj._away.load_model(path / "xg_away.ubj")
        return obj
