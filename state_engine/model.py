"""LightGBM model wrapper for the State Engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import lightgbm as lgb

from .labels import StateLabels


@dataclass(frozen=True)
class StateEngineModelConfig:
    """Configuration for the LightGBM model."""

    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 300
    max_depth: int = -1
    random_state: int = 42


class StateEngineModel:
    """Train and run a LightGBM multiclass classifier."""

    def __init__(self, config: StateEngineModelConfig | None = None) -> None:
        self.config = config or StateEngineModelConfig()
        self._model: lgb.LGBMClassifier | None = None

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Fit the LightGBM model."""
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(StateLabels),
            num_leaves=self.config.num_leaves,
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
        )
        model.fit(features, labels)
        self._model = model

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities as a DataFrame."""
        self._require_fitted_model()
        probs = self._model.predict_proba(features)
        return pd.DataFrame(
            probs,
            columns=["P(balance)", "P(transition)", "P(trend)"],
            index=features.index,
        )

    def predict_state(self, features: pd.DataFrame) -> pd.Series:
        """Predict most likely state label."""
        self._require_fitted_model()
        preds = self._model.predict(features)
        return pd.Series(preds, index=features.index)

    def save(self, path: str | Path) -> None:
        """Persist model to disk."""
        self._require_fitted_model()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.booster_.save_model(path.as_posix())

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        path = Path(path)
        booster = lgb.Booster(model_file=path.as_posix())
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(StateLabels),
        )
        model._Booster = booster
        self._model = model

    def _require_fitted_model(self) -> None:
        if self._model is None:
            raise RuntimeError("StateEngineModel must be fit or loaded before use.")


__all__ = ["StateEngineModel", "StateEngineModelConfig"]
