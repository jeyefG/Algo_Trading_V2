"""Deterministic gating logic for PA State Engine outputs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GatingThresholds:
    trend_min: float = 0.60
    trend_transition_max: float = 0.30
    balance_min: float = 0.60
    transition_min: float = 0.60


class GatingPolicy:
    """Apply ALLOW_* rules based on StateEngine probabilities."""

    def __init__(self, thresholds: GatingThresholds | None = None) -> None:
        self.thresholds = thresholds or GatingThresholds()

    def apply(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with ALLOW_* columns."""
        required = {"P(balance)", "P(transition)", "P(trend)"}
        missing = required - set(probabilities.columns)
        if missing:
            raise ValueError(f"Missing required probability columns: {sorted(missing)}")

        th = self.thresholds
        allow_trend_pullback = (probabilities["P(trend)"] >= th.trend_min) & (
            probabilities["P(transition)"] <= th.trend_transition_max
        )
        allow_balance_fade = probabilities["P(balance)"] >= th.balance_min
        allow_transition_failure = probabilities["P(transition)"] >= th.transition_min

        return pd.DataFrame(
            {
                "ALLOW_trend_pullback": allow_trend_pullback.astype(int),
                "ALLOW_balance_fade": allow_balance_fade.astype(int),
                "ALLOW_transition_failure": allow_transition_failure.astype(int),
            },
            index=probabilities.index,
        )


__all__ = ["GatingThresholds", "GatingPolicy"]
