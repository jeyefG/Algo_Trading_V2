"""Deterministic gating logic for PA State Engine outputs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .labels import StateLabels


@dataclass(frozen=True)
class GatingThresholds:
    trend_margin_min: float = 0.15
    balance_margin_min: float = 0.10
    transition_margin_min: float = 0.10
    transition_breakmag_min: float = 0.25
    transition_reentry_min: float = 1.0


class GatingPolicy:
    """Apply ALLOW_* rules based on StateEngine state and margin."""

    def __init__(self, thresholds: GatingThresholds | None = None) -> None:
        self.thresholds = thresholds or GatingThresholds()

    def apply(self, outputs: pd.DataFrame, features: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return DataFrame with ALLOW_* columns."""
        required = {"state_hat", "margin"}
        missing = required - set(outputs.columns)
        if missing:
            raise ValueError(f"Missing required output columns: {sorted(missing)}")

        th = self.thresholds
        state_hat = outputs["state_hat"]
        margin = outputs["margin"]
        allow_trend_pullback = (state_hat == StateLabels.TREND) & (margin >= th.trend_margin_min)
        allow_balance_fade = (state_hat == StateLabels.BALANCE) & (margin >= th.balance_margin_min)

        allow_transition_failure = (state_hat == StateLabels.TRANSITION) & (margin >= th.transition_margin_min)
        if features is not None:
            required_features = {"BreakMag", "ReentryCount"}
            missing_features = required_features - set(features.columns)
            if missing_features:
                raise ValueError(f"Missing required features for gating: {sorted(missing_features)}")
            allow_transition_failure &= (
                (features["BreakMag"] >= th.transition_breakmag_min)
                & (features["ReentryCount"] >= th.transition_reentry_min)
            )

        return pd.DataFrame(
            {
                "ALLOW_trend_pullback": allow_trend_pullback.astype(int),
                "ALLOW_balance_fade": allow_balance_fade.astype(int),
                "ALLOW_transition_failure": allow_transition_failure.astype(int),
            },
            index=outputs.index,
        )


__all__ = ["GatingThresholds", "GatingPolicy"]
