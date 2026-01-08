"""Label generation for State Engine states."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import pandas as pd


class StateLabels(IntEnum):
    BALANCE = 0
    TRANSITION = 1
    TREND = 2


@dataclass(frozen=True)
class StateLabeler:
    """Rule-based state labeling for offline training."""

    trend_er: float = 0.35
    trend_d: float = 1.2
    trend_a: float = 0.55

    balance_er: float = 0.20
    balance_d: float = 0.8
    balance_a: float = 0.50

    def label(self, features: pd.DataFrame) -> pd.Series:
        """Return Series of StateLabels using PA rules."""
        required = {"ER", "D", "A"}
        missing = required - set(features.columns)
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")

        er = features["ER"]
        d = features["D"]
        a = features["A"]

        trend = (er > self.trend_er) & (d > self.trend_d) & (a > self.trend_a)
        balance = (er < self.balance_er) & (d < self.balance_d) & (a < self.balance_a)

        labels = pd.Series(StateLabels.TRANSITION, index=features.index)
        labels[trend] = StateLabels.TREND
        labels[balance] = StateLabels.BALANCE
        return labels


__all__ = ["StateLabels", "StateLabeler"]
