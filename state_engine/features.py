"""Feature engineering for the PA-first State Engine.

All features are calculated using only information available at or before time t.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for PA-based features."""

    window: int = 24
    acceptance_lookback: int = 8
    enable_slopes: bool = False


class FeatureEngineer:
    """Compute PA-native features for State Engine classification."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def compute_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with engineered features.

        Parameters
        ----------
        ohlcv:
            DataFrame with columns: ["open", "high", "low", "close", "volume"].
        """
        self._validate_input(ohlcv)
        cfg = self.config

        high = ohlcv["high"]
        low = ohlcv["low"]
        close = ohlcv["close"]

        atr_w = self._atr(high, low, close, cfg.window)
        displacement = (close - close.shift(cfg.window)).abs() / atr_w
        path_length = close.diff().abs().rolling(cfg.window).sum() / atr_w
        efficiency = displacement / path_length.replace(0, np.nan)

        range_high = high.rolling(cfg.window).max()
        range_low = low.rolling(cfg.window).min()
        range_width = (range_high - range_low)
        range_w = range_width / atr_w

        close_location = (close - range_low) / range_width.replace(0, np.nan)

        acceptance = self._acceptance_ratio(
            close,
            range_low,
            range_width,
            cfg.window,
            cfg.acceptance_lookback,
        )

        reentry = self._reentry_count(close, range_low, range_high, cfg.window)
        inside_ratio = self._inside_bars_ratio(high, low, cfg.window)
        swing_counts = self._swing_counts(high, low, cfg.window)

        features = pd.DataFrame(
            {
                "D": displacement,
                "ER": efficiency,
                "A": acceptance,
                "Range_W": range_w,
                "CloseLocation": close_location,
                "ReentryCount": reentry,
                "InsideBarsRatio": inside_ratio,
                "SwingCounts": swing_counts,
            },
            index=ohlcv.index,
        )

        if cfg.enable_slopes:
            features["EfficiencySlope"] = self._slope(efficiency, cfg.window)
            features["RangeSlope"] = self._slope(range_w, cfg.window)

        return features

    def feature_names(self) -> List[str]:
        """Return the ordered list of feature names."""
        names = [
            "D",
            "ER",
            "A",
            "Range_W",
            "CloseLocation",
            "ReentryCount",
            "InsideBarsRatio",
            "SwingCounts",
        ]
        if self.config.enable_slopes:
            names.extend(["EfficiencySlope", "RangeSlope"])
        return names

    @staticmethod
    def _validate_input(ohlcv: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close"}
        missing = required - set(ohlcv.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(window).mean()

    @staticmethod
    def _acceptance_ratio(
        close: pd.Series,
        range_low: pd.Series,
        range_width: pd.Series,
        window: int,
        lookback: int,
    ) -> pd.Series:
        direction = np.sign(close - close.shift(window))
        upper_zone = range_low + (2.0 / 3.0) * range_width
        lower_zone = range_low + (1.0 / 3.0) * range_width

        up_accept = close >= upper_zone
        down_accept = close <= lower_zone
        zone_accept = np.where(direction >= 0, up_accept, down_accept)
        zone_accept = pd.Series(zone_accept, index=close.index)

        return zone_accept.rolling(lookback).mean()

    @staticmethod
    def _reentry_count(
        close: pd.Series,
        range_low: pd.Series,
        range_high: pd.Series,
        window: int,
    ) -> pd.Series:
        inside = (close >= range_low) & (close <= range_high)
        prev_inside = inside.shift(1)
        reentry = inside & (prev_inside == False)
        return reentry.rolling(window).sum()

    @staticmethod
    def _inside_bars_ratio(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        inside = (high <= prev_high) & (low >= prev_low)
        return inside.rolling(window).mean()

    @staticmethod
    def _swing_counts(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
        swing_up = (high > high.shift(1)) & (high.shift(1) > high.shift(2))
        swing_down = (low < low.shift(1)) & (low.shift(1) < low.shift(2))
        swings = swing_up | swing_down
        return swings.rolling(window).sum()

    @staticmethod
    def _slope(series: pd.Series, window: int) -> pd.Series:
        return (series - series.shift(window)) / window


__all__ = ["FeatureConfig", "FeatureEngineer"]
