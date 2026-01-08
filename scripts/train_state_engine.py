"""Train the State Engine model from CSV data.

Expected CSV columns: timestamp, open, high, low, close, volume
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from state_engine import (
    FeatureConfig,
    GatingPolicy,
    StateEngineModel,
    StateEngineModelConfig,
)
from state_engine.pipeline import DatasetBuilder


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train State Engine model.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV with OHLCV data")
    parser.add_argument("--model-out", type=Path, required=True, help="Model output path")
    args = parser.parse_args()

    ohlcv = load_ohlcv(args.csv)

    dataset_builder = DatasetBuilder(FeatureConfig())
    artifacts = dataset_builder.build(ohlcv)

    model = StateEngineModel(StateEngineModelConfig())
    model.fit(artifacts.features.dropna(), artifacts.labels.dropna())
    model.save(args.model_out)

    probs = model.predict_proba(artifacts.features.dropna())
    gating = GatingPolicy().apply(probs)

    print("Training complete.")
    print(gating.tail())


if __name__ == "__main__":
    main()
