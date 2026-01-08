"""Train the State Engine model from MetaTrader 5 data."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from state_engine import (
    FeatureConfig,
    GatingPolicy,
    MT5Connector,
    StateEngineModel,
    StateEngineModelConfig,
)
from state_engine.pipeline import DatasetBuilder

def main() -> None:
    parser = argparse.ArgumentParser(description="Train State Engine model.")
    parser.add_argument("--symbol", required=True, help="Símbolo MT5 (ej. EURUSD)")
    parser.add_argument(
        "--start",
        required=True,
        help="Fecha inicio (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="Fecha fin (YYYY-MM-DD)",
    )
    parser.add_argument("--model-out", type=Path, required=True, help="Model output path")
    args = parser.parse_args()

    connector = MT5Connector()
    try:
        start_dt = datetime.fromisoformat(args.start)
        end_dt = datetime.fromisoformat(args.end)
        ohlcv = connector.obtener_h1(args.symbol, start_dt, end_dt)
    finally:
        connector.shutdown()

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
