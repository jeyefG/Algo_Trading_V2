"""Watchdog for State Engine opportunities on new H1 bars."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd

# --- ensure project root is on PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from state_engine.features import FeatureConfig
from state_engine.gating import GatingPolicy
from state_engine.labels import StateLabels
from state_engine.model import StateEngineModel
from state_engine.mt5_connector import MT5Connector
from state_engine.pipeline import DatasetBuilder


LABEL_ORDER = [StateLabels.BALANCE, StateLabels.TRANSITION, StateLabels.TREND]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watchdog for State Engine ALLOW_* signals.")
    parser.add_argument(
        "--symbols",
        required=True,
        help="Símbolos MT5 separados por coma (ej. EURUSD,XAUUSD)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directorio donde se encuentran los modelos entrenados.",
    )
    parser.add_argument(
        "--model-template",
        default="{symbol}_state_engine.pkl",
        help="Plantilla del nombre de modelo. Usa {symbol} ya sanitizado.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=45,
        help="Días de historial H1 para calcular features.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="Intervalo de polling en segundos para nuevas velas H1.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Ejecuta un solo ciclo y termina.",
    )
    return parser.parse_args()


def safe_symbol(sym: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in sym)


def feature_config_from_metadata(metadata: dict[str, Any]) -> FeatureConfig:
    raw = metadata.get("feature_config") if isinstance(metadata, dict) else None
    if not isinstance(raw, dict):
        return FeatureConfig()
    allowed = {key for key in FeatureConfig.__dataclass_fields__}
    config = {key: value for key, value in raw.items() if key in allowed}
    return FeatureConfig(**config)


def class_distribution(labels: pd.Series) -> list[dict[str, Any]]:
    total = len(labels)
    result = []
    for label in LABEL_ORDER:
        count = int((labels == int(label)).sum())
        pct = (count / total) * 100 if total else 0.0
        result.append({"label": label.name, "count": count, "pct": pct})
    return result


def load_model(symbol: str, model_dir: Path, template: str) -> tuple[StateEngineModel, Path]:
    path = model_dir / template.format(symbol=safe_symbol(symbol))
    model = StateEngineModel()
    model.load(path)
    return model, path


def align_features(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    aligned = features.join(labels.rename("label"), how="inner")
    aligned = aligned.dropna()
    return aligned


def render_summary(
    *,
    symbol: str,
    model_path: Path,
    metadata: dict[str, Any],
    labels: pd.Series,
    outputs: pd.DataFrame,
    gating: pd.DataFrame,
    last_bar_ts: pd.Timestamp,
    server_now: pd.Timestamp,
) -> None:
    allow_any = gating.any(axis=1)
    gating_allow_rate = float(allow_any.mean()) if len(gating) else 0.0
    gating_block_rate = 1.0 - gating_allow_rate

    last_idx = outputs.index.max()
    last_allow = bool(allow_any.loc[last_idx]) if last_idx in allow_any.index else False
    last_state_hat = outputs.loc[last_idx, "state_hat"] if last_idx in outputs.index else None
    last_margin = float(outputs.loc[last_idx, "margin"]) if last_idx in outputs.index else None
    last_rules = [col for col in gating.columns if bool(gating.loc[last_idx, col])] if last_idx in gating.index else []

    label_dist = class_distribution(labels)
    baseline = max(label_dist, key=lambda r: r["count"]) if label_dist else None
    baseline_label = baseline["label"] if baseline else "NA"
    baseline_pct = baseline["pct"] if baseline else 0.0

    bar_age_minutes = (server_now - last_bar_ts).total_seconds() / 60.0
    now_utc = pd.Timestamp.utcnow().tz_localize(None)
    tick_age_min_utc = (now_utc - server_now).total_seconds() / 60.0

    trained_start = metadata.get("start")
    trained_end = metadata.get("end")
    n_samples = metadata.get("n_samples")
    n_train = metadata.get("n_train")
    n_test = metadata.get("n_test")
    accuracy = metadata.get("accuracy")
    f1_macro = metadata.get("f1_macro")

    print("=== State Engine Training Summary ===")
    print(f"Symbol: {symbol}")
    if trained_start and trained_end:
        print(f"Period: {trained_start} -> {trained_end}")
    if n_samples is not None and n_train is not None and n_test is not None:
        print(f"Samples: {n_samples} (train={n_train}, test={n_test})")
    print(f"Baseline: {baseline_label} ({baseline_pct:.2f}%)")
    if accuracy is not None and f1_macro is not None:
        print(f"Accuracy: {accuracy:.4f} | F1 Macro: {f1_macro:.4f}")
    print(f"Gating allow rate: {gating_allow_rate*100:.2f}% (block {gating_block_rate*100:.2f}%)")
    print(f"Last H1 bar used: {last_bar_ts} | age_min={bar_age_minutes:.2f}")
    print(f"Server now (tick): {server_now} | tick_age_min_vs_utc={tick_age_min_utc:.2f}")
    state_label = StateLabels(int(last_state_hat)).name if last_state_hat is not None else "NA"
    margin_value = f"{last_margin:.4f}" if last_margin is not None else "NA"
    print(f"Last bar decision: ALLOW={last_allow} | state_hat={state_label} | margin={margin_value}")
    print(f"Last bar rules fired: {last_rules if last_rules else '[]'}")
    print(f"Model saved: {model_path}")


def main() -> None:
    args = parse_args()
    symbols = [sym.strip() for sym in args.symbols.split(",") if sym.strip()]
    if not symbols:
        raise ValueError("Debe especificar al menos un símbolo en --symbols.")

    connector = MT5Connector()
    models: dict[str, StateEngineModel] = {}
    model_paths: dict[str, Path] = {}
    feature_configs: dict[str, FeatureConfig] = {}

    try:
        for symbol in symbols:
            model, path = load_model(symbol, args.model_dir, args.model_template)
            models[symbol] = model
            model_paths[symbol] = path
            feature_configs[symbol] = feature_config_from_metadata(model.metadata)

        last_seen: dict[str, pd.Timestamp] = {}

        while True:
            for symbol in symbols:
                server_now = connector.server_now(symbol).tz_localize(None)
                cutoff = server_now.floor("H")
                start = cutoff - timedelta(days=args.lookback_days)
                end = cutoff + timedelta(days=1)

                ohlcv = connector.obtener_h1(symbol, start, end)
                ohlcv = ohlcv[ohlcv.index < cutoff]
                if ohlcv.empty:
                    continue

                last_bar_ts = pd.Timestamp(ohlcv.index.max()).tz_localize(None)
                if last_seen.get(symbol) == last_bar_ts:
                    continue

                builder = DatasetBuilder(feature_config=feature_configs[symbol])
                artifacts = builder.build(ohlcv)
                aligned = align_features(artifacts.features, artifacts.labels)
                if aligned.empty:
                    last_seen[symbol] = last_bar_ts
                    continue

                features = aligned.drop(columns=["label"])
                labels = aligned["label"].astype(int)
                full_features = artifacts.full_features.loc[features.index]

                outputs = models[symbol].predict_outputs(features)
                gating_policy = GatingPolicy()
                gating = gating_policy.apply(outputs, full_features)

                last_idx = outputs.index.max()
                allow_any = gating.any(axis=1)
                if last_idx in allow_any.index and bool(allow_any.loc[last_idx]):
                    render_summary(
                        symbol=symbol,
                        model_path=model_paths[symbol],
                        metadata=models[symbol].metadata,
                        labels=labels,
                        outputs=outputs,
                        gating=gating,
                        last_bar_ts=last_bar_ts,
                        server_now=server_now,
                    )

                last_seen[symbol] = last_bar_ts

            if args.once:
                break
            time.sleep(args.poll_seconds)
    finally:
        connector.shutdown()


if __name__ == "__main__":
    main()
