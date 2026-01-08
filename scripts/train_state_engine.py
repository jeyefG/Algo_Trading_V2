"""Train the State Engine model from MetaTrader 5 data."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys

# --- ensure project root is on PYTHONPATH ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from typing import Any

import numpy as np
import pandas as pd

from state_engine.features import FeatureConfig
from state_engine.gating import GatingPolicy
from state_engine.labels import StateLabels
from state_engine.model import StateEngineModel, StateEngineModelConfig
from state_engine.mt5_connector import MT5Connector
from state_engine.pipeline import DatasetBuilder




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train State Engine model.")
    parser.add_argument("--symbol", required=True, help="Símbolo MT5 (ej. EURUSD)")
    parser.add_argument(
        "--start",
        required=True,
        help="Fecha inicio (YYYY-MM-DD) para descarga H1",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="Fecha fin (YYYY-MM-DD) para descarga H1",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Model output path. Si no se indica, se genera automáticamente desde el símbolo.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directorio base para guardar el modelo cuando --model-out no se especifica.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, WARNING)")
    parser.add_argument("--min-samples", type=int, default=2000, help="Minimum samples required to train")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/test split ratio (0-1)")
    parser.add_argument("--no-rich", action="store_true", help="Disable rich console output")
    parser.add_argument("--report-out", type=Path, help="Optional report output path (.json)")
    parser.add_argument("--class-weight-balanced", action="store_true", help="Use class_weight='balanced' in LightGBM")
    return parser.parse_args()


def try_import_rich() -> dict[str, Any] | None:
    try:
        from rich.console import Console
        from rich.table import Table

        return {"Console": Console, "Table": Table}
    except Exception:
        return None


def setup_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("state_engine")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)-8s %(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def class_distribution(labels: np.ndarray, label_order: list[StateLabels]) -> list[dict[str, Any]]:
    total = len(labels)
    result = []
    for label in label_order:
        count = int(np.sum(labels == label))
        pct = (count / total) * 100 if total else 0.0
        result.append({"label": label.name, "count": count, "pct": pct})
    return result


def confusion_matrix(labels_true: np.ndarray, labels_pred: np.ndarray, label_order: list[StateLabels]) -> np.ndarray:
    matrix = np.zeros((len(label_order), len(label_order)), dtype=int)
    for i, actual in enumerate(label_order):
        for j, predicted in enumerate(label_order):
            matrix[i, j] = int(np.sum((labels_true == actual) & (labels_pred == predicted)))
    return matrix


def f1_macro(matrix: np.ndarray) -> float:
    f1_scores = []
    for idx in range(matrix.shape[0]):
        tp = matrix[idx, idx]
        fp = matrix[:, idx].sum() - tp
        fn = matrix[idx, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def percentiles(series: pd.Series, qs: list[float]) -> dict[str, float | None]:
    """Return selected percentiles for a numeric Series.

    Parameters
    ----------
    series:
        Numeric series (non-numeric coerced to NaN).
    qs:
        Percentiles in [0, 100], e.g. [0, 50, 90, 95, 99, 100].

    Returns
    -------
    dict mapping percentile -> value (or None if empty after dropping NaN).
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {str(q): None for q in qs}
    qv = s.quantile([q / 100.0 for q in qs]).to_dict()
    return {str(q): float(qv[q / 100.0]) for q in qs}


def render_table(
    console: Any,
    table_class: Any,
    title: str,
    columns: list[str],
    rows: list[list[Any]],
) -> None:
    table = table_class(title=title)
    for col in columns:
        table.add_column(str(col))
    for row in rows:
        table.add_row(*[str(x) for x in row])
    console.print(table)


def main() -> None:
    args = parse_args()
    
    def _safe_symbol(sym: str) -> str:
    # deja letras/números/._- y reemplaza lo demás por _
        return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in sym)

    if args.model_out is None:
        sym = _safe_symbol(args.symbol)
        args.model_out = args.model_dir / f"{sym}_state_engine.pkl"
        
    rich_modules = try_import_rich() if not args.no_rich else None
    logger = setup_logging(args.log_level)

    console = None
    use_rich = bool(rich_modules)
    if use_rich and rich_modules:
        console = rich_modules["Console"]()

    def step(name: str) -> float:
        logger.info("stage=%s", name)
        return datetime.now(tz=timezone.utc).timestamp()

    def step_done(start_ts: float) -> float:
        return datetime.now(tz=timezone.utc).timestamp() - start_ts

    stage_start = step("descarga_h1")
    connector = MT5Connector()
    fecha_inicio = pd.to_datetime(args.start)
    fecha_fin = pd.to_datetime(args.end)

    ohlcv = connector.obtener_h1(args.symbol, fecha_inicio, fecha_fin)
    
    # 1) Hora servidor MT5 (inferida desde último tick)
    server_now = connector.server_now(args.symbol).tz_localize(None)
    
    # 2) Guardrail correcto: usar solo velas H1 cerradas según hora servidor
    #    La vela en formación empieza en server_now.floor("H")
    cutoff = server_now.floor("H")
    ohlcv = ohlcv[ohlcv.index < cutoff]
    
    # 3) Timestamp última vela usada
    last_bar_ts = pd.Timestamp(ohlcv.index.max()).tz_localize(None)
    
    # 4) Edad en minutos (server_now - last_bar_ts)
    bar_age_minutes = (server_now - last_bar_ts).total_seconds() / 60.0
    
    # 5) Diagnóstico: si el tick está viejo, esto será grande
    now_utc = pd.Timestamp.utcnow().tz_localize(None)
    tick_age_min_utc = (now_utc - server_now).total_seconds() / 60.0

    elapsed_download = step_done(stage_start)
    logger.info("download_rows=%s elapsed=%.2fs", len(ohlcv), elapsed_download)

    stage_start = step("build_dataset")
    feature_config = FeatureConfig()
    builder = DatasetBuilder(feature_config=feature_config)
    artifacts = builder.build(ohlcv)
    full_features = artifacts.full_features
    features_raw = artifacts.features
    labels_raw = artifacts.labels
    elapsed_build = step_done(stage_start)
    logger.info(
        "features_raw=%s labels_raw=%s elapsed=%.2fs",
        len(features_raw),
        len(labels_raw),
        elapsed_build,
    )

    stage_start = step("align_and_clean")
    aligned = features_raw.join(labels_raw.rename("label"), how="inner")
    dropped_nan = int(aligned.isna().any(axis=1).sum())
    aligned = aligned.dropna()
    features = aligned.drop(columns=["label"])
    labels = aligned["label"].astype(int)
    elapsed_clean = step_done(stage_start)
    logger.info("aligned_samples=%s dropped_nan=%s elapsed=%.2fs", len(aligned), dropped_nan, elapsed_clean)
    logger.info("n_features=%s feature_names=%s", features.shape[1], features.columns.tolist())

    if len(features) < args.min_samples:
        raise RuntimeError(f"Not enough samples to train: {len(features)} < {args.min_samples}")

    stage_start = step("split")
    n_total = len(features)
    n_train = int(n_total * args.split_ratio)
    features_train = features.iloc[:n_train]
    labels_train = labels.iloc[:n_train]
    features_test = features.iloc[n_train:]
    labels_test = labels.iloc[n_train:]
    elapsed_split = step_done(stage_start)
    logger.info(
        "n_train=%s n_test=%s split_ratio=%.2f elapsed=%.2fs",
        len(features_train),
        len(features_test),
        args.split_ratio,
        elapsed_split,
    )

    label_order = [StateLabels.BALANCE, StateLabels.TRANSITION, StateLabels.TREND]

    stage_start = step("train_model")
    model_config = StateEngineModelConfig(
        class_weight="balanced" if args.class_weight_balanced else None,
    )
    model = StateEngineModel(model_config)
    model.fit(features_train, labels_train)
    elapsed_train = step_done(stage_start)
    logger.info("train_elapsed=%.2fs", elapsed_train)

    stage_start = step("evaluate")
    preds_test = model.predict_state(features_test).to_numpy()
    labels_test_np = labels_test.to_numpy()
    matrix = confusion_matrix(labels_test_np, preds_test, label_order)
    acc = float(np.mean(preds_test == labels_test_np)) if len(labels_test_np) else 0.0
    f1 = f1_macro(matrix)
    elapsed_eval = step_done(stage_start)
    logger.info("accuracy=%.4f f1_macro=%.4f elapsed=%.2fs", acc, f1, elapsed_eval)

    stage_start = step("predict_outputs")
    outputs = model.predict_outputs(features)
    elapsed_outputs = step_done(stage_start)
    logger.info("outputs_rows=%s elapsed=%.2fs", len(outputs), elapsed_outputs)

    # Extra reporting helpers
    state_hat_dist = class_distribution(outputs["state_hat"].to_numpy(), label_order)
    q_list = [0, 50, 75, 90, 95, 99, 100]
    breakmag_p = percentiles(full_features["BreakMag"], q_list) if "BreakMag" in full_features.columns else {str(q): None for q in q_list}
    reentry_p = percentiles(full_features["ReentryCount"], q_list) if "ReentryCount" in full_features.columns else {str(q): None for q in q_list}

    stage_start = step("gating")
    gating_policy = GatingPolicy()
    gating = gating_policy.apply(outputs, full_features)
    allow_any = gating.any(axis=1)
    
    # --- "Última vela" para reporting
    last_idx = outputs.index.max()
    
    # ALLOW final (cualquier regla true)
    last_allow = bool(allow_any.loc[last_idx]) if last_idx in allow_any.index else False
    
    # Estado y margen de la última vela
    last_state_hat = int(outputs.loc[last_idx, "state_hat"]) if last_idx in outputs.index else None
    last_margin = float(outputs.loc[last_idx, "margin"]) if last_idx in outputs.index else None
    
    # Reglas específicas que dispararon en la última vela (diagnóstico)
    last_rules = [c for c in gating.columns if bool(gating.loc[last_idx, c])] if last_idx in gating.index else []

    gating_allow_rate = float(allow_any.mean()) if len(gating) else 0.0
    gating_block_rate = 1.0 - gating_allow_rate
    elapsed_gating = step_done(stage_start)
    logger.info("gating_allow_rate=%.2f%% elapsed=%.2fs", gating_allow_rate * 100, elapsed_gating)
    logger.info("gating_thresholds=%s", asdict(gating_policy.thresholds))

    stage_start = step("save_model")
    metadata = {
        "feature_names_used": features.columns.tolist(),
        "label_names": {label.name: int(label) for label in StateLabels},
        "classes": [label.name for label in StateLabels],
        "feature_config": asdict(feature_config),
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "symbol": args.symbol,
        "start": args.start,
        "end": args.end,
        "n_samples": len(features),
        "n_train": len(features_train),
        "n_test": len(features_test),
        "split_ratio": args.split_ratio,
        "accuracy": acc,
        "f1_macro": f1,
        "latest_bar": {
            "last_bar_time": str(last_bar_ts),
            "server_now": str(server_now),
            "bar_age_minutes": float(bar_age_minutes),
            "tick_age_min_vs_utc": float(tick_age_min_utc),
        },
    }
    model.save(args.model_out, metadata=metadata)
    elapsed_save = step_done(stage_start)
    logger.info("model_path=%s elapsed=%.2fs", str(args.model_out), elapsed_save)

    # Baseline (majority class)
    label_dist = class_distribution(labels.to_numpy(), label_order)
    baseline = max(label_dist, key=lambda r: r["count"]) if label_dist else None
    baseline_label = baseline["label"] if baseline else "NA"
    baseline_pct = baseline["pct"] if baseline else 0.0
    logger.info("baseline_state=%s baseline_pct=%.2f", baseline_label, baseline_pct)

    importances = model.feature_importances().head(15)

    # Guardrail detail counts (kept as in your existing report)
    gating_thresholds = gating_policy.thresholds
    transition_rule = outputs["margin"] >= gating_thresholds.transition_margin_min
    transition_break = full_features["BreakMag"] >= gating_thresholds.transition_breakmag_min
    transition_reentry = full_features["ReentryCount"] >= gating_thresholds.transition_reentry_min

    if use_rich and console:
        table_class = rich_modules["Table"]
        render_table(
            console,
            table_class,
            "Distribución de clases",
            ["Clase", "Count", "%"],
            [[row["label"], row["count"], f"{row['pct']:.2f}%"] for row in label_dist],
        )
        render_table(
            console,
            table_class,
            "Distribución state_hat (predicción)",
            ["Clase", "Count", "%"],
            [[row["label"], row["count"], f"{row['pct']:.2f}%"] for row in state_hat_dist],
        )
        render_table(
            console,
            table_class,
            "Percentiles (BreakMag / ReentryCount)",
            ["Métrica", "p0", "p50", "p75", "p90", "p95", "p99", "p100"],
            [
                [
                    "BreakMag",
                    breakmag_p.get("0"),
                    breakmag_p.get("50"),
                    breakmag_p.get("75"),
                    breakmag_p.get("90"),
                    breakmag_p.get("95"),
                    breakmag_p.get("99"),
                    breakmag_p.get("100"),
                ],
                [
                    "ReentryCount",
                    reentry_p.get("0"),
                    reentry_p.get("50"),
                    reentry_p.get("75"),
                    reentry_p.get("90"),
                    reentry_p.get("95"),
                    reentry_p.get("99"),
                    reentry_p.get("100"),
                ],
            ],
        )
        render_table(
            console,
            table_class,
            "Métricas (Test)",
            ["Accuracy", "F1 Macro"],
            [[f"{acc:.4f}", f"{f1:.4f}"]],
        )
        render_table(
            console,
            table_class,
            "Matriz de confusión",
            ["Actual \\ Pred", *[l.name for l in label_order]],
            [
                [label_order[i].name, *matrix[i].tolist()]
                for i in range(len(label_order))
            ],
        )
        render_table(
            console,
            table_class,
            "Top features (importance)",
            ["Feature", "Importance"],
            [[idx, float(val)] for idx, val in importances.items()],
        )
        # Gating summary as-is (your gating object is boolean columns per rule)
        gating_summary = []
        for col in gating.columns:
            count = int(gating[col].sum())
            pct = (count / len(gating)) * 100 if len(gating) else 0.0
            gating_summary.append([col, count, f"{pct:.2f}%"])
        render_table(
            console,
            table_class,
            "Resumen gating",
            ["Regla", "Count", "%"],
            gating_summary,
        )
        render_table(
            console,
            table_class,
            "Detalles transición (guardrails)",
            ["Condición", "Count"],
            [
                ["margin>=min", int(transition_rule.sum())],
                ["margin+breakmag", int((transition_rule & transition_break).sum())],
                ["margin+breakmag+reentry", int((transition_rule & transition_break & transition_reentry).sum())],
            ],
        )
        console.print("=== State Engine Training Summary ===")
        console.print(f"Symbol: {args.symbol}")
        console.print(f"Period: {args.start} -> {args.end}")
        console.print(f"Samples: {len(features)} (train={len(features_train)}, test={len(features_test)})")
        console.print(f"Baseline: {baseline_label} ({baseline_pct:.2f}%)")
        console.print(f"Accuracy: {acc:.4f} | F1 Macro: {f1:.4f}")
        console.print(f"Gating allow rate: {gating_allow_rate*100:.2f}% (block {gating_block_rate*100:.2f}%)")
        console.print(f"Last H1 bar used: {last_bar_ts} | age_min={bar_age_minutes:.2f}")
        console.print(f"Server now (tick): {server_now} | tick_age_min_vs_utc={tick_age_min_utc:.2f}")
        console.print(f"Last bar decision: ALLOW={last_allow} | state_hat={StateLabels(last_state_hat).name if last_state_hat is not None else 'NA'} | margin={last_margin:.4f}")
        console.print(f"Last bar rules fired: {last_rules if last_rules else '[]'}")
        console.print(f"Model saved: {args.model_out}")
    else:
        logger.info("class_distribution=%s", label_dist)
        logger.info("state_hat_distribution=%s", state_hat_dist)
        logger.info("breakmag_percentiles=%s", breakmag_p)
        logger.info("reentry_percentiles=%s", reentry_p)
        logger.info("confusion_matrix=%s", matrix.tolist())
        logger.info("top_features=%s", importances.to_dict())

    # Optional: JSON report
    report_payload = {
        "symbol": args.symbol,
        "start": args.start,
        "end": args.end,
        "n_samples": len(features),
        "n_train": len(features_train),
        "n_test": len(features_test),
        "split_ratio": args.split_ratio,
        "baseline": {"state": baseline_label, "pct": baseline_pct},
        "metrics": {"accuracy": acc, "f1_macro": f1},
        "confusion_matrix": {
            "labels": [label.name for label in label_order],
            "matrix": matrix.tolist(),
        },
        "class_distribution": label_dist,
        "state_hat_distribution": state_hat_dist,
        "guardrail_percentiles": {
            "quantiles": q_list,
            "BreakMag": breakmag_p,
            "ReentryCount": reentry_p,
        },
        "feature_importances": importances.to_dict(),
        "gating": {
            "thresholds": asdict(gating_policy.thresholds),
            "allow_rate": gating_allow_rate,
            "block_rate": gating_block_rate,
        },
        "model_path": str(args.model_out),
        "metadata": metadata,
        "training": {"class_weight": model_config.class_weight},
        "timings": {
            "download": elapsed_download,
            "build_dataset": elapsed_build,
            "align_and_clean": elapsed_clean,
            "split": elapsed_split,
            "train": elapsed_train,
            "evaluate": elapsed_eval,
            "predict_outputs": elapsed_outputs,
            "gating": elapsed_gating,
            "save_model": elapsed_save,
        },
        "latest_bar": {
            "last_bar_time": str(last_bar_ts),
            "server_now": str(server_now),
            "bar_age_minutes": float(bar_age_minutes),
            "tick_age_min_vs_utc": float(tick_age_min_utc),
        },
    }

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
