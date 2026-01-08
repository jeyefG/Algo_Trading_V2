"""Train the State Engine model from MetaTrader 5 data."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # State_Engine/
sys.path.insert(0, str(ROOT))

from state_engine import (  # noqa: E402
    FeatureConfig,
    GatingPolicy,
    GatingThresholds,
    MT5Connector,
    StateEngineModel,
    StateEngineModelConfig,
    StateLabels,
)
from state_engine.pipeline import DatasetBuilder  # noqa: E402


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
    parser.add_argument("--model-out", type=Path, required=True, help="Model output path")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, WARNING)")
    parser.add_argument("--min-samples", type=int, default=2000, help="Minimum samples required to train")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/test split ratio (0-1)")
    parser.add_argument("--no-rich", action="store_true", help="Disable rich console output")
    parser.add_argument("--report-out", type=Path, help="Optional report output path (.json)")
    return parser.parse_args()


def try_import_rich() -> dict[str, Any]:
    if importlib.util.find_spec("rich") is None:
        return {}
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table

    return {
        "Console": Console,
        "RichHandler": RichHandler,
        "Progress": Progress,
        "SpinnerColumn": SpinnerColumn,
        "BarColumn": BarColumn,
        "TextColumn": TextColumn,
        "TimeElapsedColumn": TimeElapsedColumn,
        "Table": Table,
    }


def setup_logging(level: str, use_rich: bool, rich_modules: dict[str, Any]) -> logging.Logger:
    logger = logging.getLogger("state_engine.train")
    logger.setLevel(level.upper())
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    if use_rich:
        handler = rich_modules["RichHandler"](show_time=False, show_level=True, show_path=False)
        handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
    logger.addHandler(handler)
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


def render_table(
    console: Any,
    table_class: Any,
    title: str,
    columns: list[str],
    rows: list[list[Any]],
) -> None:
    table = table_class(title=title)
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*[str(item) for item in row])
    console.print(table)


def main() -> None:
    args = parse_args()
    rich_modules = try_import_rich()
    use_rich = bool(rich_modules) and not args.no_rich
    logger = setup_logging(args.log_level, use_rich, rich_modules)
    console = rich_modules.get("Console")() if use_rich else None

    if not (0.0 < args.split_ratio < 1.0):
        raise ValueError("--split-ratio must be between 0 and 1.")

    start_time = time.perf_counter()
    progress = None
    if use_rich:
        progress = rich_modules["Progress"](
            rich_modules["SpinnerColumn"](),
            rich_modules["TextColumn"]("{task.description}"),
            rich_modules["BarColumn"](),
            rich_modules["TimeElapsedColumn"](),
        )

    if progress:
        progress.start()
        task_id = progress.add_task("Entrenando pipeline", total=9)
    else:
        task_id = None

    def step(description: str) -> float:
        logger.info("stage=%s", description)
        stage_start = time.perf_counter()
        if progress and task_id is not None:
            progress.update(task_id, description=description)
        return stage_start

    def step_done(stage_start: float) -> float:
        elapsed = time.perf_counter() - stage_start
        if progress and task_id is not None:
            progress.advance(task_id)
        return elapsed

    stage_start = step("descarga_h1")
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    connector = MT5Connector()
    try:
        ohlcv = connector.obtener_h1(args.symbol, start, end)
    finally:
        connector.shutdown()
    if ohlcv is None or ohlcv.empty:
        raise RuntimeError("OHLCV vacío: no hay datos para entrenar.")
    elapsed_download = step_done(stage_start)
    logger.info("download_rows=%s elapsed=%.2fs", len(ohlcv), elapsed_download)

    stage_start = step("build_dataset")
    feature_config = FeatureConfig()
    dataset_builder = DatasetBuilder(feature_config)
    artifacts = dataset_builder.build(ohlcv)
    elapsed_build = step_done(stage_start)
    logger.info("features_raw=%s labels_raw=%s elapsed=%.2fs", len(artifacts.features), len(artifacts.labels), elapsed_build)

    stage_start = step("align_and_clean")
    valid_mask = (
        artifacts.features.notna().all(axis=1)
        & artifacts.labels.notna()
        & artifacts.full_features.notna().all(axis=1)
    )
    n_before = len(artifacts.features)
    aligned_index = artifacts.features.index[valid_mask]
    features = artifacts.features.loc[aligned_index]
    labels = artifacts.labels.loc[aligned_index]
    full_features = artifacts.full_features.loc[aligned_index]
    n_after = len(features)
    dropped = n_before - n_after
    if n_after < args.min_samples:
        raise RuntimeError(
            f"Muestras insuficientes ({n_after}); se requieren al menos {args.min_samples}."
        )
    elapsed_align = step_done(stage_start)
    logger.info("aligned_samples=%s dropped_nan=%s elapsed=%.2fs", n_after, dropped, elapsed_align)
    logger.info("n_features=%s feature_names=%s", features.shape[1], list(features.columns))

    stage_start = step("split")
    split_idx = int(len(features) * args.split_ratio)
    if split_idx <= 0 or split_idx >= len(features):
        raise ValueError("Split ratio leaves no data for train/test.")
    features_train = features.iloc[:split_idx]
    labels_train = labels.iloc[:split_idx]
    features_test = features.iloc[split_idx:]
    labels_test = labels.iloc[split_idx:]
    elapsed_split = step_done(stage_start)
    logger.info(
        "n_train=%s n_test=%s split_ratio=%.2f elapsed=%.2fs",
        len(features_train),
        len(features_test),
        args.split_ratio,
        elapsed_split,
    )

    stage_start = step("train_model")
    model = StateEngineModel(StateEngineModelConfig())
    model.fit(features_train, labels_train)
    elapsed_train = step_done(stage_start)
    logger.info("train_elapsed=%.2fs", elapsed_train)

    stage_start = step("evaluate")
    preds_test = model.predict_state(features_test).to_numpy()
    labels_test_np = labels_test.to_numpy()
    label_order = [StateLabels.BALANCE, StateLabels.TRANSITION, StateLabels.TREND]
    matrix = confusion_matrix(labels_test_np, preds_test, label_order)
    accuracy = float(np.trace(matrix) / np.sum(matrix))
    f1 = f1_macro(matrix)
    elapsed_eval = step_done(stage_start)
    logger.info("accuracy=%.4f f1_macro=%.4f elapsed=%.2fs", accuracy, f1, elapsed_eval)

    stage_start = step("predict_outputs")
    outputs = model.predict_outputs(features)
    elapsed_outputs = step_done(stage_start)
    logger.info("outputs_rows=%s elapsed=%.2fs", len(outputs), elapsed_outputs)

    stage_start = step("gating")
    gating_policy = GatingPolicy()
    gating = gating_policy.apply(outputs, full_features)
    allow_any = gating.any(axis=1)
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
    }
    model.save(args.model_out, metadata=metadata)
    elapsed_save = step_done(stage_start)
    logger.info("model_path=%s elapsed=%.2fs", args.model_out, elapsed_save)

    if progress and task_id is not None:
        progress.stop()

    label_dist = class_distribution(labels.to_numpy(), label_order)
    baseline = max(label_dist, key=lambda row: row["count"]) if label_dist else None
    baseline_label = baseline["label"] if baseline else "N/A"
    baseline_pct = baseline["pct"] if baseline else 0.0
    logger.info("baseline_state=%s baseline_pct=%.2f", baseline_label, baseline_pct)

    gating_thresholds = gating_policy.thresholds
    trend_rule = (outputs["state_hat"] == StateLabels.TREND) & (
        outputs["margin"] >= gating_thresholds.trend_margin_min
    )
    balance_rule = (outputs["state_hat"] == StateLabels.BALANCE) & (
        outputs["margin"] >= gating_thresholds.balance_margin_min
    )
    transition_rule = (outputs["state_hat"] == StateLabels.TRANSITION) & (
        outputs["margin"] >= gating_thresholds.transition_margin_min
    )
    transition_break = full_features["BreakMag"] >= gating_thresholds.transition_breakmag_min
    transition_reentry = full_features["ReentryCount"] >= gating_thresholds.transition_reentry_min

    importances = model.feature_importances().head(15)

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
            "Métricas (Test)",
            ["Accuracy", "F1 Macro"],
            [[f"{accuracy:.4f}", f"{f1:.4f}"]],
        )
        render_table(
            console,
            table_class,
            "Matriz de confusión",
            ["Actual \\ Pred", *[label.name for label in label_order]],
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
            [[feature, f"{value:.2f}"] for feature, value in importances.items()],
        )
        render_table(
            console,
            table_class,
            "Resumen gating",
            ["Regla", "Count", "%"],
            [
                ["ALLOW_trend_pullback", int(trend_rule.sum()), f"{trend_rule.mean() * 100:.2f}%"],
                ["ALLOW_balance_fade", int(balance_rule.sum()), f"{balance_rule.mean() * 100:.2f}%"],
                ["ALLOW_transition_failure", int(gating['ALLOW_transition_failure'].sum()), f"{gating['ALLOW_transition_failure'].mean() * 100:.2f}%"],
                ["ALLOW_any", int(allow_any.sum()), f"{gating_allow_rate * 100:.2f}%"],
                ["BLOCK_any", int((~allow_any).sum()), f"{gating_block_rate * 100:.2f}%"],
            ],
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
    else:
        logger.info("class_distribution=%s", label_dist)
        logger.info("confusion_matrix=%s", matrix.tolist())
        logger.info("top_features=%s", importances.to_dict())

    report: dict[str, Any] = {
        "symbol": args.symbol,
        "start": args.start,
        "end": args.end,
        "n_rows_ohlcv": len(ohlcv),
        "n_samples": len(features),
        "n_features": features.shape[1],
        "n_train": len(features_train),
        "n_test": len(features_test),
        "nan_dropped": dropped,
        "baseline": {"label": baseline_label, "pct": baseline_pct},
        "metrics": {"accuracy": accuracy, "f1_macro": f1},
        "confusion_matrix": {
            "labels": [label.name for label in label_order],
            "matrix": matrix.tolist(),
        },
        "class_distribution": label_dist,
        "feature_importances": importances.to_dict(),
        "gating": {
            "thresholds": asdict(gating_thresholds),
            "allow_rate": gating_allow_rate,
            "block_rate": gating_block_rate,
            "counts": {
                "allow_trend_pullback": int(trend_rule.sum()),
                "allow_balance_fade": int(balance_rule.sum()),
                "allow_transition_failure": int(gating["ALLOW_transition_failure"].sum()),
                "allow_any": int(allow_any.sum()),
                "block_any": int((~allow_any).sum()),
            },
            "transition_details": {
                "margin_pass": int((transition_rule).sum()),
                "breakmag_pass": int((transition_rule & transition_break).sum()),
                "reentry_pass": int((transition_rule & transition_break & transition_reentry).sum()),
            },
        },
        "model_path": str(args.model_out),
        "metadata": metadata,
        "timings": {
            "download": elapsed_download,
            "build": elapsed_build,
            "align": elapsed_align,
            "split": elapsed_split,
            "train": elapsed_train,
            "evaluate": elapsed_eval,
            "outputs": elapsed_outputs,
            "gating": elapsed_gating,
            "save": elapsed_save,
            "total": time.perf_counter() - start_time,
        },
    }

    report_path = None
    if args.report_out:
        report_path = args.report_out
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("report_path=%s", report_path)

    summary_lines = [
        "=== State Engine Training Summary ===",
        f"Symbol: {args.symbol}",
        f"Period: {args.start} -> {args.end}",
        f"Samples: {len(features)} (train={len(features_train)}, test={len(features_test)})",
        f"Baseline: {baseline_label} ({baseline_pct:.2f}%)",
        f"Accuracy: {accuracy:.4f} | F1 Macro: {f1:.4f}",
        f"Gating allow rate: {gating_allow_rate * 100:.2f}% (block {gating_block_rate * 100:.2f}%)",
        f"Model saved: {args.model_out}",
    ]
    if report_path:
        summary_lines.append(f"Report saved: {report_path}")
    summary = "\n".join(summary_lines)

    if use_rich and console:
        console.rule("Resumen final")
        console.print(summary)
    else:
        logger.info(summary)


if __name__ == "__main__":
    main()
