from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from app.model import PROB_COLUMNS

MIN_RESOLVED = 5


def resolved_frame(preds: pd.DataFrame) -> pd.DataFrame:
    if preds is None or preds.empty:
        return pd.DataFrame()
    df = preds.copy()
    for col in ("prob_up", "actual_return", "correct", "target_up", "predicted_return", "confidence"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "status" in df.columns:
        df = df[df["status"] == "resolved"]
    df = df[df["correct"].notna() & df["prob_up"].notna()]
    if "target_date" in df.columns:
        df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")
        df = df.dropna(subset=["target_date"]).sort_values("target_date")
    return df.reset_index(drop=True)


def _window_accuracy(df: pd.DataFrame, n: int) -> dict[str, Any] | None:
    recent = df.tail(n)
    if recent.empty:
        return None
    return {
        "accuracy": round(float(recent["correct"].mean()), 4),
        "n": int(len(recent)),
        "from": str(recent["target_date"].min().date()) if "target_date" in recent else None,
        "to": str(recent["target_date"].max().date()) if "target_date" in recent else None,
    }


def _strategy_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "actual_return" not in df.columns:
        return {}
    actual = df["actual_return"].astype(float).to_numpy()
    side = np.where(df["prob_up"].to_numpy() >= 0.5, 1.0, -1.0)
    strategy = side * actual
    cumulative = np.cumprod(1 + strategy)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative / peak - 1
    std = float(np.std(strategy, ddof=1)) if len(strategy) > 1 else 0.0
    return {
        "mean_daily_return": round(float(np.mean(strategy)), 6),
        "cumulative_return": round(float(cumulative[-1] - 1), 4),
        "max_drawdown": round(float(np.min(drawdown)), 4),
        "sharpe": round(float(np.mean(strategy) / std * np.sqrt(365)), 2) if std > 0 else None,
    }


def metrics_summary(preds: pd.DataFrame) -> dict[str, Any]:
    df = resolved_frame(preds)
    if df.empty:
        return {
            "n": 0,
            "accuracy": None,
            "brier": None,
            "log_loss": None,
            "auc": None,
            "balanced_accuracy": None,
            "baseline_accuracy": None,
            "edge": None,
            "windows": {},
            "confusion": {},
            "by_confidence": [],
            "strategy": {},
        }
    y = df.get("target_up")
    if y is None or y.isna().all():
        y = (df["actual_return"] > 0).astype(float)
    y = y.astype(float)
    prob = np.clip(df["prob_up"].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
    pred = (prob >= 0.5).astype(int)
    correct = (pred == y.to_numpy().astype(int))
    accuracy = float(np.mean(correct))
    base_up = float(np.mean(y))
    baseline = max(base_up, 1 - base_up)

    eps = 1e-12
    log_loss = float(-np.mean(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps)))
    brier = float(np.mean((prob - y.to_numpy()) ** 2))

    auc: float | None = None
    if len(np.unique(y)) > 1:
        try:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(y, prob))
        except Exception:  # noqa: BLE001
            auc = None

    tp = int(np.sum((pred == 1) & (y.to_numpy() == 1)))
    tn = int(np.sum((pred == 0) & (y.to_numpy() == 0)))
    fp = int(np.sum((pred == 1) & (y.to_numpy() == 0)))
    fn = int(np.sum((pred == 0) & (y.to_numpy() == 1)))
    sensitivity = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    balanced = float(np.mean([sensitivity, specificity])) if sensitivity is not None and specificity is not None else None

    buckets = [
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.65),
        (0.65, 0.70),
        (0.70, 1.01),
    ]
    by_conf: list[dict[str, Any]] = []
    for low, high in buckets:
        mask = (df["confidence"] >= low) & (df["confidence"] < high)
        sub = df[mask]
        if sub.empty:
            continue
        by_conf.append(
            {
                "bucket": f"{low:.2f}-{min(high, 1.0):.2f}",
                "n": int(len(sub)),
                "accuracy": round(float(sub["correct"].mean()), 4),
            }
        )

    windows = {
        "30": _window_accuracy(df, 30),
        "90": _window_accuracy(df, 90),
        "180": _window_accuracy(df, 180),
    }
    return {
        "n": int(len(df)),
        "accuracy": round(accuracy, 4),
        "brier": round(brier, 4),
        "log_loss": round(log_loss, 4),
        "auc": round(auc, 4) if auc is not None else None,
        "balanced_accuracy": round(balanced, 4) if balanced is not None else None,
        "baseline_accuracy": round(float(baseline), 4),
        "edge": round(accuracy - float(baseline), 4),
        "base_up_rate": round(base_up, 4),
        "windows": windows,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "by_confidence": by_conf,
        "strategy": _strategy_metrics(df),
        "from": str(df["target_date"].min().date()),
        "to": str(df["target_date"].max().date()),
    }


def accuracy_series(preds: pd.DataFrame, window: int = 30) -> list[dict[str, Any]]:
    df = resolved_frame(preds)
    if df.empty:
        return []
    rolled = df["correct"].rolling(window, min_periods=max(5, window // 3)).mean()
    out: list[dict[str, Any]] = []
    for idx, value in rolled.items():
        if pd.isna(value):
            continue
        row = df.loc[idx]
        out.append(
            {
                "date": str(row["target_date"].date()),
                "accuracy": round(float(value), 4),
                "cumulative": round(float(df.loc[:idx, "correct"].mean()), 4),
            }
        )
    return out


def optimize_weights(preds: pd.DataFrame, window: int = 90) -> dict[str, float]:
    df = resolved_frame(preds)
    if df.empty:
        return {name: 0.25 for name in PROB_COLUMNS}
    recent = df.tail(window)
    edges: dict[str, float] = {}
    for name, col in PROB_COLUMNS.items():
        if col not in recent.columns:
            edges[name] = 0.0
            continue
        prob = pd.to_numeric(recent[col], errors="coerce")
        mask = prob.notna() & recent["target_up"].notna()
        if mask.sum() < 20:
            edges[name] = 0.0
            continue
        acc = float(((prob[mask] >= 0.5).astype(int) == recent.loc[mask, "target_up"].astype(int)).mean())
        edges[name] = max(acc - 0.5, 0.0)
    if sum(edges.values()) <= 0:
        return {name: round(1 / len(PROB_COLUMNS), 4) for name in PROB_COLUMNS}
    raw = {name: 0.08 + edge for name, edge in edges.items()}
    total = sum(raw.values())
    return {name: round(value / total, 4) for name, value in raw.items()}


def choose_confidence_threshold(preds: pd.DataFrame, window: int = 180, min_coverage: float = 0.25) -> float:
    df = resolved_frame(preds)
    if len(df) < MIN_RESOLVED:
        return 0.55
    recent = df.tail(window)
    if len(recent) < MIN_RESOLVED:
        return 0.55
    best_threshold = 0.55
    best_accuracy = -1.0
    best_coverage = 0.0
    for threshold in np.arange(0.50, 0.801, 0.01):
        mask = recent["confidence"] >= threshold
        coverage = float(mask.mean())
        if coverage < min_coverage or mask.sum() < 5:
            continue
        accuracy = float(recent.loc[mask, "correct"].mean())
        if accuracy > best_accuracy + 1e-9 or (
            abs(accuracy - best_accuracy) <= 1e-9 and coverage > best_coverage
        ):
            best_threshold = float(threshold)
            best_accuracy = accuracy
            best_coverage = coverage
    return round(best_threshold, 2)


def failure_breakdown(preds: pd.DataFrame) -> list[dict[str, Any]]:
    df = resolved_frame(preds)
    if df.empty or "failure_categories_json" not in df.columns:
        return []
    failures = df[df["correct"] == 0]
    counts: dict[str, int] = {}
    for value in failures["failure_categories_json"].dropna():
        try:
            categories = json.loads(value)
        except Exception:  # noqa: BLE001
            continue
        for category in categories:
            counts[category] = counts.get(category, 0) + 1
    total = max(len(failures), 1)
    return [
        {"category": category, "count": count, "share": round(count / total, 3)}
        for category, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]
