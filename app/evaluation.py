from __future__ import annotations

import json
from math import exp, lgamma, log, log1p, sqrt
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from app.model import PROB_COLUMNS

MIN_RESOLVED = 5
EXACT_BINOMIAL_MAX_N = 5000


def _wilson_ci(successes: float, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    phat = successes / n
    denominator = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denominator
    margin = z * sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _log_binomial_pmf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 0.0 if k == 0 else float("-inf")
    if p >= 1:
        return 0.0 if k == n else float("-inf")
    return (
        lgamma(n + 1)
        - lgamma(k + 1)
        - lgamma(n - k + 1)
        + k * log(p)
        + (n - k) * log1p(-p)
    )


def _logsumexp(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return float("-inf")
    maximum = max(finite)
    return maximum + log(sum(exp(value - maximum) for value in finite))


def binomial_test(successes: int, n: int, p0: float = 0.5) -> dict[str, Any]:
    """Exact two-sided binomial test for a hit rate against a null probability.

    For very large samples the exact test falls back to a normal approximation
    so API requests stay fast.
    """
    successes = int(successes)
    n = int(n)
    p0 = min(max(float(p0), 1e-6), 1 - 1e-6)
    if n <= 0:
        return {
            "n": 0,
            "successes": 0,
            "observed": None,
            "p0": round(p0, 4),
            "p_value": None,
            "p_value_greater": None,
            "z": None,
            "method": "none",
        }
    observed = successes / n
    lower, upper = _wilson_ci(successes, n)
    if n <= EXACT_BINOMIAL_MAX_N:
        log_obs = _log_binomial_pmf(successes, n, p0)
        tail = []
        greater = []
        for i in range(n + 1):
            log_pmf = _log_binomial_pmf(i, n, p0)
            if log_pmf <= log_obs + 1e-9:
                tail.append(log_pmf)
            if i >= successes:
                greater.append(log_pmf)
        p_two = min(1.0, max(0.0, exp(_logsumexp(tail))))
        p_greater = min(1.0, max(0.0, exp(_logsumexp(greater))))
        method = "exact"
        z = None
    else:
        se = sqrt(p0 * (1 - p0) / n)
        z = (observed - p0) / se if se > 0 else 0.0
        normal = NormalDist()
        p_two = min(1.0, max(0.0, 2 * (1 - normal.cdf(abs(z)))))
        p_greater = min(1.0, max(0.0, 1 - normal.cdf(z)))
        method = "normal"
    standard_error = sqrt(observed * (1 - observed) / n) if 0 < observed < 1 else 0.0
    return {
        "n": n,
        "successes": successes,
        "observed": round(observed, 4),
        "p0": round(p0, 4),
        "p_value": round(p_two, 6),
        "p_value_greater": round(p_greater, 6),
        "z": round(float(z), 4) if z is not None else None,
        "se_pp": round(standard_error * 100, 2),
        "ci95": [round(lower, 4) if lower is not None else None, round(upper, 4) if upper is not None else None],
        "method": method,
    }


def calibration_curve(preds: pd.DataFrame, n_bins: int = 10) -> dict[str, Any]:
    df = resolved_frame(preds)
    if df.empty:
        return {
            "n": 0,
            "bins": [],
            "ece": None,
            "mce": None,
            "brier": None,
            "brier_skill_vs_half": None,
            "brier_skill_vs_base": None,
            "mean_prob": None,
            "observed_up_rate": None,
            "sharpness": None,
        }
    prob = np.clip(df["prob_up"].astype(float).to_numpy(), 0.0, 1.0)
    y = df["target_up"].astype(float).to_numpy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    for index in range(n_bins):
        low, high = float(edges[index]), float(edges[index + 1])
        if index == n_bins - 1:
            mask = (prob >= low) & (prob <= high)
        else:
            mask = (prob >= low) & (prob < high)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "low": round(low, 2),
                    "high": round(high, 2),
                    "n": 0,
                    "mean_prob": None,
                    "observed_up": None,
                    "ci95": [None, None],
                }
            )
            continue
        mean_prob = float(np.mean(prob[mask]))
        observed = float(np.mean(y[mask]))
        lower, upper = _wilson_ci(float(y[mask].sum()), count)
        gap = abs(observed - mean_prob)
        ece += count / len(prob) * gap
        mce = max(mce, gap)
        bins.append(
            {
                "low": round(low, 2),
                "high": round(high, 2),
                "n": count,
                "mean_prob": round(mean_prob, 4),
                "observed_up": round(observed, 4),
                "ci95": [
                    round(lower, 4) if lower is not None else None,
                    round(upper, 4) if upper is not None else None,
                ],
            }
        )
    brier = float(np.mean((prob - y) ** 2))
    ref_half = float(np.mean((0.5 - y) ** 2))
    base_up = float(np.mean(y))
    ref_base = float(np.mean((base_up - y) ** 2))
    return {
        "n": int(len(prob)),
        "bins": bins,
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "brier": round(brier, 4),
        "brier_skill_vs_half": round(1 - brier / ref_half, 4) if ref_half > 0 else None,
        "brier_skill_vs_base": round(1 - brier / ref_base, 4) if ref_base > 0 else None,
        "mean_prob": round(float(np.mean(prob)), 4),
        "observed_up_rate": round(base_up, 4),
        "sharpness": round(float(np.std(prob)), 4),
    }


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
            "significance": {},
            "calibration": calibration_curve(pd.DataFrame()),
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
    correct_count = int(correct.sum())
    versus_coinflip = binomial_test(correct_count, int(len(df)), 0.5)
    versus_baseline = binomial_test(correct_count, int(len(df)), float(baseline))
    significance = {
        "n": int(len(df)),
        "correct": correct_count,
        "se_pp": round(sqrt(accuracy * (1 - accuracy) / len(df)) * 100, 2),
        "ci95": versus_coinflip["ci95"],
        "vs_coinflip": versus_coinflip,
        "vs_baseline": versus_baseline,
        "significant_vs_coinflip": (versus_coinflip["p_value"] is not None and versus_coinflip["p_value"] < 0.05),
        "significant_vs_baseline": (versus_baseline["p_value"] is not None and versus_baseline["p_value"] < 0.05),
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
        "significance": significance,
        "calibration": calibration_curve(df),
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
