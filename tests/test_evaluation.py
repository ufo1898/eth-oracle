from __future__ import annotations

import numpy as np
import pandas as pd

from app.evaluation import (
    accuracy_series,
    choose_confidence_threshold,
    failure_breakdown,
    metrics_summary,
    optimize_weights,
)


def _predictions(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    prob = np.clip(rng.normal(0.5, 0.08, n), 0.01, 0.99)
    actual = rng.normal(0.0005, 0.02, n)
    target_up = (actual > 0).astype(int)
    correct = ((prob >= 0.5).astype(int) == target_up).astype(int)
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "mode": "backtest",
            "status": "resolved",
            "target_date": dates,
            "prob_up": prob,
            "confidence": np.maximum(prob, 1 - prob),
            "actual_return": actual,
            "predicted_return": prob - 0.5,
            "target_up": target_up,
            "correct": correct,
            "p_hgb": np.clip(prob + rng.normal(0, 0.03, n), 0.01, 0.99),
            "p_rf": np.clip(prob + rng.normal(0, 0.03, n), 0.01, 0.99),
            "p_et": np.clip(prob + rng.normal(0, 0.03, n), 0.01, 0.99),
            "p_lr": np.clip(prob + rng.normal(0, 0.03, n), 0.01, 0.99),
            "failure_categories_json": [
                '["低置信度", "高波动"]' if value == 0 else None for value in correct
            ],
        }
    )


def test_metrics_summary() -> None:
    metrics = metrics_summary(_predictions())
    assert metrics["n"] == 60
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["brier"] is not None
    assert metrics["confusion"]["tp"] + metrics["confusion"]["tn"] + metrics["confusion"]["fp"] + metrics["confusion"]["fn"] == 60
    assert metrics["windows"]["30"]["n"] == 30
    assert metrics["strategy"]["cumulative_return"] is not None


def test_empty_metrics() -> None:
    metrics = metrics_summary(pd.DataFrame())
    assert metrics["n"] == 0
    assert metrics["accuracy"] is None


def test_failure_breakdown_counts() -> None:
    breakdown = failure_breakdown(_predictions())
    assert breakdown
    total = sum(item["count"] for item in breakdown)
    assert total >= len(breakdown)
    assert all(0 <= item["share"] <= 1 for item in breakdown)


def test_accuracy_series_and_threshold() -> None:
    series = accuracy_series(_predictions())
    assert series
    assert all(0 <= item["accuracy"] <= 1 for item in series)
    threshold = choose_confidence_threshold(_predictions())
    assert 0.5 <= threshold <= 0.8


def test_optimize_weights_sums_to_one() -> None:
    weights = optimize_weights(_predictions(), window=90)
    assert set(weights) == {"hgb", "rf", "et", "lr"}
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(value >= 0 for value in weights.values())
