from __future__ import annotations

import numpy as np
import pandas as pd

from app.evaluation import (
    accuracy_series,
    binomial_test,
    calibration_curve,
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
    assert metrics["significance"]["n"] == 60
    assert metrics["significance"]["se_pp"] > 0
    assert metrics["calibration"]["n"] == 60
    assert len(metrics["calibration"]["bins"]) == 10


def test_empty_metrics() -> None:
    metrics = metrics_summary(pd.DataFrame())
    assert metrics["n"] == 0
    assert metrics["accuracy"] is None
    assert metrics["significance"] == {}
    assert metrics["calibration"]["n"] == 0


def test_binomial_test_flags_edges_and_non_edges() -> None:
    strong = binomial_test(330, 600, 0.5)
    assert abs(strong["observed"] - 0.55) < 1e-9
    assert strong["p_value"] < 0.05
    assert strong["ci95"][0] < strong["observed"] < strong["ci95"][1]
    weak = binomial_test(188, 366, 0.5)
    assert weak["p_value"] > 0.05
    assert 2.0 < weak["se_pp"] < 3.0
    empty = binomial_test(0, 0)
    assert empty["p_value"] is None


def test_calibration_curve_rewards_correct_probabilities() -> None:
    n = 1000
    prob = np.full(n, 0.9)
    up = np.array([1] * 900 + [0] * 100)
    preds = pd.DataFrame(
        {
            "status": "resolved",
            "target_date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "prob_up": prob,
            "actual_return": np.where(up == 1, 0.01, -0.01),
            "target_up": up,
            "correct": (prob >= 0.5).astype(int) == up,
        }
    )
    curve = calibration_curve(preds)
    assert curve["n"] == n
    assert curve["bins"][-1]["n"] == n
    assert abs(curve["bins"][-1]["observed_up"] - 0.9) < 1e-9
    assert curve["ece"] == 0
    assert abs(curve["brier"] - 0.09) < 1e-9
    assert abs(curve["brier_skill_vs_half"] - 0.64) < 1e-9
    assert curve["sharpness"] == 0


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
