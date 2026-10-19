from __future__ import annotations

import pandas as pd

from app import promotion


def _eval(n: int = 60, accuracy: float = 0.55, brier: float = 0.24, baseline: float = 0.53) -> dict:
    return {
        "n": n,
        "accuracy": accuracy,
        "brier": brier,
        "baseline_accuracy": baseline,
        "edge": accuracy - baseline,
        "from": "2026-01-01",
        "to": "2026-02-29",
    }


def test_promotion_window_ends_before_target_is_known() -> None:
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "target_up": [1] * 19 + [None],
            "close": 100.0,
        }
    )
    window = promotion.evaluation_window(frame, as_of="2026-01-20", eval_days=5)
    assert str(window["date"].min().date()) == "2026-01-15"
    assert str(window["date"].max().date()) == "2026-01-19"
    assert window["target_up"].notna().all()


def test_decision_promotes_when_no_champion() -> None:
    decision = promotion.decide_promotion(None, _eval())
    assert decision["promote"] is True
    assert decision["decision"] == "promote"


def test_decision_keeps_champion_with_insufficient_samples() -> None:
    decision = promotion.decide_promotion(_eval(), _eval(n=10))
    assert decision["promote"] is False
    assert decision["decision"] == "keep"


def test_decision_promotes_better_accuracy_with_tolerable_brier() -> None:
    decision = promotion.decide_promotion(_eval(accuracy=0.55, brier=0.24), _eval(accuracy=0.58, brier=0.25))
    assert decision["promote"] is True


def test_decision_rejects_material_brier_regression() -> None:
    decision = promotion.decide_promotion(_eval(accuracy=0.55, brier=0.24), _eval(accuracy=0.58, brier=0.30))
    assert decision["promote"] is False
    assert "Brier" in decision["reason"]


def test_decision_rejects_no_improvement() -> None:
    decision = promotion.decide_promotion(_eval(accuracy=0.56, brier=0.24), _eval(accuracy=0.55, brier=0.23))
    assert decision["promote"] is False


def test_weekly_gate_allows_small_sample() -> None:
    gate = promotion.weekly_signal_allowed({"n": 5, "accuracy": 0.2, "baseline_accuracy": 0.5})
    assert gate["allowed"] is True


def test_weekly_gate_blocks_material_underperformance() -> None:
    metrics = {
        "n": 52,
        "accuracy": 0.346,
        "baseline_accuracy": 0.519,
        "significance": {"vs_coinflip": {"p_value": 0.036}},
    }
    gate = promotion.weekly_signal_allowed(metrics)
    assert gate["allowed"] is False
    assert "低于基线" in gate["reason"]


def test_weekly_gate_allows_healthy_model() -> None:
    metrics = {
        "n": 52,
        "accuracy": 0.55,
        "baseline_accuracy": 0.52,
        "significance": {"vs_coinflip": {"p_value": 0.4}},
    }
    gate = promotion.weekly_signal_allowed(metrics)
    assert gate["allowed"] is True
