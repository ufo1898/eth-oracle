from __future__ import annotations

from app.analysis import build_daily_report, classify_failure
from app.daily import _hydrate_prediction


def test_classify_failure_flags_expected_categories() -> None:
    prediction = {"prob_up": 0.52, "direction": "UP"}
    features = {
        "atr_14_pct": 0.08,
        "vol_30d": 0.07,
        "fng_level": 85,
        "above_sma_50": 0.0,
        "btc_ret_1d": 0.04,
        "spx_ret_1d": 0.02,
        "vix_change_1d": 3.0,
    }
    categories, reason = classify_failure(prediction, features, news_score=0.4)
    assert "低置信度" in categories
    assert "高波动" in categories
    assert "情绪极端" in categories
    assert "逆势看多" in categories
    assert "BTC联动" in categories
    assert "宏观冲击" in categories
    assert "消息面冲击" in categories
    assert "模型预测上涨，实际下跌" in reason


def test_classify_failure_has_fallback() -> None:
    categories, _ = classify_failure({"prob_up": 0.6}, {})
    assert categories == ["模型特征失效"]


def test_hydrate_prediction_parses_json_fields() -> None:
    row = {
        "features_json": '{"rsi_14": 50}',
        "drivers_json": "[]",
        "failure_categories_json": '["低置信度"]',
    }
    hydrated = _hydrate_prediction(row)
    assert hydrated["features"] == {"rsi_14": 50}
    assert hydrated["drivers"] == []
    assert hydrated["failure_categories"] == ["低置信度"]


def test_build_daily_report_contains_sections() -> None:
    report = build_daily_report(
        report_date="2026-10-01",
        latest_prediction={
            "as_of_date": "2026-09-30",
            "target_date": "2026-10-01",
            "direction": "UP",
            "prob_up": 0.56,
            "confidence": 0.56,
            "predicted_close": 2500.0,
            "predicted_return": 0.01,
            "drivers": [{"feature": "rsi_14", "direction": "up"}],
        },
        metrics={
            "n": 10,
            "accuracy": 0.5,
            "baseline_accuracy": 0.5,
            "edge": 0.0,
            "windows": {"30": {"accuracy": 0.5, "n": 10, "from": "2026-09-01"}},
            "brier": 0.25,
            "auc": 0.5,
            "strategy": {"cumulative_return": 0.0},
        },
        failure_breakdown=[{"category": "低置信度", "count": 3, "share": 0.6}],
        resolved_today=[],
        model_info={
            "version": "v-test",
            "train_start": "2020-01-01",
            "train_end": "2026-01-01",
            "n_samples": 100,
            "n_features": 10,
            "weights": {"hgb": 1.0},
            "val_metrics": {"accuracy": 0.55},
        },
        news_summary={"score": 0.1, "count": 3, "positive": 1, "negative": 1},
        news_items=[{"source": "test", "title": "t", "sentiment": 0.1}],
        refresh_status={"last_date": "2026-09-30", "rows": 100, "refreshed_at": "now", "sources": []},
        optimization={"weights": {"hgb": 1.0}, "threshold": 0.55},
    )
    assert "# ETH Oracle 日报" in report
    assert "## 下一日预测" in report
    assert "## 胜率统计" in report
    assert "失败原因分布" in report
    assert "同一笔错误可命中多个标签" in report
    assert "ETH Oracle 是统计研究工具" in report
