from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app import config, evaluation, model

log = logging.getLogger(__name__)


def evaluation_window(
    feature_df: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
    eval_days: int = 60,
    horizon_days: int = 1,
) -> pd.DataFrame:
    """Rows whose target is already known, ending one horizon before ``as_of``."""
    as_of_ts = pd.Timestamp(as_of).normalize()
    end = as_of_ts - pd.Timedelta(days=horizon_days)
    start = end - pd.Timedelta(days=max(int(eval_days), 1) - 1)
    if feature_df.empty:
        return feature_df.copy()
    frame = feature_df[(feature_df["date"] >= start) & (feature_df["date"] <= end)].copy()
    target_col = "target_up" if horizon_days == 1 else f"target_up_{horizon_days}d"
    if target_col not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame[frame[target_col].notna()].sort_values("date").reset_index(drop=True)


def evaluate_bundle(
    bundle: model.ModelBundle,
    window: pd.DataFrame,
    *,
    target_col: str = "target_up",
) -> dict[str, Any]:
    if window.empty:
        return {"n": 0, "accuracy": None, "brier": None, "baseline_accuracy": None, "edge": None}
    rows: list[dict[str, Any]] = []
    for record in window.to_dict(orient="records"):
        row = pd.Series(record)
        prediction = model.predict_one(bundle, row)
        target_up = int(record[target_col])
        predicted_up = int(float(prediction["prob_up"]) >= 0.5)
        rows.append(
            {
                "status": "resolved",
                "target_date": record["date"],
                "prob_up": prediction["prob_up"],
                "confidence": prediction["confidence"],
                "actual_return": record.get("target_ret"),
                "predicted_return": prediction["predicted_return"],
                "target_up": target_up,
                "correct": int(predicted_up == target_up),
            }
        )
    metrics = evaluation.metrics_summary(pd.DataFrame(rows))
    return {
        "n": metrics["n"],
        "accuracy": metrics["accuracy"],
        "brier": metrics["brier"],
        "baseline_accuracy": metrics["baseline_accuracy"],
        "edge": metrics["edge"],
        "from": str(window["date"].min().date()),
        "to": str(window["date"].max().date()),
    }


def decide_promotion(
    champion: dict[str, Any] | None,
    challenger: dict[str, Any] | None,
    *,
    min_n: int | None = None,
    acc_tolerance: float | None = None,
    brier_tolerance: float | None = None,
) -> dict[str, Any]:
    min_n = config.PROMOTION_MIN_N if min_n is None else int(min_n)
    acc_tolerance = config.PROMOTION_ACC_TOLERANCE if acc_tolerance is None else float(acc_tolerance)
    brier_tolerance = config.PROMOTION_BRIER_TOLERANCE if brier_tolerance is None else float(brier_tolerance)
    champ_n = int((champion or {}).get("n") or 0)
    cand_n = int((challenger or {}).get("n") or 0)
    if not challenger or cand_n < min_n:
        return {"promote": False, "decision": "keep", "reason": f"挑战者评估样本不足（{cand_n} < {min_n}）"}
    if not champion or champ_n < min_n:
        return {"promote": True, "decision": "promote", "reason": "没有可比较的在任模型，采用新模型"}

    champ_acc = float(champion["accuracy"])
    cand_acc = float(challenger["accuracy"])
    champ_brier = float(champion["brier"])
    cand_brier = float(challenger["brier"])
    delta_acc = cand_acc - champ_acc
    delta_brier = cand_brier - champ_brier
    quality_ok = cand_acc >= float(challenger["baseline_accuracy"]) - 0.02
    brier_ok = delta_brier <= brier_tolerance
    accuracy_ok = delta_acc > acc_tolerance or (
        abs(delta_acc) <= acc_tolerance and delta_brier < -1e-4
    )
    promote = bool(quality_ok and brier_ok and accuracy_ok)
    if promote:
        reason = "挑战者准确率不劣于在任模型，且 Brier 在容忍范围内"
    elif not quality_ok:
        reason = "挑战者准确率低于其基线 2pp 以上，拒绝上线"
    elif not brier_ok:
        reason = f"挑战者 Brier 恶化超过 {brier_tolerance:.3f}，拒绝上线"
    else:
        reason = "挑战者没有跑赢在任模型，保留在任模型"
    return {
        "promote": promote,
        "decision": "promote" if promote else "keep",
        "reason": reason,
        "delta_accuracy": round(delta_acc, 4),
        "delta_brier": round(delta_brier, 4),
        "quality_ok": quality_ok,
        "brier_ok": brier_ok,
        "accuracy_ok": accuracy_ok,
    }


def review_candidate(
    champion: model.ModelBundle | None,
    challenger: model.ModelBundle,
    feature_df: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
    eval_days: int | None = None,
    horizon_days: int = 1,
) -> dict[str, Any]:
    eval_days = config.PROMOTION_EVAL_DAYS if eval_days is None else int(eval_days)
    window = evaluation_window(
        feature_df,
        as_of=as_of,
        eval_days=eval_days,
        horizon_days=horizon_days,
    )
    target_col = "target_up" if horizon_days == 1 else f"target_up_{horizon_days}d"
    champion_eval = evaluate_bundle(champion, window, target_col=target_col) if champion is not None else None
    challenger_eval = evaluate_bundle(challenger, window, target_col=target_col)
    decision = decide_promotion(champion_eval, challenger_eval)
    review = {
        "eval_days": eval_days,
        "window_from": str(window["date"].min().date()) if not window.empty else None,
        "window_to": str(window["date"].max().date()) if not window.empty else None,
        "champion": champion_eval,
        "challenger": challenger_eval,
        **decision,
    }
    log.info(
        "Promotion review: champion_acc=%s challenger_acc=%s decision=%s (%s)",
        None if champion_eval is None else champion_eval.get("accuracy"),
        challenger_eval.get("accuracy"),
        decision["decision"],
        decision["reason"],
    )
    return review


def weekly_signal_allowed(
    metrics: dict[str, Any] | None,
    *,
    min_n: int | None = None,
    max_underperformance: float | None = None,
) -> dict[str, Any]:
    min_n = config.WEEKLY_MIN_EVAL_N if min_n is None else int(min_n)
    max_underperformance = (
        config.WEEKLY_MAX_UNDERPERFORMANCE if max_underperformance is None else float(max_underperformance)
    )
    n = int((metrics or {}).get("n") or 0)
    if n < min_n:
        return {
            "allowed": True,
            "reason": f"回测样本不足（{n} < {min_n}），继续输出观察",
            "n": n,
            "accuracy": (metrics or {}).get("accuracy"),
            "baseline_accuracy": (metrics or {}).get("baseline_accuracy"),
            "p_value": None,
        }
    accuracy = float(metrics["accuracy"])
    baseline = float(metrics["baseline_accuracy"])
    significance = metrics.get("significance") or {}
    p_value = (significance.get("vs_coinflip") or {}).get("p_value")
    underperform = accuracy < baseline - max_underperformance
    significantly_bad = p_value is not None and float(p_value) < 0.05 and accuracy < 0.5
    allowed = not (underperform or significantly_bad)
    if allowed:
        reason = "周回测未触发失效阈值，继续输出"
    elif underperform:
        reason = f"周准确率 {accuracy:.1%} 低于基线 {baseline:.1%} 超过 {max_underperformance:.1%}"
    else:
        reason = f"周准确率 {accuracy:.1%} 显著低于随机（p={float(p_value):.3f}）"
    return {
        "allowed": bool(allowed),
        "reason": reason,
        "n": n,
        "accuracy": round(accuracy, 4),
        "baseline_accuracy": round(baseline, 4),
        "p_value": round(float(p_value), 6) if p_value is not None else None,
    }
