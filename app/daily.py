from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app import analysis, config, db, model, promotion
from app.data import build, sentiment
from app.evaluation import (
    choose_confidence_threshold,
    failure_breakdown,
    metrics_summary,
    optimize_weights,
    resolved_frame,
)
from app.features import build_features, feature_columns

log = logging.getLogger(__name__)

KEY_ANALYSIS_FEATURES = [
    "atr_14_pct",
    "vol_30d",
    "vol_90d",
    "spx_ret_1d",
    "nasdaq_ret_1d",
    "vix_change_1d",
    "vix_level",
    "hy_spread_change_1d",
    "dxy_ret_1d",
    "us10y_change_1d",
    "fng_level",
    "funding_level",
    "above_sma_20",
    "above_sma_50",
    "above_sma_200",
    "btc_ret_1d",
    "btc_ret_5d",
    "ethbtc_ret_1d",
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "rsi_14",
    "macd_hist_norm",
    "volume_z_20",
]


def _feature_row(feature_df: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    target = pd.Timestamp(date).normalize()
    exact = feature_df[feature_df["date"] == target]
    if not exact.empty:
        return exact.iloc[-1]
    prior = feature_df[feature_df["date"] <= target]
    if prior.empty:
        return None
    return prior.iloc[-1]


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return round(number, 6)
    except (TypeError, ValueError):
        return None


def _feature_snapshot(row: pd.Series, bundle: model.ModelBundle) -> dict[str, Any]:
    keys = list(dict.fromkeys(list(bundle.importance.head(40).index) + KEY_ANALYSIS_FEATURES))
    snapshot: dict[str, Any] = {}
    for key in keys:
        if key in row.index:
            snapshot[key] = _number_or_none(row.get(key))
    return snapshot


def _hydrate_prediction(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    for key in ("features_json", "drivers_json", "failure_categories_json"):
        value = data.get(key)
        derived = key.replace("_json", "")
        if isinstance(value, str) and value:
            try:
                data[derived] = json.loads(value)
            except Exception:  # noqa: BLE001
                data[derived] = None
        elif value is None and derived not in data:
            data[derived] = None
    return data


def _build_record(
    bundle: model.ModelBundle,
    row: pd.Series,
    *,
    mode: str,
    target_date: pd.Timestamp,
    news_score: float | None,
    horizon_days: int = 1,
) -> dict[str, Any]:
    prediction = model.predict_one(bundle, row)
    drivers = model.top_drivers(bundle, row, k=6)
    return {
        "mode": mode,
        "as_of_date": str(pd.Timestamp(row["date"]).date()),
        "target_date": str(pd.Timestamp(target_date).date()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "close": float(row["close"]),
        "predicted_close": prediction["predicted_close"],
        "predicted_return": prediction["predicted_return"],
        "prob_up": prediction["prob_up"],
        "prob_raw": prediction["prob_raw"],
        "direction": prediction["direction"],
        "confidence": prediction["confidence"],
        "p_hgb": prediction.get("p_hgb"),
        "p_rf": prediction.get("p_rf"),
        "p_et": prediction.get("p_et"),
        "p_lr": prediction.get("p_lr"),
        "status": "pending",
        "horizon_days": int(horizon_days),
        "model_version": bundle.version,
        "features": _feature_snapshot(row, bundle),
        "drivers": drivers,
        "news_score": news_score,
    }


def resolve_pending(feature_df: pd.DataFrame) -> list[dict[str, Any]]:
    pending = db.get_pending_predictions()
    if not pending:
        return []
    price_map = (
        feature_df.set_index("date")["close"].astype(float).to_dict()
        if not feature_df.empty
        else {}
    )
    resolved: list[dict[str, Any]] = []
    for prediction in pending:
        target_date = pd.Timestamp(prediction["target_date"]).normalize()
        actual_close = price_map.get(target_date)
        if actual_close is None:
            continue
        entry_close = float(prediction.get("close") or 0)
        if entry_close <= 0:
            continue
        actual_return = float(actual_close) / entry_close - 1
        target_up = int(actual_return > 0)
        predicted_up = float(prediction.get("prob_up") or 0.5) >= 0.5
        correct = int((1 if predicted_up else 0) == target_up)
        features = _hydrate_prediction(prediction).get("features") or {}
        if correct:
            categories: list[str] = []
            reason = f"预测正确：模型判断{'上涨' if predicted_up else '下跌'}，实际 {actual_return:+.2%}。"
        else:
            categories, reason = analysis.classify_failure(
                {**prediction, "actual_return": actual_return},
                features,
                news_score=prediction.get("news_score"),
            )
        db.mark_resolved(
            int(prediction["id"]),
            actual_close=float(actual_close),
            actual_return=actual_return,
            target_up=target_up,
            correct=correct,
            failure_categories=categories,
            failure_reason=reason,
        )
        prediction.update(
            {
                "actual_close": float(actual_close),
                "actual_return": actual_return,
                "target_up": target_up,
                "correct": correct,
                "status": "resolved",
                "failure_categories": categories,
                "failure_reason": reason,
                "features": features,
            }
        )
        resolved.append(prediction)
        log.info(
            "Resolved live prediction %s (h%sd): actual %.2f%% correct=%s",
            prediction["as_of_date"],
            prediction.get("horizon_days") or 1,
            actual_return * 100,
            correct,
        )
    return resolved


def _mark_backtest_resolved(
    record: dict[str, Any],
    *,
    actual_close: float,
    actual_return: float,
    target_up: int,
    correct: int,
    news_score: float | None,
) -> dict[str, Any]:
    if correct:
        categories: list[str] = []
        reason = f"预测正确：实际 {actual_return:+.2%}。"
    else:
        categories, reason = analysis.classify_failure(
            {**record, "actual_return": actual_return},
            record.get("features") or {},
            news_score=news_score,
        )
    record.update(
        {
            "actual_close": float(actual_close),
            "actual_return": float(actual_return),
            "target_up": int(target_up),
            "correct": int(correct),
            "status": "resolved",
            "failure_categories": categories,
            "failure_reason": reason,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return record


def generate_backtest(
    feature_df: pd.DataFrame,
    *,
    backtest_days: int,
    rolling_weights: dict[str, float] | None,
    candidates: list[str] | None = None,
) -> int:
    if feature_df.empty:
        return 0
    end_target = pd.Timestamp(feature_df["date"].max()).normalize()
    start_target = max(
        pd.Timestamp(feature_df["date"].min()).normalize() + pd.Timedelta(days=config.MIN_TRAIN_ROWS),
        end_target - pd.Timedelta(days=backtest_days),
    )
    existing = db.get_predictions(mode="backtest", limit=100000)
    if not existing.empty:
        last = pd.to_datetime(existing["target_date"], errors="coerce").max()
        if pd.notna(last):
            start_target = max(start_target, pd.Timestamp(last).normalize() + pd.Timedelta(days=1))
    if start_target > end_target:
        return 0

    candidates = candidates or feature_columns(feature_df)
    available_dates = set(feature_df["date"].tolist())
    bundle: model.ModelBundle | None = None
    last_retrain_target: pd.Timestamp | None = None
    count = 0
    for target in pd.date_range(start_target, end_target, freq="D"):
        if target not in available_dates:
            continue
        as_of = target - pd.Timedelta(days=1)
        row = _feature_row(feature_df, as_of)
        target_rows = feature_df[feature_df["date"] == target]
        if row is None or target_rows.empty:
            continue
        if (
            bundle is None
            or last_retrain_target is None
            or (target - last_retrain_target).days >= config.RETRAIN_DAYS
        ):
            try:
                bundle = model.train_bundle(
                    feature_df,
                    as_of=as_of,
                    candidates=candidates,
                    weight_override=rolling_weights,
                )
                last_retrain_target = target
            except Exception as exc:  # noqa: BLE001
                log.warning("Backtest training failed at %s: %s", as_of.date(), exc)
                continue
        record = _build_record(
            bundle,
            row,
            mode="backtest",
            target_date=target,
            news_score=None,
            horizon_days=1,
        )
        actual_close = float(target_rows.iloc[-1]["close"])
        actual_return = actual_close / float(row["close"]) - 1
        target_up = int(actual_return > 0)
        predicted_up = float(record["prob_up"]) >= 0.5
        correct = int((1 if predicted_up else 0) == target_up)
        record = _mark_backtest_resolved(
            record,
            actual_close=actual_close,
            actual_return=actual_return,
            target_up=target_up,
            correct=correct,
            news_score=None,
        )
        db.upsert_prediction(record)
        count += 1
        if count % 30 == 0:
            log.info("Backtest progress: %s predictions through %s", count, target.date())
    log.info("Backtest generated %s predictions", count)
    return count


def generate_weekly_backtest(
    feature_df: pd.DataFrame,
    *,
    backtest_days: int,
    rolling_weights: dict[str, float] | None,
    candidates: list[str] | None = None,
    horizon_days: int = 7,
) -> int:
    """Walk-forward backtest for the multi-day (weekly) horizon.

    Target dates sit on a fixed lattice anchored at the first dataset date so
    incremental runs stay idempotent. The model is retrained every
    ``RETRAIN_DAYS`` with information strictly available ``horizon_days``
    before the prediction day.
    """
    if feature_df.empty:
        return 0
    horizon = max(int(horizon_days), 1)
    target_col = f"target_up_{horizon}d"
    target_ret_col = f"target_ret_{horizon}d"
    if target_col not in feature_df.columns or target_ret_col not in feature_df.columns:
        log.warning("Weekly targets missing from feature frame")
        return 0

    end_target = pd.Timestamp(feature_df["date"].max()).normalize()
    start_target = max(
        pd.Timestamp(feature_df["date"].min()).normalize() + pd.Timedelta(days=config.MIN_TRAIN_ROWS + horizon),
        end_target - pd.Timedelta(days=backtest_days),
    )
    existing = db.get_predictions(mode="backtest", limit=100000, horizon_days=horizon)
    existing_targets: set[pd.Timestamp] = set()
    if not existing.empty:
        parsed = pd.to_datetime(existing["target_date"], errors="coerce").dropna()
        existing_targets = {pd.Timestamp(value).normalize() for value in parsed}

    candidates = candidates or feature_columns(feature_df)
    available = set(feature_df["date"].tolist())
    anchor = pd.Timestamp(feature_df["date"].min()).normalize()
    bundle: model.ModelBundle | None = None
    last_retrain_target: pd.Timestamp | None = None
    count = 0
    for target in pd.date_range(start_target, end_target, freq="D"):
        target = pd.Timestamp(target).normalize()
        if target not in available or (target - anchor).days % horizon != 0:
            continue
        if target in existing_targets:
            continue
        as_of = target - pd.Timedelta(days=horizon)
        if as_of not in available:
            continue
        row = _feature_row(feature_df, as_of)
        target_rows = feature_df[feature_df["date"] == target]
        if row is None or target_rows.empty:
            continue
        if (
            bundle is None
            or last_retrain_target is None
            or (target - last_retrain_target).days >= config.RETRAIN_DAYS
        ):
            try:
                bundle = model.train_bundle(
                    feature_df,
                    as_of=as_of,
                    candidates=candidates,
                    weight_override=rolling_weights,
                    target_col=target_col,
                    target_ret_col=target_ret_col,
                    horizon_days=horizon,
                )
                last_retrain_target = target
            except Exception as exc:  # noqa: BLE001
                log.warning("Weekly backtest training failed at %s: %s", as_of.date(), exc)
                continue
        record = _build_record(
            bundle,
            row,
            mode="backtest",
            target_date=target,
            news_score=None,
            horizon_days=horizon,
        )
        actual_close = float(target_rows.iloc[-1]["close"])
        actual_return = actual_close / float(row["close"]) - 1
        target_up = int(actual_return > 0)
        predicted_up = float(record["prob_up"]) >= 0.5
        correct = int((1 if predicted_up else 0) == target_up)
        record = _mark_backtest_resolved(
            record,
            actual_close=actual_close,
            actual_return=actual_return,
            target_up=target_up,
            correct=correct,
            news_score=None,
        )
        db.upsert_prediction(record)
        count += 1
    log.info("Weekly backtest generated %s predictions (horizon=%sd)", count, horizon)
    return count


def run_daily(*, backtest_days: int | None = None, fetch: bool = True) -> dict[str, Any]:
    config.ensure_dirs()
    db.init_db()
    backtest_days = backtest_days or config.BACKTEST_DAYS_DEFAULT

    log.info("Fetching market and context data...")
    if fetch:
        raw_df = build.build_dataset()
    else:
        raw_df = build.load_dataset()
    feature_df = build_features(raw_df)
    if feature_df.empty:
        raise RuntimeError("Feature dataframe is empty")

    log.info("Fetching news and sentiment...")
    news_items = sentiment.fetch_news()
    db.store_news(news_items)
    news_stats = sentiment.news_summary(news_items)

    resolved = resolve_pending(feature_df)
    resolved_daily = [item for item in resolved if int(item.get("horizon_days") or 1) == 1]
    resolved_weekly = [item for item in resolved if int(item.get("horizon_days") or 1) == config.WEEKLY_HORIZON_DAYS]
    all_preds_before = db.get_predictions(limit=100000)
    rolling_weights = None
    if len(resolved_frame(all_preds_before)) >= 60:
        rolling_weights = optimize_weights(all_preds_before, window=90)
    threshold = choose_confidence_threshold(all_preds_before)

    weekly_preds_before = db.get_predictions(limit=100000, horizon_days=config.WEEKLY_HORIZON_DAYS)
    weekly_weights = None
    if len(resolved_frame(weekly_preds_before)) >= 40:
        weekly_weights = optimize_weights(weekly_preds_before, window=52)
    weekly_threshold = choose_confidence_threshold(weekly_preds_before, window=52)

    as_of = pd.Timestamp(feature_df["date"].max()).normalize()
    target_date = as_of + pd.Timedelta(days=1)
    row = _feature_row(feature_df, as_of)
    if row is None:
        raise RuntimeError(f"No feature row for as_of {as_of.date()}")
    candidates = feature_columns(feature_df)

    log.info("Training live candidate as of %s...", as_of.date())
    champion = model.load_bundle()
    candidate = model.train_bundle(
        feature_df,
        as_of=as_of,
        candidates=candidates,
        weight_override=rolling_weights,
    )
    deployed = candidate
    promotion_review: dict[str, Any] = {
        "decision": "promote",
        "promote": True,
        "reason": "没有在任模型，直接上线",
        "champion": None,
        "challenger": None,
    }
    if champion is not None:
        try:
            gate_candidate = model.train_bundle(
                feature_df,
                as_of=as_of - pd.Timedelta(days=config.PROMOTION_EVAL_DAYS),
                candidates=candidates,
                weight_override=rolling_weights,
            )
            promotion_review = promotion.review_candidate(
                champion,
                gate_candidate,
                feature_df,
                as_of=as_of,
                eval_days=config.PROMOTION_EVAL_DAYS,
            )
            if not promotion_review["promote"]:
                deployed = champion
                model.save_bundle(candidate, config.MODEL_CHALLENGER)
                log.warning(
                    "Challenger %s rejected; keeping champion %s: %s",
                    candidate.version,
                    champion.version,
                    promotion_review["reason"],
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("Promotion gate failed, keeping champion: %s", exc)
            promotion_review = {
                "decision": "keep",
                "promote": False,
                "reason": f"门禁评估失败，保留在任模型：{exc}",
                "champion": None,
                "challenger": None,
            }
            deployed = champion
    if deployed is candidate:
        model.save_bundle(candidate)
    promotion_review = {
        **promotion_review,
        "champion_version": champion.version if champion is not None else None,
        "challenger_version": candidate.version,
        "deployed_version": deployed.version,
    }
    bundle = deployed
    bundle_info = bundle.info()
    if deployed is candidate:
        db.store_model_run(
            bundle_info,
            bundle.feature_cols,
            bundle.importance.head(60).round(6).to_dict(),
        )

    live_record = _build_record(
        bundle,
        row,
        mode="live",
        target_date=target_date,
        news_score=news_stats["score"],
        horizon_days=1,
    )
    db.upsert_prediction(live_record)
    log.info(
        "Live prediction: %s -> %s direction=%s prob=%.3f predicted_close=%.2f",
        live_record["as_of_date"],
        live_record["target_date"],
        live_record["direction"],
        live_record["prob_up"],
        live_record["predicted_close"],
    )

    weekly_horizon = config.WEEKLY_HORIZON_DAYS
    weekly_bundle: model.ModelBundle | None = None
    weekly_record: dict[str, Any] | None = None
    weekly_gate_before = promotion.weekly_signal_allowed(metrics_summary(weekly_preds_before))
    if not weekly_gate_before["allowed"]:
        log.warning("Weekly live signal disabled before training: %s", weekly_gate_before["reason"])
        db.clear_predictions(mode="live", horizon_days=weekly_horizon, pending_only=True)
    try:
        if not weekly_gate_before["allowed"]:
            raise RuntimeError(weekly_gate_before["reason"])
        log.info("Training weekly model (horizon=%sd) as of %s...", weekly_horizon, as_of.date())
        weekly_bundle = model.train_bundle(
            feature_df,
            as_of=as_of,
            candidates=candidates,
            weight_override=weekly_weights,
            target_col=f"target_up_{weekly_horizon}d",
            target_ret_col=f"target_ret_{weekly_horizon}d",
            horizon_days=weekly_horizon,
        )
        model.save_bundle(weekly_bundle, config.MODEL_WEEKLY_LATEST)
        weekly_target_date = as_of + pd.Timedelta(days=weekly_horizon)
        weekly_record = _build_record(
            weekly_bundle,
            row,
            mode="live",
            target_date=weekly_target_date,
            news_score=news_stats["score"],
            horizon_days=weekly_horizon,
        )
        db.upsert_prediction(weekly_record)
        log.info(
            "Weekly live prediction: %s -> %s direction=%s prob=%.3f",
            weekly_record["as_of_date"],
            weekly_record["target_date"],
            weekly_record["direction"],
            weekly_record["prob_up"],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Weekly live model skipped: %s", exc)

    new_backtest = generate_backtest(
        feature_df,
        backtest_days=backtest_days,
        rolling_weights=rolling_weights,
        candidates=candidates,
    )
    try:
        new_weekly_backtest = generate_weekly_backtest(
            feature_df,
            backtest_days=config.WEEKLY_BACKTEST_DAYS,
            rolling_weights=weekly_weights,
            candidates=candidates,
            horizon_days=weekly_horizon,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Weekly backtest skipped: %s", exc)
        new_weekly_backtest = 0

    all_preds = db.get_predictions(limit=100000)
    weekly_preds = db.get_predictions(limit=100000, horizon_days=weekly_horizon)
    metrics = metrics_summary(all_preds)
    live_preds = all_preds[all_preds["mode"] == "live"] if not all_preds.empty else all_preds
    backtest_preds = all_preds[all_preds["mode"] == "backtest"] if not all_preds.empty else all_preds
    live_metrics = metrics_summary(live_preds)
    backtest_metrics = metrics_summary(backtest_preds)
    weekly_metrics = metrics_summary(weekly_preds)
    weekly_live_preds = weekly_preds[weekly_preds["mode"] == "live"] if not weekly_preds.empty else weekly_preds
    weekly_backtest_preds = weekly_preds[weekly_preds["mode"] == "backtest"] if not weekly_preds.empty else weekly_preds
    weekly_live_metrics = metrics_summary(weekly_live_preds)
    weekly_backtest_metrics = metrics_summary(weekly_backtest_preds)
    weekly_gate = promotion.weekly_signal_allowed(weekly_metrics)
    if not weekly_gate["allowed"] and weekly_record is not None:
        db.clear_predictions(mode="live", horizon_days=weekly_horizon, pending_only=True)
        weekly_record = None
        log.warning("Weekly live signal disabled after review: %s", weekly_gate["reason"])
    failures = failure_breakdown(all_preds)
    latest = _hydrate_prediction(db.get_latest_prediction("live"))
    latest_weekly = None
    if weekly_gate["allowed"]:
        latest_weekly = _hydrate_prediction(db.get_latest_prediction("live", horizon_days=weekly_horizon))
    model_info = {**bundle_info, "importance": bundle.importance.head(25).round(6).to_dict()}
    refresh_status = build.load_refresh_status()
    optimization = {
        "weights": {key: round(value, 4) for key, value in bundle.weights.items()},
        "threshold": threshold,
        "retrained": deployed is candidate,
        "new_backtest_predictions": new_backtest,
        "resolved_live_predictions": len(resolved_daily),
        "promotion": promotion_review,
        "weekly": {
            "weights": {
                key: round(value, 4)
                for key, value in (weekly_weights or (weekly_bundle.weights if weekly_bundle else {})).items()
            },
            "threshold": weekly_threshold,
            "new_backtest_predictions": new_weekly_backtest,
            "resolved_live_predictions": len(resolved_weekly),
            "gate": weekly_gate,
        },
    }
    report_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report = analysis.build_daily_report(
        report_date=report_date,
        latest_prediction=latest,
        metrics=metrics,
        failure_breakdown=failures,
        resolved_today=resolved,
        model_info=model_info,
        news_summary=news_stats,
        news_items=news_items,
        refresh_status=refresh_status,
        optimization=optimization,
        weekly_prediction=latest_weekly,
        weekly_metrics=weekly_metrics,
        promotion=promotion_review,
        weekly_gate=weekly_gate,
    )
    db.store_report(
        report_date,
        report,
        {
            "overall": metrics,
            "live": live_metrics,
            "backtest": backtest_metrics,
            "weekly_overall": weekly_metrics,
            "weekly_live": weekly_live_metrics,
            "weekly_backtest": weekly_backtest_metrics,
        },
    )
    report_path = config.REPORT_DIR / f"{report_date}.md"
    report_path.write_text(report, encoding="utf-8")

    return {
        "ok": True,
        "report_date": report_date,
        "data_last_date": str(as_of.date()),
        "live_prediction": _hydrate_prediction(live_record),
        "weekly_prediction": _hydrate_prediction(weekly_record),
        "resolved_live_predictions": len(resolved_daily),
        "resolved_weekly_predictions": len(resolved_weekly),
        "new_backtest_predictions": new_backtest,
        "new_weekly_backtest_predictions": new_weekly_backtest,
        "metrics": metrics,
        "live_metrics": live_metrics,
        "backtest_metrics": backtest_metrics,
        "weekly_metrics": weekly_metrics,
        "weekly_live_metrics": weekly_live_metrics,
        "weekly_backtest_metrics": weekly_backtest_metrics,
        "model": bundle_info,
        "weekly_model": weekly_bundle.info() if weekly_bundle else None,
        "promotion": promotion_review,
        "weekly_gate": weekly_gate,
        "optimization": optimization,
        "news": news_stats,
        "report_path": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ETH Oracle daily pipeline")
    parser.add_argument("--backtest-days", type=int, default=None, help="Initial walk-forward backtest length")
    parser.add_argument("--no-fetch", action="store_true", help="Use cached dataset instead of refreshing sources")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = run_daily(backtest_days=args.backtest_days, fetch=not args.no_fetch)
    print(json.dumps({k: v for k, v in result.items() if k not in ("report",)}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
