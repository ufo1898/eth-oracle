from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app import config, evaluation, model
from app.data import build, market
from app.daily import _feature_row
from app.features import build_features, feature_columns

log = logging.getLogger(__name__)

PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
]
FUNDING_COLUMNS = ["funding_mean", "funding_last", "funding_sum", "funding_count", "funding_mark"]
LABELS = {"ETHUSDT": "ETH", "BTCUSDT": "BTC", "SOLUSDT": "SOL"}


def _asset_features(shared: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if symbol == "ETHUSDT":
        frame = shared.copy()
        detail = {"source": "cached dataset", "rows": int(len(shared))}
    else:
        prices, price_status = market.update_symbol(symbol)
        funding, funding_status = market.update_funding(symbol)
        base = shared.drop(columns=[col for col in PRICE_COLUMNS if col in shared.columns]).copy()
        frame = base.merge(prices[["date"] + PRICE_COLUMNS], on="date", how="inner")
        if not funding.empty:
            frame = frame.drop(columns=[col for col in FUNDING_COLUMNS if col in frame.columns])
            frame = frame.merge(funding, on="date", how="left")
        frame = frame.sort_values("date").reset_index(drop=True)
        fill = [col for col in FUNDING_COLUMNS if col in frame.columns]
        if fill:
            frame[fill] = frame[fill].ffill()
        detail = {
            "source": "binance",
            "rows": int(len(frame)),
            "price_status": price_status,
            "funding_status": funding_status,
        }
    features = build_features(frame)
    detail["feature_rows"] = int(len(features))
    detail["first_date"] = str(features["date"].min().date()) if not features.empty else None
    detail["last_date"] = str(features["date"].max().date()) if not features.empty else None
    return features, detail


def validate_frame(feature_df: pd.DataFrame, *, days: int, retrain_days: int = 30) -> dict[str, Any]:
    if feature_df.empty:
        raise ValueError("Feature frame is empty")
    end_target = pd.Timestamp(feature_df["date"].max()).normalize()
    start_target = max(
        pd.Timestamp(feature_df["date"].min()).normalize() + pd.Timedelta(days=config.MIN_TRAIN_ROWS),
        end_target - pd.Timedelta(days=days),
    )
    candidates = feature_columns(feature_df)
    available = set(feature_df["date"].tolist())
    bundle: model.ModelBundle | None = None
    last_retrain_target: pd.Timestamp | None = None
    records: list[dict[str, Any]] = []
    for target in pd.date_range(start_target, end_target, freq="D"):
        target = pd.Timestamp(target).normalize()
        as_of = target - pd.Timedelta(days=1)
        if target not in available or as_of not in available:
            continue
        row = _feature_row(feature_df, as_of)
        target_rows = feature_df[feature_df["date"] == target]
        if row is None or target_rows.empty:
            continue
        if (
            bundle is None
            or last_retrain_target is None
            or (target - last_retrain_target).days >= retrain_days
        ):
            bundle = model.train_bundle(feature_df, as_of=as_of, candidates=candidates)
            last_retrain_target = target
        prediction = model.predict_one(bundle, row)
        actual_close = float(target_rows.iloc[-1]["close"])
        actual_return = actual_close / float(row["close"]) - 1
        target_up = int(actual_return > 0)
        predicted_up = int(float(prediction["prob_up"]) >= 0.5)
        records.append(
            {
                "mode": "backtest",
                "status": "resolved",
                "horizon_days": 1,
                "as_of_date": str(row["date"].date()),
                "target_date": str(target.date()),
                "close": float(row["close"]),
                "predicted_close": prediction["predicted_close"],
                "predicted_return": prediction["predicted_return"],
                "prob_up": prediction["prob_up"],
                "prob_raw": prediction["prob_raw"],
                "confidence": prediction["confidence"],
                "direction": prediction["direction"],
                "p_hgb": prediction.get("p_hgb"),
                "p_rf": prediction.get("p_rf"),
                "p_et": prediction.get("p_et"),
                "p_lr": prediction.get("p_lr"),
                "actual_close": actual_close,
                "actual_return": actual_return,
                "target_up": target_up,
                "correct": int(predicted_up == target_up),
                "model_version": bundle.version,
            }
        )
    frame = pd.DataFrame(records)
    metrics = evaluation.metrics_summary(frame)
    return {
        "n": metrics["n"],
        "accuracy": metrics["accuracy"],
        "baseline_accuracy": metrics["baseline_accuracy"],
        "edge": metrics["edge"],
        "brier": metrics["brier"],
        "log_loss": metrics["log_loss"],
        "auc": metrics["auc"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "significance": metrics["significance"],
        "calibration": metrics["calibration"],
        "strategy": metrics["strategy"],
        "from": metrics.get("from"),
        "to": metrics.get("to"),
        "features": len(candidates),
        "retrain_days": retrain_days,
    }


def run(symbols: list[str], *, days: int = 365, retrain_days: int = 30) -> dict[str, Any]:
    shared = build.load_dataset()
    if shared.empty:
        shared = build.build_dataset()
    results: list[dict[str, Any]] = []
    for symbol in symbols:
        label = LABELS.get(symbol, symbol.replace("USDT", ""))
        log.info("Validating %s (%s), %s days...", label, symbol, days)
        features, detail = _asset_features(shared, symbol)
        metrics = validate_frame(features, days=days, retrain_days=retrain_days)
        results.append(
            {
                "symbol": symbol,
                "label": label,
                **metrics,
                "data": detail,
            }
        )
        log.info(
            "Validated %s: n=%s acc=%s baseline=%s edge=%s p=%s",
            label,
            metrics["n"],
            metrics["accuracy"],
            metrics["baseline_accuracy"],
            metrics["edge"],
            (metrics.get("significance") or {}).get("vs_coinflip", {}).get("p_value"),
        )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days": days,
        "retrain_days": retrain_days,
        "results": results,
    }
    path = config.CACHE_DIR / "asset_validation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    log.info("Saved validation results to %s", path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward validation across crypto assets")
    parser.add_argument("--symbols", default="ETHUSDT,BTCUSDT,SOLUSDT")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--retrain-days", type=int, default=30)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    payload = run(symbols, days=args.days, retrain_days=args.retrain_days)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
