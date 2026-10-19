from __future__ import annotations

import json
import math
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__, config, daily, db
from app.data import build
from app.evaluation import (
    accuracy_series,
    choose_confidence_threshold,
    failure_breakdown,
    metrics_summary,
)
from app.model import load_bundle

@asynccontextmanager
async def lifespan(_: FastAPI):
    config.ensure_dirs()
    db.init_db()
    yield


app = FastAPI(title="ETH Oracle", version=__version__, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

JOBS: dict[str, dict[str, Any]] = {}
JOB_LOCK = threading.Lock()


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _safe_record(row: dict[str, Any] | None) -> dict[str, Any] | None:
    return _safe(daily._hydrate_prediction(row)) if row else None


def _load_cached_dataset() -> pd.DataFrame:
    if not config.DATASET_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(config.DATASET_CSV)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        return df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _metrics_payload() -> dict[str, Any]:
    preds = db.get_predictions(limit=100000)
    live = preds[preds["mode"] == "live"] if not preds.empty else preds
    backtest = preds[preds["mode"] == "backtest"] if not preds.empty else preds
    weekly = db.get_predictions(limit=100000, horizon_days=config.WEEKLY_HORIZON_DAYS)
    weekly_live = weekly[weekly["mode"] == "live"] if not weekly.empty else weekly
    weekly_backtest = weekly[weekly["mode"] == "backtest"] if not weekly.empty else weekly
    return {
        "overall": metrics_summary(preds),
        "live": metrics_summary(live),
        "backtest": metrics_summary(backtest),
        "weekly": {
            "overall": metrics_summary(weekly),
            "live": metrics_summary(weekly_live),
            "backtest": metrics_summary(weekly_backtest),
            "threshold": choose_confidence_threshold(weekly, window=52),
            "counts": db.count_predictions(horizon_days=config.WEEKLY_HORIZON_DAYS),
        },
        "accuracy_series": accuracy_series(preds, 30),
        "failure_breakdown": failure_breakdown(preds),
        "threshold": choose_confidence_threshold(preds),
        "counts": db.count_predictions(),
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    df = _load_cached_dataset()
    return {
        "ok": True,
        "version": __version__,
        "now": datetime.now(timezone.utc).isoformat(),
        "data_rows": int(len(df)),
        "data_last_date": str(df["date"].max().date()) if not df.empty else None,
        "db": str(config.DB_PATH),
        "counts": db.count_predictions(),
        "weekly_counts": db.count_predictions(horizon_days=config.WEEKLY_HORIZON_DAYS),
    }


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    latest = _safe_record(db.get_latest_prediction("live"))
    latest_weekly = _safe_record(db.get_latest_prediction("live", horizon_days=config.WEEKLY_HORIZON_DAYS))
    model_info = db.get_latest_model_run()
    bundle = load_bundle()
    if bundle is not None and model_info:
        model_info["importance"] = bundle.importance.head(25).round(6).to_dict()
        model_info["feature_cols"] = bundle.feature_cols
    preds = db.get_predictions(limit=100000)
    metrics = _metrics_payload()
    news_items = db.get_news(limit=30)
    latest_report = db.get_latest_report()
    refresh = build.load_refresh_status()
    df = _load_cached_dataset()
    return _safe(
        {
            "initialized": not preds.empty,
            "latest_prediction": latest,
            "latest_weekly_prediction": latest_weekly,
            "model": model_info,
            "metrics": metrics,
            "news": {
                "items": news_items,
                "summary": {
                    "score": round(
                        float(np.mean([float(item.get("sentiment") or 0) for item in news_items[:30]]))
                        if news_items
                        else 0.0,
                        4,
                    ),
                    "count": len(news_items),
                },
            },
            "report": latest_report,
            "data": {
                "rows": int(len(df)),
                "last_date": str(df["date"].max().date()) if not df.empty else None,
                "first_date": str(df["date"].min().date()) if not df.empty else None,
                "refresh": refresh,
            },
        }
    )


@app.get("/api/candles")
def candles(days: int = Query(240, ge=30, le=2000)) -> dict[str, Any]:
    df = _load_cached_dataset()
    if df.empty:
        return {"candles": [], "ema20": [], "ema50": [], "prediction": None, "markers": []}
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    view = df.tail(days)
    candles_data = [
        {
            "time": str(row.date.date()),
            "open": round(float(row.open), 4),
            "high": round(float(row.high), 4),
            "low": round(float(row.low), 4),
            "close": round(float(row.close), 4),
            "volume": round(float(row.volume), 4),
        }
        for row in view.itertuples()
    ]
    ema20 = [{"time": str(row.date.date()), "value": round(float(row.ema20), 4)} for row in view.itertuples()]
    ema50 = [{"time": str(row.date.date()), "value": round(float(row.ema50), 4)} for row in view.itertuples()]
    latest = _safe_record(db.get_latest_prediction("live"))
    prediction = None
    if latest:
        prediction = {
            "as_of_date": latest.get("as_of_date"),
            "target_date": latest.get("target_date"),
            "close": latest.get("close"),
            "predicted_close": latest.get("predicted_close"),
            "predicted_return": latest.get("predicted_return"),
            "prob_up": latest.get("prob_up"),
            "confidence": latest.get("confidence"),
            "direction": latest.get("direction"),
            "drivers": latest.get("drivers") or [],
            "model_version": latest.get("model_version"),
            "horizon_days": latest.get("horizon_days") or 1,
        }
    weekly = _safe_record(db.get_latest_prediction("live", horizon_days=config.WEEKLY_HORIZON_DAYS))
    weekly_prediction = None
    if weekly:
        weekly_prediction = {
            "as_of_date": weekly.get("as_of_date"),
            "target_date": weekly.get("target_date"),
            "close": weekly.get("close"),
            "predicted_close": weekly.get("predicted_close"),
            "predicted_return": weekly.get("predicted_return"),
            "prob_up": weekly.get("prob_up"),
            "confidence": weekly.get("confidence"),
            "direction": weekly.get("direction"),
            "horizon_days": weekly.get("horizon_days") or config.WEEKLY_HORIZON_DAYS,
        }

    backtest = db.get_predictions(mode="backtest", resolved_only=True, limit=60)
    markers = []
    if not backtest.empty:
        for row in backtest.sort_values("target_date").itertuples():
            if row.correct is None:
                continue
            markers.append(
                {
                    "time": str(pd.Timestamp(row.target_date).date()),
                    "position": "aboveBar" if int(row.correct) else "belowBar",
                    "color": "#22c55e" if int(row.correct) else "#ef4444",
                    "shape": "circle",
                    "text": "✓" if int(row.correct) else "✗",
                }
            )
    return _safe(
        {
            "candles": candles_data,
            "ema20": ema20,
            "ema50": ema50,
            "prediction": prediction,
            "weekly_prediction": weekly_prediction,
            "markers": markers,
        }
    )


@app.get("/api/predictions")
def predictions(
    mode: str = Query("all", pattern="^(all|live|backtest)$"),
    limit: int = Query(60, ge=1, le=1000),
    resolved_only: bool = False,
    horizon: str = Query("all", pattern="^(all|1|7)$"),
) -> dict[str, Any]:
    df = db.get_predictions(
        None if mode == "all" else mode,
        limit=limit,
        resolved_only=resolved_only,
        horizon_days=None if horizon == "all" else int(horizon),
    )
    if df.empty:
        return {"items": []}
    items = []
    for row in df.to_dict(orient="records"):
        items.append(_safe_record(row))
    return {"items": items}


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    return _safe(_metrics_payload())


@app.get("/api/failures")
def failures(limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
    preds = db.get_predictions(limit=100000)
    breakdown = failure_breakdown(preds)
    items: list[dict[str, Any]] = []
    if not preds.empty:
        wrong = preds[(preds["status"] == "resolved") & (pd.to_numeric(preds["correct"], errors="coerce") == 0)]
        wrong = wrong.sort_values("target_date", ascending=False).head(limit)
        for row in wrong.to_dict(orient="records"):
            items.append(_safe_record(row))
    return _safe({"items": items, "breakdown": breakdown})


@app.get("/api/news")
def news(limit: int = Query(50, ge=1, le=200)) -> dict[str, Any]:
    items = db.get_news(limit=limit)
    scores = [float(item.get("sentiment") or 0) for item in items]
    return _safe(
        {
            "items": items,
            "summary": {
                "score": round(float(np.mean(scores)), 4) if scores else 0.0,
                "count": len(items),
                "positive": sum(1 for score in scores if score >= 0.05),
                "negative": sum(1 for score in scores if score <= -0.05),
                "neutral": sum(1 for score in scores if -0.05 < score < 0.05),
            },
        }
    )


@app.get("/api/report")
def report() -> dict[str, Any]:
    latest = db.get_latest_report()
    if latest is None:
        return {"report": None}
    return _safe(latest)


def _run_job(job_id: str) -> None:
    with JOB_LOCK:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
    try:
        result = daily.run_daily()
        with JOB_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["result"] = _safe(result)
            JOBS[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:  # noqa: BLE001
        with JOB_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(exc)
            JOBS[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.post("/api/run")
def run_now() -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    with JOB_LOCK:
        JOBS[job_id] = {"id": job_id, "status": "queued", "created_at": datetime.now(timezone.utc).isoformat()}
    thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    thread.start()
    return {"ok": True, "job_id": job_id, "status": "queued"}


@app.get("/api/job/{job_id}")
def job(job_id: str) -> dict[str, Any]:
    with JOB_LOCK:
        job_data = JOBS.get(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="job not found")
    return _safe(job_data)


@app.get("/api/validation")
def validation() -> dict[str, Any]:
    path = config.CACHE_DIR / "asset_validation.json"
    if not path.exists():
        return {"available": False, "results": [], "note": "运行 python -m app.validate_assets 生成多标的验证结果"}
    try:
        return {"available": True, **_safe(json.loads(path.read_text(encoding="utf-8")))}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "results": [], "error": str(exc)}


@app.get("/api/model")
def model_info() -> dict[str, Any]:
    info = db.get_latest_model_run()
    bundle = load_bundle()
    if info is None and bundle is None:
        return {"model": None}
    if bundle is not None:
        info = {**(info or {}), **bundle.info(), "importance": bundle.importance.head(40).round(6).to_dict()}
    return _safe({"model": info})
