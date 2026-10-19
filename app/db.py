from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    target_date TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    close REAL,
    predicted_close REAL,
    predicted_return REAL,
    prob_up REAL,
    prob_raw REAL,
    direction TEXT,
    confidence REAL,
    p_hgb REAL,
    p_rf REAL,
    p_et REAL,
    p_lr REAL,
    actual_close REAL,
    actual_return REAL,
    target_up INTEGER,
    correct INTEGER,
    status TEXT NOT NULL DEFAULT 'pending',
    horizon_days INTEGER NOT NULL DEFAULT 1,
    model_version TEXT,
    features_json TEXT,
    drivers_json TEXT,
    failure_categories_json TEXT,
    failure_reason TEXT,
    news_score REAL,
    resolved_at TEXT,
    UNIQUE(mode, as_of_date, horizon_days)
);

CREATE INDEX IF NOT EXISTS idx_predictions_target ON predictions(target_date);
CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(status);
CREATE INDEX IF NOT EXISTS idx_predictions_horizon ON predictions(horizon_days);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    version TEXT,
    train_start TEXT,
    train_end TEXT,
    as_of TEXT,
    n_samples INTEGER,
    n_features INTEGER,
    weights_json TEXT,
    hyperparams_json TEXT,
    val_metrics_json TEXT,
    feature_list_json TEXT,
    importance_json TEXT
);

CREATE TABLE IF NOT EXISTS news_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at TEXT NOT NULL,
    published_at TEXT,
    source TEXT,
    title TEXT NOT NULL,
    url TEXT UNIQUE,
    summary TEXT,
    sentiment REAL
);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_date TEXT NOT NULL,
    created_at TEXT NOT NULL,
    content TEXT NOT NULL,
    metrics_json TEXT,
    UNIQUE(report_date)
);
"""


def _connect() -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


MIGRATION_COLUMNS = [
    "id",
    "mode",
    "as_of_date",
    "target_date",
    "created_at",
    "updated_at",
    "close",
    "predicted_close",
    "predicted_return",
    "prob_up",
    "prob_raw",
    "direction",
    "confidence",
    "p_hgb",
    "p_rf",
    "p_et",
    "p_lr",
    "actual_close",
    "actual_return",
    "target_up",
    "correct",
    "status",
    "model_version",
    "features_json",
    "drivers_json",
    "failure_categories_json",
    "failure_reason",
    "news_score",
    "resolved_at",
]


def _migrate_add_horizon(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP INDEX IF EXISTS idx_predictions_target;
        DROP INDEX IF EXISTS idx_predictions_status;
        DROP INDEX IF EXISTS idx_predictions_horizon;
        ALTER TABLE predictions RENAME TO predictions_old;
        """
    )
    conn.executescript(SCHEMA)
    columns = ", ".join(MIGRATION_COLUMNS)
    conn.execute(
        f"INSERT INTO predictions ({columns}, horizon_days) "
        f"SELECT {columns}, 1 FROM predictions_old"
    )
    conn.execute("DROP TABLE predictions_old")


def init_db() -> None:
    with _connect() as conn:
        existing = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
        ).fetchone()
        if existing:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
            if "horizon_days" not in columns:
                _migrate_add_horizon(conn)
        conn.executescript(SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(value: Any) -> str:
    def _clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(k): _clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [_clean(v) for v in item]
        if isinstance(item, float) and (math.isnan(item) or math.isinf(item)):
            return None
        if hasattr(item, "item"):
            try:
                return _clean(item.item())
            except Exception:  # noqa: BLE001
                return str(item)
        return item

    return json.dumps(_clean(value), ensure_ascii=False, allow_nan=False)


def _decode_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def upsert_prediction(record: dict[str, Any]) -> int:
    now = _now()
    payload = {
        "mode": record.get("mode", "live"),
        "as_of_date": str(record["as_of_date"]),
        "target_date": str(record["target_date"]),
        "created_at": record.get("created_at") or now,
        "updated_at": now,
        "close": record.get("close"),
        "predicted_close": record.get("predicted_close"),
        "predicted_return": record.get("predicted_return"),
        "prob_up": record.get("prob_up"),
        "prob_raw": record.get("prob_raw"),
        "direction": record.get("direction"),
        "confidence": record.get("confidence"),
        "p_hgb": record.get("p_hgb"),
        "p_rf": record.get("p_rf"),
        "p_et": record.get("p_et"),
        "p_lr": record.get("p_lr"),
        "actual_close": record.get("actual_close"),
        "actual_return": record.get("actual_return"),
        "target_up": record.get("target_up"),
        "correct": record.get("correct"),
        "status": record.get("status", "pending"),
        "horizon_days": int(record.get("horizon_days") or 1),
        "model_version": record.get("model_version"),
        "features_json": _safe_json(record.get("features") or {}),
        "drivers_json": _safe_json(record.get("drivers") or []),
        "failure_categories_json": _safe_json(record.get("failure_categories") or []),
        "failure_reason": record.get("failure_reason"),
        "news_score": record.get("news_score"),
        "resolved_at": record.get("resolved_at"),
    }
    columns = ", ".join(payload.keys())
    placeholders = ", ".join(["?"] * len(payload))
    updates = ", ".join(
        f"{key}=excluded.{key}"
        for key in payload.keys()
        if key not in ("mode", "as_of_date", "horizon_days", "created_at")
    )
    sql = (
        f"INSERT INTO predictions ({columns}) VALUES ({placeholders}) "
        f"ON CONFLICT(mode, as_of_date, horizon_days) DO UPDATE SET {updates}"
    )
    with _connect() as conn:
        cur = conn.execute(sql, list(payload.values()))
        row = conn.execute(
            "SELECT id FROM predictions WHERE mode=? AND as_of_date=? AND horizon_days=?",
            (payload["mode"], payload["as_of_date"], payload["horizon_days"]),
        ).fetchone()
        return int(row["id"]) if row else int(cur.lastrowid or 0)


def get_predictions(
    mode: str | None = None,
    *,
    limit: int = 500,
    resolved_only: bool = False,
    pending_only: bool = False,
    horizon_days: int | None = 1,
) -> pd.DataFrame:
    clauses: list[str] = []
    params: list[Any] = []
    if mode:
        clauses.append("mode = ?")
        params.append(mode)
    if horizon_days is not None:
        clauses.append("horizon_days = ?")
        params.append(int(horizon_days))
    if resolved_only:
        clauses.append("status = 'resolved'")
    if pending_only:
        clauses.append("status = 'pending'")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM predictions {where} ORDER BY target_date DESC LIMIT ?"
    params.append(limit)
    with _connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    return df


def get_latest_prediction(mode: str = "live", horizon_days: int = 1) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE mode=? AND horizon_days=? ORDER BY target_date DESC, id DESC LIMIT 1",
            (mode, int(horizon_days)),
        ).fetchone()
    return _decode_row(row)


def get_pending_predictions(horizon_days: int | None = None) -> list[dict[str, Any]]:
    if horizon_days is None:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE status='pending' ORDER BY target_date"
            ).fetchall()
    else:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE status='pending' AND horizon_days=? ORDER BY target_date",
                (int(horizon_days),),
            ).fetchall()
    return [dict(row) for row in rows]


def mark_resolved(
    prediction_id: int,
    *,
    actual_close: float,
    actual_return: float,
    target_up: int,
    correct: int,
    failure_categories: Iterable[str],
    failure_reason: str,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE predictions SET
                actual_close=?, actual_return=?, target_up=?, correct=?,
                status='resolved', failure_categories_json=?,
                failure_reason=?, resolved_at=?, updated_at=?
            WHERE id=?
            """,
            (
                actual_close,
                actual_return,
                target_up,
                correct,
                _safe_json(list(failure_categories)),
                failure_reason,
                _now(),
                _now(),
                prediction_id,
            ),
        )


def store_model_run(bundle_info: dict[str, Any], feature_list: list[str], importance: dict[str, float]) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO model_runs (
                created_at, version, train_start, train_end, as_of, n_samples, n_features,
                weights_json, hyperparams_json, val_metrics_json, feature_list_json, importance_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                bundle_info.get("version"),
                bundle_info.get("train_start"),
                bundle_info.get("train_end"),
                bundle_info.get("as_of"),
                bundle_info.get("n_samples"),
                bundle_info.get("n_features"),
                _safe_json(bundle_info.get("weights") or {}),
                _safe_json(bundle_info.get("params") or {}),
                _safe_json(bundle_info.get("val_metrics") or {}),
                _safe_json(feature_list),
                _safe_json(importance),
            ),
        )


def get_latest_model_run() -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM model_runs ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        return None
    data = dict(row)
    for key in ("weights_json", "hyperparams_json", "val_metrics_json", "feature_list_json", "importance_json"):
        raw = data.pop(key, None)
        data[key.replace("_json", "")] = _loads(raw)
    return data


def store_news(items: list[dict[str, Any]]) -> None:
    if not items:
        return
    fetched = _now()
    with _connect() as conn:
        for item in items:
            conn.execute(
                """
                INSERT INTO news_items (fetched_at, published_at, source, title, url, summary, sentiment)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    fetched_at=excluded.fetched_at,
                    published_at=excluded.published_at,
                    source=excluded.source,
                    title=excluded.title,
                    summary=excluded.summary,
                    sentiment=excluded.sentiment
                """,
                (
                    fetched,
                    item.get("published_at"),
                    item.get("source"),
                    item.get("title"),
                    item.get("url"),
                    item.get("summary"),
                    item.get("sentiment"),
                ),
            )
        conn.execute(
            "DELETE FROM news_items WHERE id NOT IN (SELECT id FROM news_items ORDER BY published_at DESC LIMIT 500)"
        )


def get_news(limit: int = 40) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM news_items ORDER BY published_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def store_report(report_date: str, content: str, metrics: dict[str, Any]) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO reports (report_date, created_at, content, metrics_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(report_date) DO UPDATE SET
                created_at=excluded.created_at,
                content=excluded.content,
                metrics_json=excluded.metrics_json
            """,
            (report_date, _now(), content, _safe_json(metrics)),
        )


def get_latest_report() -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM reports ORDER BY report_date DESC, id DESC LIMIT 1").fetchone()
    if row is None:
        return None
    data = dict(row)
    data["metrics"] = _loads(data.pop("metrics_json", None))
    return data


def count_predictions(horizon_days: int | None = 1) -> dict[str, int]:
    with _connect() as conn:
        if horizon_days is None:
            rows = conn.execute(
                "SELECT mode, status, COUNT(*) AS n FROM predictions GROUP BY mode, status"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT mode, status, COUNT(*) AS n FROM predictions WHERE horizon_days=? GROUP BY mode, status",
                (int(horizon_days),),
            ).fetchall()
    return {f"{row['mode']}:{row['status']}": int(row["n"]) for row in rows}


def clear_predictions(mode: str | None = None, horizon_days: int | None = None) -> None:
    clauses: list[str] = []
    params: list[Any] = []
    if mode:
        clauses.append("mode=?")
        params.append(mode)
    if horizon_days is not None:
        clauses.append("horizon_days=?")
        params.append(int(horizon_days))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect() as conn:
        conn.execute(f"DELETE FROM predictions {where}", params)


def _loads(raw: str | None) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return None
