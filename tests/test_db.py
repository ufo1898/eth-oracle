from __future__ import annotations

import sqlite3

import pandas as pd

from app import config, db


def _prediction(as_of: str, target: str, mode: str = "live", horizon_days: int = 1) -> dict:
    return {
        "mode": mode,
        "as_of_date": as_of,
        "target_date": target,
        "horizon_days": horizon_days,
        "created_at": "2026-10-01T00:00:00+00:00",
        "close": 2000.0,
        "predicted_close": 2020.0,
        "predicted_return": 0.01,
        "prob_up": 0.56,
        "prob_raw": 0.54,
        "direction": "UP",
        "confidence": 0.56,
        "p_hgb": 0.55,
        "p_rf": 0.54,
        "p_et": 0.58,
        "p_lr": 0.52,
        "status": "pending",
        "model_version": "v-test",
        "features": {"rsi_14": 45.0},
        "drivers": [{"feature": "rsi_14", "direction": "up", "contribution": 0.01}],
        "news_score": 0.1,
    }


def test_prediction_lifecycle(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    db.init_db()

    prediction_id = db.upsert_prediction(_prediction("2026-10-01", "2026-10-02"))
    assert prediction_id > 0
    assert len(db.get_pending_predictions()) == 1
    latest = db.get_latest_prediction("live")
    assert latest is not None
    assert latest["prob_up"] == 0.56

    db.mark_resolved(
        prediction_id,
        actual_close=2010.0,
        actual_return=0.005,
        target_up=1,
        correct=1,
        failure_categories=[],
        failure_reason="预测正确",
    )
    assert db.get_pending_predictions() == []
    resolved = db.get_predictions("live", resolved_only=True)
    assert len(resolved) == 1
    assert int(resolved.iloc[0]["correct"]) == 1
    assert db.count_predictions() == {"live:resolved": 1}


def test_upsert_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    first = db.upsert_prediction(_prediction("2026-10-01", "2026-10-02"))
    changed = _prediction("2026-10-01", "2026-10-02")
    changed["prob_up"] = 0.7
    second = db.upsert_prediction(changed)
    assert first == second
    frame = db.get_predictions("live")
    assert len(frame) == 1
    assert float(frame.iloc[0]["prob_up"]) == 0.7


def test_daily_and_weekly_predictions_are_separate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    daily_id = db.upsert_prediction(_prediction("2026-10-01", "2026-10-02"))
    weekly_id = db.upsert_prediction(_prediction("2026-10-01", "2026-10-08", horizon_days=7))
    assert daily_id != weekly_id
    assert db.count_predictions(horizon_days=1) == {"live:pending": 1}
    assert db.count_predictions(horizon_days=7) == {"live:pending": 1}
    assert len(db.get_predictions(horizon_days=1)) == 1
    assert len(db.get_predictions(horizon_days=7)) == 1
    assert db.get_latest_prediction("live", horizon_days=7)["horizon_days"] == 7
    assert len(db.get_pending_predictions(horizon_days=None)) == 2
    db.clear_predictions(horizon_days=7)
    assert db.get_predictions(horizon_days=7).empty
    assert len(db.get_predictions(horizon_days=1)) == 1


def test_init_db_migrates_old_predictions_table(tmp_path, monkeypatch) -> None:
    path = tmp_path / "old.db"
    monkeypatch.setattr(config, "DB_PATH", path)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE predictions (
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
                model_version TEXT,
                features_json TEXT,
                drivers_json TEXT,
                failure_categories_json TEXT,
                failure_reason TEXT,
                news_score REAL,
                resolved_at TEXT,
                UNIQUE(mode, as_of_date)
            );
            INSERT INTO predictions
                (mode, as_of_date, target_date, created_at, updated_at, close, status)
            VALUES ('backtest', '2026-01-01', '2026-01-02', 'x', 'x', 2000, 'resolved');
            """
        )
    db.init_db()
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
    assert "horizon_days" in columns
    frame = db.get_predictions()
    assert len(frame) == 1
    assert int(frame.iloc[0]["horizon_days"]) == 1


def test_model_run_and_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    db.store_model_run(
        {
            "version": "v-test",
            "train_start": "2020-01-01",
            "train_end": "2026-01-01",
            "as_of": "2026-01-01",
            "n_samples": 100,
            "n_features": 10,
            "weights": {"hgb": 0.5, "rf": 0.5},
            "params": {"hgb_max_iter": 10},
            "val_metrics": {"accuracy": 0.55},
        },
        ["rsi_14"],
        {"rsi_14": 0.1},
    )
    latest = db.get_latest_model_run()
    assert latest is not None
    assert latest["weights"] == {"hgb": 0.5, "rf": 0.5}
    assert latest["feature_list"] == ["rsi_14"]

    db.store_report("2026-10-01", "# 日报", {"overall": {"n": 1}})
    report = db.get_latest_report()
    assert report is not None
    assert report["content"] == "# 日报"


def test_news_storage(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    db.init_db()
    db.store_news(
        [
            {
                "published_at": "2026-10-01T00:00:00+00:00",
                "source": "test",
                "title": "test title",
                "url": "https://example.com/a",
                "summary": "summary",
                "sentiment": 0.4,
            }
        ]
    )
    items = db.get_news()
    assert len(items) == 1
    assert items[0]["sentiment"] == 0.4


def test_get_predictions_returns_empty_frame(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    frame = db.get_predictions()
    assert isinstance(frame, pd.DataFrame)
    assert frame.empty
