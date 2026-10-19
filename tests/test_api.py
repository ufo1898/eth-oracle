from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import config, db
from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.csv"
    dates = pd.date_range("2025-01-01", periods=300, freq="D")
    rng = np.random.default_rng(1)
    close = 2000 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))))
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.lognormal(10, 0.2, len(dates)),
        }
    )
    frame.to_csv(dataset, index=False)

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "api.db")
    monkeypatch.setattr(config, "DATASET_CSV", dataset)
    monkeypatch.setattr(config, "LAST_REFRESH_JSON", tmp_path / "last_refresh.json")
    monkeypatch.setattr(config, "MODEL_LATEST", tmp_path / "model_latest.joblib")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    db.init_db()
    db.upsert_prediction(
        {
            "mode": "live",
            "as_of_date": "2025-10-26",
            "target_date": "2025-10-27",
            "close": 2100.0,
            "predicted_close": 2120.0,
            "predicted_return": 0.0095,
            "prob_up": 0.57,
            "prob_raw": 0.55,
            "direction": "UP",
            "confidence": 0.57,
            "p_hgb": 0.58,
            "p_rf": 0.55,
            "p_et": 0.56,
            "p_lr": 0.53,
            "status": "pending",
            "model_version": "v-test",
            "features": {"rsi_14": 50.0},
            "drivers": [{"feature": "rsi_14", "contribution": 0.01, "direction": "up"}],
        }
    )
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_overview(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert health.json()["data_rows"] == 300

    overview = client.get("/api/overview")
    assert overview.status_code == 200
    payload = overview.json()
    assert payload["initialized"] is True
    assert payload["latest_prediction"]["prob_up"] == 0.57
    assert payload["data"]["rows"] == 300


def test_candles_and_prediction_line(client: TestClient) -> None:
    response = client.get("/api/candles?days=240")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["candles"]) == 240
    assert payload["prediction"]["target_date"] == "2025-10-27"
    assert payload["prediction"]["predicted_close"] == 2120.0
    assert payload["markers"] == []


def test_metrics_failures_news_report_and_model(client: TestClient) -> None:
    assert client.get("/api/metrics").status_code == 200
    assert client.get("/api/failures").status_code == 200
    assert client.get("/api/news").status_code == 200
    assert client.get("/api/report").status_code == 200
    model = client.get("/api/model")
    assert model.status_code == 200
    assert model.json()["model"] is None


def test_predictions_endpoint(client: TestClient) -> None:
    response = client.get("/api/predictions?mode=live&limit=10")
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["direction"] == "UP"
    assert items[0]["features"] == {"rsi_14": 50.0}


def test_job_not_found(client: TestClient) -> None:
    assert client.get("/api/job/does-not-exist").status_code == 404


def test_index_serves_dashboard(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "ETH Oracle" in response.text
