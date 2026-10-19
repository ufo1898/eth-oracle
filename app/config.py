from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


DATA_DIR = _env_path("ETORACLE_DATA_DIR", BASE_DIR / "data")
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = DATA_DIR / "models"
DB_PATH = _env_path("ETORACLE_DB", DATA_DIR / "eth_oracle.db")
REPORT_DIR = _env_path("ETORACLE_REPORT_DIR", BASE_DIR / "reports")
STATIC_DIR = BASE_DIR / "app" / "static"

PROXY = os.environ.get("ETORACLE_PROXY", "http://127.0.0.1:10900")
HOST = os.environ.get("ETORACLE_HOST", "127.0.0.1")
PORT = int(os.environ.get("ETORACLE_PORT", "8787"))

DATASET_CSV = CACHE_DIR / "dataset.csv"
LAST_REFRESH_JSON = CACHE_DIR / "last_refresh.json"
MODEL_LATEST = MODEL_DIR / "model_latest.joblib"
MODEL_WEEKLY_LATEST = MODEL_DIR / "model_weekly_latest.joblib"
MODEL_CHALLENGER = MODEL_DIR / "model_challenger.joblib"
BACKTEST_DAYS_DEFAULT = int(os.environ.get("ETORACLE_BACKTEST_DAYS", "365"))
WEEKLY_HORIZON_DAYS = 7
WEEKLY_BACKTEST_DAYS = int(os.environ.get("ETORACLE_WEEKLY_BACKTEST_DAYS", "365"))
MIN_TRAIN_ROWS = int(os.environ.get("ETORACLE_MIN_TRAIN_ROWS", "420"))
RETRAIN_DAYS = int(os.environ.get("ETORACLE_RETRAIN_DAYS", "30"))
PROMOTION_EVAL_DAYS = int(os.environ.get("ETORACLE_PROMOTION_EVAL_DAYS", "60"))
PROMOTION_MIN_N = int(os.environ.get("ETORACLE_PROMOTION_MIN_N", "30"))
PROMOTION_ACC_TOLERANCE = float(os.environ.get("ETORACLE_PROMOTION_ACC_TOLERANCE", "0.005"))
PROMOTION_BRIER_TOLERANCE = float(os.environ.get("ETORACLE_PROMOTION_BRIER_TOLERANCE", "0.02"))
WEEKLY_MIN_EVAL_N = int(os.environ.get("ETORACLE_WEEKLY_MIN_EVAL_N", "30"))
WEEKLY_MAX_UNDERPERFORMANCE = float(os.environ.get("ETORACLE_WEEKLY_MAX_UNDERPERFORMANCE", "0.02"))
# GDELT is optional and aggressively rate-limited. Disabled by default so the
# daily job does not wait on a source that is usually unavailable.
ENABLE_GDELT = os.environ.get("ETORACLE_ENABLE_GDELT", "0").lower() in {"1", "true", "yes", "on"}


def ensure_dirs() -> None:
    for path in (DATA_DIR, CACHE_DIR, MODEL_DIR, REPORT_DIR, STATIC_DIR / "vendor"):
        path.mkdir(parents=True, exist_ok=True)
