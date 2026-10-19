from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_ohlcv(n: int = 900, seed: int = 42, start: str = "2023-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    ret = rng.normal(0.001, 0.02, n)
    close = 2000 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, n)))
    volume = rng.lognormal(10, 0.3, n)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trades": volume / 2,
            "taker_buy_base": volume * 0.5,
            "taker_buy_quote": volume * 0.5 * close,
        }
    )
    external = {
        "btc_close": 30000,
        "btc_volume": 1e5,
        "btc_quote_volume": 1e9,
        "ethbtc_close": 0.05,
        "ethbtc_volume": 1e4,
        "paxg_close": 1900,
        "paxg_volume": 1e3,
        "us10y": 4.0,
        "dxy": 100,
        "spx": 4000,
        "nasdaq": 15000,
        "vix": 18,
        "hy_spread": 3.5,
        "yield_curve": 0.5,
        "funding_mean": 0.00001,
        "funding_last": 0.00001,
        "funding_sum": 0.00003,
        "funding_count": 3,
        "funding_mark": 30000,
        "fng_value": 50,
        "tvl": 1e10,
        "stablecoin_usd": 1e11,
        "wiki_views": 2000,
    }
    for col, base in external.items():
        df[col] = base * (1 + np.cumsum(rng.normal(0, 0.01, n)))
    df["fng_classification"] = "Neutral"
    return df


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def synthetic_features(synthetic_ohlcv: pd.DataFrame) -> pd.DataFrame:
    from app.features import build_features

    return build_features(synthetic_ohlcv)
