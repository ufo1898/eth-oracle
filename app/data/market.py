from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from app import config
from app.data.http import FetchError, get_json

log = logging.getLogger(__name__)

SPOT_URL = "https://api.binance.com/api/v3/klines"
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
DAY_MS = 86_400_000

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

SYMBOL_START = {
    "ETHUSDT": "2017-08-01",
    "BTCUSDT": "2017-08-01",
    "ETHBTC": "2017-08-01",
    "PAXGUSDT": "2020-01-01",
}


def _now_ms() -> int:
    return int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)


def _date_ms(value: str | pd.Timestamp) -> int:
    ts = pd.Timestamp(value, tz="UTC") if pd.Timestamp(value).tz is None else pd.Timestamp(value).tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _cache_path(name: str) -> Path:
    return config.CACHE_DIR / f"{name}.csv"


def _read_cache(path: Path, date_col: str = "date") -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
        return df
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read cache %s: %s", path, exc)
        return None


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _parse_klines(rows: list[list[Any]], symbol: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    numeric = [
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
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce")
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    df["symbol"] = symbol
    return df


def fetch_klines(symbol: str, start_date: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch all available daily klines from Binance public market data."""
    interval_ms = DAY_MS if interval == "1d" else DAY_MS
    start_ms = _date_ms(start_date)
    now_ms = _now_ms()
    rows: list[list[Any]] = []
    cursor = start_ms
    while cursor < now_ms:
        payload = get_json(
            SPOT_URL,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "limit": 1000,
            },
            timeout=25,
            retries=3,
        )
        if not isinstance(payload, list):
            raise FetchError(f"Unexpected Binance response for {symbol}: {str(payload)[:200]}")
        if not payload:
            break
        rows.extend(payload)
        last_open = int(payload[-1][0])
        next_cursor = last_open + interval_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(payload) < 1000:
            break
    if not rows:
        raise FetchError(f"No Binance klines returned for {symbol}")
    df = _parse_klines(rows, symbol)
    # Only keep completed candles. This is what the model is allowed to see.
    df = df[df["close_time"] <= now_ms]
    df = df.drop(columns=["ignore", "open_time", "close_time", "symbol"])
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return df


def update_symbol(symbol: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _cache_path(symbol)
    cached = _read_cache(path)
    start_date = SYMBOL_START.get(symbol, "2017-08-01")
    if cached is not None and not cached.empty:
        last_date = pd.Timestamp(cached["date"].max())
        start_date = (last_date - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    try:
        fresh = fetch_klines(symbol, start_date)
        if cached is not None and not cached.empty:
            merged = pd.concat([cached, fresh], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"], keep="last")
            merged = merged.sort_values("date").reset_index(drop=True)
        else:
            merged = fresh
        _atomic_csv(merged, path)
        status = {
            "source": f"binance:{symbol}",
            "ok": True,
            "rows": int(len(merged)),
            "last_date": str(merged["date"].max().date()) if not merged.empty else None,
            "cached": False,
        }
        return merged, status
    except Exception as exc:  # noqa: BLE001
        if cached is not None and not cached.empty:
            log.warning("Binance %s refresh failed, using stale cache: %s", symbol, exc)
            return cached, {
                "source": f"binance:{symbol}",
                "ok": False,
                "error": str(exc),
                "rows": int(len(cached)),
                "last_date": str(cached["date"].max().date()),
                "cached": True,
            }
        raise


def load_market() -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    frames: dict[str, pd.DataFrame] = {}
    statuses: list[dict[str, Any]] = []
    for symbol in SYMBOL_START:
        df, status = update_symbol(symbol)
        frames[symbol] = df
        statuses.append(status)
    if "ETHUSDT" not in frames or frames["ETHUSDT"].empty:
        raise FetchError("ETHUSDT market data is required but unavailable")
    return frames, statuses


def fetch_funding(start_date: str = "2019-01-01") -> pd.DataFrame:
    cursor = _date_ms(start_date)
    now_ms = _now_ms()
    rows: list[dict[str, Any]] = []
    while cursor < now_ms:
        payload = get_json(
            FUNDING_URL,
            params={"symbol": "ETHUSDT", "startTime": cursor, "limit": 1000},
            timeout=25,
            retries=3,
        )
        if not isinstance(payload, list):
            raise FetchError(f"Unexpected funding response: {str(payload)[:200]}")
        if not payload:
            break
        rows.extend(payload)
        last_time = int(payload[-1]["fundingTime"])
        if last_time + 1 <= cursor:
            break
        cursor = last_time + 1
        if len(payload) < 1000:
            break
    if not rows:
        return pd.DataFrame(columns=["date", "funding_mean", "funding_last", "funding_sum", "funding_count"])
    raw = pd.DataFrame(rows)
    raw["fundingTime"] = pd.to_numeric(raw["fundingTime"], errors="coerce")
    raw["fundingRate"] = pd.to_numeric(raw["fundingRate"], errors="coerce")
    if "markPrice" in raw.columns:
        raw["markPrice"] = pd.to_numeric(raw["markPrice"], errors="coerce")
    else:
        raw["markPrice"] = float("nan")
    raw["date"] = pd.to_datetime(raw["fundingTime"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    raw = raw.dropna(subset=["date", "fundingRate"])
    grouped = (
        raw.groupby("date", as_index=False)
        .agg(
            funding_mean=("fundingRate", "mean"),
            funding_last=("fundingRate", "last"),
            funding_sum=("fundingRate", "sum"),
            funding_count=("fundingRate", "count"),
            funding_mark=("markPrice", "last"),
        )
        .sort_values("date")
    )
    return grouped.reset_index(drop=True)


def update_funding() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _cache_path("ETHUSDT_funding")
    cached = _read_cache(path)
    start_date = "2019-01-01"
    if cached is not None and not cached.empty:
        start_date = (cached["date"].max() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    try:
        fresh = fetch_funding(start_date)
        if cached is not None and not cached.empty:
            merged = pd.concat([cached, fresh], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        else:
            merged = fresh
        merged = merged.reset_index(drop=True)
        _atomic_csv(merged, path)
        return merged, {
            "source": "binance:ETHUSDT_funding",
            "ok": True,
            "rows": int(len(merged)),
            "last_date": str(merged["date"].max().date()) if not merged.empty else None,
            "cached": False,
        }
    except Exception as exc:  # noqa: BLE001
        if cached is not None and not cached.empty:
            log.warning("Funding refresh failed, using stale cache: %s", exc)
            return cached, {
                "source": "binance:ETHUSDT_funding",
                "ok": False,
                "error": str(exc),
                "rows": int(len(cached)),
                "last_date": str(cached["date"].max().date()),
                "cached": True,
            }
        log.warning("Funding data unavailable: %s", exc)
        return pd.DataFrame(), {
            "source": "binance:ETHUSDT_funding",
            "ok": False,
            "error": str(exc),
            "rows": 0,
        }
