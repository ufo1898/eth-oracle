from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from app import config
from app.data.http import FetchError, get_json

log = logging.getLogger(__name__)

TVL_URL = "https://api.llama.fi/v2/historicalChainTvl/Ethereum"
STABLE_URL = "https://stablecoins.llama.fi/stablecoincharts/Ethereum"


def _cache(name: str) -> Path:
    return config.CACHE_DIR / f"{name}.csv"


def _read_cache(name: str) -> pd.DataFrame | None:
    path = _cache(name)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        return df
    except Exception:  # noqa: BLE001
        return None


def _write_cache(df: pd.DataFrame, name: str) -> None:
    path = _cache(name)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def fetch_tvl() -> pd.DataFrame:
    payload = get_json(TVL_URL, timeout=35, retries=3)
    if not isinstance(payload, list) or not payload:
        raise FetchError("DefiLlama TVL returned no data")
    df = pd.DataFrame(payload)
    df["date"] = pd.to_datetime(pd.to_numeric(df["date"], errors="coerce"), unit="s", utc=True)
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()
    df["tvl"] = pd.to_numeric(df["tvl"], errors="coerce")
    df = df[["date", "tvl"]].dropna().drop_duplicates(subset=["date"], keep="last").sort_values("date")
    return df.reset_index(drop=True)


def load_tvl() -> tuple[pd.DataFrame, dict[str, Any]]:
    cached = _read_cache("eth_tvl")
    try:
        fresh = fetch_tvl()
        if cached is not None and not cached.empty:
            merged = pd.concat([cached, fresh], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        else:
            merged = fresh
        merged = merged.reset_index(drop=True)
        _write_cache(merged, "eth_tvl")
        return merged, {
            "source": "defillama:tvl",
            "ok": True,
            "rows": int(len(merged)),
            "last_date": str(merged["date"].max().date()),
        }
    except Exception as exc:  # noqa: BLE001
        if cached is not None and not cached.empty:
            log.warning("TVL refresh failed, using cache: %s", exc)
            return cached, {
                "source": "defillama:tvl",
                "ok": False,
                "error": str(exc),
                "rows": int(len(cached)),
                "last_date": str(cached["date"].max().date()),
                "cached": True,
            }
        log.warning("TVL unavailable: %s", exc)
        return pd.DataFrame(), {"source": "defillama:tvl", "ok": False, "error": str(exc), "rows": 0}


def fetch_stablecoin_supply() -> pd.DataFrame:
    payload = get_json(STABLE_URL, params={"stablecoin": 1}, timeout=40, retries=3)
    if not isinstance(payload, list) or not payload:
        raise FetchError("DefiLlama stablecoin series returned no data")
    df = pd.DataFrame(payload)
    df["date"] = pd.to_datetime(pd.to_numeric(df["date"], errors="coerce"), unit="s", utc=True)
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()

    def _extract(value: Any) -> float:
        if isinstance(value, dict):
            return float(value.get("peggedUSD") or value.get("peggedusd") or 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    df["stablecoin_usd"] = df["totalCirculatingUSD"].map(_extract)
    df = df[["date", "stablecoin_usd"]].dropna().drop_duplicates(subset=["date"], keep="last").sort_values("date")
    return df.reset_index(drop=True)


def load_stablecoin_supply() -> tuple[pd.DataFrame, dict[str, Any]]:
    cached = _read_cache("eth_stablecoins")
    try:
        fresh = fetch_stablecoin_supply()
        if cached is not None and not cached.empty:
            merged = pd.concat([cached, fresh], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        else:
            merged = fresh
        merged = merged.reset_index(drop=True)
        _write_cache(merged, "eth_stablecoins")
        return merged, {
            "source": "defillama:stablecoins",
            "ok": True,
            "rows": int(len(merged)),
            "last_date": str(merged["date"].max().date()),
        }
    except Exception as exc:  # noqa: BLE001
        if cached is not None and not cached.empty:
            log.warning("Stablecoin refresh failed, using cache: %s", exc)
            return cached, {
                "source": "defillama:stablecoins",
                "ok": False,
                "error": str(exc),
                "rows": int(len(cached)),
                "last_date": str(cached["date"].max().date()),
                "cached": True,
            }
        log.warning("Stablecoin supply unavailable: %s", exc)
        return pd.DataFrame(), {"source": "defillama:stablecoins", "ok": False, "error": str(exc), "rows": 0}
