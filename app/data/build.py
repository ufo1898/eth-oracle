from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app import config
from app.data import macro, market, onchain, sentiment

log = logging.getLogger(__name__)

PRICE_COLUMNS = [
    "date",
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

EXTERNAL_COLUMNS = [
    "btc_close",
    "btc_volume",
    "btc_quote_volume",
    "ethbtc_close",
    "ethbtc_volume",
    "paxg_close",
    "paxg_volume",
    "us10y",
    "dxy",
    "spx",
    "nasdaq",
    "vix",
    "hy_spread",
    "yield_curve",
    "funding_mean",
    "funding_last",
    "funding_sum",
    "funding_count",
    "funding_mark",
    "fng_value",
    "fng_classification",
    "tvl",
    "stablecoin_usd",
    "wiki_views",
    "news_tone",
]


def _rename_market(df: pd.DataFrame, prefix: str, columns: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date"] + [f"{prefix}_{col}" for col in columns])
    keep = ["date"] + [col for col in columns if col in df.columns]
    out = df[keep].copy()
    out = out.rename(columns={col: f"{prefix}_{col}" for col in columns if col in df.columns})
    return out


def _merge(df: pd.DataFrame, other: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if other is None or other.empty:
        return df
    keep = ["date"] + [col for col in columns if col in other.columns]
    if len(keep) <= 1:
        return df
    return df.merge(other[keep], on="date", how="left")


def build_dataset(force: bool = False) -> pd.DataFrame:
    """Fetch, merge and cache the full daily feature source dataset.

    Each source degrades gracefully to its local cache; only ETH spot data is
    mandatory. The returned frame contains only completed UTC daily candles.
    """
    config.ensure_dirs()
    statuses: list[dict[str, Any]] = []

    frames, market_status = market.load_market()
    statuses.extend(market_status)
    eth = frames["ETHUSDT"]
    df = eth[PRICE_COLUMNS].copy()

    btc = _rename_market(frames.get("BTCUSDT"), "btc", ["close", "volume", "quote_volume"])
    ethbtc = _rename_market(frames.get("ETHBTC"), "ethbtc", ["close", "volume"])
    paxg = _rename_market(frames.get("PAXGUSDT"), "paxg", ["close", "volume"])
    df = _merge(df, btc, ["btc_close", "btc_volume", "btc_quote_volume"])
    df = _merge(df, ethbtc, ["ethbtc_close", "ethbtc_volume"])
    df = _merge(df, paxg, ["paxg_close", "paxg_volume"])

    macro_frames, macro_status = macro.load_macro()
    statuses.extend(macro_status)
    for name in ("us10y", "dxy", "spx", "nasdaq", "vix", "hy_spread", "yield_curve"):
        df = _merge(df, macro_frames.get(name), [name])

    funding, funding_status = market.update_funding()
    statuses.append(funding_status)
    df = _merge(df, funding, ["funding_mean", "funding_last", "funding_sum", "funding_count", "funding_mark"])

    fng, fng_status = sentiment.load_fear_greed()
    statuses.append(fng_status)
    df = _merge(df, fng, ["fng_value", "fng_classification"])

    tvl, tvl_status = onchain.load_tvl()
    statuses.append(tvl_status)
    df = _merge(df, tvl, ["tvl"])

    stable, stable_status = onchain.load_stablecoin_supply()
    statuses.append(stable_status)
    df = _merge(df, stable, ["stablecoin_usd"])

    wiki, wiki_status = sentiment.load_wikipedia_views()
    statuses.append(wiki_status)
    df = _merge(df, wiki, ["wiki_views"])

    if config.ENABLE_GDELT:
        gdelt, gdelt_status = sentiment.load_gdelt_tone()
        statuses.append(gdelt_status)
        df = _merge(df, gdelt, ["news_tone"])
    else:
        statuses.append({"source": "gdelt:tone", "ok": True, "rows": 0, "disabled": True})

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)

    # Exchange holidays / non-trading days: forward-fill external series only.
    fill_columns = [col for col in EXTERNAL_COLUMNS if col in df.columns]
    if fill_columns:
        df[fill_columns] = df[fill_columns].ffill()
    df = df.replace([np.inf, -np.inf], np.nan)

    tmp = config.DATASET_CSV.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(config.DATASET_CSV)

    refresh = {
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "first_date": str(df["date"].min().date()) if not df.empty else None,
        "last_date": str(df["date"].max().date()) if not df.empty else None,
        "sources": statuses,
    }
    config.LAST_REFRESH_JSON.write_text(json.dumps(refresh, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Dataset built: %s rows, %s -> %s", refresh["rows"], refresh["first_date"], refresh["last_date"])
    return df


def load_dataset() -> pd.DataFrame:
    if not config.DATASET_CSV.exists():
        return build_dataset()
    df = pd.read_csv(config.DATASET_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df


def load_refresh_status() -> dict[str, Any]:
    if not config.LAST_REFRESH_JSON.exists():
        return {}
    try:
        return json.loads(config.LAST_REFRESH_JSON.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
