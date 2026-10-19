from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from app import config
from app.data.http import get_text

log = logging.getLogger(__name__)

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"

SERIES = {
    "DGS10": "us10y",
    "DTWEXBGS": "dxy",
    "SP500": "spx",
    "NASDAQCOM": "nasdaq",
    "VIXCLS": "vix",
    "BAMLH0A0HYM2": "hy_spread",
    "T10Y2Y": "yield_curve",
}


def _cache_path(series_id: str) -> Path:
    return config.CACHE_DIR / f"fred_{series_id}.csv"


def _parse_fred_csv(text: str, name: str) -> pd.DataFrame:
    if "<html" in text[:400].lower():
        raise ValueError(f"FRED returned an HTML challenge for {name}")
    df = pd.read_csv(io.StringIO(text))
    if df.empty or len(df.columns) < 2:
        raise ValueError(f"Unexpected FRED CSV for {name}")
    df = df.iloc[:, :2]
    df.columns = ["date", name]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df[name] = pd.to_numeric(df[name], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    return df.sort_values("date").reset_index(drop=True)


def fetch_fred_series(series_id: str, name: str) -> pd.DataFrame:
    params = {"id": series_id}
    last_exc: Exception | None = None
    for use_proxy in (False, True):
        try:
            text = get_text(
                FRED_CSV,
                params=params,
                timeout=15,
                retries=1,
                use_proxy=use_proxy,
                allow_direct_fallback=False,
                headers={"User-Agent": "curl/8.7.1"},
            )
            return _parse_fred_csv(text, name)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.debug("FRED %s failed (proxy=%s): %s", series_id, use_proxy, exc)
    raise RuntimeError(f"FRED series {series_id} unavailable: {last_exc}")


def update_fred_series(series_id: str, name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _cache_path(series_id)
    cached = None
    if path.exists():
        try:
            cached = pd.read_csv(path)
            cached["date"] = pd.to_datetime(cached["date"], errors="coerce").dt.normalize()
            cached[name] = pd.to_numeric(cached[name], errors="coerce")
        except Exception:  # noqa: BLE001
            cached = None
    try:
        fresh = fetch_fred_series(series_id, name)
        if cached is not None and not cached.empty:
            merged = pd.concat([cached, fresh], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        else:
            merged = fresh
        merged = merged.reset_index(drop=True)
        tmp = path.with_suffix(".csv.tmp")
        merged.to_csv(tmp, index=False)
        tmp.replace(path)
        return merged, {
            "source": f"fred:{series_id}",
            "ok": True,
            "rows": int(len(merged)),
            "last_date": str(merged["date"].max().date()) if not merged.empty else None,
        }
    except Exception as exc:  # noqa: BLE001
        if cached is not None and not cached.empty:
            log.warning("FRED %s refresh failed, using cache: %s", series_id, exc)
            return cached, {
                "source": f"fred:{series_id}",
                "ok": False,
                "error": str(exc),
                "rows": int(len(cached)),
                "last_date": str(cached["date"].max().date()),
                "cached": True,
            }
        log.warning("FRED %s unavailable: %s", series_id, exc)
        return pd.DataFrame(), {
            "source": f"fred:{series_id}",
            "ok": False,
            "error": str(exc),
            "rows": 0,
        }


def load_macro() -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    frames: dict[str, pd.DataFrame] = {}
    statuses: list[dict[str, Any]] = []
    for series_id, name in SERIES.items():
        df, status = update_fred_series(series_id, name)
        frames[name] = df
        statuses.append(status)
    return frames, statuses
