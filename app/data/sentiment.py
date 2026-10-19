from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app import config
from app.data.http import FetchError, get_json, get_text

log = logging.getLogger(__name__)

FNG_URL = "https://api.alternative.me/fng/"
WIKI_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/all-agents/Ethereum/daily/{start}/{end}"
)
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

NEWS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "CryptoSlate": "https://cryptoslate.com/feed/",
}

_analyzer = SentimentIntensityAnalyzer()


def _cache(name: str) -> Path:
    return config.CACHE_DIR / f"{name}.csv"


def _read_cache(name: str) -> pd.DataFrame | None:
    path = _cache(name)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        return df
    except Exception:  # noqa: BLE001
        return None


def _write_cache(df: pd.DataFrame, name: str) -> None:
    path = _cache(name)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def fetch_fear_greed() -> pd.DataFrame:
    payload = get_json(FNG_URL, params={"limit": 0, "format": "json"}, timeout=30, retries=3)
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    if not rows:
        raise FetchError("Fear & Greed returned no data")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="s", utc=True)
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()
    df["fng_value"] = pd.to_numeric(df["value"], errors="coerce")
    df["fng_classification"] = df["value_classification"].astype(str)
    df = df[["date", "fng_value", "fng_classification"]].dropna(subset=["date", "fng_value"])
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    _write_cache(df, "fear_greed")
    return df


def load_fear_greed() -> tuple[pd.DataFrame, dict[str, Any]]:
    cached = _read_cache("fear_greed")
    try:
        fresh = fetch_fear_greed()
        if cached is not None and not cached.empty:
            merged = pd.concat([cached, fresh], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
            _write_cache(merged, "fear_greed")
        else:
            merged = fresh
        return merged, {
            "source": "alternative.me:fear_greed",
            "ok": True,
            "rows": int(len(merged)),
            "last_date": str(merged["date"].max().date()),
        }
    except Exception as exc:  # noqa: BLE001
        if cached is not None and not cached.empty:
            log.warning("Fear & Greed refresh failed, using cache: %s", exc)
            return cached, {
                "source": "alternative.me:fear_greed",
                "ok": False,
                "error": str(exc),
                "rows": int(len(cached)),
                "last_date": str(cached["date"].max().date()),
                "cached": True,
            }
        log.warning("Fear & Greed unavailable: %s", exc)
        return pd.DataFrame(), {"source": "alternative.me:fear_greed", "ok": False, "error": str(exc), "rows": 0}


def _parse_entry_date(entry: Any) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        value = getattr(entry, key, None)
        if value:
            try:
                return datetime(*value[:6], tzinfo=timezone.utc)
            except Exception:  # noqa: BLE001
                pass
    for key in ("published", "updated"):
        value = getattr(entry, key, None)
        if value:
            try:
                return pd.Timestamp(value).to_pydatetime().replace(tzinfo=timezone.utc)
            except Exception:  # noqa: BLE001
                pass
    return None


def fetch_news(limit: int = 200) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source, url in NEWS_FEEDS.items():
        try:
            text = get_text(url, timeout=20, retries=2)
            feed = feedparser.parse(text)
            for entry in feed.entries[:80]:
                title = str(getattr(entry, "title", "")).strip()
                link = str(getattr(entry, "link", "")).strip()
                summary = str(getattr(entry, "summary", "")).strip()
                key = link or title.lower()
                if not title or key in seen:
                    continue
                seen.add(key)
                published = _parse_entry_date(entry)
                text_for_sentiment = f"{title}. {summary}"[:1200]
                score = float(_analyzer.polarity_scores(text_for_sentiment)["compound"])
                items.append(
                    {
                        "source": source,
                        "title": title,
                        "url": link,
                        "summary": summary[:600],
                        "published_at": (published or datetime.now(timezone.utc)).isoformat(),
                        "sentiment": round(score, 5),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("News feed %s failed: %s", source, exc)
    items.sort(key=lambda item: item["published_at"], reverse=True)
    items = items[:limit]
    if items:
        path = config.CACHE_DIR / "news_latest.json"
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return items


def load_news_from_cache() -> list[dict[str, Any]]:
    path = config.CACHE_DIR / "news_latest.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []


def news_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"score": 0.0, "count": 0, "positive": 0, "negative": 0, "neutral": 0}
    now = datetime.now(timezone.utc)
    recent: list[dict[str, Any]] = []
    for item in items:
        try:
            published = datetime.fromisoformat(item["published_at"])
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            if (now - published).total_seconds() <= 36 * 3600:
                recent.append(item)
        except Exception:  # noqa: BLE001
            continue
    sample = recent or items[:30]
    scores = [float(item.get("sentiment", 0.0)) for item in sample]
    return {
        "score": round(sum(scores) / len(scores), 5),
        "count": len(sample),
        "positive": sum(1 for score in scores if score >= 0.05),
        "negative": sum(1 for score in scores if score <= -0.05),
        "neutral": sum(1 for score in scores if -0.05 < score < 0.05),
    }


def load_wikipedia_views() -> tuple[pd.DataFrame, dict[str, Any]]:
    cached = _read_cache("wikipedia_views")
    end = datetime.now(timezone.utc).date()
    start = end - pd.Timedelta(days=400) if cached is None or cached.empty else pd.Timestamp(cached["date"].max()).date()
    try:
        payload = get_json(
            WIKI_URL.format(start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d")),
            timeout=30,
            retries=3,
        )
        rows = payload.get("items", []) if isinstance(payload, dict) else []
        if not rows:
            raise FetchError("Wikipedia pageviews returned no items")
        fresh = pd.DataFrame(rows)
        fresh["date"] = pd.to_datetime(fresh["timestamp"].str[:8], format="%Y%m%d", errors="coerce")
        fresh["wiki_views"] = pd.to_numeric(fresh["views"], errors="coerce")
        fresh = fresh[["date", "wiki_views"]].dropna().drop_duplicates(subset=["date"], keep="last")
        if cached is not None and not cached.empty:
            merged = pd.concat([cached, fresh], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        else:
            merged = fresh.sort_values("date")
        merged = merged.reset_index(drop=True)
        _write_cache(merged, "wikipedia_views")
        return merged, {
            "source": "wikimedia:pageviews",
            "ok": True,
            "rows": int(len(merged)),
            "last_date": str(merged["date"].max().date()),
        }
    except Exception as exc:  # noqa: BLE001
        if cached is not None and not cached.empty:
            log.warning("Wikipedia pageviews refresh failed, using cache: %s", exc)
            return cached, {
                "source": "wikimedia:pageviews",
                "ok": False,
                "error": str(exc),
                "rows": int(len(cached)),
                "last_date": str(cached["date"].max().date()),
                "cached": True,
            }
        log.warning("Wikipedia pageviews unavailable: %s", exc)
        return pd.DataFrame(), {"source": "wikimedia:pageviews", "ok": False, "error": str(exc), "rows": 0}


def fetch_gdelt_tone(start_date: str = "2018-01-01") -> pd.DataFrame:
    """Optional historical news-tone series from GDELT.

    GDELT aggressively rate-limits shared IPs, so failures are expected and
    non-fatal. When available, the series is used as a model feature.
    """
    start = pd.Timestamp(start_date).strftime("%Y%m%d000000")
    end = datetime.now(timezone.utc).strftime("%Y%m%d000000")
    params = {
        "query": "ethereum",
        "mode": "timelinetone",
        "format": "json",
        "startdatetime": start,
        "enddatetime": end,
    }
    last_exc: Exception | None = None
    for use_proxy in (False, True):
        try:
            text = get_text(
                GDELT_URL,
                params=params,
                timeout=12,
                retries=1,
                use_proxy=use_proxy,
                allow_direct_fallback=False,
            )
            if text.lstrip().lower().startswith("please limit"):
                raise FetchError("GDELT rate limit")
            payload = json.loads(text)
            series = payload.get("timeline", [])
            if not series:
                raise FetchError("GDELT empty timeline")
            rows = series[0].get("data", [])
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
            df["news_tone"] = pd.to_numeric(df["value"], errors="coerce")
            df = df[["date", "news_tone"]].dropna().drop_duplicates(subset=["date"], keep="last")
            if df.empty:
                raise FetchError("GDELT empty tone series")
            _write_cache(df.sort_values("date").reset_index(drop=True), "gdelt_tone")
            return df.sort_values("date").reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2)
    log.info("GDELT unavailable, continuing without historical news tone: %s", last_exc)
    return pd.DataFrame()


def load_gdelt_tone() -> tuple[pd.DataFrame, dict[str, Any]]:
    cached = _read_cache("gdelt_tone")
    try:
        fresh = fetch_gdelt_tone(start_date=str(cached["date"].max().date()) if cached is not None and not cached.empty else "2018-01-01")
        if fresh.empty:
            raise FetchError("empty")
        if cached is not None and not cached.empty:
            merged = pd.concat([cached, fresh], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        else:
            merged = fresh
        merged = merged.reset_index(drop=True)
        _write_cache(merged, "gdelt_tone")
        return merged, {
            "source": "gdelt:tone",
            "ok": True,
            "rows": int(len(merged)),
            "last_date": str(merged["date"].max().date()),
        }
    except Exception as exc:  # noqa: BLE001
        if cached is not None and not cached.empty:
            return cached, {
                "source": "gdelt:tone",
                "ok": False,
                "error": str(exc),
                "rows": int(len(cached)),
                "last_date": str(cached["date"].max().date()),
                "cached": True,
            }
        return pd.DataFrame(), {"source": "gdelt:tone", "ok": False, "error": str(exc), "rows": 0}
