from __future__ import annotations

import json
import logging
import time
from typing import Any, Iterable

import requests

from app import config

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36 "
    "eth-oracle/0.1 (+research)"
)
RETRYABLE = {408, 425, 429, 500, 502, 503, 504, 522, 524}


class FetchError(RuntimeError):
    pass


def _proxies(enabled: bool) -> dict[str, str] | None:
    if enabled and config.PROXY:
        return {"http": config.PROXY, "https": config.PROXY}
    return None


def _proxy_modes(use_proxy: bool, allow_direct_fallback: bool) -> Iterable[bool]:
    if use_proxy and allow_direct_fallback:
        return (True, False)
    return (use_proxy,)


def get(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = 30,
    retries: int = 3,
    use_proxy: bool = True,
    allow_direct_fallback: bool = True,
    headers: dict[str, str] | None = None,
    session: requests.Session | None = None,
) -> requests.Response:
    """GET with proxy-first, direct-fallback and retry logic.

    The host machine uses an explicit local proxy. Some public endpoints work
    better without it, so the default behaviour tries both paths.
    """
    own_session = session is None
    sess = session or requests.Session()
    if own_session:
        sess.trust_env = False
    merged_headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if headers:
        merged_headers.update(headers)

    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        for proxy_enabled in _proxy_modes(use_proxy, allow_direct_fallback):
            try:
                resp = sess.get(
                    url,
                    params=params,
                    timeout=timeout,
                    headers=merged_headers,
                    proxies=_proxies(proxy_enabled),
                )
                if resp.status_code in RETRYABLE:
                    raise FetchError(f"HTTP {resp.status_code} for {url}")
                resp.raise_for_status()
                return resp
            except Exception as exc:  # noqa: BLE001 - network sources are noisy
                last_error = exc
                log.debug("GET failed (%s, proxy=%s): %s", url, proxy_enabled, exc)
                time.sleep(min(0.8 * (attempt + 1), 3.0))
    raise FetchError(f"GET failed for {url}: {last_error}") from last_error


def get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = 30,
    retries: int = 3,
    use_proxy: bool = True,
    allow_direct_fallback: bool = True,
    headers: dict[str, str] | None = None,
) -> Any:
    resp = get(
        url,
        params=params,
        timeout=timeout,
        retries=retries,
        use_proxy=use_proxy,
        allow_direct_fallback=allow_direct_fallback,
        headers=headers,
    )
    try:
        return resp.json()
    except json.JSONDecodeError as exc:
        raise FetchError(f"Invalid JSON from {url}: {resp.text[:160]}") from exc


def get_text(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = 30,
    retries: int = 3,
    use_proxy: bool = True,
    allow_direct_fallback: bool = True,
    headers: dict[str, str] | None = None,
) -> str:
    resp = get(
        url,
        params=params,
        timeout=timeout,
        retries=retries,
        use_proxy=use_proxy,
        allow_direct_fallback=allow_direct_fallback,
        headers=headers,
    )
    resp.encoding = resp.encoding or "utf-8"
    return resp.text
