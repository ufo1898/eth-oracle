from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

warnings.filterwarnings("ignore", category=PerformanceWarning)

log = logging.getLogger(__name__)

RAW_COLUMNS = {
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
    "funding_sum",
    "funding_count",
    "funding_mark",
    "fng_value",
    "fng_classification",
    "tvl",
    "stablecoin_usd",
    "wiki_views",
    "news_tone",
    "next_close",
    "target_ret",
    "target_up",
}


def _series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _pct(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods=periods, fill_method=None)


def _safe_div(a: pd.Series, b: pd.Series | float) -> pd.Series:
    if isinstance(b, pd.Series):
        return a / b.replace(0, np.nan)
    return a / b if b else pd.Series(np.nan, index=a.index)


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = _safe_div(avg_gain, avg_loss)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    return rsi.clip(0, 100)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * _safe_div(plus_dm.ewm(alpha=1 / period, adjust=False).mean(), atr)
    minus_di = 100 * _safe_div(minus_dm.ewm(alpha=1 / period, adjust=False).mean(), atr)
    dx = 100 * _safe_div((plus_di - minus_di).abs(), plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx, plus_di, minus_di


def _mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = typical * df["volume"]
    positive = money_flow.where(typical > typical.shift(1), 0.0)
    negative = money_flow.where(typical < typical.shift(1), 0.0)
    ratio = _safe_div(positive.rolling(period).sum(), negative.rolling(period).sum())
    mfi = 100 - 100 / (1 + ratio)
    return mfi.clip(0, 100)


def _rolling_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    def _last_rank(values: np.ndarray) -> float:
        if len(values) == 0 or np.isnan(values[-1]):
            return np.nan
        return float(np.mean(values[:-1] <= values[-1]))

    return series.rolling(window, min_periods=min_periods).apply(_last_rank, raw=True)


def _consecutive(series: pd.Series, direction: int) -> pd.Series:
    signed = np.sign(series.diff())
    flag = (signed == direction).astype(int)
    groups = (flag != flag.shift()).cumsum()
    return flag.groupby(groups).cumsum().astype(float)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build stationary-ish technical, macro, sentiment and on-chain features."""
    df = df.copy().sort_values("date").reset_index(drop=True)
    close, high, low, open_, volume = (
        df["close"].astype(float),
        df["high"].astype(float),
        df["low"].astype(float),
        df["open"].astype(float),
        df["volume"].astype(float),
    )

    # ---------- returns / volatility ----------
    for n in (1, 2, 3, 5, 7, 14, 30, 60, 90, 180, 365):
        df[f"ret_{n}d"] = close.pct_change(n, fill_method=None)
    log_ret = np.log(close.clip(lower=1e-12)).diff()
    for n in (7, 14, 30, 60, 90):
        df[f"vol_{n}d"] = log_ret.rolling(n, min_periods=max(3, n // 2)).std()
    df["vol_ratio_7_30"] = _safe_div(df["vol_7d"], df["vol_30d"])
    df["vol_ratio_30_90"] = _safe_div(df["vol_30d"], df["vol_90d"])
    df["downside_vol_30d"] = log_ret.where(log_ret < 0, 0.0).rolling(30, min_periods=10).std()
    df["upside_vol_30d"] = log_ret.where(log_ret > 0, 0.0).rolling(30, min_periods=10).std()
    df["return_skew_30d"] = log_ret.rolling(30, min_periods=10).skew()
    df["return_kurt_30d"] = log_ret.rolling(30, min_periods=10).kurt()

    # ---------- range / candlestick ----------
    df["hl_range"] = _safe_div(high - low, close)
    df["close_position"] = _safe_div(close - low, high - low)
    df["body_ratio"] = _safe_div(close - open_, high - low)
    df["upper_wick"] = _safe_div(high - pd.concat([open_, close], axis=1).max(axis=1), high - low)
    df["lower_wick"] = _safe_div(pd.concat([open_, close], axis=1).min(axis=1) - low, high - low)
    df["gap"] = _safe_div(open_ - close.shift(1), close.shift(1))
    df["atr_14_pct"] = _safe_div(_atr(df, 14), close)
    df["atr_30_pct"] = _safe_div(_atr(df, 30), close)
    df["atr_ratio_14_30"] = _safe_div(df["atr_14_pct"], df["atr_30_pct"])
    df["range_z_20"] = _safe_div((high - low) - (high - low).rolling(20).mean(), (high - low).rolling(20).std())

    # ---------- moving averages / trend ----------
    for n in (7, 14, 20, 30, 50, 100, 200):
        sma = close.rolling(n, min_periods=max(2, n // 3)).mean()
        df[f"close_sma_{n}"] = _safe_div(close, sma) - 1
    for n in (12, 26, 50, 200):
        ema = close.ewm(span=n, adjust=False, min_periods=max(2, n // 3)).mean()
        df[f"close_ema_{n}"] = _safe_div(close, ema) - 1
    df["sma_20_50"] = _safe_div(close.rolling(20).mean(), close.rolling(50).mean()) - 1
    df["sma_50_200"] = _safe_div(close.rolling(50).mean(), close.rolling(200).mean()) - 1
    df["sma_200_slope_20"] = _safe_div(close.rolling(200).mean(), close.rolling(200).mean().shift(20)) - 1
    df["above_sma_20"] = (close > close.rolling(20).mean()).astype(float)
    df["above_sma_50"] = (close > close.rolling(50).mean()).astype(float)
    df["above_sma_200"] = (close > close.rolling(200).mean()).astype(float)
    df["golden_cross"] = (
        (close.rolling(50).mean() > close.rolling(200).mean())
        & (close.rolling(50).mean().shift(1) <= close.rolling(200).mean().shift(1))
    ).astype(float)
    df["death_cross"] = (
        (close.rolling(50).mean() < close.rolling(200).mean())
        & (close.rolling(50).mean().shift(1) >= close.rolling(200).mean().shift(1))
    ).astype(float)
    df["dist_high_20"] = _safe_div(close, high.rolling(20).max()) - 1
    df["dist_low_20"] = _safe_div(close, low.rolling(20).min()) - 1
    df["dist_high_60"] = _safe_div(close, high.rolling(60).max()) - 1
    df["dist_low_60"] = _safe_div(close, low.rolling(60).min()) - 1
    df["dist_ath"] = _safe_div(close, close.cummax()) - 1
    df["drawdown_30"] = _safe_div(close, close.rolling(30).max()) - 1
    df["drawdown_90"] = _safe_div(close, close.rolling(90).max()) - 1
    df["consecutive_up"] = _consecutive(close, 1)
    df["consecutive_down"] = _consecutive(close, -1)

    # ---------- oscillators ----------
    for n in (6, 14, 28):
        df[f"rsi_{n}"] = _rsi(close, n)
    df["rsi_14_change_3d"] = df["rsi_14"] - df["rsi_14"].shift(3)
    df["rsi_overbought"] = (df["rsi_14"] >= 70).astype(float)
    df["rsi_oversold"] = (df["rsi_14"] <= 30).astype(float)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_raw = ema12 - ema26
    macd_signal_raw = macd_raw.ewm(span=9, adjust=False).mean()
    df["macd_norm"] = _safe_div(macd_raw, close)
    df["macd_signal_norm"] = _safe_div(macd_signal_raw, close)
    df["macd_hist_norm"] = df["macd_norm"] - df["macd_signal_norm"]
    df["macd_hist_change"] = df["macd_hist_norm"] - df["macd_hist_norm"].shift(1)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_pctb_20"] = _safe_div(close - (bb_mid - 2 * bb_std), 4 * bb_std)
    df["bb_width_20"] = _safe_div(4 * bb_std, bb_mid)
    bb_mid_50 = close.rolling(50).mean()
    bb_std_50 = close.rolling(50).std()
    df["bb_pctb_50"] = _safe_div(close - (bb_mid_50 - 2 * bb_std_50), 4 * bb_std_50)
    df["bb_width_50"] = _safe_div(4 * bb_std_50, bb_mid_50)
    low14, high14 = low.rolling(14).min(), high.rolling(14).max()
    df["stoch_k_14"] = 100 * _safe_div(close - low14, high14 - low14)
    df["stoch_d_14"] = df["stoch_k_14"].rolling(3).mean()
    df["williams_r_14"] = -100 * _safe_div(high14 - close, high14 - low14)
    adx, plus_di, minus_di = _adx(df, 14)
    df["adx_14"] = adx
    df["di_plus_14"] = plus_di
    df["di_minus_14"] = minus_di
    df["di_spread_14"] = plus_di - minus_di
    df["mfi_14"] = _mfi(df, 14)

    # ---------- volume / flow ----------
    df["volume_change_1d"] = volume.pct_change(fill_method=None)
    df["volume_z_20"] = _safe_div(volume - volume.rolling(20).mean(), volume.rolling(20).std())
    df["volume_sma_ratio_5_20"] = _safe_div(volume.rolling(5).mean(), volume.rolling(20).mean()) - 1
    quote_volume = pd.to_numeric(df.get("quote_volume"), errors="coerce") if "quote_volume" in df else volume
    trades = pd.to_numeric(df.get("trades"), errors="coerce") if "trades" in df else pd.Series(np.nan, index=df.index)
    df["quote_volume_z_20"] = _safe_div(quote_volume - quote_volume.rolling(20).mean(), quote_volume.rolling(20).std())
    df["trades_z_20"] = _safe_div(trades - trades.rolling(20).mean(), trades.rolling(20).std())
    avg_trade_size = _safe_div(quote_volume, trades)
    df["avg_trade_size_z_20"] = _safe_div(avg_trade_size - avg_trade_size.rolling(20).mean(), avg_trade_size.rolling(20).std())
    taker_buy = pd.to_numeric(df.get("taker_buy_base"), errors="coerce") if "taker_buy_base" in df else pd.Series(np.nan, index=df.index)
    df["taker_buy_ratio"] = _safe_div(taker_buy, volume)
    df["taker_buy_ratio_ma_7"] = df["taker_buy_ratio"].rolling(7).mean()
    obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    for n in (7, 30):
        df[f"obv_slope_{n}"] = _safe_div(obv - obv.shift(n), volume.rolling(n).sum())
    typical = (high + low + close) / 3
    df["vwap_dev_20"] = _safe_div(close, (typical * volume).rolling(20).sum() / volume.rolling(20).sum()) - 1

    # ---------- calendar ----------
    dates = df["date"]
    df["dow"] = dates.dt.dayofweek.astype(float)
    df["month"] = dates.dt.month.astype(float)
    df["quarter"] = dates.dt.quarter.astype(float)
    df["dom"] = dates.dt.day.astype(float)
    df["is_weekend"] = (dates.dt.dayofweek >= 5).astype(float)
    df["is_month_end"] = dates.dt.is_month_end.astype(float)
    df["is_quarter_end"] = dates.dt.is_quarter_end.astype(float)
    df["days_to_month_end"] = (dates.dt.days_in_month - dates.dt.day).astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ---------- cross-asset ----------
    btc_close = _series(df, "btc_close")
    ethbtc_close = _series(df, "ethbtc_close")
    paxg_close = _series(df, "paxg_close")
    for n in (1, 5, 20):
        df[f"btc_ret_{n}d"] = btc_close.pct_change(n, fill_method=None)
        df[f"ethbtc_ret_{n}d"] = ethbtc_close.pct_change(n, fill_method=None)
        df[f"paxg_ret_{n}d"] = paxg_close.pct_change(n, fill_method=None)
    df["btc_vol_30d"] = np.log(btc_close.clip(lower=1e-12)).diff().rolling(30, min_periods=10).std()
    df["ethbtc_sma_ratio_20"] = _safe_div(ethbtc_close, ethbtc_close.rolling(20).mean()) - 1
    df["ethbtc_sma_ratio_50"] = _safe_div(ethbtc_close, ethbtc_close.rolling(50).mean()) - 1
    df["ethbtc_trend"] = (ethbtc_close > ethbtc_close.rolling(50).mean()).astype(float)
    df["ethbtc_vol_30d"] = np.log(ethbtc_close.clip(lower=1e-12)).diff().rolling(30, min_periods=10).std()

    # ---------- macro ----------
    spx = _series(df, "spx")
    nasdaq = _series(df, "nasdaq")
    dxy = _series(df, "dxy")
    us10y = _series(df, "us10y")
    vix = _series(df, "vix")
    hy = _series(df, "hy_spread")
    yc = _series(df, "yield_curve")
    for n in (1, 5, 20):
        df[f"spx_ret_{n}d"] = spx.pct_change(n, fill_method=None)
        df[f"nasdaq_ret_{n}d"] = nasdaq.pct_change(n, fill_method=None)
        df[f"dxy_ret_{n}d"] = dxy.pct_change(n, fill_method=None)
    df["spx_vol_20d"] = spx.pct_change(fill_method=None).rolling(20, min_periods=10).std()
    df["nasdaq_vol_20d"] = nasdaq.pct_change(fill_method=None).rolling(20, min_periods=10).std()
    df["us10y_level"] = us10y
    df["us10y_change_1d"] = us10y.diff()
    df["us10y_change_5d"] = us10y.diff(5)
    df["vix_level"] = vix
    df["vix_change_1d"] = vix.diff()
    df["vix_percentile_252"] = _rolling_percentile(vix, 252, 60)
    df["hy_spread_level"] = hy
    df["hy_spread_change_1d"] = hy.diff()
    df["hy_spread_change_5d"] = hy.diff(5)
    df["yield_curve_level"] = yc
    df["yield_curve_change_1d"] = yc.diff()
    df["risk_on_composite"] = (
        df["spx_ret_5d"].fillna(0)
        + df["nasdaq_ret_5d"].fillna(0)
        - df["dxy_ret_5d"].fillna(0)
        - df["us10y_change_5d"].fillna(0) / 10
    )

    # ---------- crypto sentiment ----------
    fng = _series(df, "fng_value")
    df["fng_level"] = fng
    df["fng_change_1d"] = fng.diff()
    df["fng_change_7d"] = fng.diff(7)
    df["fng_percentile_252"] = _rolling_percentile(fng, 252, 60)
    df["fng_extreme"] = (fng - 50).abs() / 50
    funding = _series(df, "funding_mean")
    df["funding_level"] = funding * 10_000  # basis points per 8h
    df["funding_change_1d"] = funding.diff() * 10_000
    df["funding_sum_7d"] = funding.rolling(7).sum() * 10_000
    df["funding_z_30"] = _safe_div(funding - funding.rolling(30).mean(), funding.rolling(30).std())
    df["funding_positive_days_7"] = (funding > 0).rolling(7).sum()
    df["funding_last"] = _series(df, "funding_last") * 10_000

    # ---------- on-chain / attention (lagged one day to avoid same-day lookahead) ----------
    tvl = _series(df, "tvl").shift(1)
    stable = _series(df, "stablecoin_usd").shift(1)
    wiki = _series(df, "wiki_views").shift(1)
    df["tvl_log_change_1d"] = np.log(tvl.clip(lower=1e-12)).diff()
    df["tvl_log_change_7d"] = np.log(tvl.clip(lower=1e-12)).diff(7)
    df["tvl_log_change_30d"] = np.log(tvl.clip(lower=1e-12)).diff(30)
    df["tvl_z_90"] = _safe_div(tvl - tvl.rolling(90).mean(), tvl.rolling(90).std())
    df["stablecoin_log_change_1d"] = np.log(stable.clip(lower=1e-12)).diff()
    df["stablecoin_log_change_7d"] = np.log(stable.clip(lower=1e-12)).diff(7)
    df["stablecoin_log_change_30d"] = np.log(stable.clip(lower=1e-12)).diff(30)
    df["stablecoin_z_90"] = _safe_div(stable - stable.rolling(90).mean(), stable.rolling(90).std())
    df["tvl_to_stablecoin"] = _safe_div(tvl, stable)
    df["wiki_log_change_1d"] = np.log(wiki.clip(lower=1e-12)).diff()
    df["wiki_log_change_7d"] = np.log(wiki.clip(lower=1e-12)).diff(7)
    df["wiki_z_30"] = _safe_div(wiki - wiki.rolling(30).mean(), wiki.rolling(30).std())
    df["wiki_z_90"] = _safe_div(wiki - wiki.rolling(90).mean(), wiki.rolling(90).std())

    if "news_tone" in df.columns:
        tone = _series(df, "news_tone").shift(1)
        df["news_tone_level"] = tone
        df["news_tone_change_1d"] = tone.diff()
        df["news_tone_ma_7"] = tone.rolling(7).mean()

    # ---------- interactions ----------
    df["rsi14_x_trend"] = (df["rsi_14"] - 50) / 50 * df["above_sma_200"].fillna(0)
    df["fng_x_vol30"] = (fng - 50) / 50 * df["vol_30d"].fillna(0)
    df["funding_x_ret5"] = funding * 10_000 * df["ret_5d"].fillna(0)
    df["dxy_x_btc"] = df["dxy_ret_1d"].fillna(0) * df["btc_ret_1d"].fillna(0)
    df["ethbtc_x_btc"] = df["ethbtc_ret_1d"].fillna(0) * df["btc_ret_1d"].fillna(0)
    df["vol_x_trend"] = df["vol_30d"].fillna(0) * df["above_sma_200"].fillna(0)

    # ---------- targets ----------
    df["next_close"] = close.shift(-1)
    df["target_ret"] = df["next_close"] / close - 1
    df["target_up"] = (df["target_ret"] > 0).astype(float)
    df.loc[df["target_ret"].isna(), "target_up"] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def feature_columns(df: pd.DataFrame, max_missing: float = 0.35) -> list[str]:
    """Select numeric, non-constant engineered features with adequate coverage."""
    cols: list[str] = []
    n = len(df)
    for col in df.columns:
        if col in RAW_COLUMNS:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        non_null = int(df[col].notna().sum())
        if n == 0 or non_null / n < (1 - max_missing):
            continue
        if df[col].nunique(dropna=True) < 3:
            continue
        cols.append(col)
    return cols
