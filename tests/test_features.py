from __future__ import annotations

import numpy as np
import pandas as pd

from app.features import RAW_COLUMNS, build_features, feature_columns


def test_feature_columns_exclude_raw_columns(synthetic_features: pd.DataFrame) -> None:
    features = feature_columns(synthetic_features)
    assert features
    assert not set(features) & RAW_COLUMNS
    assert "rsi_14" in features
    assert "atr_14_pct" in features
    assert "btc_ret_1d" in features
    assert "funding_z_30" in features
    assert "fng_level" in features


def test_target_is_next_day_close(synthetic_ohlcv: pd.DataFrame) -> None:
    features = build_features(synthetic_ohlcv)
    expected = synthetic_ohlcv["close"].shift(-1).iloc[-2] / synthetic_ohlcv["close"].iloc[-2] - 1
    assert features["target_ret"].iloc[-2] == expected
    assert np.isnan(features["target_ret"].iloc[-1])
    assert np.isnan(features["target_up"].iloc[-1])
    assert features["target_up"].dropna().isin([0.0, 1.0]).all()


def test_weekly_target_is_seven_days_ahead(synthetic_ohlcv: pd.DataFrame) -> None:
    features = build_features(synthetic_ohlcv)
    index = len(features) - 8
    expected = synthetic_ohlcv["close"].iloc[index + 7] / synthetic_ohlcv["close"].iloc[index] - 1
    assert features["target_ret_7d"].iloc[index] == expected
    assert features["target_up_7d"].iloc[index] == int(expected > 0)
    assert features["target_ret_7d"].iloc[-7:].isna().all()
    assert features["target_up_7d"].iloc[-7:].isna().all()


def test_no_infinite_values_in_features(synthetic_features: pd.DataFrame) -> None:
    numeric = synthetic_features.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy()).any()
    last = synthetic_features.iloc[-1]
    for column in feature_columns(synthetic_features):
        assert pd.api.types.is_number(last[column]) or pd.isna(last[column])


def test_feature_engineering_is_deterministic(synthetic_ohlcv: pd.DataFrame) -> None:
    first = build_features(synthetic_ohlcv)
    second = build_features(synthetic_ohlcv)
    pd.testing.assert_frame_equal(first, second)
