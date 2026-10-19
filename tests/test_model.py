from __future__ import annotations

import pandas as pd

from app import config, model
from app.features import feature_columns


def _as_of(features: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(features["date"].max()).normalize() - pd.Timedelta(days=1)


def test_train_and_predict(synthetic_features: pd.DataFrame) -> None:
    as_of = _as_of(synthetic_features)
    bundle = model.train_bundle(synthetic_features, as_of=as_of)
    assert bundle.n_samples > config.MIN_TRAIN_ROWS
    assert bundle.feature_cols
    assert abs(sum(bundle.weights.values()) - 1.0) < 1e-6
    assert bundle.importance.index.isin(feature_columns(synthetic_features)).all()

    row = synthetic_features[synthetic_features["date"] == as_of].iloc[-1]
    prediction = model.predict_one(bundle, row)
    assert 0.0 <= prediction["prob_up"] <= 1.0
    assert 0.0 <= prediction["prob_raw"] <= 1.0
    assert prediction["direction"] in {"UP", "DOWN"}
    assert prediction["predicted_close"] > 0
    assert abs(prediction["predicted_return"]) < 0.5


def test_train_weekly_bundle_uses_lagged_targets(synthetic_features: pd.DataFrame) -> None:
    as_of = _as_of(synthetic_features)
    bundle = model.train_bundle(
        synthetic_features,
        as_of=as_of,
        target_col="target_up_7d",
        target_ret_col="target_ret_7d",
        horizon_days=7,
    )
    assert bundle.horizon_days == 7
    assert bundle.version.endswith("-h7")
    assert pd.Timestamp(bundle.train_end) <= as_of - pd.Timedelta(days=7)
    assert bundle.n_samples > config.MIN_TRAIN_ROWS


def test_top_drivers_shape(synthetic_features: pd.DataFrame) -> None:
    as_of = _as_of(synthetic_features)
    bundle = model.train_bundle(synthetic_features, as_of=as_of)
    row = synthetic_features[synthetic_features["date"] == as_of].iloc[-1]
    drivers = model.top_drivers(bundle, row, k=6)
    assert 1 <= len(drivers) <= 6
    for driver in drivers:
        assert driver["feature"] in bundle.feature_cols
        assert driver["direction"] in {"up", "down"}


def test_save_and_load_bundle(tmp_path, synthetic_features: pd.DataFrame, monkeypatch) -> None:
    monkeypatch.setattr(config, "MODEL_LATEST", tmp_path / "model_latest.joblib")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    as_of = _as_of(synthetic_features)
    bundle = model.train_bundle(synthetic_features, as_of=as_of)
    path = model.save_bundle(bundle)
    assert path.exists()
    loaded = model.load_bundle()
    assert loaded is not None
    assert loaded.version == bundle.version
    assert loaded.feature_cols == bundle.feature_cols
