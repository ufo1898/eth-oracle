from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from app import config
from app.features import feature_columns

log = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "hgb_max_iter": 320,
    "hgb_learning_rate": 0.03,
    "hgb_max_leaf_nodes": 15,
    "hgb_l2": 1.0,
    "rf_estimators": 360,
    "rf_max_depth": 6,
    "rf_min_samples_leaf": 10,
    "et_estimators": 360,
    "et_max_depth": 8,
    "et_min_samples_leaf": 8,
    "lr_c": 0.1,
}

PROB_COLUMNS = {
    "hgb": "p_hgb",
    "rf": "p_rf",
    "et": "p_et",
    "lr": "p_lr",
}


@dataclass
class ModelBundle:
    feature_cols: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    models: dict[str, Any]
    regressor: Any
    weights: dict[str, float]
    train_mean: pd.Series
    train_std: pd.Series
    importance: pd.Series
    calibrator: Any | None
    version: str
    trained_at: str
    as_of: str
    train_start: str
    train_end: str
    n_samples: int
    val_metrics: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    horizon_days: int = 1

    def info(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "trained_at": self.trained_at,
            "as_of": self.as_of,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "n_samples": self.n_samples,
            "n_features": len(self.feature_cols),
            "weights": self.weights,
            "val_metrics": self.val_metrics,
            "params": self.params,
            "horizon_days": self.horizon_days,
        }


def _new_models(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "hgb": HistGradientBoostingClassifier(
            max_iter=int(params["hgb_max_iter"]),
            learning_rate=float(params["hgb_learning_rate"]),
            max_leaf_nodes=int(params["hgb_max_leaf_nodes"]),
            l2_regularization=float(params["hgb_l2"]),
            min_samples_leaf=20,
            early_stopping=False,
            random_state=42,
        ),
        "rf": RandomForestClassifier(
            n_estimators=int(params["rf_estimators"]),
            max_depth=int(params["rf_max_depth"]),
            min_samples_leaf=int(params["rf_min_samples_leaf"]),
            max_features=0.5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "et": ExtraTreesClassifier(
            n_estimators=int(params["et_estimators"]),
            max_depth=int(params["et_max_depth"]),
            min_samples_leaf=int(params["et_min_samples_leaf"]),
            max_features=0.5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "lr": LogisticRegression(
            C=float(params["lr_c"]),
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    }


def _new_regressor(params: dict[str, Any]) -> Any:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=int(params["hgb_max_iter"]),
        learning_rate=float(params["hgb_learning_rate"]),
        max_leaf_nodes=int(params["hgb_max_leaf_nodes"]),
        l2_regularization=float(params["hgb_l2"]),
        min_samples_leaf=20,
        early_stopping=False,
        random_state=42,
    )


def _prob_up(model: Any, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return np.asarray(proba[:, classes.index(1)], dtype=float)
    return np.zeros(len(X), dtype=float)


def _fit_transform(imputer: SimpleImputer, scaler: StandardScaler, X_raw: pd.DataFrame) -> np.ndarray:
    X = imputer.transform(X_raw)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return scaler.transform(X)


def select_train_features(train_df: pd.DataFrame, candidates: list[str]) -> list[str]:
    selected: list[str] = []
    n = len(train_df)
    if n == 0:
        return selected
    for col in candidates:
        series = pd.to_numeric(train_df[col], errors="coerce")
        non_null = int(series.notna().sum())
        if non_null / n < 0.25:
            continue
        if series.nunique(dropna=True) < 3:
            continue
        selected.append(col)
    return selected


def train_bundle(
    df: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
    params: dict[str, Any] | None = None,
    validation_days: int = 120,
    candidates: list[str] | None = None,
    weight_override: dict[str, float] | None = None,
    target_col: str = "target_up",
    target_ret_col: str = "target_ret",
    horizon_days: int = 1,
) -> ModelBundle:
    """Train on information available strictly before the prediction day.

    `as_of` is the feature day used to predict a future close `horizon_days`
    later. Training can only use rows whose target is known, i.e. dates
    <= as_of - horizon_days.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    horizon_days = max(int(horizon_days), 1)
    as_of_ts = pd.Timestamp(as_of).normalize()
    train_end = as_of_ts - pd.Timedelta(days=horizon_days)
    train_df = df[(df["date"] <= train_end) & df[target_col].notna()].copy()
    if len(train_df) < config.MIN_TRAIN_ROWS:
        raise ValueError(f"Not enough training rows: {len(train_df)} < {config.MIN_TRAIN_ROWS}")

    candidates = candidates or feature_columns(df)
    feature_cols = select_train_features(train_df, candidates)
    if len(feature_cols) < 5:
        raise ValueError("Too few usable features after coverage filtering")

    X_raw = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = train_df[target_col].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Training data contains only one target class")
    y_ret = train_df[target_ret_col].astype(float).to_numpy()

    val_n = min(validation_days, max(60, len(train_df) // 4))
    fit_n = len(train_df) - val_n
    if fit_n < 250:
        val_n = max(40, len(train_df) // 5)
        fit_n = len(train_df) - val_n
    if fit_n < config.MIN_TRAIN_ROWS // 2:
        raise ValueError(f"Not enough fit rows after validation split: {fit_n}")

    fit_raw = X_raw.iloc[:fit_n]
    val_raw = X_raw.iloc[fit_n:]
    y_fit, y_val = y[:fit_n], y[fit_n:]
    ret_fit, ret_val = y_ret[:fit_n], y_ret[fit_n:]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputer.fit(fit_raw)
    X_fit_raw = imputer.transform(fit_raw)
    X_fit_raw = np.nan_to_num(X_fit_raw, nan=0.0, posinf=0.0, neginf=0.0)
    scaler.fit(X_fit_raw)
    X_fit = scaler.transform(X_fit_raw)
    X_val = _fit_transform(imputer, scaler, val_raw)

    models = _new_models(params)
    per_model_acc: dict[str, float] = {}
    val_probs: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(X_fit, y_fit)
        prob = np.clip(_prob_up(model, X_val), 1e-6, 1 - 1e-6)
        val_probs[name] = prob
        per_model_acc[name] = float(accuracy_score(y_val, (prob >= 0.5).astype(int)))

    edges = {name: max(acc - 0.5, 0.0) for name, acc in per_model_acc.items()}
    if sum(edges.values()) <= 0:
        weights = {name: 1 / len(models) for name in models}
    else:
        raw_weights = {name: 0.08 + edge for name, edge in edges.items()}
        total = sum(raw_weights.values())
        weights = {name: value / total for name, value in raw_weights.items()}

    if weight_override:
        override = {name: max(float(weight_override.get(name, 0.0)), 0.0) for name in models}
        if sum(override.values()) > 0:
            total = sum(override.values())
            weights = {name: value / total for name, value in override.items()}

    ensemble_val = sum(weights[name] * val_probs[name] for name in models)
    calibrator: LogisticRegression | None = None
    if len(np.unique(y_val)) > 1 and len(y_val) >= 50:
        calibrator = LogisticRegression(C=1.0, solver="liblinear", random_state=42)
        calibrator.fit(ensemble_val.reshape(-1, 1), y_val)
        calibrated_val = calibrator.predict_proba(ensemble_val.reshape(-1, 1))[:, 1]
    else:
        calibrated_val = ensemble_val

    # Refit every model on all available training rows for the live prediction.
    X_full = _fit_transform(imputer, scaler, X_raw)
    for model in models.values():
        model.fit(X_full, y)
    regressor = _new_regressor(params)
    regressor.fit(X_full, y_ret)
    reg_pred = regressor.predict(_fit_transform(imputer, scaler, val_raw))
    reg_mae = float(mean_absolute_error(ret_val, reg_pred)) if len(ret_val) else float("nan")

    importance = _feature_importance(models, feature_cols)
    X_full_raw = np.nan_to_num(imputer.transform(X_raw), nan=0.0, posinf=0.0, neginf=0.0)
    train_mean = pd.Series(X_full_raw.mean(axis=0), index=feature_cols)
    train_std = pd.Series(X_full_raw.std(axis=0), index=feature_cols).replace(0, np.nan)

    val_metrics = {
        "accuracy": float(accuracy_score(y_val, (calibrated_val >= 0.5).astype(int))),
        "brier": float(brier_score_loss(y_val, np.clip(calibrated_val, 1e-6, 1 - 1e-6))),
        "n": int(len(y_val)),
        "per_model_accuracy": {k: round(v, 4) for k, v in per_model_acc.items()},
        "reg_mae": reg_mae,
        "base_up_rate": float(np.mean(y_val)),
    }
    version = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
    if horizon_days != 1:
        version = f"{version}-h{horizon_days}"
    bundle = ModelBundle(
        feature_cols=feature_cols,
        imputer=imputer,
        scaler=scaler,
        models=models,
        regressor=regressor,
        weights=weights,
        train_mean=train_mean,
        train_std=train_std,
        importance=importance,
        calibrator=calibrator,
        version=version,
        trained_at=datetime.now(timezone.utc).isoformat(),
        as_of=as_of_ts.strftime("%Y-%m-%d"),
        train_start=str(train_df["date"].min().date()),
        train_end=str(train_df["date"].max().date()),
        n_samples=int(len(train_df)),
        val_metrics=val_metrics,
        params=params,
        horizon_days=horizon_days,
    )
    log.info(
        "Trained %s (horizon=%sd): rows=%s features=%s val_acc=%.3f weights=%s",
        version,
        horizon_days,
        len(train_df),
        len(feature_cols),
        val_metrics["accuracy"],
        {k: round(v, 3) for k, v in weights.items()},
    )
    return bundle


def _feature_importance(models: dict[str, Any], feature_cols: list[str]) -> pd.Series:
    importances: list[np.ndarray] = []
    for name in ("rf", "et"):
        model = models.get(name)
        values = getattr(model, "feature_importances_", None)
        if values is not None and len(values) == len(feature_cols):
            importances.append(np.asarray(values, dtype=float))
    if importances:
        raw = np.mean(importances, axis=0)
    else:
        coef = getattr(models.get("lr"), "coef_", None)
        raw = np.abs(coef[0]) if coef is not None else np.ones(len(feature_cols))
    total = float(np.sum(raw))
    if not np.isfinite(total) or total <= 0:
        raw = np.ones(len(feature_cols))
        total = float(len(feature_cols))
    return pd.Series(raw / total, index=feature_cols).sort_values(ascending=False)


def predict_one(bundle: ModelBundle, feature_row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    row = pd.Series(feature_row, dtype=object) if isinstance(feature_row, dict) else feature_row.copy()
    X_raw = pd.DataFrame([{col: pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0] for col in bundle.feature_cols}])
    X = _fit_transform(bundle.imputer, bundle.scaler, X_raw)

    probs: dict[str, float] = {}
    for name, model in bundle.models.items():
        probs[name] = float(np.clip(_prob_up(model, X)[0], 1e-6, 1 - 1e-6))
    ensemble = sum(bundle.weights[name] * probs[name] for name in bundle.models)
    if bundle.calibrator is not None:
        calibrated = float(bundle.calibrator.predict_proba(np.array([[ensemble]]))[0, 1])
    else:
        calibrated = float(ensemble)
    calibrated = float(np.clip(calibrated, 0.01, 0.99))

    pred_ret = float(bundle.regressor.predict(X)[0])
    pred_ret = float(np.clip(pred_ret, -0.25, 0.25))
    if calibrated >= 0.5 and pred_ret < 0:
        pred_ret = abs(pred_ret) * 0.25
    elif calibrated < 0.5 and pred_ret > 0:
        pred_ret = -abs(pred_ret) * 0.25

    close = float(pd.to_numeric(pd.Series([row.get("close")]), errors="coerce").iloc[0])
    predicted_close = close * (1 + pred_ret)
    return {
        "prob_up": round(calibrated, 5),
        "prob_raw": round(float(ensemble), 5),
        "direction": "UP" if calibrated >= 0.5 else "DOWN",
        "confidence": round(max(calibrated, 1 - calibrated), 5),
        "predicted_return": round(pred_ret, 6),
        "predicted_close": round(predicted_close, 4),
        **{PROB_COLUMNS[name]: round(probs[name], 5) for name in probs},
    }


def top_drivers(bundle: ModelBundle, feature_row: pd.Series, k: int = 6) -> list[dict[str, Any]]:
    drivers: list[dict[str, Any]] = []
    for feature, importance in bundle.importance.head(max(k * 3, 12)).items():
        try:
            value = pd.to_numeric(pd.Series([feature_row.get(feature)]), errors="coerce").iloc[0]
            mean = float(bundle.train_mean.get(feature, np.nan))
            std = float(bundle.train_std.get(feature, np.nan))
            if not np.isfinite(value) or not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
                continue
            z = float(np.clip((value - mean) / std, -3, 3))
            contribution = float(importance * z)
            drivers.append(
                {
                    "feature": feature,
                    "value": round(float(value), 6),
                    "z": round(z, 2),
                    "importance": round(float(importance), 5),
                    "contribution": round(contribution, 5),
                    "direction": "up" if contribution >= 0 else "down",
                }
            )
        except Exception:  # noqa: BLE001
            continue
    drivers.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return drivers[:k]


def save_bundle(bundle: ModelBundle, path: Path | None = None) -> Path:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = path or config.MODEL_LATEST
    joblib.dump(bundle, target)
    versioned = config.MODEL_DIR / f"model_{bundle.version}.joblib"
    joblib.dump(bundle, versioned)
    return target


def load_bundle(path: Path | None = None) -> ModelBundle | None:
    target = path or config.MODEL_LATEST
    if not target.exists():
        return None
    try:
        return joblib.load(target)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load model bundle: %s", exc)
        return None
