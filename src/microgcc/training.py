from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import time

import pandas as pd

from .config import FORECAST_HORIZON, VALIDATION_WEEKS, Paths
from .data import load_sales, profile_dataset
from .metrics import mae, mape, rmse, smape
from .models import Forecaster, NaiveSeasonalForecaster, all_forecasters


MANDATORY_MODEL_NAMES = ["sarima", "prophet", "xgboost", "lstm"]


def split_train_validation(df: pd.DataFrame, validation_weeks: int = VALIDATION_WEEKS) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["ds"].unique())
    if len(dates) <= validation_weeks + 30:
        raise ValueError("Not enough weekly history for the requested validation split")
    validation_dates = set(dates[-validation_weeks:])
    train = df[~df["ds"].isin(validation_dates)].copy()
    validation = df[df["ds"].isin(validation_dates)].copy()
    return train, validation


def evaluate_predictions(validation: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    merged = validation.merge(predictions, on=["state", "ds"], how="left")
    if merged["yhat"].isna().any():
        missing = merged.loc[merged["yhat"].isna(), ["state", "ds"]].head().to_dict("records")
        raise ValueError(f"Missing predictions for validation rows: {missing}")
    rows = []
    for state, state_df in merged.groupby("state"):
        rows.append({
            "state": state,
            "model": state_df["model"].iloc[0],
            "smape": smape(state_df["y"], state_df["yhat"]),
            "mae": mae(state_df["y"], state_df["yhat"]),
            "rmse": rmse(state_df["y"], state_df["yhat"]),
            "mape": mape(state_df["y"], state_df["yhat"]),
        })
    return pd.DataFrame(rows)


def train_all(data_path: str | Path, artifact_dir: str | Path, fast: bool = False) -> dict:
    paths = Paths(Path(artifact_dir))
    paths.ensure()
    df = load_sales(data_path)
    train, validation = split_train_validation(df)

    forecasters: list[Forecaster] = [NaiveSeasonalForecaster()] if fast else all_forecasters()
    all_metrics = []
    fitted: dict[str, Forecaster] = {}
    failures: dict[str, str] = {}
    attempted_models = [getattr(forecaster, "name", forecaster.__class__.__name__) for forecaster in forecasters]

    for forecaster in forecasters:
        requested_name = getattr(forecaster, "name", forecaster.__class__.__name__)
        print(f"[train] starting {requested_name}", flush=True)
        started = time.time()
        try:
            fitted_model = forecaster.fit(train)
            validation_pred = fitted_model.predict(train, len(validation["ds"].unique()))
            metrics_df = evaluate_predictions(validation, validation_pred)
            all_metrics.append(metrics_df)
            fitted[fitted_model.name] = fitted_model
            save_path = paths.models_dir / fitted_model.name
            if save_path.suffix == "":
                save_path = save_path.with_suffix(".joblib")
            fitted_model.save(save_path)
            print(f"[train] finished {requested_name} as {fitted_model.name} in {time.time() - started:.1f}s", flush=True)
        except Exception as exc:
            failures[requested_name] = str(exc)
            print(f"[train] failed {requested_name} in {time.time() - started:.1f}s: {exc}", flush=True)

    if not all_metrics:
        fallback = NaiveSeasonalForecaster().fit(train)
        validation_pred = fallback.predict(train, len(validation["ds"].unique()))
        metrics_df = evaluate_predictions(validation, validation_pred)
        all_metrics.append(metrics_df)
        fitted[fallback.name] = fallback
        fallback.save(paths.models_dir / "seasonal_naive.joblib")
        failures["mandatory_model_warning"] = "Mandatory model dependencies failed; seasonal naive fallback was used."

    metrics = pd.concat(all_metrics, ignore_index=True).sort_values(["state", "smape"])
    metrics.to_csv(paths.metrics_path, index=False)
    best = metrics.loc[metrics.groupby("state")["smape"].idxmin()].sort_values("state")

    final_forecasts = []
    for model_name, states in best.groupby("model")["state"]:
        if model_name in fitted:
            model = fitted[model_name].fit(df)
            preds = model.predict(df, FORECAST_HORIZON)
        else:
            model = NaiveSeasonalForecaster().fit(df)
            preds = model.predict(df, FORECAST_HORIZON)
        final_forecasts.append(preds[preds["state"].isin(states)])
    forecasts = pd.concat(final_forecasts, ignore_index=True).sort_values(["state", "ds"])
    forecasts.to_csv(paths.forecasts_path, index=False)

    profile = profile_dataset(df)
    registry = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "forecast_horizon_weeks": FORECAST_HORIZON,
        "validation_weeks": VALIDATION_WEEKS,
        "last_observed_date": str(df["ds"].max().date()),
        "forecast_start": str(forecasts["ds"].min().date()),
        "forecast_end": str(forecasts["ds"].max().date()),
        "primary_metric": "smape",
        "dataset_profile": profile.__dict__,
        "mandatory_models": MANDATORY_MODEL_NAMES,
        "attempted_models": attempted_models,
        "successful_models": sorted(metrics["model"].unique().tolist()),
        "additional_models": [
            "seasonal_naive",
            "ets_holt_winters",
            "random_forest_lag",
            "hist_gradient_boosting_lag",
            "ridge_lag",
            "transformer_with_pe",
            "transformer_without_pe",
        ],
        "selected_models": dict(zip(best["state"], best["model"])),
        "failures": failures,
    }
    paths.registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    paths.metadata_path.write_text(json.dumps({"profile": profile.__dict__}, indent=2), encoding="utf-8")
    return registry


def load_artifacts(artifact_dir: str | Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    paths = Paths(Path(artifact_dir))
    registry = json.loads(paths.registry_path.read_text(encoding="utf-8"))
    forecasts = pd.read_csv(paths.forecasts_path, parse_dates=["ds"])
    metrics = pd.read_csv(paths.metrics_path)
    return registry, forecasts, metrics
