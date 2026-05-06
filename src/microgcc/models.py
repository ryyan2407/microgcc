from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import WEEKLY_FREQ
from .features import feature_columns, make_supervised_features


class Forecaster(Protocol):
    name: str

    def fit(self, df: pd.DataFrame) -> "Forecaster":
        ...

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        ...

    def save(self, path: Path) -> None:
        ...


@dataclass
class NaiveSeasonalForecaster:
    name: str = "seasonal_naive"
    seasonal_period: int = 52

    def fit(self, df: pd.DataFrame) -> "NaiveSeasonalForecaster":
        return self

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        rows = []
        for state, state_df in history.groupby("state"):
            state_df = state_df.sort_values("ds")
            last_date = state_df["ds"].max()
            values = state_df["y"].to_numpy()
            for step in range(1, horizon + 1):
                idx = max(len(values) - self.seasonal_period + step - 1, 0)
                pred = values[idx] if idx < len(values) else values[-1]
                rows.append({"state": state, "ds": last_date + pd.Timedelta(weeks=step), "yhat": max(float(pred), 0.0), "model": self.name})
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)


class SarimaForecaster:
    name = "sarima"

    def __init__(self) -> None:
        self.models: dict[str, object] = {}

    def fit(self, df: pd.DataFrame) -> "SarimaForecaster":
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as exc:
            raise RuntimeError("statsmodels is required for ARIMA/SARIMA") from exc
        for state, state_df in df.groupby("state"):
            ordered = state_df.sort_values("ds")
            y = pd.Series(
                np.log1p(ordered["y"].astype(float).to_numpy()),
                index=pd.DatetimeIndex(ordered["ds"], freq=WEEKLY_FREQ),
            )
            model = ARIMA(y, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.models[state] = model.fit(method_kwargs={"maxiter": 25})
        return self

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        rows = []
        for state, result in self.models.items():
            last_date = history.loc[history["state"] == state, "ds"].max()
            pred = np.expm1(result.forecast(horizon))
            for step, value in enumerate(pred, start=1):
                rows.append({"state": state, "ds": last_date + pd.Timedelta(weeks=step), "yhat": max(float(value), 0.0), "model": self.name})
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)


class ExponentialSmoothingForecaster:
    name = "ets_holt_winters"

    def __init__(self) -> None:
        self.models: dict[str, object] = {}

    def fit(self, df: pd.DataFrame) -> "ExponentialSmoothingForecaster":
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError as exc:
            raise RuntimeError("statsmodels is required for ETS/Holt-Winters") from exc
        for state, state_df in df.groupby("state"):
            ordered = state_df.sort_values("ds")
            y = pd.Series(
                ordered["y"].astype(float).to_numpy(),
                index=pd.DatetimeIndex(ordered["ds"], freq=WEEKLY_FREQ),
            )
            try:
                model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=52, initialization_method="estimated")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.models[state] = model.fit(optimized=True)
            except Exception:
                model = ExponentialSmoothing(y, trend="add", seasonal=None, initialization_method="estimated")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.models[state] = model.fit(optimized=True)
        return self

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        rows = []
        for state, result in self.models.items():
            last_date = history.loc[history["state"] == state, "ds"].max()
            pred = result.forecast(horizon)
            for step, value in enumerate(pred, start=1):
                rows.append({"state": state, "ds": last_date + pd.Timedelta(weeks=step), "yhat": max(float(value), 0.0), "model": self.name})
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)


class ProphetForecaster:
    name = "prophet"

    def __init__(self) -> None:
        self.models: dict[str, object] = {}

    def fit(self, df: pd.DataFrame) -> "ProphetForecaster":
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise RuntimeError("prophet is required for ProphetForecaster") from exc
        for state, state_df in df.groupby("state"):
            train = state_df[["ds", "y"]].sort_values("ds").copy()
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model.add_country_holidays(country_name="US")
            model.fit(train)
            self.models[state] = model
        return self

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        rows = []
        for state, model in self.models.items():
            last_date = history.loc[history["state"] == state, "ds"].max()
            future = pd.DataFrame({"ds": pd.date_range(last_date + pd.Timedelta(weeks=1), periods=horizon, freq=WEEKLY_FREQ)})
            forecast = model.predict(future)
            for _, row in forecast.iterrows():
                rows.append({"state": state, "ds": row["ds"], "yhat": max(float(row["yhat"]), 0.0), "model": self.name})
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)


class GlobalLagForecaster:
    name = "global_lag_model"

    def __init__(self, model: object | None = None, name: str | None = None) -> None:
        self.model = model
        if name is not None:
            self.name = name
        self.state_codes: dict[str, int] = {}

    def fit(self, df: pd.DataFrame) -> "GlobalLagForecaster":
        feats = make_supervised_features(df)
        self.state_codes = dict(zip(feats["state"], feats["state_code"]))
        if self.model is None:
            self.model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, random_state=42)
        self.model.fit(feats[feature_columns()], feats["log_y"])
        return self

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        simulated = history.copy().sort_values(["state", "ds"])
        rows = []
        for _ in range(horizon):
            next_date = simulated["ds"].max() + pd.Timedelta(weeks=1)
            future_rows = [{"state": state, "ds": next_date, "y": np.nan, "was_missing": 0} for state in sorted(simulated["state"].unique())]
            candidate = pd.concat([simulated, pd.DataFrame(future_rows)], ignore_index=True)
            feats = make_supervised_features(candidate, include_target=False)
            current = feats[feats["ds"] == next_date].copy()
            current["state_code"] = current["state"].map(self.state_codes).fillna(-1).astype(int)
            preds = np.expm1(self.model.predict(current[feature_columns()]))
            new_history_rows = []
            for (_, row), pred in zip(current.iterrows(), preds):
                yhat = max(float(pred), 0.0)
                rows.append({"state": row["state"], "ds": row["ds"], "yhat": yhat, "model": self.name})
                new_history_rows.append({"state": row["state"], "ds": row["ds"], "y": yhat, "was_missing": 0})
            simulated = pd.concat([simulated, pd.DataFrame(new_history_rows)], ignore_index=True)
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)


class XGBoostForecaster(GlobalLagForecaster):
    name = "xgboost"

    def __init__(self) -> None:
        try:
            from xgboost import XGBRegressor

            model = XGBRegressor(
                n_estimators=350,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=2,
            )
            super().__init__(model=model, name="xgboost")
        except ImportError:
            super().__init__(
                model=HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, random_state=42),
                name="xgboost_fallback_histgb",
            )


class RandomForestLagForecaster(GlobalLagForecaster):
    def __init__(self) -> None:
        super().__init__(
            model=RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2, n_jobs=-1),
            name="random_forest_lag",
        )


class HistGradientBoostingLagForecaster(GlobalLagForecaster):
    def __init__(self) -> None:
        super().__init__(
            model=HistGradientBoostingRegressor(max_iter=350, learning_rate=0.04, l2_regularization=0.05, random_state=42),
            name="hist_gradient_boosting_lag",
        )


class RidgeLagForecaster(GlobalLagForecaster):
    def __init__(self) -> None:
        super().__init__(model=make_pipeline(StandardScaler(), Ridge(alpha=1.0)), name="ridge_lag")


class SequenceForecaster:
    name = "sequence_model"

    def __init__(self, sequence_length: int = 52, epochs: int = 25) -> None:
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.model = None
        self.scaler = StandardScaler()

    def _make_sequences(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for _, state_df in df.groupby("state"):
            vals = np.log1p(state_df.sort_values("ds")["y"].to_numpy(dtype=float))
            for idx in range(self.sequence_length, len(vals)):
                xs.append(vals[idx - self.sequence_length:idx].reshape(-1, 1))
                ys.append(vals[idx])
        return np.asarray(xs), np.asarray(ys)

    def _fit_fallback(self, df: pd.DataFrame, fallback_name: str) -> "SequenceForecaster":
        self.name = fallback_name
        feats = make_supervised_features(df)
        self.model = RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1)
        self.model.fit(feats[feature_columns()], feats["log_y"])
        return self

    def _build_model(self, tf, input_shape: tuple[int, int]):
        raise NotImplementedError

    def fit(self, df: pd.DataFrame) -> "SequenceForecaster":
        try:
            import tensorflow as tf
        except ImportError:
            return self._fit_fallback(df, f"{self.name}_fallback_random_forest")
        x, y = self._make_sequences(df)
        if len(x) == 0:
            raise RuntimeError(f"Not enough history for {self.name} sequences")
        x2 = self.scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        model = self._build_model(tf, (x.shape[1], x.shape[2]))
        model.compile(optimizer="adam", loss="mse")
        model.fit(
            x2,
            y,
            epochs=self.epochs,
            batch_size=32,
            validation_split=0.15,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)],
        )
        self.model = model
        return self

    def predict(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if self.name.endswith("_fallback_random_forest"):
            fallback = GlobalLagForecaster()
            fallback.model = self.model
            feats = make_supervised_features(history)
            fallback.state_codes = {state: code for state, code in zip(feats["state"], feats["state_code"])}
            fallback.name = self.name
            return fallback.predict(history, horizon)
        rows = []
        for state, state_df in history.groupby("state"):
            values = list(np.log1p(state_df.sort_values("ds")["y"].to_numpy(dtype=float)))
            last_date = state_df["ds"].max()
            for step in range(1, horizon + 1):
                seq = np.asarray(values[-self.sequence_length:]).reshape(1, self.sequence_length, 1)
                seq = self.scaler.transform(seq.reshape(-1, 1)).reshape(seq.shape)
                pred_log = float(self.model.predict(seq, verbose=0)[0][0])
                values.append(pred_log)
                rows.append({"state": state, "ds": last_date + pd.Timedelta(weeks=step), "yhat": max(float(np.expm1(pred_log)), 0.0), "model": self.name})
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        if not self.name.endswith("_fallback_random_forest"):
            path.mkdir(parents=True, exist_ok=True)
            self.model.save(path / "keras_model.keras")
            joblib.dump(
                {"sequence_length": self.sequence_length, "epochs": self.epochs, "scaler": self.scaler, "name": self.name},
                path / "metadata.joblib",
            )
        else:
            joblib.dump(self, path)


class LstmForecaster(SequenceForecaster):
    name = "lstm"

    def _build_model(self, tf, input_shape: tuple[int, int]):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(48),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(1),
        ])


class SinusoidalPositionEncoding:
    def __init__(self, sequence_length: int, d_model: int) -> None:
        self.sequence_length = sequence_length
        self.d_model = d_model

    def matrix(self) -> np.ndarray:
        positions = np.arange(self.sequence_length)[:, np.newaxis]
        dims = np.arange(self.d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(self.d_model))
        angles = positions * angle_rates
        pe = np.zeros((self.sequence_length, self.d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])
        return pe[np.newaxis, :, :]


class TransformerForecaster(SequenceForecaster):
    def __init__(self, use_positional_encoding: bool, sequence_length: int = 52, epochs: int = 18) -> None:
        self.use_positional_encoding = use_positional_encoding
        name = "transformer_with_pe" if use_positional_encoding else "transformer_without_pe"
        super().__init__(sequence_length=sequence_length, epochs=epochs)
        self.name = name

    def _build_model(self, tf, input_shape: tuple[int, int]):
        d_model = 32
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Dense(d_model)(inputs)
        if self.use_positional_encoding:
            pe = tf.constant(SinusoidalPositionEncoding(input_shape[0], d_model).matrix())
            x = tf.keras.layers.Lambda(lambda values: values + pe, name="sinusoidal_positional_encoding")(x)
        attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=8, dropout=0.1)(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attention)
        ff = tf.keras.layers.Dense(64, activation="relu")(x)
        ff = tf.keras.layers.Dropout(0.1)(ff)
        ff = tf.keras.layers.Dense(d_model)(ff)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)


def mandatory_forecasters() -> list[Forecaster]:
    return [SarimaForecaster(), ProphetForecaster(), XGBoostForecaster(), LstmForecaster()]


def all_forecasters() -> list[Forecaster]:
    return [
        NaiveSeasonalForecaster(),
        SarimaForecaster(),
        ExponentialSmoothingForecaster(),
        ProphetForecaster(),
        XGBoostForecaster(),
        RandomForestLagForecaster(),
        HistGradientBoostingLagForecaster(),
        RidgeLagForecaster(),
        LstmForecaster(),
        TransformerForecaster(use_positional_encoding=True),
        TransformerForecaster(use_positional_encoding=False),
    ]
