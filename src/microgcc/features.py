from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays
except ImportError:  # pragma: no cover - exercised only without optional deps
    holidays = None


LAG_WEEKS = [1, 7, 30]
ROLLING_WINDOWS = [4, 8]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["day_of_week"] = out["ds"].dt.dayofweek
    out["month"] = out["ds"].dt.month
    out["year"] = out["ds"].dt.year
    out["weekofyear"] = out["ds"].dt.isocalendar().week.astype(int)
    out["is_holiday_week"] = _holiday_week_flag(out["ds"])
    return out


def _holiday_week_flag(dates: pd.Series) -> pd.Series:
    if holidays is None:
        return pd.Series(0, index=dates.index, dtype=int)
    years = sorted(set(dates.dt.year.tolist()))
    us_holidays = holidays.country_holidays("US", years=years)
    flags = []
    for ds in dates:
        week = pd.date_range(ds - pd.Timedelta(days=6), ds, freq="D")
        flags.append(int(any(day.date() in us_holidays for day in week)))
    return pd.Series(flags, index=dates.index, dtype=int)


def make_supervised_features(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    out = add_calendar_features(df.sort_values(["state", "ds"]).copy())
    out["log_y"] = np.log1p(out["y"])
    for lag in LAG_WEEKS:
        out[f"lag_{lag}"] = out.groupby("state")["log_y"].shift(lag)
    for window in ROLLING_WINDOWS:
        shifted = out.groupby("state")["log_y"].shift(1)
        out[f"rolling_mean_{window}"] = shifted.groupby(out["state"]).rolling(window).mean().reset_index(level=0, drop=True)
        out[f"rolling_std_{window}"] = shifted.groupby(out["state"]).rolling(window).std().reset_index(level=0, drop=True)
    out["state_code"] = out["state"].astype("category").cat.codes
    feature_cols = feature_columns()
    if include_target:
        return out.dropna(subset=feature_cols + ["log_y"]).reset_index(drop=True)
    return out


def feature_columns() -> list[str]:
    return [
        "state_code",
        "lag_1",
        "lag_7",
        "lag_30",
        "rolling_mean_4",
        "rolling_std_4",
        "rolling_mean_8",
        "rolling_std_8",
        "day_of_week",
        "month",
        "year",
        "weekofyear",
        "is_holiday_week",
        "was_missing",
    ]

