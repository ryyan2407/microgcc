from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import WEEKLY_FREQ


REQUIRED_COLUMNS = {"State", "Date", "Total"}


@dataclass(frozen=True)
class DatasetProfile:
    row_count: int
    state_count: int
    start_date: str
    end_date: str
    week_count: int
    missing_state_weeks: int


def parse_sales_date(value: object) -> pd.Timestamp:
    text = str(value).strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(text, format=fmt)
        except ValueError:
            continue
    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Could not parse date: {value!r}")
    return parsed


def parse_total(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", "").strip()
    if text == "":
        return np.nan
    return float(text)


def load_sales(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = raw.copy()
    df["state"] = df["State"].astype(str).str.strip()
    df["ds"] = df["Date"].map(parse_sales_date)
    df["y"] = df["Total"].map(parse_total)
    df["category"] = df["Category"].astype(str).str.strip() if "Category" in df else "all"

    weekly = (
        df.groupby(["state", "ds"], as_index=False)
        .agg(y=("y", "sum"), category_count=("category", "nunique"))
        .sort_values(["state", "ds"])
    )
    return reindex_weekly(weekly)


def reindex_weekly(df: pd.DataFrame) -> pd.DataFrame:
    min_date = df["ds"].min()
    max_date = df["ds"].max()
    full_dates = pd.date_range(min_date, max_date, freq=WEEKLY_FREQ)
    frames = []
    for state, state_df in df.groupby("state", sort=True):
        state_df = state_df.set_index("ds").sort_index()
        reindexed = state_df.reindex(full_dates)
        reindexed.index.name = "ds"
        reindexed["state"] = state
        reindexed["was_missing"] = reindexed["y"].isna().astype(int)
        median = state_df["y"].median()
        reindexed["y"] = reindexed["y"].ffill().bfill().fillna(median)
        frames.append(reindexed.reset_index())
    out = pd.concat(frames, ignore_index=True)
    return out[["state", "ds", "y", "was_missing"]].sort_values(["state", "ds"]).reset_index(drop=True)


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    return DatasetProfile(
        row_count=len(df),
        state_count=df["state"].nunique(),
        start_date=str(df["ds"].min().date()),
        end_date=str(df["ds"].max().date()),
        week_count=df["ds"].nunique(),
        missing_state_weeks=int(df["was_missing"].sum()),
    )

