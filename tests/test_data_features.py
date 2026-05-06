import pandas as pd

from microgcc.data import load_sales, parse_sales_date, parse_total
from microgcc.features import make_supervised_features
from microgcc.training import split_train_validation


def test_parse_day_first_mixed_dates():
    assert parse_sales_date("1/12/2019").date().isoformat() == "2019-12-01"
    assert parse_sales_date("13-02-2022").date().isoformat() == "2022-02-13"


def test_parse_comma_total():
    assert parse_total("  109,574,036 ") == 109574036.0


def test_load_sales_profile_shape():
    df = load_sales("data.csv")
    assert df["state"].nunique() == 43
    assert df["ds"].nunique() == 188
    assert int(df["was_missing"].sum()) == 0


def test_features_and_split_are_leakage_safe():
    df = load_sales("data.csv")
    train, validation = split_train_validation(df)
    assert train["ds"].max() < validation["ds"].min()
    feats = make_supervised_features(train)
    assert {"lag_1", "lag_7", "lag_30", "rolling_mean_4", "rolling_std_8", "is_holiday_week"}.issubset(feats.columns)
    assert feats[["lag_1", "lag_7", "lag_30"]].isna().sum().sum() == 0

