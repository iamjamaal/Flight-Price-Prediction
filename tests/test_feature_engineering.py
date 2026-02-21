"""
tests/test_feature_engineering.py
===================================
Unit tests for src/feature_engineering.py — uses synthetic DataFrames only.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    create_date_features,
    encode_categoricals,
    split_data,
)


def _make_date_df(dates):
    return pd.DataFrame({"Date": pd.to_datetime(dates)})


class TestCreateDateFeatures:
    def test_creates_month_day_weekday_season_columns(self):
        df = _make_date_df(["2026-01-15"])
        result = create_date_features(df)
        for col in ("Month", "Day", "Weekday", "Season"):
            assert col in result.columns

    def test_correct_month_extracted(self):
        df = _make_date_df(["2026-07-04"])
        result = create_date_features(df)
        assert result["Month"].iloc[0] == 7

    def test_winter_season_for_december(self):
        df = _make_date_df(["2026-12-25"])
        result = create_date_features(df)
        assert result["Season"].iloc[0] == "Winter"

    def test_winter_season_for_january(self):
        df = _make_date_df(["2026-01-10"])
        result = create_date_features(df)
        assert result["Season"].iloc[0] == "Winter"

    def test_monsoon_season_for_july(self):
        df = _make_date_df(["2026-07-15"])
        result = create_date_features(df)
        assert result["Season"].iloc[0] == "Monsoon"

    def test_summer_season_for_april(self):
        df = _make_date_df(["2026-04-20"])
        result = create_date_features(df)
        assert result["Season"].iloc[0] == "Summer"

    def test_autumn_season_for_october(self):
        df = _make_date_df(["2026-10-05"])
        result = create_date_features(df)
        assert result["Season"].iloc[0] == "Autumn"

    def test_skips_gracefully_if_date_col_missing(self):
        df = pd.DataFrame({"Airline": ["Biman"], "Duration": [2.0]})
        result = create_date_features(df, date_col="Date")
        assert "Month" not in result.columns
        assert list(result.columns) == ["Airline", "Duration"]

    def test_creates_weekday_name_column(self):
        df = _make_date_df(["2026-01-05"])  # Monday
        result = create_date_features(df)
        assert "WeekdayName" in result.columns
        assert result["WeekdayName"].iloc[0] == "Monday"


class TestEncodeCategoricals:
    def _base_df(self):
        return pd.DataFrame({
            "Airline": ["Biman", "US-Bangla", "Novoair"],
            "Source": ["DAC", "CGP", "ZYL"],
            "Total Fare": [1000.0, 2000.0, 3000.0],
        })

    def test_drops_original_column_after_onehot(self):
        df = self._base_df()
        result = encode_categoricals(df, columns=["Airline"], method="onehot")
        assert "Airline" not in result.columns

    def test_creates_binary_columns(self):
        df = self._base_df()
        result = encode_categoricals(df, columns=["Airline"], method="onehot")
        airline_cols = [c for c in result.columns if c.startswith("Airline_")]
        assert len(airline_cols) >= 1

    def test_onehot_values_are_zero_or_one(self):
        df = self._base_df()
        result = encode_categoricals(df, columns=["Airline"], method="onehot")
        airline_cols = [c for c in result.columns if c.startswith("Airline_")]
        for col in airline_cols:
            assert set(result[col].unique()).issubset({0, 1})

    def test_label_encoding_produces_integers(self):
        df = self._base_df()
        result = encode_categoricals(df, columns=["Airline"], method="label")
        assert pd.api.types.is_integer_dtype(result["Airline"])

    def test_label_encoding_keeps_column(self):
        df = self._base_df()
        result = encode_categoricals(df, columns=["Airline"], method="label")
        assert "Airline" in result.columns

    def test_raises_value_error_on_unknown_method(self):
        df = self._base_df()
        with pytest.raises(ValueError, match="Unknown encoding method"):
            encode_categoricals(df, columns=["Airline"], method="invalid_method")


class TestSplitData:
    def _make_df(self, n=100):
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "Duration": rng.uniform(1, 5, n),
            "DaysBeforeDeparture": rng.integers(1, 90, n),
            "Month": rng.integers(1, 12, n),
            "Total Fare": rng.uniform(3000, 15000, n),
        })

    def test_returns_four_parts(self):
        df = self._make_df()
        result = split_data(df, target="Total Fare", save_dir=None)
        assert len(result) == 4

    def test_correct_80_20_split(self):
        df = self._make_df(100)
        X_train, X_test, y_train, y_test = split_data(df, target="Total Fare", save_dir=None)
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_target_not_in_feature_matrix(self):
        df = self._make_df()
        X_train, X_test, _, _ = split_data(df, target="Total Fare", save_dir=None)
        assert "Total Fare" not in X_train.columns
        assert "Total Fare" not in X_test.columns

    def test_raises_key_error_for_missing_target(self):
        df = self._make_df()
        with pytest.raises(KeyError):
            split_data(df, target="NonExistentColumn", save_dir=None)

    def test_drops_non_numeric_columns(self):
        df = self._make_df(50)
        df["WeekdayName"] = "Monday"
        df["Season"] = "Winter"
        X_train, _, _, _ = split_data(df, target="Total Fare", save_dir=None)
        assert "WeekdayName" not in X_train.columns
        assert "Season" not in X_train.columns
