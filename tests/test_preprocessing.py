"""
tests/test_preprocessing.py
============================
Unit tests for src/preprocessing.py — uses synthetic DataFrames only.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    normalize_columns,
    handle_missing_values,
    fix_invalid_entries,
    validate_dtypes,
)


class TestNormalizeColumns:
    def test_renames_raw_fare_columns(self):
        df = pd.DataFrame({
            "Base Fare (BDT)": [1000.0],
            "Tax & Surcharge (BDT)": [100.0],
            "Total Fare (BDT)": [1100.0],
        })
        result = normalize_columns(df)
        assert "Base Fare" in result.columns
        assert "Tax & Surcharge" in result.columns
        assert "Total Fare" in result.columns
        assert "Base Fare (BDT)" not in result.columns

    def test_renames_date_and_duration_columns(self):
        df = pd.DataFrame({
            "Departure Date & Time": ["2026-01-01"],
            "Duration (hrs)": [2.5],
            "Days Before Departure": [30],
        })
        result = normalize_columns(df)
        assert "Date" in result.columns
        assert "Duration" in result.columns
        assert "DaysBeforeDeparture" in result.columns

    def test_drops_redundant_columns(self):
        df = pd.DataFrame({
            "Source Name": ["Dhaka"],
            "Destination Name": ["Chittagong"],
            "Arrival Date & Time": ["2026-01-01 10:00"],
            "Airline": ["Biman"],
        })
        result = normalize_columns(df)
        assert "Source Name" not in result.columns
        assert "Destination Name" not in result.columns
        assert "Arrival Date & Time" not in result.columns
        assert "Airline" in result.columns  # non-redundant columns kept

    def test_idempotent_on_already_normalised_df(self):
        df = pd.DataFrame({
            "Base Fare": [1000.0],
            "Total Fare": [1100.0],
            "Airline": ["Biman"],
        })
        result = normalize_columns(df)
        assert list(result.columns) == list(df.columns)


class TestHandleMissingValues:
    def test_fills_numeric_with_median(self):
        df = pd.DataFrame({"Duration": [1.0, 2.0, np.nan, 4.0, 5.0]})
        result = handle_missing_values(df)
        assert result["Duration"].isnull().sum() == 0
        assert result["Duration"].iloc[2] == pytest.approx(3.0)  # median of [1,2,4,5]

    def test_fills_categorical_with_mode(self):
        df = pd.DataFrame({"Airline": ["Biman", "Biman", "US-Bangla", None]})
        result = handle_missing_values(df)
        assert result["Airline"].isnull().sum() == 0
        assert result["Airline"].iloc[3] == "Biman"

    def test_fallback_to_unknown_when_all_missing(self):
        df = pd.DataFrame({"Category": [None, None, None]})
        result = handle_missing_values(df)
        assert result["Category"].isnull().sum() == 0
        assert result["Category"].iloc[0] == "Unknown"

    def test_no_change_when_no_missing(self):
        df = pd.DataFrame({"Duration": [1.0, 2.0, 3.0], "Airline": ["A", "B", "C"]})
        result = handle_missing_values(df.copy())
        assert result.equals(df)


class TestFixInvalidEntries:
    def test_removes_negative_fare_rows(self):
        df = pd.DataFrame({
            "Total Fare": [500.0, -100.0, 800.0],
            "Airline": ["Biman", "US-Bangla", "Novoair"],
        })
        result = fix_invalid_entries(df)
        assert len(result) == 2
        assert (result["Total Fare"] >= 0).all()

    def test_normalises_dacca_to_dhaka(self):
        df = pd.DataFrame({
            "Source": ["Dacca", "Chittagong"],
            "Destination": ["Chittagong", "Dacca"],
        })
        result = fix_invalid_entries(df)
        assert (result["Source"] == "Dhaka").any()
        assert (result["Destination"] == "Dhaka").any()
        assert "Dacca" not in result["Source"].values
        assert "Dacca" not in result["Destination"].values

    def test_normalises_chattogram_variants(self):
        df = pd.DataFrame({
            "Source": ["Chattogram", "Chottogram"],
            "Destination": ["Dhaka", "Dhaka"],
        })
        result = fix_invalid_entries(df)
        assert (result["Source"] == "Chittagong").all()

    def test_strips_whitespace_from_string_columns(self):
        df = pd.DataFrame({
            "Airline": ["  Biman ", " US-Bangla"],
            "Source": [" DAC", "CGP "],
        })
        result = fix_invalid_entries(df)
        assert result["Airline"].iloc[0] == "Biman"
        assert result["Source"].iloc[0] == "DAC"

    def test_resets_index_after_row_removal(self):
        df = pd.DataFrame({
            "Total Fare": [-1.0, 500.0, 800.0],
        })
        result = fix_invalid_entries(df)
        assert list(result.index) == list(range(len(result)))


class TestValidateDtypes:
    def test_casts_fare_columns_to_float(self):
        df = pd.DataFrame({"Total Fare": ["1000.50", "2000.75", "3000.00"]})
        result = validate_dtypes(df)
        assert pd.api.types.is_float_dtype(result["Total Fare"])

    def test_casts_date_column_to_datetime(self):
        df = pd.DataFrame({"Date": ["2026-01-15", "2026-03-20"]})
        result = validate_dtypes(df)
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_casts_categorical_columns_to_category(self):
        df = pd.DataFrame({
            "Airline": ["Biman", "US-Bangla", "Novoair"],
            "Source": ["DAC", "CGP", "ZYL"],
            "Destination": ["CGP", "DAC", "DAC"],
        })
        result = validate_dtypes(df)
        assert str(result["Airline"].dtype) == "category"
        assert str(result["Source"].dtype) == "category"
        assert str(result["Destination"].dtype) == "category"

    def test_ignores_columns_not_present(self):
        df = pd.DataFrame({"Duration": [1.5, 2.0]})
        result = validate_dtypes(df)
        assert "Duration" in result.columns
        assert len(result) == 2
