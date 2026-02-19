"""
preprocessing.py
================
Data cleaning utilities — dropping columns, handling missing values,
fixing invalid entries, and validating data types.
Used in Phase 2 (Data Cleaning & Preprocessing).
"""

import pandas as pd
import numpy as np
from typing import Optional

# Maps raw dataset column names to the standardised names used throughout
# the pipeline.  Keys that don't appear in a given DataFrame are silently
# ignored, so this is safe to call on any version of the CSV.
_COLUMN_RENAME_MAP: dict[str, str] = {
    "Base Fare (BDT)": "Base Fare",
    "Tax & Surcharge (BDT)": "Tax & Surcharge",
    "Total Fare (BDT)": "Total Fare",
    "Departure Date & Time": "Date",
    "Duration (hrs)": "Duration",
    "Days Before Departure": "DaysBeforeDeparture",
}

# Columns that are redundant once the rename has been applied, or that
# leak information that wouldn't be available at prediction time.
_REDUNDANT_COLUMNS: list[str] = [
    "Source Name",
    "Destination Name",
    "Arrival Date & Time",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw dataset columns to the standard names used by the pipeline
    and drop columns that are redundant after renaming.

    This function is idempotent — calling it on an already-normalised
    DataFrame is a no-op.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    rename = {k: v for k, v in _COLUMN_RENAME_MAP.items() if k in df.columns}
    if rename:
        df = df.rename(columns=rename)
        print(f"Renamed columns: {rename}")

    to_drop = [c for c in _REDUNDANT_COLUMNS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"Dropped redundant columns: {to_drop}")

    return df


def drop_irrelevant_columns(
    df: pd.DataFrame,
    extra_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Normalise column names and remove columns that carry no predictive signal.

    Steps:
    1. Rename raw columns (e.g. ``Total Fare (BDT)`` → ``Total Fare``).
    2. Drop redundant columns (e.g. ``Source Name``, ``Arrival Date & Time``).
    3. Drop columns whose names contain 'unnamed' or 'index' (case-insensitive).
    4. Drop any additional columns specified via *extra_cols*.

    Parameters
    ----------
    df : pd.DataFrame
    extra_cols : list[str], optional

    Returns
    -------
    pd.DataFrame
    """
    df = normalize_columns(df)

    auto_drop = [c for c in df.columns if c.lower().startswith(("unnamed", "index"))]
    to_drop = list(set(auto_drop + (extra_cols or [])))
    existing = [c for c in to_drop if c in df.columns]

    if existing:
        df = df.drop(columns=existing)
        print(f"Dropped columns: {existing}")
    else:
        print("No extra irrelevant columns found to drop.")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values.

    Strategy:
    - Numerical columns → median (robust to outliers)
    - Categorical / object columns → mode; fallback to 'Unknown'

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    for col in df.columns:
        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue

        if df[col].dtype in ("float64", "float32", "int64", "int32"):
            fill = df[col].median()
            df[col] = df[col].fillna(fill)
            print(f"  {col}: filled {n_missing} missing with median={fill:.2f}")
        else:
            mode_vals = df[col].mode()
            fill = mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown"
            df[col] = df[col].fillna(fill)
            print(f"  {col}: filled {n_missing} missing with mode='{fill}'")

    remaining = df.isnull().sum().sum()
    print(f"Total remaining missing values: {remaining}")
    return df


def fix_invalid_entries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correct data-quality issues.

    - Negative fares → removed
    - Inconsistent city names → normalized
    - Leading/trailing whitespace → stripped from string columns

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    initial_len = len(df)

    # Strip whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Normalize common city name variants (extend as needed)
    city_map = {
        "Dacca": "Dhaka",
        "Chattogram": "Chittagong",
        "Chottogram": "Chittagong",
    }
    for col in ("Source", "Destination"):
        if col in df.columns:
            df[col] = df[col].replace(city_map)

    # Remove rows with negative fare values
    fare_cols = [c for c in df.columns if "fare" in c.lower()]
    for col in fare_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            neg_mask = df[col] < 0
            n_neg = neg_mask.sum()
            if n_neg > 0:
                df = df[~neg_mask]
                print(f"  {col}: removed {n_neg} rows with negative values")

    print(f"Rows: {initial_len:,} -> {len(df):,} (removed {initial_len - len(df):,})")
    return df.reset_index(drop=True)


def validate_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns have correct data types.

    - Fare columns → float64
    - Date column → datetime64
    - Airline, Source, Destination → category

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    # Numeric fare columns
    fare_cols = [c for c in df.columns if "fare" in c.lower() or "tax" in c.lower() or "surcharge" in c.lower()]
    for col in fare_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            print(f"  {col} -> float64")

    # Date column
    date_cols = [c for c in df.columns if "date" in c.lower()]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            print(f"  {col} -> datetime64")

    # Categorical columns
    cat_candidates = (
        "Airline", "Source", "Destination",
        "Stopovers", "Aircraft Type", "Class", "Booking Source", "Seasonality",
    )
    cat_cols = [c for c in cat_candidates if c in df.columns]
    for col in cat_cols:
        df[col] = df[col].astype("category")
        print(f"  {col} -> category")

    return df
