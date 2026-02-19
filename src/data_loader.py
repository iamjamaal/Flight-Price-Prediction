"""
data_loader.py
==============
Responsible for loading the raw dataset and performing initial inspection.
Used in Phase 1 (Problem Definition & Data Understanding).
"""

import pandas as pd
from pathlib import Path

# ── Default path to the raw dataset ──────────────────────────────────────────
DEFAULT_DATA_PATH = Path("data/raw/Flight_Price_Dataset_of_Bangladesh.csv")


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load the flight price CSV into a pandas DataFrame.

    Parameters
    ----------
    path : str or Path
        File path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Raw, unprocessed dataset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path.resolve()}. "
            "Download it from Kaggle and place it in data/raw/."
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns from {path.name}")
    return df


def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Print a comprehensive first-look summary of the DataFrame and return
    a dictionary of key statistics for programmatic use.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
        Keys: shape, dtypes, missing_counts, duplicate_rows, describe.
    """
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    print("── Column Info ─────────────────────────────────────────")
    print(df.dtypes.to_string())

    print("\n── Missing Values ──────────────────────────────────────")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"count": missing, "percent": missing_pct})
    print(missing_df[missing_df["count"] > 0].to_string() or "No missing values.")

    n_duplicates = df.duplicated().sum()
    print(f"\n── Duplicates: {n_duplicates:,} rows ──")

    print("\n── Descriptive Statistics ───────────────────────────────")
    print(df.describe().to_string())

    print("\n── First 5 Rows ────────────────────────────────────────")
    print(df.head().to_string())

    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_counts": missing.to_dict(),
        "duplicate_rows": n_duplicates,
        "describe": df.describe().to_dict(),
    }
