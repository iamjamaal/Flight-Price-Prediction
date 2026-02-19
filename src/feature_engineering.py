"""
feature_engineering.py
======================
Feature creation, categorical encoding, numerical scaling, and
train/test splitting. Used in Phase 2 (Data Cleaning & Preprocessing).
"""

import json

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def create_date_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Extract temporal features from a datetime column.

    New columns: Month, Day, Weekday (0=Mon), WeekdayName, Season.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str

    Returns
    -------
    pd.DataFrame
    """
    if date_col not in df.columns:
        print(f"Column '{date_col}' not found — skipping date features.")
        return df

    dt = pd.to_datetime(df[date_col], errors="coerce")
    df["Month"] = dt.dt.month
    df["Day"] = dt.dt.day
    df["Weekday"] = dt.dt.weekday
    df["WeekdayName"] = dt.dt.day_name()

    # Season mapping for Bangladesh climate
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Summer", 4: "Summer", 5: "Summer",
        6: "Monsoon", 7: "Monsoon", 8: "Monsoon",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    }
    df["Season"] = df["Month"].map(season_map)

    print(f"Created date features: Month, Day, Weekday, WeekdayName, Season")
    return df


def encode_categoricals(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "onehot",
) -> pd.DataFrame:
    """
    Encode categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str], optional
        Columns to encode. Auto-detected if None.
    method : 'onehot' or 'label'

    Returns
    -------
    pd.DataFrame
    """
    if columns is None:
        columns = [
            c for c in (
                "Airline", "Source", "Destination",
                "Stopovers", "Aircraft Type", "Class", "Booking Source",
                "Seasonality", "WeekdayName", "Season",
            )
            if c in df.columns
        ]

    if method == "onehot":
        df = pd.get_dummies(df, columns=columns, drop_first=True, dtype=int)
        print(f"One-hot encoded: {columns}  →  {df.shape[1]} total columns")
    elif method == "label":
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        print(f"Label encoded: {columns}")
    else:
        raise ValueError(f"Unknown encoding method: {method}")

    return df


def scale_numericals(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Apply StandardScaler to numerical feature columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str], optional
        Columns to scale. Auto-detected from fare/tax columns if None.
    scaler : StandardScaler, optional
        Pre-fitted scaler for transform-only (e.g., on test set).

    Returns
    -------
    tuple[pd.DataFrame, StandardScaler]
        Scaled DataFrame and the fitted scaler (for reuse on test data).
    """
    if columns is None:
        columns = [
            c for c in df.columns
            if any(k in c.lower() for k in ("fare", "tax", "surcharge"))
            and "total" not in c.lower()  # don't scale the target
        ]

    if not columns:
        print("No columns to scale.")
        return df, scaler or StandardScaler()

    fit_new = scaler is None
    if fit_new:
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        print(f"Fit & transformed: {columns}")
    else:
        df[columns] = scaler.transform(df[columns])
        print(f"Transformed (pre-fitted scaler): {columns}")

    return df, scaler


def split_data(
    df: pd.DataFrame,
    target: str = "Total Fare",
    test_size: float = 0.20,
    random_state: int = 42,
    save_dir: str | Path | None = "data/processed",
) -> tuple:
    """
    Split into train/test sets and optionally save to disk.

    Also persists the training feature column names to
    ``<save_dir>/train_columns.json`` so the inference API can align
    its one-hot encoded input to the exact feature set the model expects.

    Parameters
    ----------
    df : pd.DataFrame
    target : str
    test_size : float
    random_state : int
    save_dir : str or Path, optional

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    target_cols = [c for c in df.columns if c == target]
    if not target_cols:
        raise KeyError(f"Target column '{target}' not found in DataFrame.")

    y = df[target]
    X = df.drop(columns=[target])

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)
        print(f"Dropped non-numeric columns before split: {non_numeric}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Train: {X_train.shape[0]:,} samples  |  Test: {X_test.shape[0]:,} samples")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        X_train.to_csv(save_dir / "X_train.csv", index=False)
        X_test.to_csv(save_dir / "X_test.csv", index=False)
        y_train.to_csv(save_dir / "y_train.csv", index=False)
        y_test.to_csv(save_dir / "y_test.csv", index=False)
        save_training_columns(X_train.columns.tolist(), save_dir / "train_columns.json")
        print(f"Saved splits to {save_dir}/")

    return X_train, X_test, y_train, y_test


# ── Training-column persistence for inference alignment ──────────────────────


def save_training_columns(columns: list[str], path: str | Path) -> None:
    """Write the ordered list of training feature names to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(columns))
    print(f"Saved {len(columns)} training column names → {path}")


def load_training_columns(path: str | Path = "data/processed/train_columns.json") -> list[str]:
    """Read the ordered list of training feature names from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Training columns file not found at {path}. "
            "Run the preprocessing notebook (02_*) first."
        )
    return json.loads(path.read_text())


def align_features(df: pd.DataFrame, training_columns: list[str]) -> pd.DataFrame:
    """
    Align a one-hot encoded DataFrame to the training feature set.

    Adds missing columns (filled with 0) and drops extra columns so the
    result has exactly the same columns in the same order as training data.
    """
    return df.reindex(columns=training_columns, fill_value=0)


# ── Scaler persistence for inference ─────────────────────────────────────────

SCALER_PATH = Path("data/processed/scaler.joblib")
SCALER_COLUMNS_PATH = Path("data/processed/scaler_columns.json")


def save_scaler(scaler: StandardScaler, columns: list[str], path: str | Path = SCALER_PATH) -> None:
    """Persist a fitted StandardScaler and the column names it was fitted on."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    cols_path = path.with_suffix(".columns.json")
    cols_path.write_text(json.dumps(columns))
    print(f"Saved scaler → {path}  (columns: {columns})")


def load_scaler(path: str | Path = SCALER_PATH) -> tuple[StandardScaler, list[str]]:
    """Load a fitted StandardScaler and the column names it expects."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {path}. "
            "Run the preprocessing notebook (02_*) first."
        )
    scaler = joblib.load(path)
    cols_path = path.with_suffix(".columns.json")
    columns = json.loads(cols_path.read_text())
    return scaler, columns
