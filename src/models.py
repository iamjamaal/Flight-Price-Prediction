"""
models.py
=========
Model training, hyperparameter tuning, serialization, and loading.
Used in Phases 4 and 5 (Baseline + Advanced Modeling).
"""

import datetime
import json
import warnings

warnings.filterwarnings("ignore", message="Loky-backed parallel loops")

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

# Lazy imports for optional heavy libraries
_OPTIONAL_MODELS = {}



def _get_xgboost():
    if "xgboost" not in _OPTIONAL_MODELS:
        from xgboost import XGBRegressor
        _OPTIONAL_MODELS["xgboost"] = XGBRegressor
    return _OPTIONAL_MODELS["xgboost"]


# ── Model Registry
MODEL_REGISTRY = {
    "linear_regression": lambda: LinearRegression(),
    "ridge": lambda: Ridge(alpha=1.0),
    "lasso": lambda: Lasso(alpha=0.001, max_iter=50_000),
    "decision_tree": lambda: DecisionTreeRegressor(random_state=42),
    "random_forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
    "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=20, random_state=42),
    "xgboost": lambda: _get_xgboost()(n_estimators=100, random_state=42, n_jobs=1),
}



# ── Hyperparameter grids for tuning 
PARAM_GRIDS = {
    "ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "lasso": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    "decision_tree": {
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
    "gradient_boosting": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    },
    "xgboost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
    },
}



def train_model(name: str, X_train, y_train):
    """
    Instantiate and fit a model from the registry.

    Parameters
    ----------
    name : str
        Key in MODEL_REGISTRY.
    X_train, y_train : array-like

    Returns
    -------
    Fitted estimator.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")

    model = MODEL_REGISTRY[name]()
    model.fit(X_train, y_train)
    print(f"Trained: {name} ({type(model).__name__})")
    return model


def tune_model(
    name: str,
    X_train,
    y_train,
    cv: int = 5,
    n_iter: int = 30,
    scoring: str = "r2",
):
    """
    Run RandomizedSearchCV to find optimal hyperparameters.

    Parameters
    ----------
    name : str
    X_train, y_train : array-like
    cv : int
    n_iter : int
    scoring : str

    Returns
    -------
    Best fitted estimator.
    """
    if name not in PARAM_GRIDS:
        raise ValueError(f"No param grid for '{name}'. Available: {list(PARAM_GRIDS)}")

    base_model = MODEL_REGISTRY[name]()
    search = RandomizedSearchCV(
        base_model,
        PARAM_GRIDS[name],
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=42,
        n_jobs=1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"Best params for {name}: {search.best_params_}")
    print(f"Best CV {scoring}: {search.best_score_:.4f}")
    return search.best_estimator_




def save_model(model, path: str | Path) -> None:
    """Serialize a trained model to disk using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved -> {path}")


def save_model_versioned(
    model,
    model_name: str,
    metrics: dict,
    models_dir: str | Path,
) -> None:
    """
    Save a versioned copy of the model and update the model registry.

    Saves:
    - models/<model_name>_v<YYYYMMDD_HHMMSS>.joblib  (timestamped copy)
    - models/<model_name>.joblib                      (canonical alias, backward compat)
    - models/model_registry.json                      (appended with new entry)

    Parameters
    ----------
    model : fitted estimator
    model_name : str
    metrics : dict
        e.g. {"r2": 0.95, "mae": 320.5, "rmse": 510.2}
    models_dir : str or Path
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = models_dir / f"{model_name}_v{timestamp}.joblib"
    canonical_path = models_dir / f"{model_name}.joblib"

    joblib.dump(model, versioned_path)
    print(f"Model saved (versioned)  -> {versioned_path}")

    joblib.dump(model, canonical_path)
    print(f"Model saved (canonical)  -> {canonical_path}")

    registry_path = models_dir / "model_registry.json"
    registry: list = []
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))

    entry = {
        "version": timestamp,
        "model_name": model_name,
        "versioned_path": str(versioned_path),
        "canonical_path": str(canonical_path),
        "metrics": metrics,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    registry.append(entry)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Registry updated         -> {registry_path}")


def load_model(path: str | Path):
    """Load a serialized model from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model file at {path}")
    model = joblib.load(path)
    print(f"Model loaded <- {path}")
    return model
