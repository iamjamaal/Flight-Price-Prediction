"""
evaluation.py
=============
Regression metrics, model comparison tables, and cross-validation helpers.
Used in Phases 4 and 5.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score




def evaluate_model(model, X_test, y_test) -> dict:
    """
    Compute core regression metrics on the test set.

    Parameters
    ----------
    model : fitted estimator
    X_test, y_test : array-like

    Returns
    -------
    dict
        Keys: r2, mae, rmse, predictions.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "predictions": y_pred,
    }
    return metrics



def print_metrics(metrics: dict) -> None:
    """Pretty-print evaluation metrics."""
    print(f"  R²   = {metrics['r2']:.4f}")
    print(f"  MAE  = {metrics['mae']:.2f}")
    print(f"  RMSE = {metrics['rmse']:.2f}")



def cross_validate_model(
    model,
    X,
    y,
    cv: int = 5,
    scoring: str = "r2",
) -> dict:
    """
    Perform k-fold cross-validation and return summary statistics.

    Parameters
    ----------
    model : estimator (unfitted or fitted — sklearn clones internally)
    X, y : array-like
    cv : int
    scoring : str

    Returns
    -------
    dict
        Keys: scores, mean, std.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    result = {
        "scores": scores,
        "mean": scores.mean(),
        "std": scores.std(),
    }
    print(f"  CV {scoring} = {result['mean']:.4f} ± {result['std']:.4f}  (k={cv})")
    return result


def build_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a model comparison DataFrame from evaluation results.

    Parameters
    ----------
    results : dict
        {model_name: metrics_dict} where each metrics_dict has r2, mae, rmse.

    Returns
    -------
    pd.DataFrame
        Sorted by R² descending.
    """
    rows = []
    for name, m in results.items():
        rows.append({
            "Model": name,
            "R²": round(m["r2"], 4),
            "MAE": round(m["mae"], 2),
            "RMSE": round(m["rmse"], 2),
        })
    df = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)
    return df
