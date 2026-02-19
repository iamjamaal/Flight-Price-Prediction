"""
pipeline.py
===========
Standalone task functions for the Airflow-orchestrated ML pipeline.
Each function encapsulates one pipeline phase and can be called
independently by Airflow PythonOperators.

All file paths are resolved relative to PROJECT_ROOT so the functions
work identically inside Docker containers and in local environments.
"""

import json
import logging
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Loky-backed parallel loops")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

logger = logging.getLogger("airflow.task")

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent)
)
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def _ensure_dirs():
    """Create output directories if they don't exist."""
    for d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _configure_viz():
    """Point the visualization module's FIGURE_DIR at the project path."""
    import src.visualization as viz
    viz.FIGURE_DIR = FIGURES_DIR
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _eval_bdt(model, X_test, y_test_log):
    """Predict in log space, inverse-transform, and return BDT-scale metrics."""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    preds_log = model.predict(X_test)
    y_bdt = np.expm1(y_test_log)
    preds_bdt = np.expm1(preds_log)
    preds_bdt = np.maximum(preds_bdt, 0)
    return {
        "r2": r2_score(y_bdt, preds_bdt),
        "mae": mean_absolute_error(y_bdt, preds_bdt),
        "rmse": np.sqrt(mean_squared_error(y_bdt, preds_bdt)),
        "predictions": preds_bdt,
    }


# ── Phase 1 ──────────────────────────────────────────────────────────────────

def load_and_validate(**context):
    """Load the raw CSV and persist a quality summary."""
    from src.data_loader import load_dataset, inspect_dataset

    _ensure_dirs()
    csv_path = DATA_RAW / "Flight_Price_Dataset_of_Bangladesh.csv"
    df = load_dataset(csv_path)
    summary = inspect_dataset(df)

    serializable = {
        "shape": list(summary["shape"]),
        "missing_counts": {k: int(v) for k, v in summary["missing_counts"].items()},
        "duplicate_rows": int(summary["duplicate_rows"]),
    }
    out = DATA_PROCESSED / "data_summary.json"
    out.write_text(json.dumps(serializable, indent=2))
    logger.info("Phase 1 complete — %d rows, %d cols", *summary["shape"])


# ── Phase 2 ──────────────────────────────────────────────────────────────────

def clean_and_preprocess(**context):
    """Clean, engineer features, encode, scale, and produce train/test splits."""
    from src.data_loader import load_dataset
    from src.preprocessing import (
        drop_irrelevant_columns,
        handle_missing_values,
        fix_invalid_entries,
        validate_dtypes,
    )
    from src.feature_engineering import (
        create_date_features,
        encode_categoricals,
        scale_numericals,
        split_data,
        save_scaler,
    )

    _ensure_dirs()
    df = load_dataset(DATA_RAW / "Flight_Price_Dataset_of_Bangladesh.csv")
    df = drop_irrelevant_columns(df)
    df = handle_missing_values(df)
    df = fix_invalid_entries(df)
    df = validate_dtypes(df)
    df = create_date_features(df)

    df.to_csv(DATA_PROCESSED / "cleaned_with_features.csv", index=False)

    leak_cols = ["Base Fare", "Tax & Surcharge"]
    df_model = df.drop(columns=[c for c in leak_cols if c in df.columns])
    logger.info("Dropped leakage columns: %s", leak_cols)

    df_model["Total Fare"] = np.log1p(df_model["Total Fare"])
    logger.info("Applied log1p transform to Total Fare (skew reduction)")

    df_encoded = encode_categoricals(df_model.copy())
    df_scaled, scaler = scale_numericals(df_encoded.copy())

    scale_cols = [
        c for c in df_encoded.columns
        if any(k in c.lower() for k in ("fare", "tax", "surcharge"))
        and "total" not in c.lower()
    ]
    save_scaler(scaler, scale_cols, path=DATA_PROCESSED / "scaler.joblib")

    X_train, X_test, y_train, y_test = split_data(
        df_scaled,
        target="Total Fare",
        test_size=0.20,
        random_state=42,
        save_dir=str(DATA_PROCESSED),
    )
    logger.info(
        "Phase 2 complete — train %s, test %s", X_train.shape, X_test.shape
    )


# ── Phase 3 ──────────────────────────────────────────────────────────────────

def generate_eda_report(**context):
    """Produce EDA visualizations and KPI statistics."""
    import matplotlib.pyplot as plt
    from src.visualization import (
        plot_correlation_heatmap,
        plot_fare_distribution,
        plot_fare_by_airline,
        plot_seasonal_fares,
    )

    _ensure_dirs()
    _configure_viz()
    df = pd.read_csv(DATA_PROCESSED / "cleaned_with_features.csv")

    for fn in (
        plot_correlation_heatmap,
        plot_fare_distribution,
        plot_fare_by_airline,
        plot_seasonal_fares,
    ):
        fn(df)
        plt.close("all")

    kpis = {
        "avg_fare_by_airline": (
            df.groupby("Airline")["Total Fare"]
            .mean()
            .sort_values(ascending=False)
            .round(2)
            .to_dict()
        ),
        "popular_routes": {
            f"{s} → {d}": int(c)
            for (s, d), c in df.groupby(["Source", "Destination"])
            .size()
            .sort_values(ascending=False)
            .head(10)
            .items()
        },
        "expensive_routes": {
            f"{s} → {d}": round(float(v), 2)
            for (s, d), v in df.groupby(["Source", "Destination"])["Total Fare"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
            .items()
        },
    }
    if "Season" in df.columns:
        kpis["avg_fare_by_season"] = (
            df.groupby("Season")["Total Fare"]
            .mean()
            .sort_values(ascending=False)
            .round(2)
            .to_dict()
        )

    (DATA_PROCESSED / "eda_kpis.json").write_text(json.dumps(kpis, indent=2))
    logger.info("Phase 3 complete — figures saved to %s", FIGURES_DIR)


# ── Phase 4 ──────────────────────────────────────────────────────────────────

def train_baseline_model(**context):
    """Train a Linear Regression baseline and persist metrics."""
    import matplotlib.pyplot as plt
    from src.models import train_model, save_model
    from src.evaluation import print_metrics
    from src.visualization import plot_actual_vs_predicted, plot_residuals

    _ensure_dirs()
    _configure_viz()

    X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").squeeze()
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").squeeze()

    model = train_model("linear_regression", X_train, y_train)
    metrics = _eval_bdt(model, X_test, y_test)
    print_metrics(metrics)

    plot_actual_vs_predicted(
        np.expm1(y_test), metrics["predictions"],
        save_as="baseline_actual_vs_predicted.png",
    )
    plt.close("all")
    plot_residuals(
        np.expm1(y_test), metrics["predictions"],
        save_as="baseline_residuals.png",
    )
    plt.close("all")

    save_model(model, str(MODELS_DIR / "linear_regression_baseline.joblib"))

    baseline = {
        "r2": round(metrics["r2"], 4),
        "mae": round(metrics["mae"], 2),
        "rmse": round(metrics["rmse"], 2),
    }
    (DATA_PROCESSED / "baseline_metrics.json").write_text(
        json.dumps(baseline, indent=2)
    )
    logger.info(
        "Phase 4 complete — R²=%.4f  MAE=%.2f  RMSE=%.2f",
        metrics["r2"], metrics["mae"], metrics["rmse"],
    )


# ── Phase 5 ──────────────────────────────────────────────────────────────────

def train_advanced_models(**context):
    """Train, tune, and compare multiple regression models."""
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.metrics import r2_score
    from src.models import train_model, tune_model, save_model
    from src.evaluation import (
        print_metrics,
        cross_validate_model,
        build_comparison_table,
    )

    _ensure_dirs()
    _configure_viz()

    X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").squeeze()
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").squeeze()

    # --- Base models ---
    model_names = [
        "linear_regression", "ridge", "lasso",
        "decision_tree", "random_forest",
    ]
    trained = {}
    results = {}
    for name in model_names:
        m = train_model(name, X_train, y_train)
        metrics = _eval_bdt(m, X_test, y_test)
        print_metrics(metrics)
        trained[name] = m
        results[name] = metrics

    try:
        xgb = train_model("xgboost", X_train, y_train)
        results["xgboost"] = _eval_bdt(xgb, X_test, y_test)
        trained["xgboost"] = xgb
    except Exception as exc:
        logger.warning("XGBoost skipped: %s", exc)

    # --- Hyperparameter tuning ---
    best_rf = tune_model("random_forest", X_train, y_train, cv=3, n_iter=10)
    results["random_forest_tuned"] = _eval_bdt(best_rf, X_test, y_test)
    trained["random_forest_tuned"] = best_rf

    if "xgboost" in trained:
        best_xgb = tune_model("xgboost", X_train, y_train, cv=3, n_iter=10)
        results["xgboost_tuned"] = _eval_bdt(best_xgb, X_test, y_test)
        trained["xgboost_tuned"] = best_xgb

    # --- Cross-validation for top models ---
    for name in ["ridge", "random_forest"]:
        if name in trained:
            cross_validate_model(trained[name], X_train, y_train, cv=5)

    # --- Comparison table ---
    comparison = build_comparison_table(results)
    comparison.to_csv(DATA_PROCESSED / "model_comparison.csv", index=False)

    # --- Bias-variance tradeoff plot (log-space friendly alpha range) ---
    alphas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    ridge_tr, ridge_te = [], []
    lasso_tr, lasso_te = [], []
    for a in alphas:
        r = Ridge(alpha=a).fit(X_train, y_train)
        ridge_tr.append(r2_score(y_train, r.predict(X_train)))
        ridge_te.append(r2_score(y_test, r.predict(X_test)))
        la = Lasso(alpha=a, max_iter=50_000).fit(X_train, y_train)
        lasso_tr.append(r2_score(y_train, la.predict(X_train)))
        lasso_te.append(r2_score(y_test, la.predict(X_test)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].semilogx(alphas, ridge_tr, "o-", label="Train")
    axes[0].semilogx(alphas, ridge_te, "s--", label="Test")
    axes[0].set(title="Ridge: Bias-Variance Tradeoff",
                xlabel="Alpha", ylabel="R²")
    axes[0].legend()
    axes[1].semilogx(alphas, lasso_tr, "o-", label="Train")
    axes[1].semilogx(alphas, lasso_te, "s--", label="Test")
    axes[1].set(title="Lasso: Bias-Variance Tradeoff",
                xlabel="Alpha", ylabel="R²")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(
        str(FIGURES_DIR / "bias_variance_tradeoff.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close("all")

    # --- Select and save best ---
    best_name = comparison.iloc[0]["Model"]
    save_model(trained[best_name], str(MODELS_DIR / "best_model.joblib"))
    logger.info(
        "Phase 5 complete — best model: %s (R²=%.4f)",
        best_name, comparison.iloc[0]["R²"],
    )


# ── Phase 6 ──────────────────────────────────────────────────────────────────

def interpret_and_report(**context):
    """Feature importance, coefficients, and business insight report."""
    import matplotlib.pyplot as plt
    from src.models import load_model
    from src.visualization import plot_feature_importance, plot_coefficients

    _ensure_dirs()
    _configure_viz()

    best_model = load_model(str(MODELS_DIR / "best_model.joblib"))
    lr_model = load_model(str(MODELS_DIR / "linear_regression_baseline.joblib"))
    X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    feature_names = X_train.columns.tolist()

    top_features = {}
    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(best_model, feature_names, top_n=15)
        plt.close("all")
        imp = pd.Series(best_model.feature_importances_, index=feature_names)
        top_features = imp.nlargest(15).round(4).to_dict()

    plot_coefficients(lr_model, feature_names, top_n=15)
    plt.close("all")

    airline_feats = [f for f in feature_names if "Airline" in f]
    airline_coefs = {
        f: round(float(lr_model.coef_[feature_names.index(f)]), 4)
        for f in airline_feats
    }

    seasonal_feats = [f for f in feature_names if "Season" in f or "Month" in f]
    route_feats = [f for f in feature_names if "Source" in f or "Destination" in f]
    seasonal_importance, route_importance = {}, {}
    if hasattr(best_model, "feature_importances_"):
        imp = pd.Series(best_model.feature_importances_, index=feature_names)
        seasonal_importance = (
            imp[seasonal_feats].sort_values(ascending=False).round(4).to_dict()
        )
        route_importance = (
            imp[route_feats]
            .sort_values(ascending=False)
            .head(10)
            .round(4)
            .to_dict()
        )

    comparison_path = DATA_PROCESSED / "model_comparison.csv"
    comparison = (
        pd.read_csv(comparison_path).to_dict(orient="records")
        if comparison_path.exists()
        else []
    )

    report = {
        "top_features": top_features,
        "airline_coefficients": airline_coefs,
        "seasonal_importance": seasonal_importance,
        "route_importance": route_importance,
        "model_comparison": comparison,
    }
    out = DATA_PROCESSED / "interpretation_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Phase 6 complete — report at %s", out)
