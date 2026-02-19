"""
visualization.py
================
All plotting utilities for EDA, model evaluation, and interpretation.
Used across Phases 3–6.

Every function accepts a matplotlib Axes and returns it, following the
'pass-axes' pattern so callers can compose multi-panel figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FIGURE_DIR = Path("reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)


def _save(fig, filename: str | None):
    """Save figure to reports/figures/ if a filename is provided."""
    if filename:
        path = FIGURE_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {path}")


# ── EDA Visualizations ──────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, save_as: str | None = "correlation_heatmap.png"):
    """Correlation heatmap of all numerical columns."""
    numeric = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    _save(fig, save_as)
    plt.show()
    return ax


def plot_fare_distribution(df: pd.DataFrame, col: str = "Total Fare", save_as: str | None = "fare_distribution.png"):
    """Histogram with KDE of the target variable."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[col], kde=True, bins=50, ax=ax)
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    _save(fig, save_as)
    plt.show()
    return ax


def plot_fare_by_airline(df: pd.DataFrame, save_as: str | None = "fare_by_airline.png"):
    """Side-by-side bar chart (mean fare) and boxplot (spread) per airline."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    order = df.groupby("Airline")["Total Fare"].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x="Airline", y="Total Fare", order=order, ax=axes[0], errorbar=None)
    axes[0].set_title("Average Fare by Airline")
    axes[0].tick_params(axis="x", rotation=45)

    sns.boxplot(data=df, x="Airline", y="Total Fare", order=order, ax=axes[1])
    axes[1].set_title("Fare Distribution by Airline")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    _save(fig, save_as)
    plt.show()
    return axes


def plot_seasonal_fares(df: pd.DataFrame, save_as: str | None = "seasonal_fares.png"):
    """Boxplot and monthly line chart of fare variation."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if "Season" in df.columns:
        season_order = ["Winter", "Summer", "Monsoon", "Autumn"]
        existing = [s for s in season_order if s in df["Season"].unique()]
        sns.boxplot(data=df, x="Season", y="Total Fare", order=existing, ax=axes[0])
        axes[0].set_title("Fare by Season")

    if "Month" in df.columns:
        monthly = df.groupby("Month")["Total Fare"].mean()
        axes[1].plot(monthly.index, monthly.values, marker="o")
        axes[1].set_title("Average Fare by Month")
        axes[1].set_xlabel("Month")
        axes[1].set_ylabel("Average Total Fare")
        axes[1].set_xticks(range(1, 13))

    fig.tight_layout()
    _save(fig, save_as)
    plt.show()
    return axes


# ── Model Evaluation Visualizations ─────────────────────────────────────────

def plot_actual_vs_predicted(y_true, y_pred, save_as: str | None = "actual_vs_predicted.png"):
    """Scatter plot of actual vs predicted values with 45-degree reference."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.4, edgecolors="k", linewidth=0.3)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Fare")
    ax.set_ylabel("Predicted Fare")
    ax.set_title("Actual vs Predicted Fares")
    ax.legend()
    _save(fig, save_as)
    plt.show()
    return ax


def plot_residuals(y_true, y_pred, save_as: str | None = "residuals.png"):
    """Residual histogram and residuals-vs-predicted scatter."""
    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(residuals, kde=True, bins=50, ax=axes[0])
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Residual (Actual − Predicted)")

    axes[1].scatter(y_pred, residuals, alpha=0.4, edgecolors="k", linewidth=0.3)
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set_title("Residuals vs Predicted")
    axes[1].set_xlabel("Predicted Fare")
    axes[1].set_ylabel("Residual")

    fig.tight_layout()
    _save(fig, save_as)
    plt.show()
    return axes


# ── Interpretation Visualizations ────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list[str], top_n: int = 15, save_as: str | None = "feature_importance.png"):
    """Bar chart of top-N feature importances from a tree-based model."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in idx][::-1],
        importances[idx][::-1],
    )
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    _save(fig, save_as)
    plt.show()
    return ax


def plot_coefficients(model, feature_names: list[str], top_n: int = 15, save_as: str | None = "coefficients.png"):
    """Horizontal bar chart of linear model coefficients."""
    coefs = pd.Series(model.coef_, index=feature_names)
    top = coefs.abs().nlargest(top_n).index
    subset = coefs[top].sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["green" if v > 0 else "red" for v in subset]
    ax.barh(subset.index, subset.values, color=colors)
    ax.set_title(f"Top {top_n} Model Coefficients")
    ax.set_xlabel("Coefficient Value")
    _save(fig, save_as)
    plt.show()
    return ax
