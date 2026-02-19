# Execution Plan — Flight Fare Prediction

> This document is the **step-by-step implementation guide**. Each action is
> numbered, described, and annotated with the file it affects. Work through
> them sequentially; every step builds on the previous one.
>
> **Orchestration model:** The entire ML pipeline (Phases 1–6) is
> orchestrated by an **Apache Airflow DAG**. Notebooks exist for interactive
> development and documentation; the production pipeline runs through
> Airflow.

---

## PHASE 0 — Environment & Infrastructure Setup

### 0.1 Install Docker Desktop

- Download from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
- Verify installation:

```bash
docker --version
docker compose version
```

### 0.2 Download the Dataset

- Go to Kaggle and download `Flight_Price_Dataset_of_Bangladesh.csv`
- Place it in `data/raw/Flight_Price_Dataset_of_Bangladesh.csv`

### 0.3 Build Docker Images

> **Important:** All `docker compose` commands must be run from the
> **project root** directory (where `docker-compose.yml` lives).

Build the Jupyter notebook image (for interactive development):

```bash
docker compose build notebook
```

Build the Airflow image (for pipeline orchestration):

```bash
docker compose --profile airflow build
```

- Both images install Python 3.11 and all dependencies from `requirements.txt`
- The Airflow image is based on `apache/airflow:2.8.1-python3.11`
- Expected build time: 3–8 minutes (first build)

### 0.4 Initialize the Airflow Database (One-Time)

```bash
docker compose --profile airflow run --rm airflow-init
```

This runs database migrations and creates the Airflow admin user:

- Username: `admin`
- Password: `admin`

### 0.5 Start the Airflow Environment

```bash
docker compose --profile airflow up -d
```

This starts three containers:

- **PostgreSQL** — Airflow metadata database
- **Airflow Webserver** — UI at **[http://localhost:8081](http://localhost:8081)**
- **Airflow Scheduler** — monitors and triggers DAG runs

### 0.6 Verify the Airflow UI

1. Navigate to **[http://localhost:8081](http://localhost:8081)**
2. Log in with `admin` / `admin`
3. Confirm the **flight_fare_prediction** DAG is visible and active

### 0.7 (Optional) Start Jupyter for Interactive Development

If you want to explore data or develop interactively alongside Airflow:

```bash
docker compose up notebook
```

Open browser at [http://localhost:8888](http://localhost:8888) — no token required.

### 0.8 Initialize Git Repository

```bash
git init
git add .
git commit -m "Initial project scaffold with Docker, Airflow, and documentation"
```

### 0.9 Verification Checklist

- `docker ps` shows Airflow webserver, scheduler, and postgres running
- Airflow UI opens at [http://localhost:8081](http://localhost:8081)
- `flight_fare_prediction` DAG is visible in the UI
- Dataset CSV exists in `data/raw/`
- (Optional) Jupyter opens at [http://localhost:8888](http://localhost:8888)

---

## PHASE 1 — Problem Definition & Data Understanding

> **Airflow task:** `load_and_validate` → calls `src.pipeline.load_and_validate()`
> **Notebook:** `notebooks/01_problem_definition_data_understanding.ipynb`
> **Source module:** `src/data_loader.py`

### What the Airflow Task Does

The `load_and_validate` task automatically:

1. Loads the raw CSV from `data/raw/`
2. Runs `inspect_dataset()` to compute shape, missing values, duplicates
3. Persists a quality summary to `data/processed/data_summary.json`

### 1.1 Write the Business Context (Notebook — Markdown Cell)

- Explain WHY fare prediction matters for airlines and travel platforms
- State the business question: *"Can we predict the total fare of a flight
given its airline, route, date, and fare components?"*

### 1.2 Define the ML Task (Notebook — Markdown Cell)

- Task type: Supervised Regression
- Target variable: `Total Fare`
- Feature candidates: Airline, Source, Destination, Date, Base Fare,
Tax & Surcharge
- Success metric: R², MAE, RMSE

### 1.3 Load & Inspect the Dataset (Notebook — Code Cell)

```python
from src.data_loader import load_dataset, inspect_dataset

df = load_dataset("data/raw/Flight_Price_Dataset_of_Bangladesh.csv")
inspect_dataset(df)
```

- `load_dataset()` reads the CSV with pandas
- `inspect_dataset()` prints `.info()`, `.describe()`, `.head()`, `.shape`

### 1.4 Catalog Data Quality Issues (Notebook — Code + Markdown)

- Count missing values per column: `df.isnull().sum()`
- Identify outliers with boxplots on numerical columns
- List columns with wrong dtypes (e.g., fare stored as string)
- Check for duplicate rows: `df.duplicated().sum()`

### 1.5 Document Assumptions & Limitations (Notebook — Markdown Cell)

- Note dataset geography (Bangladesh domestic/regional flights)
- Note temporal coverage (date range of the data)
- State any columns you plan to drop and why

### 1.6 Commit Checkpoint

```bash
git add . && git commit -m "Phase 1: Problem definition and data understanding complete"
```

---

## PHASE 2 — Data Cleaning & Preprocessing

> **Airflow task:** `clean_and_preprocess` → calls `src.pipeline.clean_and_preprocess()`
> **Notebook:** `notebooks/02_data_cleaning_preprocessing.ipynb`
> **Source modules:** `src/preprocessing.py`, `src/feature_engineering.py`

### What the Airflow Task Does

The `clean_and_preprocess` task automatically:

1. Loads the raw CSV
2. Renames raw columns (e.g., `Total Fare (BDT)` → `Total Fare`)
3. Drops redundant columns (`Source Name`, `Destination Name`, `Arrival Date & Time`)
4. Imputes missing values (median for numeric, mode for categorical)
5. Fixes invalid entries (negative fares, inconsistent city names)
6. Validates and converts data types
7. Engineers date features (Month, Day, Weekday, Season)
8. Saves cleaned data to `data/processed/cleaned_with_features.csv`
9. One-hot encodes categorical variables
10. Scales numerical features (StandardScaler on Base Fare, Tax & Surcharge)
11. Saves the fitted scaler to `data/processed/scaler.joblib`
12. Performs 80/20 train-test split and saves to `data/processed/`

### 2.1 Drop Irrelevant Columns (Notebook — Code Cell)

```python
from src.preprocessing import drop_irrelevant_columns
df = drop_irrelevant_columns(df)
```

- Renames raw columns to standard names used throughout the pipeline
- Drops: "Unnamed", "Index", or any column that is just a row counter

### 2.2 Handle Missing Values

```python
from src.preprocessing import handle_missing_values
df = handle_missing_values(df)
```

- Numerical columns → impute with **median** (robust to outliers)
- Categorical columns → impute with **mode** or "Unknown"
- Log how many rows/values were imputed

### 2.3 Correct Invalid Entries

```python
from src.preprocessing import fix_invalid_entries
df = fix_invalid_entries(df)
```

- Remove rows where Base Fare or Total Fare is negative
- Normalize city names (e.g., "Dacca" → "Dhaka", strip whitespace)
- Report number of corrections made

### 2.4 Validate & Convert Data Types

```python
from src.preprocessing import validate_dtypes
df = validate_dtypes(df)
```

- Base Fare, Tax & Surcharge, Total Fare → `float64`
- Date → `datetime64`
- Airline, Source, Destination → `category`

### 2.5 Feature Engineering

```python
from src.feature_engineering import create_date_features, encode_categoricals, scale_numericals
df = create_date_features(df)
df_encoded = encode_categoricals(df)
df_scaled, scaler = scale_numericals(df_encoded)
```

- Extract: Month, Day, Weekday, Season from Date
- One-Hot Encode: Airline, Source, Destination, Stopovers, etc.
- Scale: StandardScaler on Base Fare, Tax & Surcharge

### 2.6 Train/Test Split

```python
from src.feature_engineering import split_data
X_train, X_test, y_train, y_test = split_data(df_scaled, target="Total Fare")
```

- 80% train / 20% test, `random_state=42` for reproducibility
- Save splits to `data/processed/`

### 2.7 Before/After Summary (Notebook — Markdown Cell)


| Metric         | Before | After |
| -------------- | ------ | ----- |
| Rows           | ?      | ?     |
| Columns        | ?      | ?     |
| Missing values | ?      | 0     |
| Duplicate rows | ?      | 0     |


### 2.8 Commit Checkpoint

```bash
git add . && git commit -m "Phase 2: Data cleaning and preprocessing complete"
```

---

## PHASE 3 — Exploratory Data Analysis (EDA)

> **Airflow task:** `generate_eda_report` → calls `src.pipeline.generate_eda_report()`
> **Notebook:** `notebooks/03_exploratory_data_analysis.ipynb`
> **Source module:** `src/visualization.py`

### What the Airflow Task Does

The `generate_eda_report` task automatically:

1. Reads the cleaned dataset from `data/processed/cleaned_with_features.csv`
2. Generates all visualizations (correlation heatmap, fare distribution,
  fare by airline, seasonal fares)
3. Saves all figures to `reports/figures/`
4. Computes KPIs and saves them to `data/processed/eda_kpis.json`

### 3.1 Descriptive Statistics

```python
df.groupby("Airline")["Total Fare"].describe()
df.groupby("Source")["Total Fare"].describe()
```

### 3.2 Correlation Analysis

```python
from src.visualization import plot_correlation_heatmap
plot_correlation_heatmap(df)
```

- Examine multicollinearity between Base Fare and Total Fare
- Save figure to `reports/figures/correlation_heatmap.png`

### 3.3 Fare Distribution

```python
from src.visualization import plot_fare_distribution
plot_fare_distribution(df)
```

- Histogram of Total Fare with KDE overlay
- Note skewness — may need log transform for modeling

### 3.4 Fare by Airline (Bar Chart + Boxplot)

```python
from src.visualization import plot_fare_by_airline
plot_fare_by_airline(df)
```

- Average fare per airline (bar chart)
- Fare spread per airline (boxplot)

### 3.5 Seasonal Fare Variation

```python
from src.visualization import plot_seasonal_fares
plot_seasonal_fares(df)
```

- Boxplot: fare by season (Winter, Summer, Monsoon, Autumn)
- Line chart: average fare by month

### 3.6 KPI Dashboard

- Average fare per airline
- Most popular route (highest frequency)
- Top 5 most expensive routes
- Seasonal peaks and troughs

### 3.7 Commit Checkpoint

```bash
git add . && git commit -m "Phase 3: Exploratory data analysis complete"
```

---

## PHASE 4 — Baseline Model Development

> **Airflow task:** `train_baseline_model` → calls `src.pipeline.train_baseline_model()`
> **Notebook:** `notebooks/04_baseline_model_development.ipynb`
> **Source modules:** `src/models.py`, `src/evaluation.py`

### What the Airflow Task Does

The `train_baseline_model` task automatically:

1. Loads train/test splits from `data/processed/`
2. Trains a Linear Regression model
3. Evaluates R², MAE, RMSE on the test set
4. Generates actual-vs-predicted and residual plots in `reports/figures/`
5. Saves the model to `models/linear_regression_baseline.joblib`
6. Saves metrics to `data/processed/baseline_metrics.json`

> **Note:** This task runs **in parallel** with `generate_eda_report`
> after the preprocessing task completes.

### 4.1 Train Linear Regression (Notebook — Code Cell)

```python
from src.models import train_model
model = train_model("linear_regression", X_train, y_train)
```

### 4.2 Evaluate on Test Set

```python
from src.evaluation import evaluate_model, print_metrics
metrics = evaluate_model(model, X_test, y_test)
print_metrics(metrics)
```

- Output: R², MAE, RMSE

### 4.3 Actual vs Predicted Plot

```python
from src.visualization import plot_actual_vs_predicted
plot_actual_vs_predicted(y_test, model.predict(X_test))
```

### 4.4 Residual Analysis

```python
from src.visualization import plot_residuals
plot_residuals(y_test, model.predict(X_test))
```

- Histogram of residuals (should be ~normal)
- Residuals vs predicted (should be ~random scatter)

### 4.5 Save Baseline Model

```python
from src.models import save_model
save_model(model, "models/linear_regression_baseline.joblib")
```

### 4.6 Set Improvement Target (Notebook — Markdown Cell)

- Record baseline R², MAE, RMSE
- Set target: e.g., "Improve R² by at least 0.05 in Phase 5"

### 4.7 Commit Checkpoint

```bash
git add . && git commit -m "Phase 4: Baseline model (Linear Regression) complete"
```

---

## PHASE 5 — Advanced Modeling & Optimization

> **Airflow task:** `train_advanced_models` → calls `src.pipeline.train_advanced_models()`
> **Notebook:** `notebooks/05_advanced_modeling_optimization.ipynb`
> **Source modules:** `src/models.py`, `src/evaluation.py`

### What the Airflow Task Does

The `train_advanced_models` task automatically:

1. Loads train/test splits from `data/processed/`
2. Trains: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest
3. Attempts XGBoost (gracefully skipped if unavailable)
4. Tunes Random Forest and XGBoost with `RandomizedSearchCV` (5-fold CV)
5. Runs cross-validation on Ridge and Random Forest
6. Builds a model comparison table → `data/processed/model_comparison.csv`
7. Generates bias-variance tradeoff plots → `reports/figures/`
8. Selects the best model by R² → `models/best_model.joblib`

### 5.1 Train Multiple Models (Notebook — Code Cell)

```python
models_to_train = [
    "ridge", "lasso", "decision_tree",
    "random_forest", "xgboost"
]
results = {}
for name in models_to_train:
    model = train_model(name, X_train, y_train)
    results[name] = evaluate_model(model, X_test, y_test)
```

### 5.2 Hyperparameter Tuning

```python
from src.models import tune_model
best_rf = tune_model("random_forest", X_train, y_train, cv=5)
best_xgb = tune_model("xgboost", X_train, y_train, cv=5)
```

- Use `RandomizedSearchCV` for efficiency
- Log best parameters found

### 5.3 Cross-Validation

```python
from src.evaluation import cross_validate_model
cv_scores = cross_validate_model(best_rf, X_train, y_train, cv=5)
```

### 5.4 Model Comparison Table

```python
from src.evaluation import build_comparison_table
comparison_df = build_comparison_table(results)
comparison_df
```


| Model             | R²  | MAE | RMSE |
| ----------------- | --- | --- | ---- |
| Linear Regression | ?   | ?   | ?    |
| Ridge             | ?   | ?   | ?    |
| Lasso             | ?   | ?   | ?    |
| Decision Tree     | ?   | ?   | ?    |
| Random Forest     | ?   | ?   | ?    |
| XGBoost           | ?   | ?   | ?    |


### 5.5 Regularization & Bias-Variance Analysis

- Plot Ridge/Lasso coefficients vs alpha
- Show how regularization reduces overfitting
- Plot training vs validation error curves

### 5.6 Select Best Model

- Choose the model with best R² that avoids overfitting
- Save to `models/best_model.joblib`

### 5.7 Commit Checkpoint

```bash
git add . && git commit -m "Phase 5: Advanced modeling and optimization complete"
```

---

## PHASE 6 — Model Interpretation & Insights

> **Airflow task:** `interpret_and_report` → calls `src.pipeline.interpret_and_report()`
> **Notebook:** `notebooks/06_model_interpretation_insights.ipynb`

### What the Airflow Task Does

The `interpret_and_report` task automatically:

1. Loads the best model and the baseline Linear Regression model
2. Generates feature importance plot (tree-based) → `reports/figures/`
3. Generates coefficient plot (linear model) → `reports/figures/`
4. Extracts airline, seasonal, and route importance rankings
5. Saves the full interpretation report to `data/processed/interpretation_report.json`

### 6.1 Feature Importance (Tree-Based Models)

```python
from src.visualization import plot_feature_importance
plot_feature_importance(best_model, feature_names)
```

### 6.2 Linear Model Coefficients

```python
from src.visualization import plot_coefficients
plot_coefficients(ridge_model, feature_names)
```

### 6.3 Answer Business Questions (Notebook — Markdown Cells)

1. **What factors most influence fare prices?**
  → Rank by feature importance scores
2. **How do airlines differ in pricing strategy?**
  → Compare fare distributions and model predictions per airline
3. **Do certain seasons or routes show higher fares?**
  → Cross-reference EDA findings with model behavior

### 6.4 Non-Technical Stakeholder Summary (Notebook — Markdown Cell)

- 3–5 bullet points summarizing the key findings
- Clear, data-backed recommendations
- Suggested business actions

### 6.5 Commit Checkpoint

```bash
git add . && git commit -m "Phase 6: Model interpretation and insights complete"
```

---

## PHASE 7 — Running the Full Pipeline via Airflow

> **DAG:** `dags/flight_fare_pipeline.py`
> **Pipeline tasks:** `src/pipeline.py`
> **Docker:** `Dockerfile.airflow`, `docker-compose.yml` (airflow profile)

This phase describes how to run the complete ML pipeline (Phases 1–6) as
a single Airflow DAG. Each phase maps to a dedicated Airflow task backed
by functions in `src/pipeline.py`, which in turn call the reusable modules
in `src/`.

### 7.1 Architecture Overview

The DAG `flight_fare_prediction` defines six tasks:

```
load_and_validate
       │
clean_and_preprocess
     ┌───┴───┐
eda_report  baseline_model     ← run in parallel
     └───┬───┘
train_advanced_models
       │
interpret_and_report
```


| Task                    | Calls                                  | Phase |
| ----------------------- | -------------------------------------- | ----- |
| `load_and_validate`     | `src.pipeline.load_and_validate()`     | 1     |
| `clean_and_preprocess`  | `src.pipeline.clean_and_preprocess()`  | 2     |
| `generate_eda_report`   | `src.pipeline.generate_eda_report()`   | 3     |
| `train_baseline_model`  | `src.pipeline.train_baseline_model()`  | 4     |
| `train_advanced_models` | `src.pipeline.train_advanced_models()` | 5     |
| `interpret_and_report`  | `src.pipeline.interpret_and_report()`  | 6     |


**Schedule:** `@weekly` (every Sunday at midnight). Can also be triggered
manually from the Airflow web UI.

### 7.2 Key Design Decisions

- **PythonOperator tasks** call functions in `src/pipeline.py`, which in
turn call the existing `src/` modules — no code duplication.
- **Headless matplotlib** (`Agg` backend) so figures render without a display.
- **File-based communication** between tasks (`data/processed/`, `models/`,
`reports/figures/`) — no XCom size limits.
- **Idempotent tasks** — every run safely overwrites previous outputs.
- `**@weekly` schedule** with `catchup=False` for rolling retraining.

### 7.3 File Reference


| File                           | Purpose                                                                                                                |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `src/pipeline.py`              | Task functions — each wraps one pipeline phase with headless matplotlib, absolute paths, and JSON artifact persistence |
| `dags/flight_fare_pipeline.py` | Airflow DAG definition with PythonOperators and dependency chain                                                       |
| `Dockerfile.airflow`           | Docker image based on `apache/airflow:2.8.1-python3.11` with project dependencies                                      |
| `docker-compose.yml`           | Airflow services (postgres, init, webserver, scheduler) under the `airflow` profile                                    |


### 7.4 Start the Airflow Environment

**Prerequisites:** Docker Desktop running, dataset in `data/raw/`.

If you already started Airflow in Phase 0, skip to Step 4.

#### Step 1 — Build the Airflow image

```bash
docker compose --profile airflow build
```

#### Step 2 — Initialize Airflow (one-time setup)

```bash
docker compose --profile airflow run --rm airflow-init
```

This runs database migrations and creates an admin user:

- Username: `admin`
- Password: `admin`

#### Step 3 — Start Airflow services

```bash
docker compose --profile airflow up -d
```

This starts:

- **PostgreSQL** — Airflow metadata database (port not exposed)
- **Airflow Webserver** — UI at **[http://localhost:8081](http://localhost:8081)**
- **Airflow Scheduler** — Polls for DAG runs

#### Step 4 — Trigger the pipeline

1. Navigate to **[http://localhost:8081](http://localhost:8081)**
2. Log in with `admin` / `admin`
3. Find the **flight_fare_prediction** DAG
4. The DAG is unpaused by default — click **Trigger DAG** (▶) to run
  immediately, or wait for the weekly schedule

### 7.5 Monitor a Pipeline Run

- **Graph view:** Visual task dependency graph with live status colors
- **Task logs:** Click any task → **Log** tab to see full Python output
- **Expected runtime:** 5–15 minutes depending on hardware (the advanced
modeling phase with hyperparameter tuning is the slowest)

### 7.6 Pipeline Artifacts

After a successful run, the following files are produced:


| Artifact              | Path                                              |
| --------------------- | ------------------------------------------------- |
| Data quality summary  | `data/processed/data_summary.json`                |
| Cleaned dataset       | `data/processed/cleaned_with_features.csv`        |
| Train/test splits     | `data/processed/X_train.csv`, `y_train.csv`, etc. |
| Training column names | `data/processed/train_columns.json`               |
| Fitted scaler         | `data/processed/scaler.joblib`                    |
| EDA KPIs              | `data/processed/eda_kpis.json`                    |
| Baseline metrics      | `data/processed/baseline_metrics.json`            |
| Model comparison      | `data/processed/model_comparison.csv`             |
| Interpretation report | `data/processed/interpretation_report.json`       |
| Baseline model        | `models/linear_regression_baseline.joblib`        |
| Best model            | `models/best_model.joblib`                        |
| All figures           | `reports/figures/*.png`                           |


### 7.7 Scheduled Retraining

The DAG is configured with `schedule="@weekly"` and `catchup=False`.
To enable automatic retraining:

1. Place updated data in `data/raw/Flight_Price_Dataset_of_Bangladesh.csv`
2. Ensure Airflow services are running
3. The scheduler will trigger the next run automatically

To change the frequency, edit `schedule` in
`dags/flight_fare_pipeline.py` (e.g., `"@daily"`, `"0 2 * * MON"`).

### 7.8 Stopping Airflow

```bash
docker compose --profile airflow down
```

To also remove volumes (database, logs):

```bash
docker compose --profile airflow down -v
```

### 7.9 Commit Checkpoint

```bash
git add . && git commit -m "Phase 7: Airflow pipeline orchestration verified"
```

---

## PHASE 8 (Stretch) — Flask API Deployment

### 8.1 Build Flask API

- `app/app.py` with `/predict` and `/health` endpoints
- Accepts JSON: `{"airline": "...", "source": "...", "destination": "...", "date": "..."}`
- Returns: `{"predicted_fare": 12345.67}`
- Loads the best model and scaler produced by the Airflow pipeline

### 8.2 Start the API

```bash
docker compose --profile api up api
```

### 8.3 Test the API

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" \
  -d '{"airline":"Biman","source":"Dhaka","destination":"Chittagong","date":"2026-03-15"}'
```

### 8.4 Commit Checkpoint

```bash
git add . && git commit -m "Phase 8: Flask API deployment complete"
```

---

## Summary of Git Commit Checkpoints


| Commit | Phase | Message                                                          |
| ------ | ----- | ---------------------------------------------------------------- |
| 1      | 0     | Initial project scaffold with Docker, Airflow, and documentation |
| 2      | 1     | Phase 1: Problem definition and data understanding complete      |
| 3      | 2     | Phase 2: Data cleaning and preprocessing complete                |
| 4      | 3     | Phase 3: Exploratory data analysis complete                      |
| 5      | 4     | Phase 4: Baseline model (Linear Regression) complete             |
| 6      | 5     | Phase 5: Advanced modeling and optimization complete             |
| 7      | 6     | Phase 6: Model interpretation and insights complete              |
| 8      | 7     | Phase 7: Airflow pipeline orchestration verified                 |
| 9      | 8     | Phase 8: Flask API deployment complete                           |


---

## Troubleshooting

### Docker build fails with "requirements.txt not found"

You are running `docker compose build` from the wrong directory. All
commands must be run from the **project root** (where `docker-compose.yml`
and `requirements.txt` live).

### `docker-compose` returns "no configuration file provided"

Modern Docker Desktop uses `docker compose` (without the hyphen). Replace
`docker-compose` with `docker compose` in all commands.

### "Total Fare" KeyError in notebooks

The raw dataset column is `Total Fare (BDT)`. You must call
`drop_irrelevant_columns(df)` first — it renames `Total Fare (BDT)` →
`Total Fare`. The Airflow pipeline handles this automatically; the error
only occurs in notebooks if preprocessing steps are skipped or run out of
order.

### Airflow DAG not visible in the UI

- Ensure the `dags/` folder is mounted: check that `docker-compose.yml`
has `./dags:/opt/airflow/dags` in the volumes section.
- Check scheduler logs: `docker compose --profile airflow logs airflow-scheduler`
- The DAG file must be parseable — check for import errors in the logs.

### Airflow task fails

- Click the failed task in the Graph view → **Log** tab for the full
Python traceback.
- The most common cause is a missing file from an earlier task. Re-trigger
the DAG from the beginning if inter-task files are missing.

