# Flight Fare Prediction Using Machine Learning

An end-to-end machine learning pipeline for predicting flight fares based on
airline, route, travel date, and pricing components. Built using the
**Flight_Price_Dataset_of_Bangladesh** from Kaggle.

The entire pipeline is **orchestrated by Apache Airflow**, running data
ingestion, cleaning, EDA, model training, optimization, and interpretation
as a single automated DAG.

---

## Business Context

Airlines and travel platforms need accurate fare estimates to power dynamic
pricing strategies and recommendation engines. This project frames that need as
a **supervised regression** problem where the target variable is **Total Fare**
and the feature space includes airline identity, source/destination cities, travel
dates, base fare, and tax & surcharge amounts.

---

## Project Structure

```
Flight-Price-Prediction/
├── data/
│   ├── raw/                    # Original untouched dataset
│   └── processed/              # Cleaned data, splits, metrics (Airflow output)
├── notebooks/
│   ├── 01_problem_definition_data_understanding.ipynb
│   ├── 02_data_cleaning_preprocessing.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_baseline_model_development.ipynb
│   ├── 05_advanced_modeling_optimization.ipynb
│   └── 06_model_interpretation_insights.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Dataset loading & initial inspection
│   ├── preprocessing.py        # Cleaning, imputation, type conversion
│   ├── feature_engineering.py  # New features, encoding, scaling
│   ├── models.py               # Model training, tuning, persistence
│   ├── evaluation.py           # Metrics, comparison tables
│   ├── visualization.py        # All plotting utilities
│   └── pipeline.py             # Airflow-callable task functions
├── dags/
│   └── flight_fare_pipeline.py # Airflow DAG definition
├── app/
│   └── app.py                  # Flask REST API (stretch challenge)
├── models/                     # Serialized trained models (.joblib)
├── reports/
│   └── figures/                # Saved charts and plots
├── docs/
│   ├── ROADMAP.md              # Phased delivery roadmap with status
│   └── EXECUTION_PLAN.md       # Detailed step-by-step execution guide
├── Dockerfile                  # Jupyter / API image
├── Dockerfile.airflow          # Airflow image
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quick Start — Running the Pipeline with Airflow

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Dataset file `Flight_Price_Dataset_of_Bangladesh.csv` placed in `data/raw/`

> **Important:** All commands must be run from the **project root** directory
> (where `docker-compose.yml` lives). Use `docker compose` (no hyphen).

### 1. Build the Airflow Docker image
```bash
docker compose --profile airflow build
```

### 2. Initialize the Airflow database (one-time)
```bash
docker compose --profile airflow run --rm airflow-init
```
Creates the database schema and an admin user (`admin` / `admin`).

### 3. Start Airflow services
```bash
docker compose --profile airflow up -d
```
This starts PostgreSQL, the Airflow webserver, and the scheduler.

### 4. Trigger the pipeline
1. Open **http://localhost:8081** in your browser
2. Log in with `admin` / `admin`
3. Find the **`flight_fare_prediction`** DAG (already active by default)
4. Click **Trigger DAG** (▶) to run the full pipeline

### 5. Monitor progress
Switch to **Graph** view in the Airflow UI to watch tasks execute:
```
load_and_validate → clean_and_preprocess → [eda_report | baseline_model] → train_advanced_models → interpret_and_report
```
Click any task → **Log** tab for full Python output.

### Stopping Airflow
```bash
docker compose --profile airflow down
```

---

## Optional: Interactive Development with Jupyter

For exploring data or running notebook cells interactively:
```bash
docker compose up notebook
```
Open your browser at **http://localhost:8888** — no token required.

---

## Optional: Flask Prediction API (Stretch)

After the Airflow pipeline has produced a trained model:
```bash
docker compose --profile api up api
```
The REST API will be available at **http://localhost:5000**.

Test it:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"airline":"Biman","source":"Dhaka","destination":"Chittagong","date":"2026-03-15"}'
```

---

## Airflow Pipeline

The entire ML workflow is orchestrated as an **Apache Airflow DAG** for
automated, scheduled retraining. Each pipeline phase maps to a dedicated
Airflow task backed by functions in `src/pipeline.py`.

### Task Dependency Graph

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

### Key Design Decisions

- **PythonOperator tasks** call functions in `src/pipeline.py`, which in
  turn call the existing `src/` modules — no code duplication.
- **Headless matplotlib** (`Agg` backend) so figures render without a display.
- **File-based communication** between tasks (`data/processed/`, `models/`,
  `reports/figures/`) — no XCom size limits.
- **Idempotent tasks** — every run safely overwrites previous outputs.
- **`@weekly` schedule** with `catchup=False` for rolling retraining.

### Pipeline Artifacts

After a successful DAG run, these outputs are produced:

| Artifact | Path |
|----------|------|
| Data quality summary | `data/processed/data_summary.json` |
| Cleaned dataset | `data/processed/cleaned_with_features.csv` |
| Train/test splits | `data/processed/X_train.csv`, `y_train.csv`, etc. |
| Fitted scaler | `data/processed/scaler.joblib` |
| EDA KPIs | `data/processed/eda_kpis.json` |
| Baseline metrics | `data/processed/baseline_metrics.json` |
| Model comparison | `data/processed/model_comparison.csv` |
| Interpretation report | `data/processed/interpretation_report.json` |
| Baseline model | `models/linear_regression_baseline.joblib` |
| Best model | `models/best_model.joblib` |
| All figures | `reports/figures/*.png` |

See [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md) for the full
step-by-step guide including troubleshooting.

---

## Dataset

| Field             | Description                                  |
|-------------------|----------------------------------------------|
| Airline           | Name of the airline carrier                  |
| Source            | Departure city                               |
| Destination       | Arrival city                                 |
| Date              | Date of travel                               |
| Base Fare         | Ticket price before taxes                    |
| Tax & Surcharge   | Government taxes and fuel surcharges         |
| **Total Fare**    | **Target variable** (Base Fare + Tax)        |

---

## ML Pipeline Overview

| Phase | Airflow Task | Notebook | Description |
|-------|-------------|----------|-------------|
| 1 | `load_and_validate` | `01_problem_definition_data_understanding` | Load data, inspect schema, document assumptions |
| 2 | `clean_and_preprocess` | `02_data_cleaning_preprocessing` | Handle missing values, fix types, engineer features |
| 3 | `generate_eda_report` | `03_exploratory_data_analysis` | Statistical summaries, visual trends, KPIs |
| 4 | `train_baseline_model` | `04_baseline_model_development` | Linear Regression baseline with R², MAE, RMSE |
| 5 | `train_advanced_models` | `05_advanced_modeling_optimization` | Ridge, Lasso, Decision Tree, Random Forest, XGBoost |
| 6 | `interpret_and_report` | `06_model_interpretation_insights` | Feature importance, business insights, recommendations |

---

## Evaluation Metrics

- **R² (Coefficient of Determination)** — proportion of variance explained
- **MAE (Mean Absolute Error)** — average absolute prediction error
- **RMSE (Root Mean Squared Error)** — penalizes large errors more heavily

---

## Technologies

- Python 3.11 | pandas | NumPy | scikit-learn
- Matplotlib | Seaborn | Plotly
- XGBoost | LightGBM
- Jupyter Notebook
- Docker & Docker Compose
- Apache Airflow 2.8 (pipeline orchestration & scheduled retraining)
- PostgreSQL 15 (Airflow metadata backend)
- Flask (REST API)

---

## License

This project is developed for educational purposes as part of the
AmaliTech DEM09 Data Science Module Lab.
