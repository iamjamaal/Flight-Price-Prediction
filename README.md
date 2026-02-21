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
dates, travel class, booking lead time, and temporal features derived from the
departure date.

> **Note on data leakage:** `Base Fare` and `Tax & Surcharge` are present in the
> raw dataset but are **excluded from all model features** — they form an
> arithmetic identity with the target (`Total Fare = Base Fare + Tax & Surcharge`)
> and would cause leakage. Including them inflates R² to ~1.0 artificially.

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
│   ├── constants.py            # Shared constants (SEASON_MAP, CITY_NAME_ALIASES)
│   ├── data_loader.py          # Dataset loading & initial inspection
│   ├── preprocessing.py        # Cleaning, imputation, type conversion
│   ├── feature_engineering.py  # New features, encoding, scaling
│   ├── models.py               # Model training, tuning, versioned persistence
│   ├── evaluation.py           # Metrics, comparison tables
│   ├── visualization.py        # All plotting utilities
│   └── pipeline.py             # Airflow-callable task functions
├── dags/
│   └── flight_fare_pipeline.py # Airflow DAG definition
├── app/
│   ├── app.py                  # Flask REST API with Swagger UI & rate limiting
│   └── streamlit_app.py        # Streamlit web app (stretch challenge)
├── tests/
│   ├── test_preprocessing.py   # pytest — preprocessing unit tests
│   ├── test_feature_engineering.py  # pytest — feature engineering tests
│   └── test_api.py             # pytest — Flask API tests (fully mocked)
├── models/                     # Serialized trained models (.joblib)
├── reports/
│   └── figures/                # Saved charts and plots
├── docs/
│   ├── ROADMAP.md                  # Phased delivery roadmap with status
│   ├── EXECUTION_PLAN.md           # Detailed step-by-step execution guide
│   ├── IMPROVEMENTS.md             # Pipeline improvement log
│   └── LAB_REQUIREMENTS_EVALUATION.md  # Assessment criteria checklist
├── PROJECT_DOCUMENTATION.md    # Full step-by-step implementation writeup
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
4. Click **Trigger DAG**  to run the full pipeline

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

## Optional: Streamlit Web App (Stretch)

After the Airflow pipeline has produced a trained model, launch the interactive
Streamlit web application:

```bash
docker compose --profile streamlit up streamlit
```

Open your browser at **http://localhost:8501** to access the app.

**Features:**
- Interactive flight details form
- Real-time fare predictions
- KPI dashboard with airline and seasonal insights
- Model performance metrics

To run locally without Docker:
```bash
streamlit run app/streamlit_app.py
```

---

## Optional: Flask REST API (Stretch)

For programmatic access, use the Flask REST API:

```bash
docker compose --profile api up api
```

The REST API will be available at **http://localhost:5000**.

Test it:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"airline":"Biman Bangladesh Airlines","source":"DAC","destination":"CGP","date":"2026-03-15"}'
```

**API features:**
- `POST /predict` — fare prediction with input validation against known training values (400 if unknown airline/route)
- `GET /health` — health check
- `GET /apidocs/` — interactive Swagger UI (OpenAPI 2.0)
- Rate limited to **30 requests/minute** per IP (in-memory, no Redis required)
- Debug mode controlled by `FLASK_DEBUG` environment variable (off by default)

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
| API input validation values | `data/processed/known_values.json` |
| EDA KPIs | `data/processed/eda_kpis.json` |
| Baseline metrics | `data/processed/baseline_metrics.json` |
| Model comparison | `data/processed/model_comparison.csv` |
| Interpretation report | `data/processed/interpretation_report.json` |
| Baseline model | `models/linear_regression_baseline.joblib` |
| Best model (canonical) | `models/best_model.joblib` |
| Best model (versioned) | `models/best_model_v<YYYYMMDD_HHMMSS>.joblib` |
| Model audit registry | `models/model_registry.json` |
| All figures | `reports/figures/*.png` |

See [`docs/EXECUTION_PLAN.md`](docs/EXECUTION_PLAN.md) for the full
step-by-step guide including troubleshooting.

For a detailed writeup of how each project step was implemented — including
code, metrics, and findings — see [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md).

---

## Key Findings

| Finding | Detail |
|---|---|
| Best model R² | **0.8935** (Linear Regression and Ridge — tied) |
| Leakage fix | Removing `Base Fare` & `Tax & Surcharge` dropped R² from ~1.0 to honest 0.89 |
| Log₁p transform | Linearised the right-skewed fare distribution (skewness 1.58); enabled linear models to match ensemble methods |
| Winter fare premium | **+16.2%** over Autumn — December–February is the highest-demand window |
| Airline spread | ~7,400 BDT between most expensive (IndiGo) and cheapest (Vistara) carriers |
| Most expensive route | SPD → BKK: 117,952 BDT average |
| Dataset quality | 57,000 records — zero missing values, zero duplicates |

### Final Model Ranking (pipeline run — log₁p-scale metrics)

> **Note:** XGBoost requires a separate `pip install xgboost` and was not present
> in this environment. All other models ran to completion.

| Rank | Model | R² | MAE | RMSE |
|---|---|---|---|---|
| 1 | Linear Regression | **0.8935** | 0.35 | 0.46 |
| 1 | Ridge | **0.8935** | 0.35 | 0.46 |
| 3 | Random Forest (Tuned) | 0.8932 | 0.35 | 0.46 |
| 4 | Lasso | 0.8931 | 0.35 | 0.46 |
| 5 | Random Forest | 0.8878 | 0.36 | 0.47 |
| 6 | Decision Tree | 0.778 | 0.48 | 0.66 |

---

## Dataset

| Field             | Description                                  | Used as Feature |
|-------------------|----------------------------------------------|---|
| Airline           | Name of the airline carrier                  | Yes |
| Source            | Departure airport code                       | Yes |
| Destination       | Arrival airport code                         | Yes |
| Date              | Departure date (decomposed to Month, Day, Weekday, Season) | Yes (engineered) |
| Duration (hrs)    | Flight duration                              | Yes |
| Stopovers         | Direct / 1 Stop / 2+ Stops                  | Yes |
| Aircraft Type     | Boeing, Airbus, etc.                         | Yes |
| Class             | Economy / Business / First                   | Yes |
| Booking Source    | Online, travel agent, etc.                   | Yes |
| Days Before Dep.  | Booking lead time in days                    | Yes |
| Seasonality       | Pre-labelled season (cross-checked with Date) | Yes |
| Base Fare         | Ticket price before taxes                    | **No — leakage** |
| Tax & Surcharge   | Government taxes and fuel surcharges         | **No — leakage** |
| **Total Fare**    | **Target variable** (Base Fare + Tax)        | Target only |

---

## ML Pipeline Overview

| Phase | Airflow Task | Notebook | Description |
|-------|-------------|----------|-------------|
| 1 | `load_and_validate` | `01_problem_definition_data_understanding` | Load 57,000 records, inspect schema, document assumptions |
| 2 | `clean_and_preprocess` | `02_data_cleaning_preprocessing` | Impute, fix types, engineer date features, remove leakage columns, apply `log1p` transform, one-hot encode, StandardScale, 80/20 split |
| 3 | `generate_eda_report` | `03_exploratory_data_analysis` | Statistical summaries, 4 visualisations, KPI JSON (airline fares, popular routes, seasonal trends) |
| 4 | `train_baseline_model` | `04_baseline_model_development` | Linear Regression baseline — R²=0.8935 on log-scale target |
| 5 | `train_advanced_models` | `05_advanced_modeling_optimization` | Ridge, Lasso, Decision Tree, Random Forest (+ XGBoost when installed); RandomizedSearchCV tuning; bias–variance analysis; versioned model save |
| 6 | `interpret_and_report` | `06_model_interpretation_insights` | Airline coefficients, seasonal/route importance, stakeholder summary, leakage documentation |

---

## Evaluation Metrics

- **R² (Coefficient of Determination)** — proportion of variance explained
- **MAE (Mean Absolute Error)** — average absolute prediction error
- **RMSE (Root Mean Squared Error)** — penalizes large errors more heavily

---

## Technologies

- Python 3.11 | pandas | NumPy | scikit-learn
- Matplotlib | Seaborn | Plotly
- XGBoost (optional) | LightGBM
- Jupyter Notebook
- Docker & Docker Compose
- Apache Airflow 2.8 (pipeline orchestration & scheduled retraining)
- PostgreSQL 15 (Airflow metadata backend)
- Flask | flask-limiter | flasgger (REST API + rate limiting + Swagger UI)
- Streamlit (Web App)
- pytest (automated test suite — 51 tests)

---


