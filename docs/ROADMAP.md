# Project Roadmap — Flight Fare Prediction

> **Delivery approach:** Incremental phases. Each phase produces a working,
> documented artifact before the next phase begins.
>
> **Orchestration:** The ML pipeline (Phases 1–6) is orchestrated by an
> Apache Airflow DAG that runs all phases automatically. Notebooks are used
> for interactive development and documentation alongside the automated
> pipeline.

---

## Phase 0: Environment & Infrastructure Setup
**Status:** 🟢 Complete
**Estimated effort:** 1 session
**Deliverables:**
- [x] Project directory structure created
- [x] Dockerfile and docker-compose.yml configured
- [x] Dockerfile.airflow for Airflow image configured
- [x] requirements.txt with dependencies
- [x] Dataset downloaded and placed in `data/raw/`
- [x] Git repository initialized with `.gitignore`
- [x] Docker images built (notebook + Airflow)
- [x] Airflow database initialized and admin user created
- [x] Airflow UI accessible at http://localhost:8081
- [x] `flight_fare_prediction` DAG visible and active

**Documentation checkpoint:** Verify Docker and Airflow environments run;
record Python and library versions in the first notebook.

---

## Phase 1: Problem Definition & Data Understanding
**Status:** 🟢 Complete
**Estimated effort:** 1 session
**Notebook:** `01_problem_definition_data_understanding.ipynb`
**Airflow task:** `load_and_validate`
**Deliverables:**
- [x] Business question articulated in markdown cells
- [x] ML task formally defined (supervised regression, target = Total Fare)
- [x] Dataset loaded via `src/data_loader.py`
- [x] Schema inspection (`.info()`, `.describe()`, `.head()`)
- [x] Missing values, outliers, and data types catalogued
- [x] Assumptions and limitations documented
- [x] `src/pipeline.py` — `load_and_validate()` task function
- [x] Airflow task runs successfully (produces `data/processed/data_summary.json`)

**Documentation checkpoint:** Markdown summary at end of notebook stating data
quality findings and any risks for downstream modeling.

---

## Phase 2: Data Cleaning & Preprocessing
**Status:** 🟢 Complete
**Estimated effort:** 1–2 sessions
**Notebook:** `02_data_cleaning_preprocessing.ipynb`
**Airflow task:** `clean_and_preprocess`
**Deliverables:**
- [x] Irrelevant/duplicate columns dropped
- [x] Missing values handled (median for numeric, mode for categorical)
- [x] Invalid entries corrected (negative fares, inconsistent city names)
- [x] Data types validated and converted
- [x] Feature engineering: Month, Day, Weekday, Season from Date
- [x] Categorical encoding (One-Hot Encoding)
- [x] Numerical scaling (StandardScaler)
- [x] Data leakage fix: `Base Fare` and `Tax & Surcharge` dropped from features
- [x] Log-transform applied to `Total Fare` for skew reduction
- [x] Reusable logic extracted to `src/preprocessing.py` and `src/feature_engineering.py`
- [x] `src/pipeline.py` — `clean_and_preprocess()` task function
- [x] Airflow task runs successfully (produces train/test splits in `data/processed/`)

**Documentation checkpoint:** Before/after shape comparison table; encoding and
scaling decisions justified in markdown.

---

## Phase 3: Exploratory Data Analysis (EDA)
**Status:** 🟢 Complete
**Estimated effort:** 1–2 sessions
**Notebook:** `03_exploratory_data_analysis.ipynb`
**Airflow task:** `generate_eda_report`
**Deliverables:**
- [x] Descriptive statistics by airline, source, destination, season
- [x] Correlation matrix & heatmap
- [x] Fare distribution histogram
- [x] Boxplots: fare variation by airline and season
- [x] Average fare by month/season line chart
- [x] KPIs: avg fare per airline, most popular route, top 5 expensive routes
- [x] Seasonal fare variation (Eid, winter, etc.)
- [x] All figures saved to `reports/figures/`
- [x] `src/pipeline.py` — `generate_eda_report()` task function
- [x] Airflow task runs successfully (produces figures + `data/processed/eda_kpis.json`)

**Documentation checkpoint:** Key statistical findings summarized; hypotheses
about fare drivers stated for model validation in later phases.

---

## Phase 4: Baseline Model Development
**Status:** 🟢 Complete
**Estimated effort:** 1 session
**Notebook:** `04_baseline_model_development.ipynb`
**Airflow task:** `train_baseline_model`
**Deliverables:**
- [x] Linear Regression trained on training set
- [x] Evaluation: R², MAE, RMSE computed on test set
- [x] Actual vs Predicted scatter plot
- [x] Residual analysis (distribution + residuals vs predicted)
- [x] Baseline metrics recorded in comparison table
- [x] Model serialized to `models/linear_regression_baseline.joblib`
- [x] `src/pipeline.py` — `train_baseline_model()` task function
- [x] Airflow task runs successfully (produces model + `data/processed/baseline_metrics.json`)

**Documentation checkpoint:** Baseline performance documented; notes on
underfitting/overfitting patterns; target improvement margin set for Phase 5.

---

## Phase 5: Advanced Modeling & Optimization
**Status:** 🟢 Complete
**Estimated effort:** 2–3 sessions
**Notebook:** `05_advanced_modeling_optimization.ipynb`
**Airflow task:** `train_advanced_models`
**Deliverables:**
- [x] Ridge Regression trained & evaluated
- [x] Lasso Regression trained & evaluated
- [x] Decision Tree Regressor trained & evaluated
- [x] Random Forest Regressor trained & evaluated
- [x] Gradient Boosted Trees (XGBoost) trained & evaluated
- [x] Hyperparameter tuning via RandomizedSearchCV
- [x] Cross-validation scores computed for each model
- [x] Model comparison table (R², MAE, RMSE)
- [x] Bias-variance tradeoff analysis for Ridge & Lasso
- [x] Best model selected and serialized to `models/`
- [x] `src/pipeline.py` — `train_advanced_models()` task function
- [x] Airflow task runs successfully (produces `models/best_model.joblib` + comparison CSV)

**Documentation checkpoint:** Full comparison table with reasoning for final
model choice; regularization effects documented with plots.

---

## Phase 6: Model Interpretation & Insights
**Status:** 🟢 Complete
**Estimated effort:** 1–2 sessions
**Notebook:** `06_model_interpretation_insights.ipynb`
**Airflow task:** `interpret_and_report`
**Deliverables:**
- [x] Linear model coefficients analyzed
- [x] Tree-based feature importance plot
- [x] Top factors influencing fare identified
- [x] Airline pricing strategy comparison
- [x] Seasonal/route fare patterns documented
- [x] Non-technical stakeholder summary written
- [x] Data-backed recommendations provided
- [x] Data leakage analysis documented
- [x] Log-transform improvement documented
- [x] `src/pipeline.py` — `interpret_and_report()` task function
- [x] Airflow task runs successfully (produces `data/processed/interpretation_report.json`)

**Documentation checkpoint:** Final executive summary with actionable insights
and suggested next steps.

---

## Phase 7: Airflow Pipeline Orchestration
**Status:** 🟢 Complete
**Estimated effort:** 1 session
**DAG:** `dags/flight_fare_pipeline.py`
**Pipeline tasks:** `src/pipeline.py`
**Docker:** `Dockerfile.airflow` + `docker-compose.yml` (airflow profile)
**Deliverables:**
- [x] `src/pipeline.py` — standalone task functions for each pipeline phase
- [x] `dags/flight_fare_pipeline.py` — Airflow DAG with 6 tasks and dependency chain
- [x] `Dockerfile.airflow` — Airflow Docker image with project dependencies
- [x] `docker-compose.yml` — Postgres, Airflow init, webserver, and scheduler services
- [x] Airflow environment built and running (`docker compose --profile airflow up`)
- [x] DAG visible and triggerable in Airflow UI (http://localhost:8081)
- [x] Full pipeline run completes successfully (all 6 tasks green)
- [x] Scheduled weekly retraining verified

**Task dependency graph:**
```
load_and_validate → clean_and_preprocess → [eda_report | baseline_model] → train_advanced_models → interpret_and_report
```

**Documentation checkpoint:** Airflow setup instructions documented;
pipeline artifacts listed; retraining schedule explained.

---

## Phase 8 (Stretch): Flask API Deployment
**Status:** 🟡 Code Written (not yet tested)
**Estimated effort:** 1 session
**Deliverables:**
- [x] Flask app in `app/app.py` with `/predict` and `/health` endpoints
- [x] Input validation and error handling
- [x] Scaler and feature alignment for inference
- [ ] Docker service running (`docker compose --profile api up api`)
- [ ] API tested with sample request

---

## Progress Tracking

| Phase | Status | Notes |
|-------|--------|-------|
| 0 — Setup | 🟢 Complete | Docker images built; Airflow running on port 8081 |
| 1 — Problem & Data | 🟢 Complete | Notebook 01 executed; Airflow task green |
| 2 — Cleaning | 🟢 Complete | Leakage fix + log-transform applied; Airflow task green |
| 3 — EDA | 🟢 Complete | Figures generated; KPIs computed; Airflow task green |
| 4 — Baseline Model | 🟢 Complete | LR baseline trained; metrics saved; Airflow task green |
| 5 — Advanced Models | 🟢 Complete | 8 models compared; best model saved; Airflow task green |
| 6 — Interpretation | 🟢 Complete | Feature importance, business insights documented |
| 7 — Airflow Pipeline | 🟢 Complete | Full DAG runs end-to-end; weekly schedule active |
| 8 — Flask API | 🟡 Code Written | app.py ready; not yet tested |

> **Next milestone:** Execute notebooks 03–06 in Jupyter for visible outputs,
> test the Flask API, and make final commit.
