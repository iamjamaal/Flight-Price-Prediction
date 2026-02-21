# Lab Requirements Evaluation — Flight Fare Prediction

This document evaluates the project against the **Module Lab: Flight Fare Prediction Using Machine Learning** completion requirements (as of the lab specification).

---

## Summary

| Category | Status | Notes |
|----------|--------|--------|
| **Step 1: Problem Definition & Data Understanding** | ✅ Met | Business goal, ML task, load/inspect, initial observations all present. |
| **Step 2: Data Cleaning & Preprocessing** | ✅ Met | Cleaning, encoding, scaling, split implemented; one minor optional gap. |
| **Step 3: EDA** | ✅ Met | Descriptive stats, visual analysis, KPIs and suggested visualizations covered. |
| **Step 4: Baseline Model** | ✅ Met | Linear Regression, R²/MAE/RMSE, actual vs predicted, residuals. |
| **Step 5: Advanced Modeling & Optimization** | ✅ Met | Multiple models, tuning, comparison table, bias–variance. |
| **Step 6: Model Interpretation & Insights** | ✅ Met | Feature importance, coefficients, business questions, stakeholder summary. |
| **Suggested Visualizations** | ✅ Met | All six suggested charts/plots implemented. |
| **Stretch Challenges** | ✅ Met | Flask API and Airflow integration both implemented. |
| **Evaluation Criteria** | ✅ Addressed | Implementation, readability, EDA, model comparison, insights, structure. |

**Overall:** The project **meets or exceeds** the lab requirements. Data leakage was identified and fixed; log-transform and documentation are in place.

---

## Step 1: Problem Definition & Data Understanding

| Requirement | Evidence | Status |
|-------------|----------|--------|
| Understand the business goal | README “Business Context”; Notebook 01 and pipeline narrative describe airlines/travel platforms needing fare estimates. | ✅ Met |
| Define the ML task: Supervised Regression, target = Total Fare, features listed | README and Notebook 01 state task type, target, and feature set (Airline, Source, Destination, Date, etc.). | ✅ Met |
| Load dataset: `Flight_Price_Dataset_of_Bangladesh.csv` | `src/data_loader.py` — `load_dataset()` with default path `data/raw/Flight_Price_Dataset_of_Bangladesh.csv`; Kaggle referenced in README. | ✅ Met |
| Inspect structure: `.info()`, `.describe()`, `.head()` | `inspect_dataset()` in `data_loader.py` prints dtypes, shape, missing values, duplicates, describe, head. | ✅ Met |
| Initial observations: missing data, outliers, categoricals, numerical ranges | Phase 1 notebook and pipeline document missing values, duplicates, and schema; ROADMAP/EXECUTION_PLAN reference data quality. | ✅ Met |
| Document assumptions and limitations | Notebook 01 and docs (e.g. EDA hypotheses, leakage note) capture assumptions and limitations. | ✅ Met |

---

## Step 2: Data Cleaning & Preprocessing

### A. Cleaning Data

| Requirement | Evidence | Status |
|-------------|----------|--------|
| Drop irrelevant/duplicate columns (e.g. Unnamed, Index) | `preprocessing.py`: `drop_irrelevant_columns()` drops columns whose names start with “unnamed” or “index”; redundant columns listed. | ✅ Met |
| Missing values: numerical → mean/median; categorical → “Unknown” or mode | `preprocessing.py`: `handle_missing_values()` — median for numeric, mode (fallback “Unknown”) for categorical. | ✅ Met |
| Invalid entries: negative fares → replace or remove; inconsistent city names → normalize | `preprocessing.py`: `fix_invalid_entries()` removes rows with negative fare values; city map normalizes e.g. Dacca→Dhaka, Chattogram/Chottogram→Chittagong. | ✅ Met |
| Validate data types: fare columns → float; date → datetime | `preprocessing.py`: `validate_dtypes()` converts fare/tax columns to numeric, date columns to datetime, categoricals to category. | ✅ Met |

### B. Feature Engineering

| Requirement | Evidence | Status |
|-------------|----------|--------|
| Total Fare = Base Fare + Tax & Surcharge (if missing) | Not explicitly implemented; missing Total Fare would be imputed by median in `handle_missing_values()`. Lab treats this as an optional derivation when target is missing. | ⚠️ Minor gap |
| Month, Day, Weekday, Season from Date | `feature_engineering.py`: `create_date_features()` adds Month, Day, Weekday, WeekdayName, Season (Bangladesh climate mapping). | ✅ Met |
| Encode categoricals: One-Hot or Label (Airline, Source, Destination) | `feature_engineering.py`: `encode_categoricals()` uses `pd.get_dummies()` (one-hot) for Airline, Source, Destination, Season, etc.; Label Encoding supported. | ✅ Met |
| Scale numericals: StandardScaler or MinMaxScaler | `feature_engineering.py`: `scale_numericals()` uses StandardScaler; scaler persisted and used in pipeline/API. | ✅ Met |
| Train–test split (e.g. 80/20) | `feature_engineering.py`: `split_data()` uses `train_test_split()` with configurable `test_size`; pipeline uses 80/20 and saves splits. | ✅ Met |

---

## Step 3: Exploratory Data Analysis (EDA)

| Requirement | Evidence | Status |
|-------------|----------|--------|
| Descriptive statistics: fares by airline, source, destination, season | Pipeline and Notebook 03: groupbys and EDA KPIs; `eda_kpis.json` includes avg_fare_by_airline, avg_fare_by_season; popular and expensive routes. | ✅ Met |
| Correlations among numerical features | `visualization.py`: `plot_correlation_heatmap()`; used in EDA report and notebook. | ✅ Met |
| Distributions of fares, base fares, taxes | `plot_fare_distribution()`; fare distribution in EDA. | ✅ Met |
| Boxplots: fare variation across airlines | `plot_fare_by_airline()` — bar chart (mean fare) and boxplot by airline. | ✅ Met |
| Average fare by month/season | `plot_seasonal_fares()` — boxplot by season and line chart by month. | ✅ Met |
| Correlation heatmap (multicollinearity) | `plot_correlation_heatmap()` — heatmap of numeric columns. | ✅ Met |
| KPI: Average fare per airline | `eda_kpis.json`: `avg_fare_by_airline`. | ✅ Met |
| KPI: Most popular route (frequency) | `eda_kpis.json`: `popular_routes` (top 10). | ✅ Met |
| KPI: Seasonal fare variation (e.g. Eid, winter) | `eda_kpis.json`: `avg_fare_by_season`; winter premium documented in notebooks. | ✅ Met |
| KPI: Top 5 most expensive routes | `eda_kpis.json`: `expensive_routes` (top 5); Notebook 03 and 06 reference them. | ✅ Met |

---

## Step 4: Model Development (Baseline)

| Requirement | Evidence | Status |
|-------------|----------|--------|
| Linear Regression as baseline | `models.py`: MODEL_REGISTRY includes `linear_regression` (sklearn); pipeline and Notebook 04 train it. | ✅ Met |
| Evaluate: R², MAE, RMSE | `evaluation.py`: `evaluate_model()` returns r2, mae, rmse; `print_metrics()` displays them; pipeline and notebooks use them (log-space post leakage fix). | ✅ Met |
| Actual vs predicted visualization | `visualization.py`: `plot_actual_vs_predicted()`; pipeline saves `baseline_actual_vs_predicted.png`; Notebook 04 uses it. | ✅ Met |
| Residual analysis (under/overfitting) | `visualization.py`: `plot_residuals()` (histogram + residuals vs predicted); pipeline saves `baseline_residuals.png`; Notebook 04 documents findings. | ✅ Met |
| Document key findings | Notebook 04 and Phase 4 summary describe baseline performance, leakage fix, and log-transform. | ✅ Met |

---

## Step 5: Advanced Modeling & Optimization

| Requirement | Evidence | Status |
|-------------|----------|--------|
| Ridge Regression | `models.py`: Ridge in MODEL_REGISTRY; pipeline and Notebook 05 train it. | ✅ Met |
| Lasso Regression | `models.py`: Lasso in MODEL_REGISTRY (alpha tuned for log-space); pipeline and Notebook 05. | ✅ Met |
| Decision Tree Regressor | `models.py`: DecisionTreeRegressor in MODEL_REGISTRY; pipeline and Notebook 05. | ✅ Met |
| Random Forest Regressor | `models.py`: RandomForestRegressor in MODEL_REGISTRY; pipeline and Notebook 05. | ✅ Met |
| Gradient Boosted Trees (optional) | `models.py`: XGBRegressor (and GradientBoostingRegressor) in MODEL_REGISTRY; Notebook 05 uses XGBoost. XGBoost requires `pip install xgboost`; gracefully skipped if absent. | ✅ Met |
| GridSearchCV or RandomizedSearchCV | `models.py`: `tune_model()` uses RandomizedSearchCV; pipeline and Notebook 05 tune RF and XGBoost. | ✅ Met |
| Compare models with cross-validation | `evaluation.py`: `cross_validate_model()`; Notebook 05 runs CV; pipeline compares models. | ✅ Met |
| Comparison table: Model, R², MAE, RMSE | `evaluation.py`: `build_comparison_table()`; pipeline writes `model_comparison.csv`; Notebook 05 displays table. | ✅ Met |
| Identify best model (accuracy vs simplicity) | Pipeline and Notebook 05 select best from comparison table; best model saved as `best_model.joblib`. | ✅ Met |
| Regularization & bias–variance: Ridge and Lasso effect; plot tradeoff | Pipeline and Notebook 05: bias–variance plot (Ridge/Lasso vs alpha); `bias_variance_tradeoff.png` saved. | ✅ Met |

---

## Step 6: Model Interpretation & Insights

| Requirement | Evidence | Status |
|-------------|----------|--------|
| Linear models: examine coefficients | `visualization.py`: `plot_coefficients()`; Notebook 06 and pipeline plot linear (baseline) coefficients. | ✅ Met |
| Tree-based: plot feature importances | `visualization.py`: `plot_feature_importance()`; Notebook 06 and pipeline use it when best model is tree-based. | ✅ Met |
| What factors most influence fare prices? | Notebook 06: “What factors most influence fare prices?” with airline, route, class, days before departure, season. | ✅ Met |
| How do airlines differ in pricing strategy? | Notebook 06: airline tiers (Premium/Mid-tier/Budget), coefficients, EDA fare-by-airline. | ✅ Met |
| Do certain seasons or routes show higher fares? | Notebook 06: winter premium, top expensive routes (SPD→BKK, CXB→YYZ, etc.). | ✅ Met |
| Summarize for non-technical stakeholders | Notebook 06: “Non-Technical Stakeholder Summary” with executive summary, recommendations, next steps. | ✅ Met |
| Data-backed recommendations | Notebook 06: recommendations (use corrected model, focus on Winter, collect richer features, etc.). | ✅ Met |

---

## Suggested Visualizations (Lab List)

| Visualization | Evidence | Status |
|---------------|----------|--------|
| Average Fare by Airline (bar chart) | `plot_fare_by_airline()` — left panel bar chart; `fare_by_airline.png`. | ✅ Met |
| Total Fare Distribution (histogram) | `plot_fare_distribution()`; `fare_distribution.png`. | ✅ Met |
| Fare Variation Across Seasons (boxplot) | `plot_seasonal_fares()` — season boxplot; `seasonal_fares.png`. | ✅ Met |
| Feature Correlation Heatmap | `plot_correlation_heatmap()`; `correlation_heatmap.png`. | ✅ Met |
| Feature Importance (RF or Linear) | `plot_feature_importance()` (tree), `plot_coefficients()` (linear); `feature_importance.png`, `coefficients.png`. | ✅ Met |
| Predicted vs Actual Fares (scatter) | `plot_actual_vs_predicted()`; `baseline_actual_vs_predicted.png`. | ✅ Met |

---

## Stretch Challenges

| Challenge | Evidence | Status |
|-----------|----------|--------|
| Integrate model into Flask or Streamlit app | `app/app.py`: Flask app with `POST /predict`, `GET /health`, `GET /`; Swagger UI at `/apidocs/`; rate limiting (30/min); input validation against known training values; `FLASK_DEBUG` env var; README documents curl example; Docker profile `api`. | ✅ Met |
| Connect to Airflow pipeline for scheduled retraining | `dags/flight_fare_pipeline.py`: full DAG; `schedule="@weekly"`; README and EXECUTION_PLAN describe trigger and monitoring. Phase 2 now also produces `known_values.json` for API validation. | ✅ Met |
| Deploy locally as simple REST API | Flask app run via `docker compose --profile api up api` on port 5000. Interactive docs at `http://localhost:5000/apidocs/`. | ✅ Met |

---

## Evaluation Criteria (Lab)

| Criterion | Assessment |
|-----------|------------|
| Correct implementation of each project step | Steps 1–6 and suggested visualizations are implemented in code and notebooks; pipeline runs end-to-end. |
| Code readability and documentation | Modular `src/` (constants, data_loader, preprocessing, feature_engineering, models, evaluation, visualization, pipeline); docstrings; README, EXECUTION_PLAN, ROADMAP. Constants centralized in `src/constants.py`. |
| Quality of EDA and feature engineering | EDA includes distributions, by-airline/season, heatmap, KPIs; feature engineering includes date features, one-hot, scaling, leak removal, log-transform. |
| Model performance and comparison rigor | Multiple models, R²/MAE/RMSE, comparison table, CV, tuning, bias–variance plot; leakage removed and log-space metrics documented. Model versioning with `model_registry.json` audit trail. |
| Clarity of insights and visualization quality | Interpretation notebook and pipeline report address business questions; figures saved at 150 dpi; stakeholder summary and recommendations. |
| Structure and completeness of deliverables | Six notebooks (01–06), pipeline artifacts (splits, models, figures, JSON reports), docs, Docker, Airflow DAG, Flask API (Swagger UI, rate limiting), pytest suite (51 tests). |

---

## Optional Gaps / Improvements

1. **Total Fare when missing:** Lab suggests “Total Fare = Base Fare + Tax & Surcharge (if missing)”. The project imputes missing numerics with median and does not derive Total Fare from Base + Tax. Acceptable if the dataset has no missing Total Fare; otherwise consider adding this derivation in preprocessing.
2. **MinMaxScaler:** Lab mentions “StandardScaler or MinMaxScaler”. Only StandardScaler is used; no functional gap, but MinMaxScaler could be noted as an alternative in docs.

---

## Conclusion

The project **fully satisfies** the Module Lab requirements for Steps 1–6, suggested visualizations, stretch challenges, and evaluation criteria. Data leakage was identified and remedied (Base Fare and Tax & Surcharge removed from features), and a log-transform was applied and documented. The codebase is structured, documented, and orchestrated via Airflow with a Flask API for predictions.

**Recommendation:** Submit as-is; optionally add a one-line derivation of Total Fare from Base + Tax when Total Fare is missing if the raw data ever contains such gaps.
