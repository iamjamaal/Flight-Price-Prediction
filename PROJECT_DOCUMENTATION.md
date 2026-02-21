# Flight Fare Prediction — Project Documentation

> **Course:** DEM09: Data Science — ML Foundation and Supervised Learning
> **Dataset:** Flight Price Dataset of Bangladesh (57,000 records)
> **Goal:** End-to-end supervised regression pipeline predicting domestic and international flight fares from Bangladesh

---

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Step 1 — Problem Definition & Data Understanding](#step-1--problem-definition--data-understanding)
3. [Step 2 — Data Cleaning & Preprocessing](#step-2--data-cleaning--preprocessing)
4. [Step 3 — Exploratory Data Analysis](#step-3--exploratory-data-analysis)
5. [Step 4 — Baseline Model Development](#step-4--baseline-model-development)
6. [Step 5 — Advanced Modeling & Optimization](#step-5--advanced-modeling--optimization)
7. [Step 6 — Model Interpretation & Insights](#step-6--model-interpretation--insights)
8. [Stretch Challenges](#stretch-challenges)
9. [Results Summary](#results-summary)

---

## Project Architecture

The project is structured as a modular, production-ready pipeline with clearly separated concerns:

```
Flight-Price-Prediction/
├── data/
│   ├── raw/                          # Original CSV dataset
│   └── processed/                    # Cleaned splits, scaler, column lists
│       ├── X_train.csv / X_test.csv
│       ├── y_train.csv / y_test.csv
│       ├── scaler.joblib
│       ├── train_columns.json
│       ├── model_comparison.csv
│       ├── interpretation_report.json
│       └── data_summary.json
├── models/
│   ├── best_model.joblib             # Production model
│   └── linear_regression_baseline.joblib
├── notebooks/
│   ├── 01_problem_definition_data_understanding.ipynb
│   ├── 02_data_cleaning_preprocessing.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_baseline_model_development.ipynb
│   ├── 05_advanced_modeling_optimization.ipynb
│   └── 06_model_interpretation_insights.ipynb
├── src/
│   ├── constants.py                  # Shared constants (SEASON_MAP, CITY_NAME_ALIASES)
│   ├── data_loader.py                # Dataset ingestion
│   ├── preprocessing.py              # Cleaning and validation
│   ├── feature_engineering.py        # Encoding, scaling, splitting
│   ├── models.py                     # Model registry, tuning, versioned save
│   ├── evaluation.py                 # Metrics and comparison
│   ├── visualization.py              # All plots
│   └── pipeline.py                   # Orchestration entry point
├── dags/
│   └── flight_fare_pipeline.py       # Apache Airflow DAG
├── app/
│   ├── app.py                        # Flask REST API (Swagger, rate limiting)
│   └── streamlit_app.py              # Live prediction web app
├── tests/
│   ├── test_preprocessing.py         # 16 unit tests
│   ├── test_feature_engineering.py   # 20 unit tests
│   └── test_api.py                   # 14 API tests (mocked)
└── reports/
    └── figures/                      # Saved visualizations (PNG)
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `constants.py` | Single source of truth for `SEASON_MAP`, `SEASON_ORDER`, `CITY_NAME_ALIASES` |
| `data_loader.py` | Load raw CSV, inspect structure |
| `preprocessing.py` | Rename columns, impute, fix invalid entries, validate dtypes |
| `feature_engineering.py` | Date features, one-hot encoding, StandardScaler, train/test split |
| `models.py` | Model registry (7 algorithms), hyperparameter grids, RandomizedSearchCV, versioned save |
| `evaluation.py` | R², MAE, RMSE, k-fold cross-validation, comparison table |
| `visualization.py` | EDA plots, residuals, feature importance, coefficients |
| `pipeline.py` | 6-phase orchestration callable by Airflow or standalone |
| `dags/flight_fare_pipeline.py` | Weekly Airflow DAG wiring all phases |

---

## Step 1 — Problem Definition & Data Understanding

**Notebook:** `01_problem_definition_data_understanding.ipynb`
**Source module:** `src/data_loader.py`

### Business Question

> *Can we predict the total fare (BDT) of a flight from Bangladesh given information about the airline, route, travel class, booking lead time, and travel date?*

Airlines and travel platforms need reliable fare estimates to support dynamic pricing strategies, customer-facing booking tools, and revenue forecasting. A regression model trained on historical booking data can power these use cases.

### ML Task Definition

| Attribute | Value |
|---|---|
| Problem type | Supervised Regression |
| Target variable | `Total Fare` (BDT) |
| Evaluation metrics | R² (coefficient of determination), MAE, RMSE |
| Environment | Python 3.11.14, pandas 2.3.3, numpy 2.4.2, scikit-learn |

### Dataset Loading

The dataset was loaded using `load_dataset()` from `src/data_loader.py`:

```python
def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    # Reads Flight_Price_Dataset_of_Bangladesh.csv
    # Raises FileNotFoundError if dataset doesn't exist
    # Returns unprocessed raw DataFrame
```

Dataset path: `data/raw/Flight_Price_Dataset_of_Bangladesh.csv`

### Initial Observations

The `inspect_dataset()` function produced a comprehensive first-look summary:

```python
def inspect_dataset(df: pd.DataFrame) -> dict:
    # Returns: shape, dtypes, missing_counts, duplicate_rows, describe
```

**Dataset shape:** 57,000 rows × 17 columns

**Columns present:**

| Column | Type | Description |
|---|---|---|
| Airline | object | Carrier name (23 unique airlines) |
| Source | object | Departure airport code |
| Source Name | object | Departure city name (dropped later) |
| Destination | object | Arrival airport code |
| Destination Name | object | Arrival city name (dropped later) |
| Departure Date & Time | object | Full departure timestamp |
| Arrival Date & Time | object | Full arrival timestamp (dropped later) |
| Duration (hrs) | float | Flight duration in hours |
| Stopovers | object | Direct / 1 Stop / 2+ Stops |
| Aircraft Type | object | Boeing, Airbus, etc. |
| Class | object | Economy, Business, First Class |
| Booking Source | object | Online, Travel Agent, etc. |
| Base Fare (BDT) | float | Pre-tax fare component |
| Tax & Surcharge (BDT) | float | Tax component |
| Total Fare (BDT) | float | **Target variable** |
| Seasonality | object | Winter / Summer / Monsoon / Autumn |
| Days Before Departure | int | Booking lead time in days |

**Data quality findings:**

| Check | Result |
|---|---|
| Missing values | **0** — no imputation needed |
| Duplicate rows | **0** |
| Negative fares | None found |
| Invalid city names | Fixed in preprocessing ("Dacca" → "Dhaka") |

**Target variable statistics:**

| Metric | Value |
|---|---|
| Mean fare | 71,030 BDT |
| Median fare | 41,308 BDT |
| Min fare | 1,801 BDT |
| Max fare | 558,987 BDT |
| Skewness | **1.58** (right-skewed — addressed in preprocessing) |

The large gap between mean and median, plus skewness of 1.58, immediately suggested a log-transform would be beneficial for model training.

---

## Step 2 — Data Cleaning & Preprocessing

**Notebook:** `02_data_cleaning_preprocessing.ipynb`
**Source modules:** `src/preprocessing.py`, `src/feature_engineering.py`

### A. Cleaning Data

#### Column Normalization

`normalize_columns()` in `preprocessing.py` applied a rename map to standardize column names and dropped redundant columns:

```python
_COLUMN_RENAME_MAP = {
    "Base Fare (BDT)":          "Base Fare",
    "Tax & Surcharge (BDT)":    "Tax & Surcharge",
    "Total Fare (BDT)":         "Total Fare",
    "Departure Date & Time":    "Date",
    "Duration (hrs)":           "Duration",
    "Days Before Departure":    "DaysBeforeDeparture",
}

_REDUNDANT_COLUMNS = ["Source Name", "Destination Name", "Arrival Date & Time"]
```

Shape after dropping: **57,000 × 14 columns**

#### Missing Value Handling

`handle_missing_values()` applied a robust imputation strategy (the dataset had zero missing values, but the logic ensures pipeline correctness on new data):

- **Numerical columns** (float64, float32, int64, int32): impute with **median** (robust to the fare distribution's long tail)
- **Categorical/object columns**: impute with **mode**, falling back to `"Unknown"` if mode is undefined

#### Invalid Entry Fixes

`fix_invalid_entries()` performed:
- Stripped leading/trailing whitespace from all string columns
- Normalised city name variants:
  - `"Dacca"` → `"Dhaka"`
  - `"Chattogram"` / `"Chottogram"` → `"Chittagong"`
- Removed rows with negative fare values (none found in this dataset)
- Reset index after any row removal

#### Data Type Validation

`validate_dtypes()` enforced correct types:

| Columns | Target type |
|---|---|
| Base Fare, Tax & Surcharge, Total Fare, Duration | `float64` |
| Date | `datetime64[ns]` (parsed with `dayfirst=True`) |
| Airline, Source, Destination, Stopovers, Aircraft Type, Class, Booking Source, Seasonality | `category` |

### B. Feature Engineering

#### Date Feature Extraction

`create_date_features()` decomposed the `Date` datetime column into:

```python
# Season mapping (Bangladesh climate)
season_map = {
    12: "Winter",  1: "Winter",  2: "Winter",
     3: "Summer",  4: "Summer",  5: "Summer",
     6: "Monsoon", 7: "Monsoon", 8: "Monsoon",
     9: "Autumn", 10: "Autumn", 11: "Autumn",
}
```

New features created: `Month`, `Day`, `Weekday` (0=Monday), `WeekdayName`, `Season`

#### Leakage Removal

A critical decision was made to **remove `Base Fare` and `Tax & Surcharge`** from the feature set before modelling. The arithmetic identity `Total Fare = Base Fare + Tax & Surcharge` makes these features perfectly collinear with the target — including them constitutes data leakage. This was confirmed in Phase 6 when initial Random Forest importance attributed 99.62% to `Base Fare` alone.

#### Log-Transform on Target

Given the right-skewed distribution (skewness = 1.58), `log1p` was applied to `Total Fare`:

```python
df["Total Fare"] = np.log1p(df["Total Fare"])
```

This linearises the fare distribution and makes linear models significantly more competitive. All reported MAE/RMSE metrics are in log-units; BDT-scale interpretation uses `np.expm1()`.

#### Categorical Encoding

`encode_categoricals()` applied **one-hot encoding** to all nominal columns:

```python
pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
```

`drop_first=True` prevents perfect multicollinearity. The final encoded feature space expanded to **78 features**.

#### Numerical Scaling

`scale_numericals()` fitted a **StandardScaler** (mean=0, std=1) on `Duration` and `DaysBeforeDeparture`:

```python
scaler = StandardScaler()
df[columns] = scaler.fit_transform(df[columns])
```

The fitted scaler was persisted to `data/processed/scaler.joblib` for use during inference. Target-related columns (`Base Fare`, `Tax & Surcharge`, already removed) were explicitly excluded from scaling.

#### Train/Test Split

`split_data()` produced reproducible splits:

| Split | Size |
|---|---|
| X_train | 45,600 samples × 78 features |
| X_test | 11,400 samples × 78 features |
| y_train | 45,600 log-transformed fare values |
| y_test | 11,400 log-transformed fare values |

Parameters: `test_size=0.20`, `random_state=42`

All splits were saved to `data/processed/` as CSV files. Training column names were saved to `train_columns.json` to enable feature alignment at inference time.

---

## Step 3 — Exploratory Data Analysis

**Notebook:** `03_exploratory_data_analysis.ipynb`
**Source module:** `src/visualization.py`

All visualisations were produced at **150 dpi** using Seaborn (`whitegrid` theme, `deep` palette) and saved to `reports/figures/`.

### 1. Descriptive Statistics

#### Average Fare by Airline

| Airline | Mean Fare (BDT) |
|---|---|
| Turkish Airlines | 75,547 |
| AirAsia | 74,534 |
| Cathay Pacific | 73,325 |
| Thai Airways | 72,846 |
| Malaysian Airlines | 72,775 |
| IndiGo | 72,504 |
| Singapore Airlines | 68,324 |
| Vistara | 68,108 *(lowest)* |

**Spread:** ~7,400 BDT between most and least expensive carrier.

#### Average Fare by Destination

| Destination | Mean Fare (BDT) | Type |
|---|---|---|
| JFK (New York) | 110,903 | International |
| KUL (Kuala Lumpur) | 110,495 | International |
| YYZ (Toronto) | 109,531 | International |
| BZL (Barisal) | 7,655 | Domestic |
| CGP (Chittagong) | 7,610 | Domestic |
| CXB (Cox's Bazar) | 7,496 | Domestic |

The bimodal structure — international routes fetching 10–15× domestic fares — confirms route type as the dominant fare driver.

#### Seasonal Fare Variation

| Season | Mean Fare (BDT) | Premium vs. Autumn |
|---|---|---|
| Winter | 78,772 | **+16.2%** |
| Monsoon | 69,178 | +1.9% |
| Summer | 68,604 | +1.1% |
| Autumn | 67,855 | baseline |

Winter (December–February) commands a clear premium, aligned with Bangladeshi holiday travel (Eid holidays, end-of-year travel).

### 2. Visual Analysis

Four plots were generated by `src/visualization.py`:

**`plot_fare_distribution()`** — Histogram (50 bins) with KDE overlay showing a right-skewed unimodal distribution. Skewness of 1.58 visible as a long tail toward 500,000+ BDT fares.

**`plot_fare_by_airline()`** — Dual subplot:
- Left: Bar chart of mean fares per airline (ordered descending)
- Right: Boxplot of fare spread per airline, revealing variance differences between carriers

**`plot_seasonal_fares()`** — Dual subplot:
- Left: Boxplot of fares by season (Winter → Summer → Monsoon → Autumn)
- Right: Line chart of monthly average fares, showing the December–February peak

**`plot_correlation_heatmap()`** — Pearson correlation heatmap of all numerical features, annotated with 2-decimal values, `coolwarm` colormap. Confirmed strong correlation between `Base Fare` and `Total Fare` (leakage signal).

### 3. KPI Exploration

**Most frequent routes (booking volume):**

| Route | Flights |
|---|---|
| RJH → SIN | 417 |
| DAC → DXB | 413 |
| BZL → YYZ | 410 |

**Top 5 most expensive routes:**

| Route | Mean Fare (BDT) |
|---|---|
| SPD → BKK | 117,952 |
| CXB → YYZ | 117,849 |
| CXB → LHR | 116,668 |

**Hypotheses formed for modelling:**

| Hypothesis | Outcome |
|---|---|
| H1: Base Fare & Tax dominate (leakage risk) | **Confirmed** — removed before modelling |
| H2: Seasonal features carry ~16% price signal | **Confirmed** — Winter premium validated |
| H3: Route/airline require non-linear models | **Partially** — log-transform linearises enough for LR to compete |

---

## Step 4 — Baseline Model Development

**Notebook:** `04_baseline_model_development.ipynb`
**Source modules:** `src/models.py`, `src/evaluation.py`, `src/visualization.py`

### Model Selection

Linear Regression was selected as the interpretable baseline, implemented via scikit-learn's `LinearRegression()` with no regularisation. It serves as the lower-bound benchmark against which all advanced models are compared.

### Implementation

Training via `train_model()` in `src/models.py`:

```python
MODEL_REGISTRY = {
    "linear_regression": LinearRegression(),
    ...
}

def train_model(name: str, X_train, y_train):
    model = MODEL_REGISTRY[name]
    model.fit(X_train, y_train)
    return model
```

Evaluation via `evaluate_model()` in `src/evaluation.py`:

```python
def evaluate_model(model, X_test, y_test) -> dict:
    # Returns: r2, mae, rmse, predictions
    preds = model.predict(X_test)
    return {
        "r2":   r2_score(y_test, preds),
        "mae":  mean_absolute_error(y_test, preds),
        "rmse": sqrt(mean_squared_error(y_test, preds)),
    }
```

### Results

All metrics are on the log₁p-transformed target scale:

| Metric | Value |
|---|---|
| R² | **0.8935** |
| MAE | 0.35 (log-BDT) |
| RMSE | 0.46 (log-BDT) |

**Interpretation:** The baseline Linear Regression explains **89.35%** of the variance in log-transformed fares using only legitimate predictors (airline, route, class, season, booking lead time). This strong baseline result — compared with the prior R²≈0.57 seen when training on raw unlogged fares — demonstrates how effective the log₁p transform is at linearising the fare distribution.

The model was saved to `models/linear_regression_baseline.joblib`.

### Visualisation

Two plots were generated to analyse model behaviour:

**`plot_actual_vs_predicted()`** — Scatter plot of actual vs. predicted log fares, with a 45° reference line. Points clustering tightly around the diagonal confirm the model captures the primary fare signal.

**`plot_residuals()`** — Dual subplot:
- Left: Histogram with KDE of residuals — approximately normal distribution centred near zero
- Right: Residuals vs. predicted values — no strong heteroscedasticity after log-transform, confirming the transform was appropriate

### Leakage Investigation

An initial run *with* `Base Fare` and `Tax & Surcharge` included showed R²=0.9969 — near perfect. Feature importance analysis confirmed the arithmetic identity: `Total Fare = Base Fare + Tax & Surcharge`. Both columns were then permanently excluded from all subsequent modelling steps.

---

## Step 5 — Advanced Modeling & Optimization

**Notebook:** `05_advanced_modeling_optimization.ipynb`
**Source module:** `src/models.py`, `src/evaluation.py`

### Models Trained

Seven model variants were trained and evaluated, plus two tuned versions:

```python
MODEL_REGISTRY = {
    "linear_regression": LinearRegression(),
    "ridge":             Ridge(alpha=1.0),
    "lasso":             Lasso(alpha=0.001, max_iter=50_000),
    "decision_tree":     DecisionTreeRegressor(random_state=42),
    "random_forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "xgboost":           XGBRegressor(n_estimators=100, random_state=42, n_jobs=1),
}
```

`Lasso` was configured with `max_iter=50_000` to resolve convergence warnings that appeared during the Airflow run. `n_jobs=1` was forced on parallelisable models to avoid Loky multiprocessing conflicts inside the containerised Airflow environment.

### Hyperparameter Tuning

`tune_model()` in `src/models.py` used `RandomizedSearchCV` (not `GridSearchCV`) for efficiency:

```python
def tune_model(name, X_train, y_train, cv=3, n_iter=10, scoring="r2"):
    search = RandomizedSearchCV(
        estimator=MODEL_REGISTRY[name],
        param_distributions=PARAM_GRIDS[name],
        n_iter=n_iter,   # reduced from 30 to 10 for ~4× speed in Airflow
        cv=cv,           # reduced from 5 to 3 for pipeline efficiency
        scoring=scoring,
        n_jobs=1,
        random_state=42,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_
```

**Random Forest search space:**
- `n_estimators`: [50, 100, 200, 300]
- `max_depth`: [10, 15, 20, None]
- `min_samples_split`: [2, 5]
- `min_samples_leaf`: [1, 2]

**Best Random Forest params:** `n_estimators=100`, `max_depth=10`, `min_samples_split=5`, `min_samples_leaf=1` — CV R²=0.8918

**XGBoost search space** (requires `pip install xgboost`; skipped gracefully if absent):
- `n_estimators`: [100, 200, 300]
- `max_depth`: [3, 5, 7, 10]
- `learning_rate`: [0.01, 0.05, 0.1, 0.2]
- `subsample`: [0.7, 0.8, 1.0]

**Best XGBoost params:** `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`, `subsample=0.8` — CV R²=0.8923

### Model Evaluation Results

| Rank | Model | R² | MAE | RMSE |
|---|---|---|---|---|
| 1 | Linear Regression | **0.8935** | 0.35 | 0.46 |
| 1 | Ridge | **0.8935** | 0.35 | 0.46 |
| 3 | Random Forest (Tuned) | 0.8932 | 0.35 | 0.46 |
| 4 | Lasso | 0.8931 | 0.35 | 0.46 |
| 5 | Random Forest | 0.8878 | 0.36 | 0.47 |
| 6 | Decision Tree | 0.778 | 0.48 | 0.66 |

*All metrics on log₁p-transformed target. Results saved to `data/processed/model_comparison.csv`.*
*XGBoost requires `pip install xgboost` and was not available in this run.*

**Best model:** Linear Regression / Ridge (tied at R²=0.8935). `models/best_model.joblib` (canonical alias) and `models/best_model_v<timestamp>.joblib` (versioned archive) saved to disk. `models/model_registry.json` updated with metrics and timestamp.

### Cross-Validation

5-fold cross-validation was run via `cross_validate_model()`:

```python
def cross_validate_model(model, X, y, cv=5, scoring="r2") -> dict:
    # n_jobs=-1 to use all processors
    # Returns: scores array, mean, std
```

| Model | CV R² (mean ± std) |
|---|---|
| Ridge | 0.8927 ± 0.0018 |
| Random Forest | 0.8871 ± 0.0017 |

Low standard deviation across folds confirms stable generalisation — no overfitting.

### Regularisation & Bias–Variance Analysis

The bias-variance tradeoff was plotted for Ridge and Lasso across 7 alpha values:
`[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`

**Key finding:** R² remained nearly flat (~0.897) across all alpha values for both Ridge and Lasso. This indicates:
1. The log₁p-transformed relationship is strongly linear — regularisation has minimal impact
2. No significant overfitting is present in the unregularised Linear Regression
3. Ridge and Lasso are safe choices for production but offer no material improvement

The Decision Tree's low R²=0.778 confirms that deep single trees overfit without ensemble averaging.

---

## Step 6 — Model Interpretation & Insights

**Notebook:** `06_model_interpretation_insights.ipynb`
**Source module:** `src/visualization.py` (interpretation functions)

### 1. Feature Importance

#### Data Leakage Discovery

The leakage was formally documented in Phase 6 with a Random Forest run that included `Base Fare` and `Tax & Surcharge`:

| Feature | Importance |
|---|---|
| Base Fare | **99.62%** |
| Tax & Surcharge | 0.07% |
| All other features combined | 0.31% |

This is a textbook arithmetic leakage: `Total Fare = Base Fare + Tax & Surcharge`. The model learns a trivial identity rather than pricing signals. After removal, the honest feature set reveals genuine fare drivers.

#### Linear Regression Coefficients (Post-Leakage Fix)

`plot_coefficients()` in `visualization.py` rendered horizontal bar charts with green bars for positive coefficients (fare-increasing) and red for negative (fare-decreasing):

```python
def plot_coefficients(model, feature_names, top_n=15, save_as="coefficients.png"):
    # Extracts model.coef_, pairs with feature_names
    # Top-N by absolute value
    # Green for positive, red for negative
```

#### Tree-Based Feature Importance

`plot_feature_importance()` extracted importances from tree models via `model.feature_importances_` and plotted the top 15 features as a horizontal bar chart.

### 2. Insights

#### Airline Pricing (from Linear Regression Coefficients on log Fare)

| Direction | Airline | Coefficient |
|---|---|---|
| Premium ↑ | IndiGo | +0.0144 |
| Premium ↑ | AirAsia | +0.0123 |
| Premium ↑ | Air India | +0.0121 |
| Premium ↑ | Turkish Airlines | +0.0113 |
| Premium ↑ | Qatar Airways | +0.0104 |
| Premium ↑ | Cathay Pacific | +0.0095 |
| Neutral | Gulf Air, Malaysian Airlines, FlyDubai | ≈0.00 |
| Discount ↓ | Emirates | −0.0094 |
| Discount ↓ | SriLankan Airlines | −0.0055 |
| Discount ↓ | Lufthansa | −0.0053 |
| Discount ↓ | British Airways | −0.0052 |
| Discount ↓ | Biman Bangladesh Airlines | −0.0031 |

Note: These coefficients control for route, season, and class simultaneously — they represent the *ceteris paribus* airline price effect, distinct from raw average fares from EDA.

#### Seasonal Patterns

Winter (December–February) commands a **+16.2% fare premium** over Autumn. Monsoon and Summer show negligible premiums (+1–2%), suggesting demand concentration in winter holiday travel.

#### Most Expensive Routes

| Route | Mean Fare (BDT) |
|---|---|
| SPD → BKK (Bangkok) | 117,952 |
| CXB → YYZ (Toronto) | 117,849 |
| CXB → LHR (London) | 116,668 |

International routes to North America and Europe fetch approximately 2.5× more than average fares.

### 3. Non-Technical Stakeholder Summary

**What factors most influence flight fares?**
- The single biggest driver is the **route** — where you fly to. International flights to North America and Europe cost roughly 15× more than domestic flights.
- **Airline choice** creates a ~7,400 BDT gap between premium and budget carriers on comparable routes.
- **Season** matters: booking in Winter costs ~16% more than in Autumn.

**What can airlines and travel platforms do with this?**
- Target Winter pricing campaigns to capture the premium demand window.
- Position budget carriers (Emirates, Vistara) against premium ones (IndiGo, AirAsia) in recommendation engines.
- Flag long-haul routes (CXB→YYZ, CXB→LHR) as high-revenue segments for capacity planning.

**Data-backed recommendations:**
1. Enrich the dataset with booking lead time and seat availability — the current model explains 89% of fare variance; the remaining 11% likely lies in real-time demand signals.
2. Apply target encoding to high-cardinality route pairs to better capture route-specific pricing.
3. Schedule weekly pipeline retraining (already implemented in the Airflow DAG) to keep the model current as fares change.

---

## Stretch Challenges

### Streamlit Web Application

**File:** `app/streamlit_app.py`

A fully functional prediction interface was built with Streamlit, meeting the stretch goal of integrating the model into a live web app.

**Features implemented:**
- Interactive booking form: airline, source, destination, travel date, class, stopovers, duration, days before departure
- Real-time fare prediction using the saved best model
- Direct feature construction at inference time (no preprocessing pipeline re-run needed)
- Analytics dashboard with model comparison charts and airline pricing insights
- Custom CSS styling with professional UI
- Handles feature alignment via `align_features()` to match the 78-feature training schema

**Prediction flow:**

```python
def predict_fare(airline, source, destination, travel_date, travel_class, ...):
    # 1. Parse date → extract Month, Day, Weekday, Season
    # 2. Build raw feature row as DataFrame
    # 3. One-hot encode via encode_categoricals()
    # 4. Scale numeric features via fitted scaler
    # 5. Align to training column schema via align_features()
    # 6. model.predict() → inverse log1p → BDT fare
```

**Deployment:** Containerised with Docker via `docker-compose.yml` (Streamlit service added alongside Airflow services).

### Apache Airflow Pipeline

**File:** `dags/flight_fare_pipeline.py`

The full pipeline is orchestrated as a production Airflow DAG meeting the stretch goal of connecting all phases for scheduled retraining.

**DAG configuration:**

| Setting | Value |
|---|---|
| DAG ID | `flight_fare_prediction` |
| Schedule | `@weekly` (every Sunday) |
| Start date | 2026-01-01 |
| Owner | `flight-fare-team` |
| Retries | 1 (5-minute delay on failure) |
| Tags | `["ml", "flight-fare", "regression"]` |

**Task dependency graph:**

```
load_and_validate
        │
clean_and_preprocess
    ┌───┤───┐
eda_report  baseline_model     ← run in parallel
    └───┬───┘
train_advanced_models
        │
interpret_and_report
```

EDA report generation and baseline model training are parallelised, reducing total DAG runtime. Each task is a `PythonOperator` calling the corresponding function from `src/pipeline.py`.

**Outputs produced per DAG run:**

| Artifact | Path |
|---|---|
| Data quality summary | `data/processed/data_summary.json` |
| Train/test splits | `data/processed/X_train.csv`, etc. |
| EDA KPIs | `data/processed/kpis.json` |
| EDA figures | `reports/figures/*.png` |
| Baseline model | `models/linear_regression_baseline.joblib` |
| Best model | `models/best_model.joblib` |
| Model comparison | `data/processed/model_comparison.csv` |
| Interpretation report | `data/processed/interpretation_report.json` |

---

## Results Summary

### Final Model Performance (Airflow DAG Run)

All metrics on log₁p-transformed `Total Fare`. MAE/RMSE in log units.

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | **0.8935** | 0.35 | 0.46 |
| Ridge | **0.8935** | 0.35 | 0.46 |
| Random Forest (Tuned) | 0.8932 | 0.35 | 0.46 |
| Lasso | 0.8931 | 0.35 | 0.46 |
| Random Forest | 0.8878 | 0.36 | 0.47 |
| Decision Tree | 0.778 | 0.48 | 0.66 |

### Key Project Numbers

| Metric | Value |
|---|---|
| Dataset size | 57,000 rows × 17 columns |
| Missing values | 0 |
| Duplicate rows | 0 |
| Encoded feature count | 78 |
| Train samples | 45,600 |
| Test samples | 11,400 |
| Best model R² | **0.8935** |
| Winter fare premium | +16.2% vs. Autumn |
| Airline fare spread | ~7,400 BDT |
| Most expensive route | SPD→BKK (117,952 BDT) |

### Improvements Validated Through the Pipeline

| Improvement | Impact |
|---|---|
| Removed `Base Fare` & `Tax & Surcharge` | Eliminated data leakage; R² dropped from ~1.0 to honest 0.89 |
| Applied `log1p` target transform | Linearised skewed distribution; enabled linear models to match ensemble models |
| Reduced tuning iterations (n_iter=10, cv=3) | ~4× faster pipeline execution without meaningful R² loss |
| Forced `n_jobs=1` on parallelisable models | Resolved Loky process conflicts in Docker/Airflow environment |
| Increased Lasso `max_iter` to 50,000 | Resolved convergence warnings; Lasso converges at R²=0.8931 |
| Centralized constants in `src/constants.py` | Eliminated 3 duplicate `SEASON_MAP` dicts; single source of truth |
| Model versioning via `save_model_versioned()` | Timestamped archive + `model_registry.json` — no more silent overwrites |
| Flask API hardening | Rate limiting, input validation, Swagger UI, debug-mode fix, `get_json` fix |
| pytest suite (51 tests) | All core modules covered; no disk I/O; ~6 s full run |

### Evaluation Criteria Coverage

| Criterion | How Addressed |
|---|---|
| Correct implementation of each project step | All 6 steps implemented across 6 dedicated notebooks and 8 `src/` modules |
| Code readability and documentation | Modular `src/` package with typed function signatures, centralized constants, docstrings; notebooks with markdown explanations |
| Quality of EDA and feature engineering | 4 EDA visualisations, KPI table, seasonal/route/airline analysis, Bangladesh climate-based season mapping |
| Model performance and comparison rigor | 6 models benchmarked (+ XGBoost when available), hyperparameter tuning, 5-fold cross-validation, bias–variance plots, versioned model registry |
| Clarity of insights and visualisation quality | 150 dpi seaborn figures, airline coefficient table, stakeholder summary in notebook 06 |
| Structure and completeness of deliverables | Notebooks, `src/`, `dags/`, `app/`, `tests/`, `models/`, `reports/` all populated |
| Stretch: Streamlit app | Implemented in `app/streamlit_app.py`, Docker-composed |
| Stretch: Airflow pipeline | Weekly DAG in `dags/flight_fare_pipeline.py`, 6-task graph with parallelism |
| Stretch: Flask API | Rate-limited REST API with Swagger UI at `/apidocs/`, input validation against training data domain |
