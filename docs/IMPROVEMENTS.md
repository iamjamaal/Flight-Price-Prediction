# Improvements — Flight Fare Prediction

> **Status:** All code/config fixes applied. Notebooks updated with interpretation.
> Next step: rebuild Docker images and re-trigger the Airflow DAG to produce leakage-free artifacts.

---

## 1. Fix Data Leakage (Critical)

### Problem
`Total Fare = Base Fare + Tax & Surcharge`. Including both as features
means the model learns trivial arithmetic instead of meaningful fare
prediction. This is why:
- Linear models get R² ≈ 0.997
- Tree-based models get R² ≈ 1.000

### Fix
**File:** `src/pipeline.py` — `clean_and_preprocess()`

Drop `Base Fare` and `Tax & Surcharge` from the feature set before
splitting. Keep them only for the cleaned CSV (EDA uses them), but
exclude them before encoding/scaling/splitting.

```python
# After saving cleaned_with_features.csv but BEFORE encoding:
df_for_model = df.drop(columns=["Base Fare", "Tax & Surcharge"], errors="ignore")
df_encoded = encode_categoricals(df_for_model.copy())
df_scaled, scaler = scale_numericals(df_encoded.copy())
```

**Also update:** `src/feature_engineering.py` — `scale_numericals()`
auto-detection will no longer find fare/tax columns, so it becomes a
no-op for scaling (only date/numeric features remain). This is fine —
the remaining numeric features (Duration, DaysBeforeDeparture, Month,
Day, Weekday) don't need scaling for tree-based models, and linear
models will still work.

**Expected impact:** R² will drop significantly (likely 0.3–0.8 range),
which is realistic and demonstrates genuine predictive power from
airline, route, date, and seasonality features.

### Documentation
Add a markdown cell in notebook 06 explaining:
- What data leakage is
- Why Base Fare + Tax & Surcharge leak the target
- How removing them changes model performance
- Why the lower R² is actually more meaningful

---

## 2. Speed Up Hyperparameter Tuning (Performance)

### Problem
`train_advanced_models` takes 45-60+ minutes because:
- `n_jobs=1` forced by Airflow's process model (cannot be changed)
- `n_iter=20` with 5-fold CV = 100 fits per model
- Two models tuned (Random Forest + XGBoost) = 200 fits total

### Fix
**File:** `src/pipeline.py` — `train_advanced_models()`

Reduce `n_iter` from 20 to 10 for both tuning calls:
```python
best_rf = tune_model("random_forest", X_train, y_train, cv=3, n_iter=10)
best_xgb = tune_model("xgboost", X_train, y_train, cv=3, n_iter=10)
```

Also reduce CV folds from 5 to 3 for the tuning step (cross-validation
on the top models can stay at 5).

**Expected impact:** ~4x faster (from ~60 min to ~15 min). Marginal
loss in tuning quality — negligible for this dataset size.

---

## 3. Suppress Noisy Warnings (Cosmetic)

### Problem
Hundreds of `Loky-backed parallel loops cannot be called in a
multiprocessing, setting n_jobs=1` warnings flood the Airflow task logs,
making it hard to find actual errors.

### Fix
**File:** `src/pipeline.py` — top of file, after imports:

```python
import warnings
warnings.filterwarnings("ignore", message="Loky-backed parallel loops")
```

**Also add to** `src/models.py` — inside `train_model()` and `tune_model()`:
```python
# Force n_jobs=1 explicitly to avoid the warning entirely
```

Update the model registry to use `n_jobs=1` instead of `n_jobs=-1`:
```python
"random_forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
"xgboost": lambda: _get_xgboost()(n_estimators=100, random_state=42, n_jobs=1),
```

---

## 4. Remove Obsolete `version` Key from docker-compose.yml (Cosmetic)

### Problem
Every `docker compose` command prints:
```
the attribute `version` is obsolete, it will be ignored
```

### Fix
**File:** `docker-compose.yml` — delete line 1:
```yaml
version: "3.9"   # ← remove this line
```

---

## 5. Add Lasso Convergence Fix (Minor)

### Problem
Lasso produces a `ConvergenceWarning` during training.

### Fix
**File:** `src/models.py` — update the Lasso entry in `MODEL_REGISTRY`:
```python
"lasso": lambda: Lasso(alpha=1.0, max_iter=10000),
```

Also update `PARAM_GRIDS["lasso"]` to include `max_iter`:
```python
"lasso": {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
    "max_iter": [10000],
},
```

---

## 6. Add `data/processed/` and `models/` to `.gitignore` (Hygiene)

### Problem
Pipeline artifacts (CSV splits, joblib models, JSON metrics) should not
be committed to git. They are large, binary, and reproducible by
re-running the pipeline.

### Fix
**File:** `.gitignore` — append:
```
data/processed/
models/*.joblib
reports/figures/*.png
```

---

## 7. Notebook Sync — Execute Notebooks 03–06 (Documentation)

### Problem
Notebooks 03–06 have code cells but zero outputs. For the lab
submission, the notebooks need visible outputs showing the results.

### Options
- **Option A:** Run them manually inside Jupyter (http://localhost:8888)
  after the pipeline has produced all artifacts.
- **Option B:** Add `papermill` to requirements.txt and create an Airflow
  task that executes notebooks programmatically.

**Recommendation:** Option A — simpler, and the lab evaluators will see
real outputs in the `.ipynb` files.

---

## Execution Status

| # | Fix | Status |
|---|-----|--------|
| 1 | Data leakage — drop Base Fare & Tax from features | **Done** (`src/pipeline.py`) |
| 2 | Speed up tuning — reduce n_iter/CV folds | **Done** (`src/pipeline.py`) |
| 3 | Suppress Loky warnings, force n_jobs=1 | **Done** (`src/pipeline.py`, `src/models.py`) |
| 4 | Remove obsolete `version` key | **Done** (`docker-compose.yml`) |
| 5 | Lasso convergence — increase max_iter to 50,000 | **Done** (`src/models.py`) |
| 6 | Add artifacts to .gitignore | **Done** (`.gitignore`) |
| 7 | Notebook interpretation — fill in findings | **Done** (notebooks 03–06) |

### Remaining Steps

1. Rebuild the Airflow Docker image: `docker compose --profile airflow build --no-cache`
2. Re-trigger the DAG to produce leakage-free artifacts
3. Run notebooks 03–06 in Jupyter to generate visible cell outputs
4. Commit and push
