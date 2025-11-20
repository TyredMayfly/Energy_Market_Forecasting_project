# Full Hyperparameter Search and Analysis Pipeline

This document describes the comprehensive hyperparameter search and analysis pipeline for the Energy Market Forecasting project.

## Overview

The pipeline consists of two main components:

1. **Full Search Orchestrator** (`scripts/run_full_hyperparam_search.py`): Runs hyperparameter search across all or filtered model × market combinations
2. **Results Analyzer** (`scripts/analyze_hyperparams.py`): Performs in-depth analysis of search results and generates recommendations

## Quick Start

### 1. Run Full Hyperparameter Search

```bash
# Run search for all combinations (default: 30 iterations, 3 CV splits)
python -m scripts.run_full_hyperparam_search

# Filter by specific markets
python -m scripts.run_full_hyperparam_search --markets day_ahead imbalance_shortage

# Filter by specific models
python -m scripts.run_full_hyperparam_search --models hist_gradient_boosting random_forest

# Customize search intensity
python -m scripts.run_full_hyperparam_search --n-iter 50 --cv-splits 5

# Update defaults configuration after search
python -m scripts.run_full_hyperparam_search --update-defaults
```

### 2. Analyze Results

```bash
# Analyze all results
python -m scripts.analyze_hyperparams

# Generate visualization plots
python -m scripts.analyze_hyperparams --generate-plots

# Filter analysis by markets
python -m scripts.analyze_hyperparams --markets day_ahead
```

## Pipeline Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────┐
│  Raw Market Data (CSV files)                            │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Feature Engineering                                     │
│  - Time-based features (hour, day, month)               │
│  - Lag features (price history)                         │
│  - Rolling statistics (mean, std)                       │
│  - Weather features                                     │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Full Hyperparameter Search Orchestrator                │
│  (run_full_hyperparam_search.py)                       │
│                                                          │
│  For each (market, model) combination:                  │
│  ├─ Run RandomizedSearchCV with TimeSeriesSplit        │
│  ├─ Track all trial results                            │
│  └─ Save results:                                       │
│     ├─ artifacts/hyperparameter_search/                │
│     │   ├─ summary.csv (global)                        │
│     │   └─ <market>/                                   │
│     │       ├─ <model>_latest.json                     │
│     │       ├─ <model>_trials_latest.csv               │
│     │       └─ <model>_<timestamp>.json                │
│     └─ hyperparams_defaults.json (--update-defaults)   │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Results Analyzer (analyze_hyperparams.py)              │
│                                                          │
│  1. Per-Market Analysis                                 │
│     - Best model per market                             │
│     - Performance gaps between models                   │
│     - Overfitting detection (CV vs test score)          │
│                                                          │
│  2. Per-Model Analysis                                  │
│     - Average rank across markets                       │
│     - Best/worst markets per model                      │
│     - Win rate (1st place finishes)                     │
│                                                          │
│  3. Hyperparameter Importance                           │
│     - Numerical: Spearman correlation with score        │
│     - Categorical: Mean score per value                 │
│     - Identify optimal parameter ranges                 │
│                                                          │
│  4. Generate Recommendations                            │
│     - Best model + hyperparameters per market           │
│     - Global insights and patterns                      │
│                                                          │
│  5. Visualization (optional)                            │
│     - Model comparison plots                            │
│     - Hyperparameter importance charts                  │
└─────────────────────────────────────────────────────────┘
```

### Output Structure

```
artifacts/hyperparameter_search/
├── summary.csv                          # Global summary of all searches
├── hyperparams_defaults.json            # Recommended defaults (if --update-defaults)
├── plots/                               # Visualizations (if --generate-plots)
│   ├── model_comparison_by_market.png
│   └── <market>_<model>_param_importance.png
│
├── day_ahead/
│   ├── random_forest_latest.json
│   ├── random_forest_trials_latest.csv
│   ├── random_forest_20250120_143022.json
│   ├── linear_regression_latest.json
│   ├── linear_regression_trials_latest.csv
│   └── ...
│
├── imbalance_shortage/
│   ├── random_forest_latest.json
│   ├── random_forest_trials_latest.csv
│   └── ...
│
└── regulation_state/
    ├── xgboost_classifier_latest.json
    ├── xgboost_classifier_trials_latest.csv
    └── ...
```

### File Formats

#### summary.csv

Global summary with one row per search:

| Column | Description |
|--------|-------------|
| `market` | Market type (e.g., day_ahead) |
| `model` | Model type (e.g., random_forest) |
| `metric` | Scoring metric (e.g., neg_mean_squared_error) |
| `best_cv_score` | Best cross-validation score |
| `test_score` | Score on held-out test set |
| `n_samples_train` | Training samples used |
| `n_samples_test` | Test samples used |
| `n_features` | Number of features |
| `n_iter` | Number of search iterations |
| `cv_splits` | Number of CV splits |
| `timestamp` | Search timestamp (UTC) |
| `param_*` | Best hyperparameter values |

#### <model>_latest.json

Best parameters and metadata for a specific combination:

```json
{
  "best_params": {
    "n_estimators": 150,
    "max_depth": 8,
    "min_samples_split": 5
  },
  "best_score": -95.5,
  "metadata": {
    "model_type": "random_forest",
    "market_type": "day_ahead",
    "search_timestamp": "2025-01-20T14:30:22",
    "n_samples_train": 10000,
    "n_samples_test": 2000,
    "n_features": 25,
    "scoring": "neg_mean_squared_error",
    "best_cv_score": -95.5,
    "test_score": -98.2,
    "cv_results": {
      "mean_test_scores": [-100.0, -95.5, -105.0, ...],
      "std_test_scores": [5.0, 4.5, 6.0, ...],
      "params": [...]
    }
  }
}
```

#### <model>_trials_latest.csv

Per-trial results for detailed analysis:

| Column | Description |
|--------|-------------|
| `trial_id` | Trial number (0-indexed) |
| `mean_cv_score` | Mean score across CV folds |
| `std_cv_score` | Standard deviation across folds |
| `<param_1>` | Value of hyperparameter 1 |
| `<param_2>` | Value of hyperparameter 2 |
| ... | Additional hyperparameters |

## Model × Market Combinations

The pipeline supports these combinations:

### Regression Markets
- **day_ahead** (Day-Ahead Market Prices)
  - `linear_regression`
  - `random_forest`
  - `hist_gradient_boosting`

- **imbalance_shortage** (Imbalance Shortage Prices)
  - `linear_regression`
  - `random_forest`
  - `hist_gradient_boosting`

- **imbalance_surplus** (Imbalance Surplus Prices)
  - `linear_regression`
  - `random_forest`
  - `hist_gradient_boosting`

### Classification Markets
- **regulation_state** (System Regulation State)
  - `xgboost_classifier`

**Total: 10 valid combinations**

## Hyperparameter Search Spaces

### Linear Regression
```python
{
    "fit_intercept": [True, False],
    "positive": [True, False],
}
```
**Search Space Size:** ~4 combinations

### Random Forest
```python
{
    "n_estimators": randint(50, 300),
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
}
```
**Search Space Size:** Infinite (continuous distributions)

### Histogram Gradient Boosting
```python
{
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [3, 5, 7, None],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 50, 100],
    "max_iter": [300, 500, 800],
    "l2_regularization": [0.0, 0.1, 1.0],
}
```
**Search Space Size:** ~3,240 combinations

### XGBoost Classifier
```python
{
    "n_estimators": randint(50, 300),
    "max_depth": randint(3, 10),
    "learning_rate": uniform(0.01, 0.29),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "gamma": uniform(0, 5),
    "min_child_weight": randint(1, 10),
}
```
**Search Space Size:** Infinite (continuous distributions)

## Search Configuration

### Cross-Validation Strategy

- **Method:** `TimeSeriesSplit` from scikit-learn
- **Why:** Preserves temporal ordering, preventing data leakage
- **Default Splits:** 3
- **How it works:**
  ```
  Fold 1: Train [----] Test [--]
  Fold 2: Train [--------] Test [--]
  Fold 3: Train [------------] Test [--]
  ```

### Scoring Metrics

- **Regression:** `neg_mean_squared_error` (higher is better)
- **Classification:** `accuracy`

### Train/Test Split

- **Method:** Chronological split (respects time series nature)
- **Default:** 80% train / 20% test
- **Purpose:** Final evaluation on unseen recent data

### Parallelization

- **Default:** `-1` (use all CPU cores)
- **Recommendation:** Use fewer cores if system resources are limited
- **Override:** `--n-jobs 4`

## Analysis Features

### 1. Per-Market Analysis

**Identifies:**
- Best performing model per market
- Top 3 models with performance gaps
- Potential overfitting (large CV-test score difference)

**Example Output:**
```
Market: Day-Ahead Market (day_ahead)
  Metric: neg_mean_squared_error
  Best Model: hist_gradient_boosting
    Test Score: -95.5000
    CV Score: -93.2000
    Training Samples: 10000
    Features: 25

  Top Models:
    1. hist_gradient_boosting    Score: -95.5000  (Δ +0.0000, +0.0%)
    2. random_forest              Score: -98.3000  (Δ -2.8000, -2.9%)
    3. linear_regression          Score: -125.7000 (Δ -30.2000, -31.6%)
```

### 2. Per-Model Analysis

**Calculates:**
- Average rank across all markets
- Win rate (% of markets where model ranks 1st)
- Best and worst markets for each model

**Example Output:**
```
Model: Histogram Gradient Boosting (hist_gradient_boosting)
  Average Rank: 1.33
  Average Test Score: -85.2345 (±12.3456)
  Markets Tested: 3
  Best Market: day_ahead (score: -75.5000)
  Worst Market: imbalance_surplus (score: -95.0000)
  1st Place Finishes: 2/3
```

### 3. Hyperparameter Importance

**For Numerical Parameters:**
- Computes Spearman correlation with CV score
- Identifies parameters that strongly influence performance
- Suggests optimal value ranges

**For Categorical Parameters:**
- Groups trials by parameter value
- Computes mean score per value
- Identifies best category

**Example Output:**
```
hist_gradient_boosting × day_ahead:
  Trials: 30
  Top Influential Hyperparameters:
    learning_rate: correlation=-0.654 (p=0.0001)
    max_iter: correlation=+0.482 (p=0.0073)
    max_depth: correlation=-0.312 (p=0.0934)
    l2_regularization: correlation=+0.156 (p=0.4120)
```

### 4. Global Recommendations

**Provides:**
- Recommended model and hyperparameters per market
- Rationale based on importance analysis
- Overall best-performing models
- Common hyperparameter patterns

**Example Output:**
```
Day-Ahead Market (day_ahead):
  Recommended Model: Histogram Gradient Boosting
  Expected neg_mean_squared_error: -95.5000
  Recommended Hyperparameters:
    learning_rate: 0.05
    max_iter: 500
    max_depth: 7
    max_leaf_nodes: 31
    min_samples_leaf: 50
    l2_regularization: 0.0
  Key Insight: lower learning_rate correlates with better performance (r=-0.654)
```

## Usage Examples

### Example 1: Quick Search for One Market

```bash
# Search only day-ahead market with moderate iterations
python -m scripts.run_full_hyperparam_search \
    --markets day_ahead \
    --n-iter 20 \
    --cv-splits 3
```

**Expected Runtime:** ~5-10 minutes (3 models × 20 iterations)

### Example 2: Intensive Search for All Combinations

```bash
# Comprehensive search with high iterations
python -m scripts.run_full_hyperparam_search \
    --n-iter 100 \
    --cv-splits 5 \
    --update-defaults
```

**Expected Runtime:** ~2-4 hours (10 combinations × 100 iterations × 5 folds)

### Example 3: Compare Gradient Boosting Models

```bash
# Search only boosting models
python -m scripts.run_full_hyperparam_search \
    --models hist_gradient_boosting xgboost_classifier \
    --n-iter 50

# Analyze results
python -m scripts.analyze_hyperparams --generate-plots
```

### Example 4: Rapid Prototyping

```bash
# Minimal search for quick testing
python -m scripts.run_full_hyperparam_search \
    --markets day_ahead \
    --models random_forest \
    --n-iter 5 \
    --cv-splits 2
```

**Expected Runtime:** ~30 seconds

## Integration with Training Pipeline

### Loading Best Hyperparameters

The `forecast_service.py` automatically loads tuned hyperparameters when available:

```python
from app.services.hyperparameter_service import load_best_params

# Load best params for a combination
best_params = load_best_params(
    market_type="day_ahead",
    model_type="random_forest"
)

if best_params:
    # Use tuned hyperparameters
    model = RandomForestPriceModel(**best_params)
else:
    # Fall back to defaults from config
    model = RandomForestPriceModel()
```

### Using Defaults Configuration

If you ran the search with `--update-defaults`, you can programmatically access recommendations:

```python
import json
from pathlib import Path

defaults_path = Path("artifacts/hyperparameter_search/hyperparams_defaults.json")
with open(defaults_path) as f:
    defaults = json.load(f)

# Get recommendation for day_ahead market
day_ahead_config = defaults["day_ahead"]
best_model_type = max(
    day_ahead_config.items(),
    key=lambda x: x[1]["best_score"]
)[0]

print(f"Best model for day_ahead: {best_model_type}")
print(f"Parameters: {day_ahead_config[best_model_type]['best_params']}")
```

## Performance Considerations

### Memory Usage

- **Per Search:** ~100-500 MB depending on dataset size
- **Trial Data:** Stored incrementally to disk
- **Recommendation:** Close other applications for large searches

### CPU Utilization

- **Default:** Uses all available cores (`n_jobs=-1`)
- **Override:** Limit cores with `--n-jobs` if needed
- **Note:** More cores = faster but higher power usage

### Storage Requirements

- **Per Combination:** ~10-50 KB (JSON + CSV)
- **Full Pipeline (10 combinations):** ~500 KB - 1 MB
- **With Plots:** Add ~2-5 MB

### Estimated Runtimes

Based on typical dataset sizes (~10,000 samples):

| Configuration | Combinations | Runtime |
|---------------|--------------|---------|
| Quick (5 iter, 2 splits) | 1 | ~30 sec |
| Default (30 iter, 3 splits) | 1 | ~3-5 min |
| Intensive (100 iter, 5 splits) | 1 | ~15-25 min |
| Full Pipeline (30 iter, 3 splits) | 10 | ~30-50 min |
| Full Pipeline (100 iter, 5 splits) | 10 | ~2.5-4 hours |

## Best Practices

### 1. Start Small, Scale Up

```bash
# Step 1: Quick test with one combination
python -m scripts.run_full_hyperparam_search \
    --markets day_ahead --models random_forest --n-iter 5

# Step 2: Moderate search for selected combinations
python -m scripts.run_full_hyperparam_search \
    --markets day_ahead imbalance_shortage --n-iter 30

# Step 3: Full intensive search
python -m scripts.run_full_hyperparam_search --n-iter 100 --cv-splits 5
```

### 2. Monitor Progress

The orchestrator logs progress for each combination:
- ✓ Completed combinations
- ✗ Failed combinations with error details
- Best scores achieved

### 3. Analyze Incrementally

You can run analysis even if search is still running:

```bash
# In one terminal: run search
python -m scripts.run_full_hyperparam_search

# In another terminal: analyze partial results
python -m scripts.analyze_hyperparams
```

### 4. Iterate on Insights

1. Run initial broad search (30 iterations)
2. Analyze results to identify promising parameter ranges
3. Run focused search on best models with tighter ranges
4. Update production defaults

### 5. Version Control

```bash
# Save search results with git
git add artifacts/hyperparameter_search/summary.csv
git add artifacts/hyperparameter_search/hyperparams_defaults.json
git commit -m "Update hyperparameter search results - Jan 2025"
```

## Troubleshooting

### Issue: Search fails for specific combination

**Symptoms:** Error logged for one market × model combination

**Solutions:**
1. Check if market has sufficient data (min 720 samples recommended)
2. Verify model-market compatibility (regression vs classification)
3. Review error traceback in logs

### Issue: Analysis shows no trial data

**Symptoms:** "No trial data available" warnings

**Solutions:**
1. Ensure you ran search with recent version of orchestrator
2. Check that `save_search_results_with_trials()` is being called
3. Verify `*_trials_latest.csv` files exist in market directories

### Issue: Plots not generating

**Symptoms:** No plots directory created

**Solutions:**
1. Verify matplotlib is installed: `pip install matplotlib`
2. Use `--generate-plots` flag explicitly
3. Check logs for import errors

### Issue: Out of memory during search

**Symptoms:** Process killed or memory error

**Solutions:**
1. Reduce `--n-iter` (try 20 instead of 100)
2. Reduce `--cv-splits` (try 3 instead of 5)
3. Limit parallel jobs: `--n-jobs 2`
4. Search one market at a time: `--markets day_ahead`

## API Reference

### run_full_hyperparam_search.py

```
Usage: python -m scripts.run_full_hyperparam_search [OPTIONS]

Options:
  --markets [MARKET ...]        Filter by markets
  --models [MODEL ...]          Filter by models
  --n-iter INT                  Iterations per combination (default: 30)
  --cv-splits INT               CV splits (default: 3)
  --test-size FLOAT            Test set ratio (default: 0.2)
  --random-state INT            Random seed (default: 42)
  --n-jobs INT                  Parallel jobs (default: -1)
  --output-dir PATH             Output directory
  --update-defaults             Create hyperparams_defaults.json
```

### analyze_hyperparams.py

```
Usage: python -m scripts.analyze_hyperparams [OPTIONS]

Options:
  --results-dir PATH            Results directory
  --markets [MARKET ...]        Filter analysis by markets
  --generate-plots              Generate visualization plots
  --output-report PATH          Save report to file (not implemented)
```

## See Also

- [HYPERPARAMETER_SEARCH.md](HYPERPARAMETER_SEARCH.md) - Original hyperparameter search documentation
- [README.md](../README.md) - Project overview and setup
- [PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md) - Full project context
- [QUICKSTART.md](../QUICKSTART.md) - Quick start guide
