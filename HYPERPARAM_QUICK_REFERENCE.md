# Hyperparameter Search Quick Reference

## Commands

### Run Full Search

```bash
# All combinations (default: 30 iterations, 3 CV splits)
python -m scripts.run_full_hyperparam_search

# Quick test
python -m scripts.run_full_hyperparam_search --markets day_ahead --models random_forest --n-iter 5

# Intensive search with defaults update
python -m scripts.run_full_hyperparam_search --n-iter 100 --cv-splits 5 --update-defaults

# Filter by market
python -m scripts.run_full_hyperparam_search --markets day_ahead imbalance_shortage

# Filter by model
python -m scripts.run_full_hyperparam_search --models hist_gradient_boosting random_forest
```

### Analyze Results

```bash
# Basic analysis
python -m scripts.analyze_hyperparams

# With visualization plots
python -m scripts.analyze_hyperparams --generate-plots

# Filter by market
python -m scripts.analyze_hyperparams --markets day_ahead
```

### Run Examples

```bash
# View all examples
python examples/full_hyperparam_pipeline_example.py

# Run individual model search
python -m scripts.search_hyperparameters --market day_ahead --model random_forest
```

## Python API

```python
from app.services.hyperparameter_service import (
    get_model_market_combinations,
    load_best_params,
    run_hyperparameter_search,
    save_search_results_with_trials,
)

# Get all valid combinations
combinations = get_model_market_combinations()
# Returns: [('day_ahead', 'linear_regression'), ...]

# Load tuned hyperparameters
best_params = load_best_params("day_ahead", "random_forest")
# Returns: {'n_estimators': 150, 'max_depth': 8, ...} or None

# Run search for one combination
best_params, best_score, metadata = run_hyperparameter_search(
    model_type="random_forest",
    market_type="day_ahead",
    n_iter=30,
    cv_splits=3,
)

# Save results with trial data
save_search_results_with_trials(
    market_type="day_ahead",
    model_type="random_forest",
    best_params=best_params,
    best_score=best_score,
    metadata=metadata,
)
```

## Data Analysis

```python
import pandas as pd

# Load summary
summary = pd.read_csv("artifacts/hyperparameter_search/summary.csv")

# Best model per market
summary.loc[summary.groupby("market")["test_score"].idxmax()]

# Load trial data
trials = pd.read_csv(
    "artifacts/hyperparameter_search/day_ahead/random_forest_trials_latest.csv"
)

# Parameter correlations
trials[["n_estimators", "mean_cv_score"]].corr()
```

## File Locations

```
artifacts/hyperparameter_search/
├── summary.csv                              # All search results
├── hyperparams_defaults.json                # Recommended configs
├── plots/                                   # Visualizations
│   ├── model_comparison_by_market.png
│   └── <market>_<model>_param_importance.png
└── <market>/
    ├── <model>_latest.json                  # Best params
    ├── <model>_trials_latest.csv            # Trial data
    └── <model>_<timestamp>.json             # Timestamped backup
```

## Model × Market Combinations (10 total)

### Regression Markets (9 combinations)
- **day_ahead**: linear_regression, random_forest, hist_gradient_boosting
- **imbalance_shortage**: linear_regression, random_forest, hist_gradient_boosting
- **imbalance_surplus**: linear_regression, random_forest, hist_gradient_boosting

### Classification Market (1 combination)
- **regulation_state**: xgboost_classifier

## Common Options

### run_full_hyperparam_search.py
- `--markets [MARKET ...]` - Filter by markets
- `--models [MODEL ...]` - Filter by models
- `--n-iter INT` - Iterations per combination (default: 30)
- `--cv-splits INT` - CV splits (default: 3)
- `--test-size FLOAT` - Test set ratio (default: 0.2)
- `--random-state INT` - Random seed (default: 42)
- `--n-jobs INT` - Parallel jobs (default: -1 = all cores)
- `--output-dir PATH` - Output directory
- `--update-defaults` - Create hyperparams_defaults.json

### analyze_hyperparams.py
- `--results-dir PATH` - Results directory
- `--markets [MARKET ...]` - Filter analysis by markets
- `--generate-plots` - Generate visualization plots

## Typical Workflow

```bash
# 1. Quick test (30 seconds)
python -m scripts.run_full_hyperparam_search \
    --markets day_ahead --models random_forest --n-iter 5

# 2. Moderate search (30-50 minutes)
python -m scripts.run_full_hyperparam_search --n-iter 30

# 3. Analyze (instant)
python -m scripts.analyze_hyperparams --generate-plots

# 4. Intensive search (2-4 hours)
python -m scripts.run_full_hyperparam_search \
    --n-iter 100 --cv-splits 5 --update-defaults

# 5. Final analysis
python -m scripts.analyze_hyperparams --generate-plots
```

## Expected Runtimes

| Config | Combinations | Runtime |
|--------|--------------|---------|
| Quick (5 iter, 2 splits) | 1 | ~30 sec |
| Default (30 iter, 3 splits) | 1 | ~3-5 min |
| Default (30 iter, 3 splits) | 10 | ~30-50 min |
| Intensive (100 iter, 5 splits) | 10 | ~2.5-4 hours |

## Troubleshooting

### Out of Memory
```bash
# Reduce iterations
--n-iter 20

# Reduce CV splits
--cv-splits 3

# Limit parallel jobs
--n-jobs 2

# Search one market at a time
--markets day_ahead
```

### No Trial Data
- Ensure using `save_search_results_with_trials()` (automatic in orchestrator)
- Check `*_trials_latest.csv` files exist

### Missing Plots
```bash
# Install matplotlib
pip install matplotlib

# Use --generate-plots flag
python -m scripts.analyze_hyperparams --generate-plots
```

## Documentation

- **[FULL_HYPERPARAM_PIPELINE.md](docs/FULL_HYPERPARAM_PIPELINE.md)** - Complete guide
- **[HYPERPARAMETER_SEARCH.md](docs/HYPERPARAMETER_SEARCH.md)** - Individual search
- **[examples/full_hyperparam_pipeline_example.py](examples/full_hyperparam_pipeline_example.py)** - Code examples
- **[FULL_HYPERPARAM_PIPELINE_SUMMARY.md](FULL_HYPERPARAM_PIPELINE_SUMMARY.md)** - Implementation summary
