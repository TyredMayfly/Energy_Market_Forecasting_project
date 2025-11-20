# Hyperparameter Search Pipeline

This document describes the hyperparameter search pipeline for tuning all model × market combinations in the Market Forecasting application.

## Overview

The hyperparameter search pipeline provides:
- **Automated tuning** of all supported model × market combinations
- **Time-series cross-validation** to respect temporal ordering
- **Persistent storage** of best hyperparameters
- **Automatic loading** of tuned parameters during training
- **Comprehensive logging** and result tracking

## Quick Start

### Run Search for All Combinations

```bash
python scripts/search_hyperparameters.py
```

This will:
1. Enumerate all valid model × market combinations from configuration
2. Run hyperparameter search for each using cross-validation
3. Save results to `artifacts/hyperparameter_search/`
4. Log summary of all searches

### Run Search for Specific Market

```bash
python scripts/search_hyperparameters.py --market day_ahead
```

### Run Search for Specific Model

```bash
python scripts/search_hyperparameters.py --model random_forest
```

### Customize Search Parameters

```bash
python scripts/search_hyperparameters.py \
    --n-iter 100 \
    --cv-splits 10 \
    --test-size 0.2 \
    --random-state 42
```

### List Available Tuned Models

```bash
python scripts/search_hyperparameters.py --list
```

## Architecture

### Components

1. **`app/services/hyperparameter_service.py`**
   - Core hyperparameter search functionality
   - Functions:
     - `get_param_grid(model_type)` - Define search space for each model
     - `run_hyperparameter_search()` - Execute search with CV
     - `save_search_results()` - Persist results to disk
     - `load_best_params()` - Load tuned parameters
     - `get_model_market_combinations()` - Enumerate valid combinations

2. **`scripts/search_hyperparameters.py`**
   - Command-line interface for running searches
   - Supports filtering by market, model, or both
   - Provides detailed progress logging and summaries

3. **`artifacts/hyperparameter_search/`**
   - Storage for search results
   - Organized by market type
   - Contains both timestamped and "latest" versions

4. **`app/services/forecast_service.py`**
   - Modified to automatically load tuned parameters
   - Falls back to defaults if no tuned params available

### Search Strategy

The pipeline uses **RandomizedSearchCV** with **TimeSeriesSplit**:

- **TimeSeriesSplit**: Respects temporal ordering of data (critical for time series)
- **RandomizedSearch**: Samples from parameter distributions (more efficient than grid search)
- **Train-Test Split**: Final evaluation on held-out test set (chronological split)

### Parameter Grids

#### Linear Regression
- `fit_intercept`: [True, False]

#### Random Forest
- `n_estimators`: 50-300 (uniform distribution)
- `max_depth`: 5-30 (uniform distribution)
- `min_samples_split`: 2-20 (uniform distribution)
- `min_samples_leaf`: 1-10 (uniform distribution)
- `max_features`: ["sqrt", "log2", None]
- `bootstrap`: [True, False]

#### Histogram Gradient Boosting
- `learning_rate`: [0.03, 0.05, 0.1]
- `max_depth`: [3, 5, 7, None] (None = unlimited depth)
- `max_leaf_nodes`: [15, 31, 63]
- `min_samples_leaf`: [20, 50, 100]
- `max_iter`: [300, 500, 800]
- `l2_regularization`: [0.0, 0.1, 1.0]

#### XGBoost Classifier
- `n_estimators`: 50-300 (uniform distribution)
- `max_depth`: 3-12 (uniform distribution)
- `learning_rate`: 0.01-0.3 (uniform distribution)
- `subsample`: 0.6-1.0 (uniform distribution)
- `colsample_bytree`: 0.6-1.0 (uniform distribution)
- `gamma`: 0-0.5 (uniform distribution)
- `reg_alpha`: 0-1 (L1 regularization)
- `reg_lambda`: 0.5-2.5 (L2 regularization)

## Valid Model × Market Combinations

The system automatically validates compatibility:

### Regression Markets
- `day_ahead` (Day-Ahead Market prices)
  - Compatible models: `linear_regression`, `random_forest`, `hist_gradient_boosting`
- `imbalance_shortage` (Imbalance shortage prices)
  - Compatible models: `linear_regression`, `random_forest`, `hist_gradient_boosting`
- `imbalance_surplus` (Imbalance surplus prices)
  - Compatible models: `linear_regression`, `random_forest`, `hist_gradient_boosting`

### Classification Markets
- `regulation_state` (System regulation state)
  - Compatible models: `xgboost_classifier`

**Note**: `persistence` model is excluded (no hyperparameters to tune).

## Result Storage

### Directory Structure

```
artifacts/hyperparameter_search/
├── day_ahead/
│   ├── linear_regression_latest.json
│   ├── linear_regression_20250115_143022.json
│   ├── random_forest_latest.json
│   └── random_forest_20250115_145531.json
├── imbalance_shortage/
│   ├── linear_regression_latest.json
│   └── random_forest_latest.json
└── regulation_state/
    └── xgboost_classifier_latest.json
```

### Result File Format

Each JSON file contains:

```json
{
  "best_params": {
    "n_estimators": 150,
    "max_depth": 12,
    "min_samples_split": 5
  },
  "best_score": -125.43,
  "metadata": {
    "model_type": "random_forest",
    "market_type": "day_ahead",
    "search_timestamp": "2025-01-15T14:30:22.123456",
    "n_samples_train": 10779,
    "n_samples_test": 2695,
    "n_features": 25,
    "feature_names": ["price_lag_1", "price_lag_2", ...],
    "n_iter": 50,
    "cv_splits": 5,
    "test_size_ratio": 0.2,
    "scoring": "neg_mean_squared_error",
    "best_cv_score": -125.43,
    "test_score": -132.18,
    "cv_results": {
      "mean_test_scores": [...],
      "std_test_scores": [...],
      "params": [...]
    }
  }
}
```

## Using Tuned Hyperparameters

### Automatic Loading (Recommended)

The `ForecastService` automatically loads tuned hyperparameters when available:

```python
from app.services.forecast_service import ForecastService

service = ForecastService()

# Automatically uses tuned params if available, otherwise uses defaults
service.train_model(
    market_type="day_ahead",
    model_type="random_forest"
)
```

### Manual Loading

```python
from app.services.hyperparameter_service import load_best_params

params = load_best_params(
    market_type="day_ahead",
    model_type="random_forest"
)

if params:
    print(f"Tuned parameters: {params}")
else:
    print("No tuned parameters found, using defaults")
```

### Programmatic Search

```python
from app.services.hyperparameter_service import (
    run_hyperparameter_search,
    save_search_results
)

best_params, best_score, metadata = run_hyperparameter_search(
    model_type="random_forest",
    market_type="day_ahead",
    n_iter=100,
    cv_splits=10
)

save_search_results(
    market_type="day_ahead",
    model_type="random_forest",
    best_params=best_params,
    best_score=best_score,
    metadata=metadata
)
```

## Testing

Comprehensive unit tests are provided in `tests/unit/test_hyperparameter_search.py`:

```bash
# Run hyperparameter search tests
pytest tests/unit/test_hyperparameter_search.py -v

# Run all tests
pytest
```

Test coverage includes:
- Parameter grid generation
- Model-market compatibility validation
- Result saving and loading
- Directory structure creation
- Combination enumeration
- Integration tests

## Command-Line Reference

```bash
# Search all combinations
python scripts/search_hyperparameters.py

# Search specific market
python scripts/search_hyperparameters.py --market day_ahead

# Search specific model
python scripts/search_hyperparameters.py --model random_forest

# Search specific combination
python scripts/search_hyperparameters.py --market day_ahead --model random_forest

# Customize search
python scripts/search_hyperparameters.py --n-iter 100 --cv-splits 10

# List available tuned models
python scripts/search_hyperparameters.py --list

# Help
python scripts/search_hyperparameters.py --help
```

### Arguments

- `--market`: Filter by market type (e.g., `day_ahead`, `imbalance_shortage`)
- `--model`: Filter by model type (e.g., `random_forest`, `xgboost_classifier`)
- `--n-iter`: Number of parameter settings to sample (default: 50)
- `--cv-splits`: Number of cross-validation splits (default: 5)
- `--test-size`: Ratio of data for final testing (default: 0.2)
- `--random-state`: Random seed for reproducibility (default: 42)
- `--n-jobs`: Number of parallel jobs, -1 for all cores (default: -1)
- `--verbose`: Verbosity level (0=silent, 1=progress, 2=detailed) (default: 1)
- `--list`: List all available tuned models and exit

## Performance Considerations

### Computational Cost

- **Random Forest**: Most expensive (many trees × many parameters)
  - Typical search time: 10-30 minutes per market
- **XGBoost**: Moderate cost (boosting iterations × tree depth)
  - Typical search time: 5-15 minutes per market
- **Linear Regression**: Very fast (few parameters)
  - Typical search time: 1-2 minutes per market

### Recommendations

1. **Start small**: Use default `--n-iter 50` for initial searches
2. **Increase gradually**: Use `--n-iter 100` or more for production
3. **Monitor CV splits**: More splits = more robust, but slower (5-10 recommended)
4. **Use parallelization**: Default `--n-jobs -1` uses all CPU cores
5. **Schedule overnight**: Run full search pipeline during off-hours

### Example Workflow

```bash
# Quick exploratory search (fast)
python scripts/search_hyperparameters.py --n-iter 20 --cv-splits 3

# Standard production search (balanced)
python scripts/search_hyperparameters.py --n-iter 50 --cv-splits 5

# Thorough production search (slow but robust)
python scripts/search_hyperparameters.py --n-iter 100 --cv-splits 10
```

## Best Practices

1. **Always use time-series CV**: Never use standard k-fold for time series
2. **Keep test set chronological**: Final evaluation on most recent data
3. **Monitor overfitting**: Compare CV score vs. test score in metadata
4. **Re-tune periodically**: As new data arrives, re-run searches quarterly
5. **Version control results**: Commit `*_latest.json` files for reproducibility
6. **Log everything**: All searches logged with full metadata

## Troubleshooting

### "No valid combinations found"
- Check market and model names match configuration
- Verify `MARKET_TYPES` and `MODEL_TYPES` in `app/core/config.py`
- Use `--list` to see available tuned models

### "No data available for market"
- Ensure data files exist in `data/` directory
- Check data file paths in `MARKET_TYPES` configuration
- Verify data has been downloaded and processed

### "Model cannot be used for classification/regression"
- Models have strict compatibility requirements
- Linear regression and random forest: regression only
- XGBoost classifier: classification only
- Check market `target_type` in configuration

### Search takes too long
- Reduce `--n-iter` (default 50 → 20)
- Reduce `--cv-splits` (default 5 → 3)
- Filter to specific market or model
- Ensure `--n-jobs -1` for parallelization

### Memory errors
- Reduce `--cv-splits` to lower memory usage
- Process one market at a time: `--market day_ahead`
- Reduce `n_estimators` in parameter grid (edit `hyperparameter_service.py`)

## Future Enhancements

Potential improvements:
- Support for Bayesian optimization (Optuna)
- Distributed search across multiple machines
- Automatic re-tuning triggers based on model drift
- Hyperparameter importance analysis
- Meta-learning across markets

## References

- [scikit-learn RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
- [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [Random Forest Tuning](https://scikit-learn.org/stable/modules/ensemble.html#parameters)
