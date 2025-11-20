# Hyperparameter Search Results

This directory contains the results of hyperparameter searches for all model × market combinations.

## Directory Structure

```
hyperparameter_search/
├── day_ahead/
│   ├── linear_regression_latest.json
│   ├── linear_regression_20250101_120000.json
│   ├── random_forest_latest.json
│   └── random_forest_20250101_130000.json
├── imbalance_shortage/
│   ├── linear_regression_latest.json
│   └── random_forest_latest.json
├── imbalance_surplus/
│   └── ...
└── regulation_state/
    └── xgboost_classifier_latest.json
```

## File Format

Each JSON file contains:
- `best_params`: Dictionary of best hyperparameters found
- `best_score`: Cross-validation score achieved
- `metadata`: Search configuration and detailed results
  - Model and market types
  - Number of samples, features
  - Cross-validation settings
  - Test set performance
  - Full CV results

## Using Tuned Hyperparameters

The `*_latest.json` files are automatically loaded by the forecast service when available.
To manually load parameters:

```python
from app.services.hyperparameter_service import load_best_params

params = load_best_params(market_type="day_ahead", model_type="random_forest")
```

## Running Hyperparameter Search

See `scripts/search_hyperparameters.py` for running new searches.
