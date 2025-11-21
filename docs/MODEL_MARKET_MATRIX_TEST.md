# Model × Market Matrix Integration Test - Implementation Summary

## Overview

A comprehensive integration test suite has been implemented to systematically validate all model × market combinations in the forecasting system.

## Files Created/Modified

### New Files
1. **`tests/integration/test_model_market_matrix.py`** - Main test module (380+ lines)
2. **`tests/integration/README.md`** - Documentation for integration tests

### Modified Files
1. **`pyproject.toml`** - Added `slow` marker configuration

## Test Implementation Details

### Test Functions

#### `test_model_market_combination_performance(market_type, model_type)`
Main parametrized test that validates each (market, model) combination:
- **13 combinations tested** (3 markets × 4 regression models + 1 market × 1 classifier)
- Generates synthetic but realistic data (2000 samples, 75/25 train/test split)
- Trains model with best available hyperparameters or sensible defaults
- Validates predictions are non-NaN and non-Inf
- Checks performance meets lenient thresholds

#### `test_all_markets_have_valid_combinations()`
Verifies each market has at least one compatible model type

#### `test_all_models_have_valid_combinations()`
Verifies each model has at least one compatible market type

#### `test_combination_count()`
Validates expected number of combinations and lists them

### Helper Functions

- **`list_all_markets()`** - Dynamically gets markets from config
- **`list_all_model_types()`** - Dynamically gets models from config
- **`get_model_market_combinations()`** - Generates valid combinations based on task type
- **`prepare_data_for_market()`** - Creates synthetic training/test data
- **`build_model_for_market_and_type()`** - Instantiates models with hyperparameters
- **`compute_rmse()`** - Regression metric
- **`compute_f1_macro()`** - Classification metric

## Performance Thresholds (Lenient)

### Regression (RMSE in EUR/MWh)
- Day-ahead: < 500
- Imbalance shortage: < 1000
- Imbalance surplus: < 1000

### Classification (F1-score)
- Regulation state: > 0.2

These thresholds ensure models are learning something meaningful without enforcing strict quality requirements.

## Model × Market Combinations

| Market Type | Target Type | Compatible Models |
|-------------|------------|-------------------|
| day_ahead | regression | persistence, linear_regression, random_forest, hist_gradient_boosting |
| imbalance_shortage | regression | persistence, linear_regression, random_forest, hist_gradient_boosting |
| imbalance_surplus | regression | persistence, linear_regression, random_forest, hist_gradient_boosting |
| regulation_state | classification | xgboost_classifier |

**Total: 13 valid combinations**

## Usage

### Run all tests
```bash
pytest tests/integration/test_model_market_matrix.py -m slow -v
```

### Exclude slow tests (CI)
```bash
pytest tests/integration/test_model_market_matrix.py -m "not slow"
```

### Run specific combination
```bash
pytest tests/integration/test_model_market_matrix.py::test_model_market_combination_performance[day_ahead-linear_regression] -v
```

## Test Results

✅ **All 16 tests passing** (13 combinations + 3 validation tests)
- Execution time: ~8-25 seconds for full suite
- No warnings or errors
- Proper marker configuration

## Key Features

### Dynamic Configuration
- Uses existing `MARKET_TYPES` and `MODEL_TYPES` from `app/core/config.py`
- Automatically detects new models/markets when added to config
- No hardcoded market/model names

### Realistic Synthetic Data
- Time series patterns with appropriate noise levels
- Different volatility for different market types
- Proper temporal ordering for train/test split
- Realistic feature engineering (lags, time features, weather)

### Hyperparameter Integration
- Attempts to load tuned hyperparameters via `load_best_params()`
- Falls back to sensible defaults if not available
- Uses smaller defaults for testing speed

### Robust Error Handling
- Skips combinations if data preparation fails
- Skips if insufficient samples
- Uses `pytest.importorskip` for optional dependencies (XGBoost)
- Clear failure messages with context

### Informative Output
- Prints performance metrics for each combination
- Shows which threshold was used
- Classification models print detailed classification report

## Design Principles

1. **Comprehensive Coverage** - Tests every valid combination automatically
2. **Fast Execution** - Uses synthetic data and reduced model sizes
3. **Lenient Thresholds** - Catches catastrophic failures, not optimization issues
4. **Maintainability** - Self-updating when config changes
5. **CI-Friendly** - Can be excluded with marker for faster pipelines

## Future Extensions

To add new models or markets:
1. Add to `app/core/config.py` in `MARKET_TYPES` or `MODEL_TYPES`
2. Import model class in test file
3. Add to model instantiation logic in `build_model_for_market_and_type()`
4. Optionally adjust thresholds if needed

Test will automatically pick up new combinations!

## Integration with Existing Architecture

- Uses existing model classes unchanged
- Compatible with hyperparameter service
- Follows same feature engineering patterns
- Respects model-market compatibility rules
- Aligns with existing test fixtures and patterns
