# Integration Tests

This directory contains integration tests that validate the end-to-end functionality of the market forecasting system.

## Test Files

### `test_model_market_matrix.py`

Comprehensive integration test that systematically validates all model × market combinations.

**What it tests:**
- Every valid combination of model type and market type
- Data preparation for each market
- Model instantiation with hyperparameters
- Training on synthetic but realistic data
- Prediction generation
- Performance validation using lenient thresholds

**Metrics used:**
- **RMSE** for regression targets (day-ahead, imbalance prices)
- **F1-score (macro)** for classification targets (regulation state)

**Thresholds:**
- Day-ahead prices: RMSE < 500 EUR/MWh
- Imbalance prices: RMSE < 1000 EUR/MWh
- Regulation state: F1-score > 0.2

These are intentionally lenient to catch catastrophic failures while allowing for natural variation in model performance on synthetic data.

**Model × Market Combinations tested (13 total):**

| Market Type | Compatible Models |
|-------------|------------------|
| day_ahead | persistence, linear_regression, random_forest, hist_gradient_boosting |
| imbalance_shortage | persistence, linear_regression, random_forest, hist_gradient_boosting |
| imbalance_surplus | persistence, linear_regression, random_forest, hist_gradient_boosting |
| regulation_state | xgboost_classifier |

## Running the Tests

### Run all integration tests (including slow tests):
```bash
pytest tests/integration/test_model_market_matrix.py -m slow -v
```

### Exclude slow tests (useful for CI):
```bash
pytest tests/integration/test_model_market_matrix.py -m "not slow"
```

### Run a specific combination:
```bash
pytest tests/integration/test_model_market_matrix.py::test_model_market_combination_performance[day_ahead-linear_regression] -v
```

### Run with coverage:
```bash
pytest tests/integration/test_model_market_matrix.py -m slow --cov=app --cov-report=html
```

## Test Duration

The full matrix test takes approximately **8-25 seconds** to run all 16 test cases (13 model×market combinations + 3 validation tests).

## Purpose

This test ensures that:
1. The pipeline doesn't have catastrophic failures
2. All model types can be instantiated correctly
3. All market types have working data preparation
4. Models produce valid (non-NaN, non-Inf) predictions
5. Models achieve at least minimal performance (better than random)

It is **not** designed to validate optimal model quality or enforce strict performance requirements.

## Maintenance

When adding new models or markets:
1. Add the model class to the imports
2. Add the model type to `MODEL_TYPES` in `app/core/config.py`
3. Add the market type to `MARKET_TYPES` in `app/core/config.py`
4. Update the threshold dictionary if needed
5. The test will automatically pick up the new combinations

No changes to the test code are needed thanks to the dynamic combination generation from the configuration.
