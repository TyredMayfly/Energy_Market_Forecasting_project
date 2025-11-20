# Hyperparameter Search - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Understand What's Available

View all valid model × market combinations:

```bash
python examples\hyperparameter_search_example.py
```

This shows you:
- 7 valid model × market combinations
- Parameter grids for each model type
- Example usage patterns

### Step 2: Run a Quick Search

Start with a fast exploratory search:

```bash
python scripts\search_hyperparameters.py --n-iter 20 --cv-splits 3
```

This will:
- Search all 7 combinations
- Use 20 iterations per search (fast)
- Use 3-fold cross-validation
- Complete in ~30-60 minutes

**For even faster testing**, search just one combination:

```bash
python scripts\search_hyperparameters.py --market day_ahead --model linear_regression --n-iter 20 --cv-splits 3
```

This completes in ~1-2 minutes.

### Step 3: Check Results

List available tuned models:

```bash
python scripts\search_hyperparameters.py --list
```

Results are saved in `hyperparameters/`:

```
hyperparameters/
├── day_ahead/
│   ├── linear_regression_latest.json  ← Best params for this combo
│   └── random_forest_latest.json
└── ...
```

### Step 4: Use Tuned Parameters

**Automatic** (recommended):

```python
from app.services.forecast_service import ForecastService

service = ForecastService()

# Automatically uses tuned params if available
service.train_model(
    market_type="day_ahead",
    model_type="random_forest"
)
```

**Manual**:

```python
from app.services.hyperparameter_service import load_best_params

params = load_best_params(
    market_type="day_ahead",
    model_type="random_forest"
)

print(f"Tuned parameters: {params}")
```

## 📊 Common Usage Patterns

### Pattern 1: Quick Exploration
```bash
# Fast search with reduced iterations
python scripts\search_hyperparameters.py --n-iter 20 --cv-splits 3
```

**Use when**: First time exploring, development, testing

### Pattern 2: Standard Production Search
```bash
# Balanced search (default settings)
python scripts\search_hyperparameters.py
```

**Use when**: Regular production use, quarterly retuning

### Pattern 3: Thorough Tuning
```bash
# Comprehensive search with more iterations
python scripts\search_hyperparameters.py --n-iter 100 --cv-splits 10
```

**Use when**: Critical applications, final production deployment

### Pattern 4: Targeted Search
```bash
# Search only one market
python scripts\search_hyperparameters.py --market day_ahead

# Search only one model type
python scripts\search_hyperparameters.py --model random_forest

# Search specific combination
python scripts\search_hyperparameters.py --market day_ahead --model random_forest
```

**Use when**: Updating specific models, investigating performance issues

## 🎯 What Each Model Does

### Linear Regression
- **What**: Simple linear relationships
- **Tunable**: `fit_intercept` (True/False)
- **Search Time**: 1-2 minutes
- **Best For**: Quick baselines, interpretable models

### Random Forest
- **What**: Ensemble of decision trees
- **Tunable**: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`
- **Search Time**: 10-30 minutes
- **Best For**: Capturing non-linear patterns, feature importance

### Histogram Gradient Boosting
- **What**: Efficient gradient boosting with histogram-based algorithm
- **Tunable**: `learning_rate`, `max_depth`, `max_leaf_nodes`, `min_samples_leaf`, `max_iter`, `l2_regularization`
- **Search Time**: 10-25 minutes
- **Best For**: Large datasets, fast training, non-linear patterns

### XGBoost Classifier
- **What**: Gradient boosting for classification
- **Tunable**: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`
- **Search Time**: 5-15 minutes
- **Best For**: Regulation state prediction (UP, DOWN, BALANCED)

## 📁 Where Things Are

```
Market_Forecasting_example/
├── app/services/hyperparameter_service.py  ← Core functions
├── scripts/search_hyperparameters.py       ← CLI script
├── hyperparameters/                        ← Results saved here
├── tests/unit/test_hyperparameter_search.py ← Tests
├── examples/hyperparameter_search_example.py ← Examples
└── docs/HYPERPARAMETER_SEARCH.md           ← Full docs
```

## ⚙️ CLI Options Reference

```bash
python scripts\search_hyperparameters.py [OPTIONS]

Options:
  --market MARKET          Filter by market (e.g., day_ahead)
  --model MODEL            Filter by model (e.g., random_forest)
  --n-iter N               Number of iterations (default: 50)
  --cv-splits N            CV splits (default: 5)
  --test-size RATIO        Test set ratio (default: 0.2)
  --random-state N         Random seed (default: 42)
  --n-jobs N               Parallel jobs (default: -1, all cores)
  --verbose LEVEL          Verbosity (0-2, default: 1)
  --list                   List available tuned models
  --help                   Show help message
```

## 🔍 Understanding Results

Each result file (e.g., `random_forest_latest.json`) contains:

```json
{
  "best_params": {
    "n_estimators": 150,
    "max_depth": 12,
    "min_samples_split": 5
  },
  "best_score": -125.43,  // CV score (neg MSE for regression)
  "metadata": {
    "model_type": "random_forest",
    "market_type": "day_ahead",
    "search_timestamp": "2025-01-15T14:30:22",
    "n_samples_train": 10779,
    "n_samples_test": 2695,
    "best_cv_score": -125.43,  // Cross-validation
    "test_score": -132.18      // Final test set
  }
}
```

**Key metrics**:
- `best_cv_score`: Performance during cross-validation
- `test_score`: Performance on held-out test data
- Compare these to detect overfitting

## 🐛 Troubleshooting

### "No data available for market"
**Solution**: Ensure data files exist in `data/` directory

### "No valid combinations found"
**Solution**: Check market and model names match config
```bash
python examples\hyperparameter_search_example.py  # Shows valid combos
```

### Search takes too long
**Solution**: Reduce iterations or splits
```bash
python scripts\search_hyperparameters.py --n-iter 20 --cv-splits 3
```

### Memory errors
**Solution**: Search one market at a time
```bash
python scripts\search_hyperparameters.py --market day_ahead
```

## 📚 Next Steps

1. **Read full docs**: `docs/HYPERPARAMETER_SEARCH.md`
2. **Run examples**: `python examples\hyperparameter_search_example.py`
3. **Run tests**: `pytest tests/unit/test_hyperparameter_search.py -v`
4. **Start searching**: `python scripts\search_hyperparameters.py --n-iter 20 --cv-splits 3`

## 💡 Pro Tips

1. **Start small**: Use `--n-iter 20 --cv-splits 3` for first run
2. **Check results**: Use `--list` to see what's been tuned
3. **Re-tune quarterly**: As new data arrives, re-run searches
4. **Monitor overfitting**: Compare CV score vs. test score in results
5. **Schedule overnight**: Run thorough searches during off-hours
6. **Use parallelization**: Default `--n-jobs -1` uses all cores

## ✅ Validation

Everything working? Run these checks:

```bash
# 1. View examples
python examples\hyperparameter_search_example.py

# 2. Check CLI help
python scripts\search_hyperparameters.py --help

# 3. List available models (should be empty initially)
python scripts\search_hyperparameters.py --list

# 4. Run quick test search (1-2 minutes)
python scripts\search_hyperparameters.py --market day_ahead --model linear_regression --n-iter 20 --cv-splits 3

# 5. Verify results saved
python scripts\search_hyperparameters.py --list

# 6. Run all tests
pytest tests/unit/test_hyperparameter_search.py -v
```

All passing? You're ready to go! 🎉

---

**For detailed information**, see: `docs/HYPERPARAMETER_SEARCH.md`
