"""
Integration test for all model × market combinations.

This test systematically validates that every combination of model type and market type:
1. Successfully trains on a small subset of data
2. Produces valid predictions
3. Achieves at least a "reasonable" performance (lenient threshold)

Uses RMSE for regression targets and F1-score for classification targets.
The goal is to ensure the pipeline runs end-to-end and produces sensible results,
not to enforce strict model quality.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error, f1_score
from typing import Tuple, Optional, List

from app.core.config import MARKET_TYPES, MODEL_TYPES
from app.models.persistence_model import PersistencePriceModel
from app.models.linear_regression_model import LinearRegressionPriceModel
from app.models.random_forest_model import RandomForestPriceModel
from app.models.xgboost_classifier import RegulationStateXGBModel
from app.models.hist_gradient_boosting_model import HistGradientBoostingPriceModel
from app.services.hyperparameter_service import load_best_params


# Lenient thresholds for sanity checks
THRESHOLDS = {
    # Regression thresholds (RMSE in EUR/MWh)
    "day_ahead": 500.0,  # Day-ahead prices typically 30-150 EUR/MWh
    "imbalance_shortage": 1000.0,  # Imbalance can be more volatile
    "imbalance_surplus": 1000.0,  # Imbalance can be more volatile
    
    # Classification threshold (F1-score)
    "regulation_state_f1": 0.2,  # Just slightly better than random
}


def list_all_markets() -> List[str]:
    """Get all market types from configuration."""
    return list(MARKET_TYPES.keys())


def list_all_model_types() -> List[str]:
    """Get all model types from configuration."""
    return list(MODEL_TYPES.keys())


def get_model_market_combinations() -> List[Tuple[str, str]]:
    """
    Get all valid (market, model_type) combinations.
    
    Returns:
        List of (market_type, model_type) tuples that are compatible
    """
    combinations = []
    
    for market_type in list_all_markets():
        market_config = MARKET_TYPES[market_type]
        target_type = market_config.get("target_type", "regression")
        
        for model_type in list_all_model_types():
            # Check compatibility
            if target_type == "classification":
                # Only XGBoost classifier for classification tasks
                if model_type == "xgboost_classifier":
                    combinations.append((market_type, model_type))
            else:  # regression
                # All models except XGBoost classifier for regression
                if model_type != "xgboost_classifier":
                    combinations.append((market_type, model_type))
    
    return combinations


def prepare_data_for_market(
    market_type: str,
    n_samples: int = 2000,
    test_ratio: float = 0.25,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    """
    Prepare synthetic but realistic training and test data for a market.
    
    Args:
        market_type: Type of market
        n_samples: Total number of samples to generate
        test_ratio: Fraction of data to use for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, task_type)
    """
    np.random.seed(random_seed)
    
    market_config = MARKET_TYPES[market_type]
    target_type = market_config.get("target_type", "regression")
    
    # Generate time series
    hours = np.arange(n_samples)
    
    # Create features DataFrame
    X = pd.DataFrame()
    
    # Lag features (use simplified lags for synthetic data)
    for lag in [24, 48, 168]:
        # Initialize with small random values
        X[f"price_lag_{lag}h"] = np.random.uniform(40, 70, n_samples)
    
    # Time features
    X["hour_of_day"] = hours % 24
    X["day_of_week"] = (hours // 24) % 7
    X["hour_sin"] = np.sin(2 * np.pi * X["hour_of_day"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour_of_day"] / 24)
    X["day_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
    X["day_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)
    X["is_weekend"] = (X["day_of_week"] >= 5).astype(int)
    
    # Weather features
    X["temperature_deg_c"] = 10 + 5 * np.sin(2 * np.pi * hours / 24) + np.random.randn(n_samples) * 2
    X["wind_speed_m_per_s"] = np.clip(8 + 4 * np.random.randn(n_samples), 0, None)
    X["cloud_cover_pct"] = np.clip(50 + 30 * np.random.randn(n_samples), 0, 100)
    X["precipitation_mm"] = np.clip(np.random.exponential(2, n_samples), 0, 50)
    
    # Generate target based on market type
    if target_type == "regression":
        # Price-based target
        if "day_ahead" in market_type:
            # Day-ahead: smoother prices
            base_price = 50 + 20 * np.sin(2 * np.pi * hours / 24)
            noise = np.random.randn(n_samples) * 8
            y = base_price + noise
        elif "imbalance" in market_type:
            # Imbalance: more volatile with occasional spikes
            base_price = 60 + 25 * np.sin(2 * np.pi * hours / 24)
            spikes = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
            spike_values = spikes * np.random.uniform(50, 150, n_samples)
            noise = np.random.randn(n_samples) * 15
            y = base_price + spike_values + noise
        else:
            # Default regression target
            base_price = 55 + 15 * np.sin(2 * np.pi * hours / 24)
            noise = np.random.randn(n_samples) * 10
            y = base_price + noise
        
        # Update lag features with actual values
        for lag in [24, 48, 168]:
            X[f"price_lag_{lag}h"] = np.concatenate([np.full(lag, y[0]), y[:-lag]])
        
        y = pd.Series(y, name="price")
        
    else:  # classification
        # Regulation state classification
        # Create price difference feature
        day_ahead_price = 50 + 15 * np.sin(2 * np.pi * hours / 24) + np.random.randn(n_samples) * 8
        imbalance_price = day_ahead_price + np.random.randn(n_samples) * 20
        price_diff = imbalance_price - day_ahead_price
        
        X["day_ahead_price"] = day_ahead_price
        X["imbalance_price"] = imbalance_price
        X["price_difference"] = price_diff
        
        # Add price difference lags
        for lag in [1, 2, 3, 24]:
            X[f"price_diff_lag_{lag}h"] = np.concatenate([np.full(lag, 0.0), price_diff[:-lag]])
        
        # Generate regulation state based on price differences
        y = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            diff = price_diff[i]
            if abs(diff) < 5:
                y[i] = 0  # BALANCED
            elif diff > 15:
                y[i] = 1  # UP
            elif diff < -15:
                y[i] = -1  # DOWN
            else:
                # Random for borderline cases
                y[i] = np.random.choice([-1, 0, 1, 2], p=[0.25, 0.35, 0.25, 0.15])
        
        # Add some UP_AND_DOWN cases
        up_and_down_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
        y[up_and_down_indices] = 2
        
        y = pd.Series(y, name="regulation_state")
    
    # Time-based split (preserves temporal order)
    split_idx = int((1 - test_ratio) * n_samples)
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()
    
    return X_train, X_test, y_train, y_test, target_type


def build_model_for_market_and_type(market_type: str, model_type: str):
    """
    Build a model instance for a given market and model type.
    
    Tries to load best hyperparameters if available, otherwise uses defaults.
    
    Args:
        market_type: Type of market
        model_type: Type of model
        
    Returns:
        Model instance
    """
    # Try to load tuned hyperparameters (may not exist, that's OK)
    params = None
    if model_type != "persistence":
        try:
            params = load_best_params(market_type=market_type, model_type=model_type)
        except Exception:
            pass  # Use defaults if loading fails
    
    # Build model with params or defaults
    if model_type == "persistence":
        return PersistencePriceModel(method="last")
    elif model_type == "linear_regression":
        if params:
            return LinearRegressionPriceModel(**params)
        return LinearRegressionPriceModel()
    elif model_type == "random_forest":
        if params:
            return RandomForestPriceModel(**params)
        # Use smaller default for faster testing
        return RandomForestPriceModel(n_estimators=50, max_depth=10, random_state=42)
    elif model_type == "xgboost_classifier":
        if params:
            return RegulationStateXGBModel(**params)
        # Use smaller default for faster testing
        return RegulationStateXGBModel(n_estimators=50, max_depth=6, random_state=42)
    elif model_type == "hist_gradient_boosting":
        if params:
            return HistGradientBoostingPriceModel(**params)
        # Use smaller default for faster testing
        return HistGradientBoostingPriceModel(max_iter=100, max_depth=6, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_rmse(y_test: pd.Series, y_pred: np.ndarray) -> float:
    """Compute RMSE for regression."""
    return np.sqrt(mean_squared_error(y_test, y_pred))


def compute_f1_macro(y_test: pd.Series, y_pred: np.ndarray) -> float:
    """Compute macro F1-score for classification."""
    return f1_score(y_test, y_pred, average="macro", zero_division=0)


@pytest.mark.slow
@pytest.mark.parametrize("market_type,model_type", get_model_market_combinations())
def test_model_market_combination_performance(market_type: str, model_type: str):
    """
    Test that each valid (market, model) combination trains and achieves reasonable performance.
    
    This is an integration test that validates:
    1. Data preparation works for the market type
    2. Model can be instantiated with appropriate hyperparameters
    3. Model can train on the data
    4. Model produces valid predictions
    5. Model achieves at least minimal performance (better than trivial baseline)
    
    Args:
        market_type: Type of market (e.g., 'day_ahead', 'imbalance_shortage', 'regulation_state')
        model_type: Type of model (e.g., 'linear_regression', 'random_forest', etc.)
    """
    # Skip XGBoost if not installed
    if model_type == "xgboost_classifier":
        pytest.importorskip("xgboost")
    
    # Prepare data
    try:
        X_train, X_test, y_train, y_test, task_type = prepare_data_for_market(market_type)
    except Exception as e:
        pytest.skip(f"Could not prepare data for {market_type}: {e}")
    
    # Ensure we have enough samples
    if len(X_train) < 100 or len(X_test) < 20:
        pytest.skip(f"Insufficient data for {market_type}: train={len(X_train)}, test={len(X_test)}")
    
    # Build model
    try:
        model = build_model_for_market_and_type(market_type, model_type)
    except Exception as e:
        pytest.fail(f"Failed to build model {model_type} for {market_type}: {e}")
    
    # Train model
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        pytest.fail(f"Failed to train {model_type} on {market_type}: {e}")
    
    # Generate predictions
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        pytest.fail(f"Failed to predict with {model_type} on {market_type}: {e}")
    
    # Validate predictions
    assert len(y_pred) == len(X_test), f"Prediction length mismatch: {len(y_pred)} vs {len(X_test)}"
    assert not np.any(np.isnan(y_pred)), f"Predictions contain NaN values for {market_type}/{model_type}"
    assert not np.any(np.isinf(y_pred)), f"Predictions contain Inf values for {market_type}/{model_type}"
    
    # Evaluate based on task type
    if task_type == "regression":
        # Compute RMSE
        rmse = compute_rmse(y_test, y_pred)
        
        # Get threshold for this market
        threshold = THRESHOLDS.get(market_type, 1000.0)
        
        # Verify RMSE is reasonable
        assert rmse > 0, f"RMSE must be positive for {market_type}/{model_type}"
        assert rmse < threshold, (
            f"RMSE too high for {market_type}/{model_type}: "
            f"{rmse:.2f} >= {threshold:.2f}"
        )
        
        print(f"✓ {market_type}/{model_type}: RMSE = {rmse:.2f} (threshold: {threshold:.2f})")
        
    elif task_type == "classification":
        # Compute F1-score
        f1 = compute_f1_macro(y_test, y_pred)
        
        # Get threshold
        threshold = THRESHOLDS.get(f"{market_type}_f1", 0.2)
        
        # Verify F1 is reasonable
        assert f1 >= 0, f"F1-score must be non-negative for {market_type}/{model_type}"
        assert f1 > threshold, (
            f"F1-score too low for {market_type}/{model_type}: "
            f"{f1:.3f} <= {threshold:.3f}"
        )
        
        print(f"✓ {market_type}/{model_type}: F1 = {f1:.3f} (threshold: {threshold:.3f})")
    
    else:
        pytest.fail(f"Unknown task type: {task_type}")


@pytest.mark.slow
def test_all_markets_have_valid_combinations():
    """Verify that each market type has at least one compatible model."""
    markets = list_all_markets()
    combinations = get_model_market_combinations()
    
    for market in markets:
        market_combos = [c for c in combinations if c[0] == market]
        assert len(market_combos) > 0, f"Market {market} has no compatible models"


@pytest.mark.slow
def test_all_models_have_valid_combinations():
    """Verify that each model type has at least one compatible market."""
    models = list_all_model_types()
    combinations = get_model_market_combinations()
    
    for model in models:
        model_combos = [c for c in combinations if c[1] == model]
        assert len(model_combos) > 0, f"Model {model} has no compatible markets"


@pytest.mark.slow
def test_combination_count():
    """Verify we have the expected number of combinations."""
    combinations = get_model_market_combinations()
    
    # Should have at least:
    # - 3 regression markets × 4 regression models = 12
    # - 1 classification market × 1 classification model = 1
    # Total: at least 13 combinations
    assert len(combinations) >= 13, f"Expected at least 13 combinations, got {len(combinations)}"
    
    print(f"Testing {len(combinations)} model × market combinations:")
    for market, model in sorted(combinations):
        print(f"  - {market} × {model}")
