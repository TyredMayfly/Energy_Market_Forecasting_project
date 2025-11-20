"""
Hyperparameter search service for model tuning.

Provides functions to:
- Define parameter grids for each model type
- Run hyperparameter search using cross-validation
- Save and load best hyperparameters
- Track search results and metadata
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.stats import randint, uniform

from app.core.config import MARKET_TYPES, MODEL_TYPES, settings
from app.core.logging import get_logger
from app.models.linear_regression_model import LinearRegressionPriceModel
from app.models.random_forest_model import RandomForestPriceModel
from app.models.xgboost_classifier import RegulationStateXGBModel
from app.models.hist_gradient_boosting_model import HistGradientBoostingPriceModel
from app.services.feature_engineering import build_features_and_target

logger = get_logger(__name__)


def get_param_grid(model_type: str) -> Dict[str, Union[List, object]]:
    """
    Get hyperparameter grid for a specific model type.

    Args:
        model_type: Type of model ('linear_regression', 'random_forest', 'xgboost_classifier')

    Returns:
        Dictionary of parameter distributions for RandomizedSearchCV

    Raises:
        ValueError: If model_type is unknown or doesn't support hyperparameter tuning
    """
    if model_type == "persistence":
        raise ValueError("Persistence model does not support hyperparameter tuning")

    if model_type == "linear_regression":
        # Linear regression has very few hyperparameters
        return {
            "fit_intercept": [True, False],
        }

    elif model_type == "random_forest":
        # Random forest has many tunable hyperparameters
        return {
            "n_estimators": randint(50, 300),  # Number of trees
            "max_depth": randint(5, 30),  # Maximum depth of trees
            "min_samples_split": randint(2, 20),  # Minimum samples to split a node
            "min_samples_leaf": randint(1, 10),  # Minimum samples at leaf node
            "max_features": ["sqrt", "log2", None],  # Features to consider for split
            "bootstrap": [True, False],  # Whether to use bootstrap samples
        }

    elif model_type == "xgboost_classifier":
        # XGBoost classifier hyperparameters
        return {
            "n_estimators": randint(50, 300),  # Number of boosting rounds
            "max_depth": randint(3, 12),  # Maximum tree depth
            "learning_rate": uniform(0.01, 0.3),  # Step size shrinkage
            "subsample": uniform(0.6, 0.4),  # Subsample ratio (0.6 to 1.0)
            "colsample_bytree": uniform(0.6, 0.4),  # Column subsample ratio (0.6 to 1.0)
            "gamma": uniform(0, 0.5),  # Minimum loss reduction for split
            "reg_alpha": uniform(0, 1),  # L1 regularization
            "reg_lambda": uniform(0.5, 2),  # L2 regularization (0.5 to 2.5)
        }

    elif model_type == "hist_gradient_boosting":
        # HistGradientBoosting regressor hyperparameters
        return {
            "learning_rate": [0.03, 0.05, 0.1],  # Learning rate
            "max_depth": [3, 5, 7, None],  # Maximum tree depth (None = unlimited)
            "max_leaf_nodes": [15, 31, 63],  # Maximum number of leaves
            "min_samples_leaf": [20, 50, 100],  # Minimum samples per leaf
            "max_iter": [300, 500, 800],  # Number of boosting iterations
            "l2_regularization": [0.0, 0.1, 1.0],  # L2 regularization
        }

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_hyperparameter_search(
    model_type: str,
    market_type: str,
    n_iter: int = 50,
    cv_splits: int = 5,
    test_size_ratio: float = 0.2,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1,
) -> Tuple[Dict, float, Dict]:
    """
    Run hyperparameter search for a model on a specific market.

    Args:
        model_type: Type of model to tune
        market_type: Type of market to train on
        n_iter: Number of parameter settings to sample
        cv_splits: Number of cross-validation splits
        test_size_ratio: Ratio of data to use for testing (e.g., 0.2 = 20%)
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 = all cores)
        verbose: Verbosity level for search

    Returns:
        Tuple of (best_params, best_score, metadata)

    Raises:
        ValueError: If model or market type is invalid, or incompatible
    """
    logger.info(f"Starting hyperparameter search for {model_type} on {market_type}")

    # Validate market type
    if market_type not in MARKET_TYPES:
        raise ValueError(f"Unknown market type: {market_type}")

    # Validate model-market compatibility
    market_config = MARKET_TYPES[market_type]
    target_type = market_config.get("target_type", "regression")

    regression_models = {"linear_regression", "random_forest", "hist_gradient_boosting"}
    classification_models = {"xgboost_classifier"}

    if target_type == "classification" and model_type not in classification_models:
        raise ValueError(
            f"Model '{model_type}' cannot be used for classification target '{market_type}'"
        )
    if target_type == "regression" and model_type not in regression_models:
        raise ValueError(
            f"Model '{model_type}' cannot be used for regression target '{market_type}'"
        )

    # Load data and build features
    logger.info(f"Loading data for {market_type}")
    X, y = build_features_and_target(market_type=market_type, include_weather=True)

    if X.empty or y.empty:
        raise ValueError(f"No data available for {market_type}")

    # Remove timestamp column if present
    feature_cols = [col for col in X.columns if col != "timestamp_utc"]
    X_features = X[feature_cols]

    logger.info(f"Loaded {len(X_features)} samples with {len(feature_cols)} features")

    # Split data into train and test sets (chronological split for time series)
    split_idx = int(len(X_features) * (1 - test_size_ratio))
    X_train, X_test = X_features.iloc[:split_idx], X_features.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Get parameter grid
    param_grid = get_param_grid(model_type)

    # Create base model instance
    if model_type == "linear_regression":
        base_model = LinearRegressionPriceModel()
    elif model_type == "random_forest":
        base_model = RandomForestPriceModel()
    elif model_type == "xgboost_classifier":
        base_model = RegulationStateXGBModel(feature_columns=feature_cols)
    elif model_type == "hist_gradient_boosting":
        base_model = HistGradientBoostingPriceModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create time series cross-validator
    # Use TimeSeriesSplit to respect temporal ordering
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    # Determine scoring metric
    if target_type == "classification":
        scoring = "accuracy"
    else:
        scoring = "neg_mean_squared_error"

    logger.info(f"Running randomized search with {n_iter} iterations, {cv_splits} CV splits")

    # Run randomized search
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=tscv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
        return_train_score=True,
    )

    # Fit search on training data
    search.fit(X_train, y_train)

    # Get best parameters and score
    best_params = search.best_params_
    best_cv_score = search.best_score_

    # Evaluate on test set
    best_model = search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    
    # Calculate RMSE for regression models
    test_rmse = None
    if target_type == "regression":
        y_pred = best_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    logger.info(f"Best CV score ({scoring}): {best_cv_score:.4f}")
    logger.info(f"Test score: {test_score:.4f}")
    if test_rmse is not None:
        logger.info(f"Test RMSE: {test_rmse:.4f}")
    logger.info(f"Best parameters: {best_params}")

    # Compile metadata
    metadata = {
        "model_type": model_type,
        "market_type": market_type,
        "search_timestamp": datetime.utcnow().isoformat(),
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "n_iter": n_iter,
        "cv_splits": cv_splits,
        "test_size_ratio": test_size_ratio,
        "random_state": random_state,
        "scoring": scoring,
        "best_cv_score": float(best_cv_score),
        "test_score": float(test_score),
        "test_rmse": float(test_rmse) if test_rmse is not None else None,
        "best_params": {k: _serialize_param(v) for k, v in best_params.items()},
        "cv_results": {
            "mean_test_scores": search.cv_results_["mean_test_score"].tolist(),
            "std_test_scores": search.cv_results_["std_test_score"].tolist(),
            "params": [
                {k: _serialize_param(v) for k, v in p.items()}
                for p in search.cv_results_["params"]
            ],
        },
    }

    return best_params, best_cv_score, metadata


def _serialize_param(value):
    """Convert parameter value to JSON-serializable format."""
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif value is None:
        return None
    else:
        return value


def save_search_results(
    market_type: str,
    model_type: str,
    best_params: Dict,
    best_score: float,
    metadata: Dict,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save hyperparameter search results to disk.

    Args:
        market_type: Type of market
        model_type: Type of model
        best_params: Best hyperparameters found
        best_score: Best cross-validation score
        metadata: Search metadata (samples, features, etc.)
        output_dir: Directory to save results (default: hyperparameters/)

    Returns:
        Path to saved results file
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "hyperparameters"

    # Create directory structure
    market_dir = output_dir / market_type
    market_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with timestamp
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_type}_{timestamp}.json"
    filepath = market_dir / filename

    # Also save a "latest" file for easy access
    latest_filepath = market_dir / f"{model_type}_latest.json"

    # Prepare output data
    output_data = {
        "best_params": best_params,
        "best_score": best_score,
        "metadata": metadata,
    }

    # Save both timestamped and latest versions
    with open(filepath, "w") as f:
        json.dump(output_data, f, indent=2)

    with open(latest_filepath, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved hyperparameter search results to {filepath}")
    logger.info(f"Also saved to {latest_filepath}")

    return filepath


def load_best_params(
    market_type: str,
    model_type: str,
    search_dir: Optional[Path] = None,
) -> Optional[Dict]:
    """
    Load best hyperparameters for a model-market combination.

    Args:
        market_type: Type of market
        model_type: Type of model
        search_dir: Directory containing search results (default: hyperparameters/)

    Returns:
        Dictionary of best hyperparameters, or None if not found
    """
    if search_dir is None:
        search_dir = Path(__file__).parent.parent.parent / "hyperparameters"

    # Try to load the "latest" file
    latest_filepath = search_dir / market_type / f"{model_type}_latest.json"

    if not latest_filepath.exists():
        logger.debug(f"No tuned hyperparameters found at {latest_filepath}")
        return None

    try:
        with open(latest_filepath, "r") as f:
            data = json.load(f)

        best_params = data.get("best_params", {})
        logger.info(f"Loaded tuned hyperparameters for {model_type} on {market_type}")
        return best_params

    except Exception as e:
        logger.error(f"Error loading hyperparameters from {latest_filepath}: {e}")
        return None


def list_available_tuned_models(
    search_dir: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """
    List all available tuned model-market combinations.

    Args:
        search_dir: Directory containing search results (default: hyperparameters/)

    Returns:
        List of dictionaries with 'market_type', 'model_type', and 'filepath'
    """
    if search_dir is None:
        search_dir = Path(__file__).parent.parent.parent / "hyperparameters"

    if not search_dir.exists():
        return []

    available = []

    # Scan each market directory
    for market_dir in search_dir.iterdir():
        if not market_dir.is_dir():
            continue

        market_type = market_dir.name

        # Look for "latest" files
        for filepath in market_dir.glob("*_latest.json"):
            model_type = filepath.stem.replace("_latest", "")
            available.append(
                {
                    "market_type": market_type,
                    "model_type": model_type,
                    "filepath": str(filepath),
                }
            )

    return available


def get_model_market_combinations() -> List[Tuple[str, str]]:
    """
    Get all valid model × market combinations from configuration.

    Returns:
        List of (market_type, model_type) tuples
    """
    combinations = []

    for market_type, market_config in MARKET_TYPES.items():
        target_type = market_config.get("target_type", "regression")

        # Skip persistence model (no hyperparameters to tune)
        for model_type in MODEL_TYPES.keys():
            if model_type == "persistence":
                continue

            # Check compatibility
            if target_type == "classification":
                if model_type == "xgboost_classifier":
                    combinations.append((market_type, model_type))
            else:  # regression
                if model_type in {"linear_regression", "random_forest", "hist_gradient_boosting"}:
                    combinations.append((market_type, model_type))

    return combinations


def save_search_results_with_trials(
    market_type: str,
    model_type: str,
    best_params: Dict,
    best_score: float,
    metadata: Dict,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    Save hyperparameter search results including per-trial data.

    Args:
        market_type: Type of market
        model_type: Type of model
        best_params: Best hyperparameters found
        best_score: Best cross-validation score
        metadata: Search metadata including trials
        output_dir: Directory to save results (default: hyperparameters/)

    Returns:
        Tuple of (best_json_path, trials_csv_path)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "hyperparameters"

    # Create directory structure
    market_dir = output_dir / market_type
    market_dir.mkdir(parents=True, exist_ok=True)

    # Save best parameters JSON (reuse existing function)
    best_json_path = save_search_results(
        market_type=market_type,
        model_type=model_type,
        best_params=best_params,
        best_score=best_score,
        metadata=metadata,
        output_dir=output_dir,
    )

    # Save per-trial CSV
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    trials_csv_path = market_dir / f"{model_type}_trials_{timestamp}.csv"
    latest_trials_csv_path = market_dir / f"{model_type}_trials_latest.csv"

    # Extract trial-level data from cv_results
    cv_results = metadata.get("cv_results", {})
    params_list = cv_results.get("params", [])
    mean_test_scores = cv_results.get("mean_test_scores", [])
    std_test_scores = cv_results.get("std_test_scores", [])

    if params_list and mean_test_scores:
        # Create DataFrame with trial data
        import pandas as pd

        trials_data = []
        for idx, (params, mean_score, std_score) in enumerate(
            zip(params_list, mean_test_scores, std_test_scores)
        ):
            trial_row = {
                "trial_id": idx,
                "mean_cv_score": mean_score,
                "std_cv_score": std_score,
                **params,
            }
            trials_data.append(trial_row)

        trials_df = pd.DataFrame(trials_data)

        # Save both timestamped and latest versions
        trials_df.to_csv(trials_csv_path, index=False)
        trials_df.to_csv(latest_trials_csv_path, index=False)

        logger.info(f"Saved trial data to {trials_csv_path}")
        logger.info(f"Also saved to {latest_trials_csv_path}")
    else:
        logger.warning(f"No trial data available for {model_type} × {market_type}")
        trials_csv_path = None

    return best_json_path, trials_csv_path
