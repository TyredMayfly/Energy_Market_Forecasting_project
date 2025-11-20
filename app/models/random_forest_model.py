"""
Random forest forecasting model.

Uses sklearn's RandomForestRegressor with conservative parameters for interpretability.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from app.core.config import MODEL_TYPES
from app.core.logging import get_logger

logger = get_logger(__name__)


class RandomForestPriceModel:
    """
    Random forest model for price forecasting.

    Uses an ensemble of decision trees with limited depth to maintain
    interpretability while capturing non-linear relationships.
    """

    def __init__(
        self,
        n_estimators: int = None,
        max_depth: int = None,
        min_samples_split: int = None,
        min_samples_leaf: int = None,
        max_features: str = None,
        bootstrap: bool = None,
        random_state: int = None,
        **kwargs,  # Accept additional kwargs for compatibility
    ):
        """
        Initialize the random forest model.

        Args:
            n_estimators: Number of trees (default from config)
            max_depth: Maximum tree depth (default from config)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples per leaf
            max_features: Number of features to consider for splits
            bootstrap: Whether to use bootstrap samples
            random_state: Random seed (default from config)
            **kwargs: Additional parameters for compatibility
        """
        # Use defaults from config if not specified
        config = MODEL_TYPES.get("random_forest", {})

        self.n_estimators = n_estimators or config.get("n_estimators", 50)
        self.max_depth = max_depth or config.get("max_depth", 10)
        self.min_samples_split = min_samples_split or config.get("min_samples_split", 2)
        self.min_samples_leaf = min_samples_leaf or config.get("min_samples_leaf", 1)
        self.max_features = max_features or config.get("max_features", "sqrt")
        self.bootstrap = bootstrap if bootstrap is not None else config.get("bootstrap", True)
        self.random_state = random_state or config.get("random_state", 42)

        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
        )

        self.feature_names_ = None
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestPriceModel":
        """
        Fit the random forest model.

        Args:
            X: Feature matrix
            y: Target values (prices)

        Returns:
            Self
        """
        if len(X) == 0:
            raise ValueError("Cannot fit model with empty data")

        # Store feature names (excluding timestamp if present)
        feature_cols = [col for col in X.columns if col != "timestamp_utc"]
        self.feature_names_ = feature_cols

        # Extract feature matrix
        X_train = X[feature_cols].values

        # Fit model
        self.model_.fit(X_train, y.values)
        self.is_fitted_ = True

        # Calculate training score
        train_score = self.model_.score(X_train, y.values)

        logger.info(
            f"Random forest fitted with {len(X)} samples, "
            f"{len(feature_cols)} features, "
            f"{self.n_estimators} trees, "
            f"R² score: {train_score:.4f}"
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate forecasts using the random forest model.

        Args:
            X: Feature matrix

        Returns:
            Array of forecasted prices
        """
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet")

        # Extract features
        X_pred = X[self.feature_names_].values

        # Generate predictions
        predictions = self.model_.predict(X_pred)

        logger.debug(f"Generated {len(predictions)} random forest forecasts")

        return predictions

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate R² score on the given data.

        Args:
            X: Feature matrix
            y: True target values

        Returns:
            R² score
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first")

        X_eval = X[self.feature_names_].values
        return self.model_.score(X_eval, y.values)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances from the random forest.

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first")

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names_,
                "importance": self.model_.feature_importances_,
            }
        )

        importance_df = importance_df.sort_values("importance", ascending=False)

        return importance_df

    def get_params(self, deep: bool = True) -> dict:
        """Get model parameters (sklearn compatibility)."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "RandomForestPriceModel":
        """Set model parameters (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        # Recreate model with new params
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.is_fitted_ = False
        return self
