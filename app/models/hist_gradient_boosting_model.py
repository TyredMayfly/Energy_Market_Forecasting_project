"""
Histogram-based Gradient Boosting forecasting model.

Uses sklearn's HistGradientBoostingRegressor for efficient gradient boosting
with native support for missing values and categorical features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from app.core.logging import get_logger

logger = get_logger(__name__)


class HistGradientBoostingPriceModel:
    """
    Histogram-based Gradient Boosting model for price forecasting.

    Uses scikit-learn's HistGradientBoostingRegressor, which is optimized for
    speed and memory efficiency while maintaining high accuracy. This model
    uses histogram-based splitting for faster training on large datasets.

    Features:
    - Fast training with histogram-based algorithm
    - Native handling of missing values
    - Efficient memory usage
    - Strong performance on non-linear relationships

    Example usage:
        >>> model = HistGradientBoostingPriceModel(
        ...     learning_rate=0.05,
        ...     max_depth=6,
        ...     max_iter=500
        ... )
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        loss: str = "squared_error",
        learning_rate: float = 0.05,
        max_iter: int = 500,
        max_depth: int = 6,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 50,
        l2_regularization: float = 0.0,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize the histogram-based gradient boosting model.

        Args:
            loss: Loss function to optimize ('squared_error', 'absolute_error', 'poisson')
            learning_rate: Learning rate shrinks contribution of each tree (0.01-0.3)
            max_iter: Maximum number of boosting iterations (trees to build)
            max_depth: Maximum depth of each tree (None for unlimited)
            max_leaf_nodes: Maximum number of leaves for each tree
            min_samples_leaf: Minimum number of samples per leaf
            l2_regularization: L2 regularization parameter (0.0 = no regularization)
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters passed to HistGradientBoostingRegressor
        """
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state

        # Create model instance
        self.model_ = HistGradientBoostingRegressor(
            loss=loss,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_state,
            **kwargs,
        )

        self.feature_names_ = None
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistGradientBoostingPriceModel":
        """
        Fit the histogram-based gradient boosting model.

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
            f"HistGradientBoosting fitted with {len(X)} samples, "
            f"{len(feature_cols)} features, "
            f"{self.max_iter} max iterations, "
            f"learning_rate={self.learning_rate}, "
            f"R² score: {train_score:.4f}"
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate forecasts using the histogram-based gradient boosting model.

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

        logger.debug(f"Generated {len(predictions)} HistGradientBoosting forecasts")

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
            raise ValueError("Model has not been fitted yet")

        X_eval = X[self.feature_names_].values
        return self.model_.score(X_eval, y.values)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances from the trained model.

        Note: HistGradientBoostingRegressor does not provide feature_importances_
        by default. This method uses permutation importance as an alternative,
        but for now returns a placeholder indicating this feature is not available.

        Returns:
            DataFrame with feature names and their importance scores

        Raises:
            ValueError: HistGradientBoostingRegressor does not support feature importances
        """
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet")

        # HistGradientBoostingRegressor doesn't provide feature_importances_
        # You would need to use permutation_importance or similar for feature importance
        raise NotImplementedError(
            "HistGradientBoostingRegressor does not provide built-in feature importances. "
            "Use sklearn.inspection.permutation_importance for feature importance analysis."
        )

    def get_params(self, deep: bool = True) -> dict:
        """Get model parameters (sklearn compatibility)."""
        return {
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "max_depth": self.max_depth,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "HistGradientBoostingPriceModel":
        """Set model parameters (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        # Recreate model with new params
        self.model_ = HistGradientBoostingRegressor(
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )
        self.is_fitted_ = False
        return self
