"""
Linear regression forecasting model.

Uses sklearn's LinearRegression with lag features, time features, and weather data.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from app.core.logging import get_logger

logger = get_logger(__name__)


class LinearRegressionPriceModel:
    """
    Linear regression model for price forecasting.

    Uses multiple features including:
    - Lagged price values
    - Time-of-day and day-of-week indicators
    - Weather variables (temperature, wind, radiation)
    """

    def __init__(self, fit_intercept: bool = True):
        """
        Initialize the linear regression model.

        Args:
            fit_intercept: Whether to calculate the intercept
        """
        self.fit_intercept = fit_intercept
        self.model_ = LinearRegression(fit_intercept=fit_intercept)
        self.feature_names_ = None
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRegressionPriceModel":
        """
        Fit the linear regression model.

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

        logger.info(
            f"Linear regression fitted with {len(X)} samples, "
            f"{len(feature_cols)} features, "
            f"R² score: {self.model_.score(X_train, y.values):.4f}"
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate forecasts using the linear regression model.

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

        logger.debug(f"Generated {len(predictions)} linear regression forecasts")

        return predictions

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature coefficients as a measure of importance.

        Returns:
            DataFrame with feature names and coefficients
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first")

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names_,
                "coefficient": self.model_.coef_,
            }
        )

        importance_df["abs_coefficient"] = importance_df["coefficient"].abs()
        importance_df = importance_df.sort_values("abs_coefficient", ascending=False)

        return importance_df

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate R² score on given data.

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

    def get_params(self, deep: bool = True) -> dict:
        """Get model parameters (sklearn compatibility)."""
        return {"fit_intercept": self.fit_intercept}

    def set_params(self, **params) -> "LinearRegressionPriceModel":
        """Set model parameters (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
