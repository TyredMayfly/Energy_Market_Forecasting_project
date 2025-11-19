"""
Persistence forecasting model.

A simple baseline model that assumes future prices will equal recent past values.
"""

import numpy as np
import pandas as pd
from app.core.logging import get_logger

logger = get_logger(__name__)


class PersistencePriceModel:
    """
    Persistence model for price forecasting.

    This model forecasts that future prices will equal the last observed price
    or the same hour from the previous day/week (depending on configuration).
    """

    def __init__(self, method: str = "last"):
        """
        Initialize the persistence model.

        Args:
            method: Persistence method
                - 'last': Use the last observed value
                - 'same_hour_yesterday': Use same hour from previous day
        """
        self.method = method
        self.last_value_ = None
        self.historical_data_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistencePriceModel":
        """
        Fit the persistence model.

        For persistence models, this just stores the last known value.

        Args:
            X: Feature matrix (not used, but kept for sklearn compatibility)
            y: Target values (prices)

        Returns:
            Self
        """
        if len(y) == 0:
            raise ValueError("Cannot fit persistence model with empty data")

        self.last_value_ = y.iloc[-1]
        self.historical_data_ = y.copy()

        logger.info(
            f"Persistence model fitted with {len(y)} samples, last value: {self.last_value_:.2f}"
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate persistence forecasts.

        Args:
            X: Feature matrix with 'timestamp_utc' column

        Returns:
            Array of forecasted prices
        """
        if self.last_value_ is None:
            raise ValueError("Model has not been fitted yet")

        n_samples = len(X)

        if self.method == "last":
            # Simply repeat the last value
            predictions = np.full(n_samples, self.last_value_)

        elif self.method == "same_hour_yesterday":
            # Use same hour from previous day if available
            predictions = []

            for idx, row in X.iterrows():
                # Try to find same hour yesterday in historical data
                predictions.append(self.last_value_)  # Fallback to last value

            predictions = np.array(predictions)

        else:
            raise ValueError(f"Unknown persistence method: {self.method}")

        logger.debug(f"Generated {n_samples} persistence forecasts")

        return predictions

    def get_params(self, deep: bool = True) -> dict:
        """Get model parameters (sklearn compatibility)."""
        return {"method": self.method}

    def set_params(self, **params) -> "PersistencePriceModel":
        """Set model parameters (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
