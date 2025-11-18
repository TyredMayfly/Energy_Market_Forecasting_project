"""
Tests for forecasting models.
"""

import numpy as np
import pandas as pd
import pytest
from app.models.linear_regression_model import LinearRegressionPriceModel
from app.models.persistence_model import PersistencePriceModel
from app.models.random_forest_model import RandomForestPriceModel


class TestPersistenceModel:
    """Test suite for PersistenceModel."""

    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        # Create sample data
        n = 100
        X = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2025-01-01", periods=n, freq="h"),
                "feature1": np.random.randn(n),
            }
        )
        y = pd.Series(np.random.uniform(30, 60, n))

        model = PersistencePriceModel(method="last")
        model.fit(X, y)

        # Predict
        X_test = X.tail(10)
        predictions = model.predict(X_test)

        # Should repeat last value
        assert len(predictions) == 10
        assert np.all(predictions == y.iloc[-1])

    def test_fit_with_empty_data(self):
        """Test that fitting with empty data raises error."""
        model = PersistencePriceModel()

        X = pd.DataFrame()
        y = pd.Series(dtype=float)

        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_predict_before_fit(self):
        """Test that predicting before fitting raises error."""
        model = PersistencePriceModel()

        X = pd.DataFrame({"feature1": [1, 2, 3]})

        with pytest.raises(ValueError):
            model.predict(X)


class TestLinearRegressionModel:
    """Test suite for LinearRegressionModel."""

    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        # Create sample data with clear relationship
        n = 100
        X = pd.DataFrame(
            {
                "lag1": np.random.randn(n),
                "lag24": np.random.randn(n),
                "hour_of_day": np.random.randint(0, 24, n),
            }
        )
        # y somewhat related to features
        y = pd.Series(X["lag1"] * 2 + X["lag24"] + np.random.randn(n) * 0.1)

        model = LinearRegressionPriceModel()
        model.fit(X, y)

        # Predict
        predictions = model.predict(X.head(10))

        assert len(predictions) == 10
        assert model.is_fitted_

    def test_get_feature_importance(self):
        """Test feature importance retrieval."""
        n = 100
        X = pd.DataFrame(
            {
                "lag1": np.random.randn(n),
                "lag24": np.random.randn(n),
            }
        )
        y = pd.Series(np.random.randn(n))

        model = LinearRegressionPriceModel()
        model.fit(X, y)

        importance_df = model.get_feature_importance()

        assert not importance_df.empty
        assert "feature" in importance_df.columns
        assert "coefficient" in importance_df.columns


class TestRandomForestModel:
    """Test suite for RandomForestModel."""

    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        n = 100
        X = pd.DataFrame(
            {
                "lag1": np.random.randn(n),
                "lag24": np.random.randn(n),
                "hour_of_day": np.random.randint(0, 24, n),
            }
        )
        y = pd.Series(np.random.uniform(30, 60, n))

        model = RandomForestPriceModel(n_estimators=10, max_depth=5)
        model.fit(X, y)

        predictions = model.predict(X.head(10))

        assert len(predictions) == 10
        assert model.is_fitted_

    def test_get_feature_importance(self):
        """Test feature importance retrieval."""
        n = 100
        X = pd.DataFrame(
            {
                "lag1": np.random.randn(n),
                "lag24": np.random.randn(n),
            }
        )
        y = pd.Series(np.random.randn(n))

        model = RandomForestPriceModel(n_estimators=10)
        model.fit(X, y)

        importance_df = model.get_feature_importance()

        assert not importance_df.empty
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns

    def test_get_params(self):
        """Test parameter retrieval."""
        model = RandomForestPriceModel(n_estimators=25, max_depth=8)

        params = model.get_params()

        assert params["n_estimators"] == 25
        assert params["max_depth"] == 8
