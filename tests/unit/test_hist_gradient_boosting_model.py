"""
Unit tests for HistGradientBoostingPriceModel.

Tests cover:
- Model initialization with default and custom parameters
- Fitting and prediction
- Feature importance extraction
- Error handling for invalid inputs
"""

import numpy as np
import pandas as pd
import pytest

from app.models.hist_gradient_boosting_model import HistGradientBoostingPriceModel


class TestHistGradientBoostingPriceModel:
    """Tests for HistGradientBoostingPriceModel."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training and test data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 5

        # Create features
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )

        # Create target with some relationship to features
        y = pd.Series(
            X["feature_0"] * 2.5 + X["feature_1"] * -1.3 + np.random.randn(n_samples) * 0.1,
            name="target",
        )

        # Split into train and test
        split_idx = 150
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        return X_train, X_test, y_train, y_test

    def test_initialization_with_defaults(self):
        """Test model initialization with default parameters."""
        model = HistGradientBoostingPriceModel()

        assert model.loss == "squared_error"
        assert model.learning_rate == 0.05
        assert model.max_iter == 500
        assert model.max_depth == 6
        assert model.max_leaf_nodes == 31
        assert model.min_samples_leaf == 50
        assert model.l2_regularization == 0.0
        assert model.random_state == 42
        assert not model.is_fitted_

    def test_initialization_with_custom_params(self):
        """Test model initialization with custom parameters."""
        model = HistGradientBoostingPriceModel(
            learning_rate=0.1,
            max_iter=300,
            max_depth=5,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=0.5,
        )

        assert model.learning_rate == 0.1
        assert model.max_iter == 300
        assert model.max_depth == 5
        assert model.max_leaf_nodes == 15
        assert model.min_samples_leaf == 100
        assert model.l2_regularization == 0.5

    def test_fit_with_valid_data(self, sample_data):
        """Test fitting model with valid data."""
        X_train, _, y_train, _ = sample_data

        model = HistGradientBoostingPriceModel(max_iter=100)  # Reduced for speed
        model.fit(X_train, y_train)

        assert model.is_fitted_
        assert model.feature_names_ == list(X_train.columns)
        assert len(model.feature_names_) == X_train.shape[1]

    def test_fit_with_empty_data(self):
        """Test that fitting with empty data raises ValueError."""
        X = pd.DataFrame()
        y = pd.Series(dtype=float)

        model = HistGradientBoostingPriceModel()

        with pytest.raises(ValueError, match="Cannot fit model with empty data"):
            model.fit(X, y)

    def test_predict_after_fit(self, sample_data):
        """Test prediction after fitting."""
        X_train, X_test, y_train, _ = sample_data

        model = HistGradientBoostingPriceModel(max_iter=100)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
        assert predictions.dtype == np.float64

    def test_predict_before_fit_raises_error(self, sample_data):
        """Test that prediction before fitting raises ValueError."""
        _, X_test, _, _ = sample_data

        model = HistGradientBoostingPriceModel()

        with pytest.raises(ValueError, match="Model has not been fitted yet"):
            model.predict(X_test)

    def test_score_method(self, sample_data):
        """Test the score method returns R² score."""
        X_train, X_test, y_train, y_test = sample_data

        model = HistGradientBoostingPriceModel(max_iter=100)
        model.fit(X_train, y_train)

        # Get R² score on test data
        score = model.score(X_test, y_test)

        assert isinstance(score, float)
        # Score should be between -inf and 1, but for this synthetic data should be positive
        assert score > -1  # Very loose bound
        assert score <= 1

    def test_score_before_fit_raises_error(self, sample_data):
        """Test that score before fitting raises ValueError."""
        _, X_test, _, y_test = sample_data

        model = HistGradientBoostingPriceModel()

        with pytest.raises(ValueError, match="Model has not been fitted yet"):
            model.score(X_test, y_test)

    def test_get_feature_importance(self, sample_data):
        """Test that feature importance raises NotImplementedError for HistGradientBoosting."""
        X_train, _, y_train, _ = sample_data

        model = HistGradientBoostingPriceModel(max_iter=100)
        model.fit(X_train, y_train)

        # HistGradientBoostingRegressor doesn't provide built-in feature importances
        with pytest.raises(
            NotImplementedError,
            match="does not provide built-in feature importances",
        ):
            model.get_feature_importance()

    def test_get_feature_importance_before_fit_raises_error(self):
        """Test that feature importance before fitting raises ValueError."""
        model = HistGradientBoostingPriceModel()

        with pytest.raises(ValueError, match="Model has not been fitted yet"):
            model.get_feature_importance()

    def test_model_with_timestamp_column(self, sample_data):
        """Test that model correctly handles and excludes timestamp column."""
        X_train, X_test, y_train, _ = sample_data

        # Add timestamp column
        X_train_with_ts = X_train.copy()
        X_train_with_ts["timestamp_utc"] = pd.date_range("2025-01-01", periods=len(X_train), freq="15min")

        X_test_with_ts = X_test.copy()
        X_test_with_ts["timestamp_utc"] = pd.date_range(
            "2025-01-01",
            periods=len(X_test),
            freq="15min",
        )

        model = HistGradientBoostingPriceModel(max_iter=100)
        model.fit(X_train_with_ts, y_train)

        # Feature names should exclude timestamp
        assert "timestamp_utc" not in model.feature_names_
        assert len(model.feature_names_) == X_train.shape[1]

        # Should be able to predict with timestamp in X_test
        predictions = model.predict(X_test_with_ts)
        assert len(predictions) == len(X_test)

    def test_model_performance_on_synthetic_data(self, sample_data):
        """Test that model achieves reasonable performance on synthetic data."""
        X_train, X_test, y_train, y_test = sample_data

        model = HistGradientBoostingPriceModel(max_iter=100, learning_rate=0.1)
        model.fit(X_train, y_train)

        # Get predictions
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        # Model should fit training data well
        assert train_score > 0.8

        # Model should generalize reasonably (not perfect due to noise)
        assert test_score > 0.5

    def test_model_with_different_loss_functions(self, sample_data):
        """Test model with different loss functions."""
        X_train, X_test, y_train, _ = sample_data

        # Test squared error (default)
        model_se = HistGradientBoostingPriceModel(loss="squared_error", max_iter=50)
        model_se.fit(X_train, y_train)
        pred_se = model_se.predict(X_test)

        # Test absolute error
        model_ae = HistGradientBoostingPriceModel(loss="absolute_error", max_iter=50)
        model_ae.fit(X_train, y_train)
        pred_ae = model_ae.predict(X_test)

        # Both should produce predictions
        assert len(pred_se) == len(X_test)
        assert len(pred_ae) == len(X_test)

        # Predictions may differ slightly due to different loss functions
        assert not np.allclose(pred_se, pred_ae, rtol=0.01)

    def test_model_with_max_depth_none(self, sample_data):
        """Test model with unlimited depth (max_depth=None)."""
        X_train, X_test, y_train, _ = sample_data

        model = HistGradientBoostingPriceModel(max_depth=None, max_iter=50)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_model_reproducibility_with_random_state(self, sample_data):
        """Test that models with same random_state produce identical results."""
        X_train, X_test, y_train, _ = sample_data

        # Train two models with same random state
        model1 = HistGradientBoostingPriceModel(random_state=42, max_iter=50)
        model1.fit(X_train, y_train)
        pred1 = model1.predict(X_test)

        model2 = HistGradientBoostingPriceModel(random_state=42, max_iter=50)
        model2.fit(X_train, y_train)
        pred2 = model2.predict(X_test)

        # Predictions should be identical
        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_model_with_kwargs(self, sample_data):
        """Test that model accepts additional kwargs for HistGradientBoostingRegressor."""
        X_train, _, y_train, _ = sample_data

        # Pass additional valid sklearn parameters
        model = HistGradientBoostingPriceModel(
            max_iter=50,
            early_stopping=False,  # Additional kwarg
            validation_fraction=0.1,  # Additional kwarg
        )

        # Should fit without errors
        model.fit(X_train, y_train)
        assert model.is_fitted_
