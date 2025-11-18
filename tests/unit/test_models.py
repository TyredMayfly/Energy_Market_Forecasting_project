"""
Tests for forecasting models.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from app.models.persistence_model import PersistencePriceModel
from app.models.linear_regression_model import LinearRegressionPriceModel
from app.models.random_forest_model import RandomForestPriceModel


@pytest.fixture
def simple_training_data():
    """Create simple training data for model tests."""
    n_samples = 100
    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
    })
    y = pd.Series(50 + 10 * np.random.randn(n_samples))
    return X, y


@pytest.fixture
def simple_test_data():
    """Create simple test data."""
    n_samples = 20
    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
    })
    return X


class TestPersistenceModel:
    """Test suite for PersistenceModel."""

    def test_initialization_default(self):
        """Test default initialization."""
        model = PersistencePriceModel()
        
        assert model.method == "last"
        assert model.last_value_ is None

    def test_initialization_with_method(self):
        """Test initialization with specific method."""
        model = PersistencePriceModel(method="same_hour_yesterday")
        
        assert model.method == "same_hour_yesterday"

    def test_fit_basic(self, simple_training_data):
        """Test basic model fitting."""
        X, y = simple_training_data
        model = PersistencePriceModel()
        
        result = model.fit(X, y)
        
        # Should return self
        assert result is model
        # Should store last value
        assert model.last_value_ is not None

    def test_fit_with_empty_data(self):
        """Test fitting with empty data raises error."""
        X = pd.DataFrame()
        y = pd.Series(dtype=float)
        model = PersistencePriceModel()
        
        with pytest.raises(ValueError, match="empty data"):
            model.fit(X, y)

    def test_fit_stores_last_value(self, simple_training_data):
        """Test that fit stores the last target value."""
        X, y = simple_training_data
        model = PersistencePriceModel()
        
        model.fit(X, y)
        
        assert model.last_value_ == y.iloc[-1]

    def test_predict_basic(self, simple_training_data, simple_test_data):
        """Test basic prediction."""
        X_train, y_train = simple_training_data
        X_test = simple_test_data
        
        model = PersistencePriceModel()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)

    def test_predict_last_method(self, simple_training_data, simple_test_data):
        """Test prediction with 'last' method."""
        X_train, y_train = simple_training_data
        X_test = simple_test_data
        
        model = PersistencePriceModel(method="last")
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        # All predictions should equal last training value
        assert np.all(predictions == y_train.iloc[-1])

    def test_predict_before_fit_raises_error(self, simple_test_data):
        """Test that predict before fit raises error."""
        model = PersistencePriceModel()
        
        with pytest.raises(ValueError, match="not been fitted"):
            model.predict(simple_test_data)

    def test_predict_returns_numpy_array(self, simple_training_data, simple_test_data):
        """Test that predict returns numpy array."""
        X_train, y_train = simple_training_data
        X_test = simple_test_data
        
        model = PersistencePriceModel()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert isinstance(predictions, np.ndarray)

    def test_persistence_with_different_y_values(self):
        """Test persistence model with specific y values."""
        X = pd.DataFrame({'feature_1': [1, 2, 3, 4, 5]})
        y = pd.Series([10, 20, 30, 40, 50])
        
        model = PersistencePriceModel()
        model.fit(X, y)
        
        X_test = pd.DataFrame({'feature_1': [6, 7, 8]})
        predictions = model.predict(X_test)
        
        # Should all be 50 (last value)
        assert np.all(predictions == 50)


class TestLinearRegressionModel:
    """Test suite for LinearRegressionModel."""

    def test_initialization_default(self):
        """Test default initialization."""
        model = LinearRegressionPriceModel()
        
        assert model.fit_intercept is True
        assert model.is_fitted_ is False

    def test_initialization_without_intercept(self):
        """Test initialization without intercept."""
        model = LinearRegressionPriceModel(fit_intercept=False)
        
        assert model.fit_intercept is False

    def test_fit_basic(self, simple_training_data):
        """Test basic model fitting."""
        X, y = simple_training_data
        model = LinearRegressionPriceModel()
        
        result = model.fit(X, y)
        
        # Should return self
        assert result is model
        # Should be marked as fitted
        assert model.is_fitted_ is True

    def test_fit_with_empty_data(self):
        """Test fitting with empty data raises error."""
        X = pd.DataFrame()
        y = pd.Series(dtype=float)
        model = LinearRegressionPriceModel()
        
        with pytest.raises(ValueError, match="empty data"):
            model.fit(X, y)

    def test_fit_stores_feature_names(self, simple_training_data):
        """Test that fit stores feature names."""
        X, y = simple_training_data
        model = LinearRegressionPriceModel()
        
        model.fit(X, y)
        
        assert model.feature_names_ is not None
        assert len(model.feature_names_) == X.shape[1]

    def test_predict_basic(self, simple_training_data, simple_test_data):
        """Test basic prediction."""
        X_train, y_train = simple_training_data
        X_test = simple_test_data
        
        model = LinearRegressionPriceModel()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)

    def test_predict_before_fit_raises_error(self, simple_test_data):
        """Test that predict before fit raises error."""
        model = LinearRegressionPriceModel()
        
        with pytest.raises(ValueError, match="not been fitted"):
            model.predict(simple_test_data)

    def test_predict_shape(self, simple_training_data, simple_test_data):
        """Test prediction output shape."""
        X_train, y_train = simple_training_data
        X_test = simple_test_data
        
        model = LinearRegressionPriceModel()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert predictions.shape == (len(X_test),)

    def test_predict_with_mismatched_features(self, simple_training_data):
        """Test prediction with wrong number of features."""
        X_train, y_train = simple_training_data
        model = LinearRegressionPriceModel()
        model.fit(X_train, y_train)
        
        # Test data with different features
        X_test_wrong = pd.DataFrame({
            'feature_1': [1, 2, 3],
            'feature_2': [4, 5, 6],
        })
        
        with pytest.raises(KeyError):
            model.predict(X_test_wrong)

    def test_linear_relationship(self):
        """Test that model learns linear relationship."""
        # Create data with known linear relationship: y = 2*x1 + 3*x2 + 1
        n = 100
        X = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
        })
        y = pd.Series(2 * X['x1'] + 3 * X['x2'] + 1)
        
        model = LinearRegressionPriceModel()
        model.fit(X, y)
        
        # Predict on training data
        predictions = model.predict(X)
        
        # Should fit very well
        mse = np.mean((predictions - y) ** 2)
        assert mse < 1e-10  # Should be nearly perfect

    def test_get_feature_importance(self, simple_training_data):
        """Test getting feature coefficients."""
        X, y = simple_training_data
        model = LinearRegressionPriceModel()
        model.fit(X, y)
        
        importances = model.get_feature_importance()
        
        assert isinstance(importances, pd.DataFrame)
        assert "feature" in importances.columns
        assert "coefficient" in importances.columns
        assert len(importances) == X.shape[1]


class TestRandomForestModel:
    """Test suite for RandomForestModel."""

    def test_initialization_default(self):
        """Test default initialization."""
        model = RandomForestPriceModel()
        
        assert model.n_estimators > 0
        assert model.max_depth > 0
        assert model.random_state is not None
        assert model.is_fitted_ is False

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        model = RandomForestPriceModel(
            n_estimators=100,
            max_depth=5,
            random_state=123
        )
        
        assert model.n_estimators == 100
        assert model.max_depth == 5
        assert model.random_state == 123

    def test_fit_basic(self, simple_training_data):
        """Test basic model fitting."""
        X, y = simple_training_data
        model = RandomForestPriceModel(n_estimators=10)  # Small for speed
        
        result = model.fit(X, y)
        
        # Should return self
        assert result is model
        # Should be marked as fitted
        assert model.is_fitted_ is True

    def test_fit_with_empty_data(self):
        """Test fitting with empty data raises error."""
        X = pd.DataFrame()
        y = pd.Series(dtype=float)
        model = RandomForestPriceModel()
        
        with pytest.raises(ValueError, match="empty data"):
            model.fit(X, y)

    def test_fit_stores_feature_names(self, simple_training_data):
        """Test that fit stores feature names."""
        X, y = simple_training_data
        model = RandomForestPriceModel(n_estimators=10)
        
        model.fit(X, y)
        
        assert model.feature_names_ is not None
        assert len(model.feature_names_) == X.shape[1]

    def test_predict_basic(self, simple_training_data, simple_test_data):
        """Test basic prediction."""
        X_train, y_train = simple_training_data
        X_test = simple_test_data
        
        model = RandomForestPriceModel(n_estimators=10)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)

    def test_predict_before_fit_raises_error(self, simple_test_data):
        """Test that predict before fit raises error."""
        model = RandomForestPriceModel()
        
        with pytest.raises(ValueError, match="not been fitted"):
            model.predict(simple_test_data)

    def test_predict_shape(self, simple_training_data, simple_test_data):
        """Test prediction output shape."""
        X_train, y_train = simple_training_data
        X_test = simple_test_data
        
        model = RandomForestPriceModel(n_estimators=10)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert predictions.shape == (len(X_test),)

    def test_predict_range(self, simple_training_data, simple_test_data):
        """Test that predictions are in reasonable range."""
        X_train, y_train = simple_training_data
        X_test = simple_test_data
        
        model = RandomForestPriceModel(n_estimators=10)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        # Predictions should be roughly in range of training data
        y_min, y_max = y_train.min(), y_train.max()
        margin = (y_max - y_min) * 0.5
        
        assert predictions.min() >= y_min - margin
        assert predictions.max() <= y_max + margin

    def test_get_feature_importance(self, simple_training_data):
        """Test getting feature importances."""
        X, y = simple_training_data
        model = RandomForestPriceModel(n_estimators=10)
        model.fit(X, y)
        
        importances = model.get_feature_importance()
        
        assert isinstance(importances, pd.DataFrame)
        assert "feature" in importances.columns
        assert "importance" in importances.columns
        assert len(importances) == X.shape[1]
        
        # Importances should sum to ~1.0
        assert np.isclose(importances["importance"].sum(), 1.0)

    def test_nonlinear_relationship(self):
        """Test that model can capture nonlinear relationships."""
        # Create data with nonlinear relationship: y = x^2
        n = 200
        X = pd.DataFrame({
            'x': np.linspace(-5, 5, n),
        })
        y = pd.Series(X['x'] ** 2 + np.random.randn(n) * 0.5)
        
        model = RandomForestPriceModel(n_estimators=50, max_depth=10)
        model.fit(X, y)
        
        # Predict
        predictions = model.predict(X)
        
        # Should fit reasonably well
        mse = np.mean((predictions - y) ** 2)
        naive_mse = np.var(y)  # Variance is MSE of constant prediction
        
        # Model should be better than naive mean prediction
        assert mse < naive_mse * 0.5

    def test_random_state_reproducibility(self, simple_training_data):
        """Test that same random state gives same results."""
        X, y = simple_training_data
        
        model1 = RandomForestPriceModel(n_estimators=10, random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict(X)
        
        model2 = RandomForestPriceModel(n_estimators=10, random_state=42)
        model2.fit(X, y)
        pred2 = model2.predict(X)
        
        # Should be identical
        assert np.allclose(pred1, pred2)


class TestModelComparison:
    """Test suite comparing different models."""

    def test_all_models_have_fit_predict(self):
        """Test that all models have fit and predict methods."""
        models = [
            PersistencePriceModel(),
            LinearRegressionPriceModel(),
            RandomForestPriceModel(n_estimators=10),
        ]
        
        for model in models:
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')
            assert callable(model.fit)
            assert callable(model.predict)

    def test_all_models_work_with_same_data(self, simple_training_data, simple_test_data):
        """Test that all models can fit and predict with same data."""
        X_train, y_train = simple_training_data
        X_test = simple_test_data
        
        models = [
            PersistencePriceModel(),
            LinearRegressionPriceModel(),
            RandomForestPriceModel(n_estimators=10),
        ]
        
        for model in models:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            assert len(predictions) == len(X_test)
            assert isinstance(predictions, np.ndarray)

    def test_models_give_different_predictions(self, simple_training_data, simple_test_data):
        """Test that different models give different predictions."""
        X_train, y_train = simple_training_data
        X_test = simple_test_data
        
        persistence = PersistencePriceModel()
        persistence.fit(X_train, y_train)
        pred_persistence = persistence.predict(X_test)
        
        linear = LinearRegressionPriceModel()
        linear.fit(X_train, y_train)
        pred_linear = linear.predict(X_test)
        
        # Predictions should be different
        assert not np.allclose(pred_persistence, pred_linear)
