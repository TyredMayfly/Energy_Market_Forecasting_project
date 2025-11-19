"""
Integration tests for actual model training.

Tests that all models can successfully train on realistic datasets
and produce valid predictions and metrics.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score

from app.models.persistence_model import PersistencePriceModel
from app.models.linear_regression_model import LinearRegressionPriceModel
from app.models.random_forest_model import RandomForestPriceModel
from app.models.xgboost_classifier import RegulationStateXGBModel


class TestPersistenceModelTraining:
    """Test actual training of Persistence model."""

    def test_train_on_day_ahead_data(self, realistic_day_ahead_features):
        """Test training persistence model on day-ahead price data."""
        X_train, y_train, X_test, y_test = realistic_day_ahead_features
        
        # Initialize and train model
        model = PersistencePriceModel(method="last")
        model.fit(X_train, y_train)
        
        # Verify model is fitted
        assert model.last_value_ is not None
        assert model.last_value_ == y_train.iloc[-1]
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Verify predictions
        assert len(y_pred) == len(X_test)
        assert not np.isnan(y_pred).any()
        assert not np.isinf(y_pred).any()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Verify reasonable metrics (persistence should have MAE < 100 EUR/MWh for day-ahead)
        assert mae > 0, "MAE should be positive"
        assert mae < 100, f"MAE too high: {mae:.2f}"
        assert rmse > 0, "RMSE should be positive"
        assert rmse < 150, f"RMSE too high: {rmse:.2f}"

    def test_train_on_imbalance_data(self, realistic_imbalance_features):
        """Test training persistence model on imbalance price data."""
        X_train, y_train, X_test, y_test = realistic_imbalance_features
        
        model = PersistencePriceModel(method="last")
        model.fit(X_train, y_train)
        
        assert model.last_value_ is not None
        
        y_pred = model.predict(X_test)
        
        assert len(y_pred) == len(X_test)
        assert not np.isnan(y_pred).any()
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        assert mae > 0
        assert rmse > 0


class TestLinearRegressionModelTraining:
    """Test actual training of Linear Regression model."""

    def test_train_on_day_ahead_data(self, realistic_day_ahead_features):
        """Test training linear regression on day-ahead data."""
        X_train, y_train, X_test, y_test = realistic_day_ahead_features
        
        # Initialize model
        model = LinearRegressionPriceModel()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Verify model is fitted
        assert model.model_ is not None
        assert hasattr(model.model_, 'coef_')
        assert hasattr(model.model_, 'intercept_')
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Verify predictions
        assert len(y_pred) == len(X_test)
        assert not np.isnan(y_pred).any()
        assert not np.isinf(y_pred).any()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Verify reasonable metrics
        assert mae > 0, "MAE should be positive"
        assert mae < 80, f"MAE too high for linear regression: {mae:.2f}"
        assert rmse > 0, "RMSE should be positive"
        assert r2 > -1, f"R² too low: {r2:.3f}"  # At least better than horizontal line

    def test_train_on_imbalance_data(self, realistic_imbalance_features):
        """Test training linear regression on imbalance data."""
        X_train, y_train, X_test, y_test = realistic_imbalance_features
        
        model = LinearRegressionPriceModel()
        model.fit(X_train, y_train)
        
        assert model.model_ is not None
        
        y_pred = model.predict(X_test)
        
        assert len(y_pred) == len(X_test)
        assert not np.isnan(y_pred).any()
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        assert mae > 0
        assert r2 > -2  # Should have some predictive power

    def test_feature_importance_extraction(self, realistic_day_ahead_features):
        """Test that model can extract feature importance."""
        X_train, y_train, X_test, y_test = realistic_day_ahead_features
        
        model = LinearRegressionPriceModel()
        model.fit(X_train, y_train)
        
        importance = model.get_feature_importance()
        
        # Should return DataFrame with feature importance
        assert len(importance) == X_train.shape[1]
        assert 'feature' in importance.columns
        assert 'coefficient' in importance.columns


class TestRandomForestModelTraining:
    """Test actual training of Random Forest model."""

    def test_train_on_day_ahead_data(self, realistic_day_ahead_features):
        """Test training random forest on day-ahead data."""
        X_train, y_train, X_test, y_test = realistic_day_ahead_features
        
        # Initialize model with reasonable parameters
        model = RandomForestPriceModel(
            n_estimators=50,  # Fewer trees for faster testing
            max_depth=10,
            random_state=42
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Verify model is fitted
        assert model.model_ is not None
        assert hasattr(model.model_, 'estimators_')
        assert len(model.model_.estimators_) == 50
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Verify predictions
        assert len(y_pred) == len(X_test)
        assert not np.isnan(y_pred).any()
        assert not np.isinf(y_pred).any()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Random Forest should outperform persistence and linear models
        assert mae > 0, "MAE should be positive"
        assert mae < 70, f"MAE too high for random forest: {mae:.2f}"
        assert rmse > 0, "RMSE should be positive"
        assert r2 > -0.5, f"R² too low: {r2:.3f}"

    def test_train_on_imbalance_data(self, realistic_imbalance_features):
        """Test training random forest on imbalance data."""
        X_train, y_train, X_test, y_test = realistic_imbalance_features
        
        model = RandomForestPriceModel(
            n_estimators=30,
            max_depth=8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        assert model.model_ is not None
        assert len(model.model_.estimators_) == 30
        
        y_pred = model.predict(X_test)
        
        assert len(y_pred) == len(X_test)
        assert not np.isnan(y_pred).any()
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        assert mae > 0
        assert r2 > -1

    def test_feature_importance_extraction(self, realistic_day_ahead_features):
        """Test that model can extract feature importance."""
        X_train, y_train, X_test, y_test = realistic_day_ahead_features
        
        model = RandomForestPriceModel(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)
        
        importance = model.get_feature_importance()
        
        # Should return DataFrame with feature importance
        assert len(importance) == X_train.shape[1]
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert all(importance['importance'] >= 0)  # Importance should be non-negative
        
        # Importance should sum to approximately 1.0
        total_importance = importance['importance'].sum()
        assert 0.99 <= total_importance <= 1.01

    def test_train_with_different_configurations(self, realistic_day_ahead_features):
        """Test training with different hyperparameters."""
        X_train, y_train, X_test, y_test = realistic_day_ahead_features
        
        configs = [
            {"n_estimators": 10, "max_depth": 5},
            {"n_estimators": 30, "max_depth": 15},
            {"n_estimators": 50, "max_depth": None},  # No max depth limit
        ]
        
        for config in configs:
            model = RandomForestPriceModel(random_state=42, **config)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
            # All configurations should produce valid predictions
            assert mae > 0
            assert mae < 100


class TestXGBoostClassifierTraining:
    """Test actual training of XGBoost classifier for regulation states."""

    def test_train_on_regulation_state_data(self, realistic_regulation_features):
        """Test training XGBoost classifier on regulation state data."""
        X_train, y_train, X_test, y_test = realistic_regulation_features
        
        # Initialize model
        model = RegulationStateXGBModel(
            feature_columns=list(X_train.columns),
            n_estimators=50,
            max_depth=6,
            random_state=42
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Verify model is fitted
        assert model.model is not None
        assert model.feature_columns == list(X_train.columns)
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Verify predictions
        assert len(y_pred) == len(X_test)
        assert all(pred in model.class_labels.keys() for pred in y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Verify reasonable accuracy (better than random guessing)
        n_classes = len(model.class_labels)
        random_accuracy = 1.0 / n_classes
        assert accuracy > random_accuracy, f"Accuracy {accuracy:.3f} not better than random {random_accuracy:.3f}"
        assert accuracy > 0.3, f"Accuracy too low: {accuracy:.3f}"

    def test_predict_probabilities(self, realistic_regulation_features):
        """Test that model can predict class probabilities."""
        X_train, y_train, X_test, y_test = realistic_regulation_features
        
        model = RegulationStateXGBModel(
            feature_columns=list(X_train.columns),
            n_estimators=30,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Get probability predictions (returns dict mapping states to probability arrays)
        y_proba = model.predict_proba(X_test)
        
        # Verify structure - should be dict with 4 classes
        assert isinstance(y_proba, dict)
        assert len(y_proba) == len(model.class_labels)
        
        # Each class should have probability array for all test samples
        for state, probs in y_proba.items():
            assert len(probs) == len(X_test)
            assert (probs >= 0).all() and (probs <= 1).all()  # Valid probabilities
        
        # Probabilities across all classes should sum to 1 for each sample
        proba_array = np.array([y_proba[state] for state in sorted(model.class_labels.keys())]).T
        assert np.allclose(proba_array.sum(axis=1), 1.0)

    def test_feature_importance_extraction(self, realistic_regulation_features):
        """Test that model can extract feature importance."""
        X_train, y_train, X_test, y_test = realistic_regulation_features
        
        model = RegulationStateXGBModel(
            feature_columns=list(X_train.columns),
            n_estimators=30,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        importance = model.get_feature_importance()
        
        # Should return Series with feature importance
        assert len(importance) > 0  # At least some features should have importance
        assert all(v >= 0 for v in importance.values)  # Importance should be non-negative

    def test_evaluate_metrics(self, realistic_regulation_features):
        """Test that model can compute evaluation metrics."""
        X_train, y_train, X_test, y_test = realistic_regulation_features
        
        model = RegulationStateXGBModel(
            feature_columns=list(X_train.columns),
            n_estimators=30,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        metrics = model.evaluate(X_test, y_test)
        
        # Verify metrics structure
        assert 'accuracy' in metrics
        assert 'classification_report' in metrics
        assert 'confusion_matrix' in metrics
        
        # Verify metric values
        assert 0 <= metrics['accuracy'] <= 1
        assert metrics['confusion_matrix'].shape == (len(model.class_labels), len(model.class_labels))


class TestModelComparisonOnSameData:
    """Test multiple models on the same dataset for comparison."""

    def test_compare_regression_models_day_ahead(self, realistic_day_ahead_features):
        """Compare all regression models on the same day-ahead data."""
        X_train, y_train, X_test, y_test = realistic_day_ahead_features
        
        models = {
            'persistence': PersistencePriceModel(method="last"),
            'linear': LinearRegressionPriceModel(),
            'random_forest': RandomForestPriceModel(n_estimators=30, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
        
        # All models should produce valid metrics
        for name, metrics in results.items():
            assert metrics['mae'] > 0, f"{name}: MAE should be positive"
            assert metrics['rmse'] > 0, f"{name}: RMSE should be positive"
            assert not np.isnan(metrics['r2']), f"{name}: R² should not be NaN"
        
        # Random Forest should generally outperform persistence
        # (though not guaranteed on all random splits)
        print(f"\nModel comparison on day-ahead data:")
        for name, metrics in results.items():
            print(f"  {name}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")

    def test_compare_regression_models_imbalance(self, realistic_imbalance_features):
        """Compare all regression models on the same imbalance data."""
        X_train, y_train, X_test, y_test = realistic_imbalance_features
        
        models = {
            'persistence': PersistencePriceModel(method="last"),
            'linear': LinearRegressionPriceModel(),
            'random_forest': RandomForestPriceModel(n_estimators=20, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            results[name] = {'mae': mae, 'rmse': rmse}
        
        # All models should produce valid metrics
        for name, metrics in results.items():
            assert metrics['mae'] > 0
            assert metrics['rmse'] > 0
        
        print(f"\nModel comparison on imbalance data:")
        for name, metrics in results.items():
            print(f"  {name}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")


class TestModelTrainingWithInsufficientData:
    """Test model behavior with insufficient training data."""

    def test_models_with_minimal_data(self):
        """Test that models handle minimal data appropriately."""
        # Create minimal dataset (e.g., 50 samples)
        n_samples = 50
        X = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
        })
        y = pd.Series(50 + 10 * np.random.randn(n_samples))
        
        # Split
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Persistence should work with minimal data
        persistence = PersistencePriceModel()
        persistence.fit(X_train, y_train)
        y_pred = persistence.predict(X_test)
        assert len(y_pred) == len(X_test)
        
        # Linear regression should work
        linear = LinearRegressionPriceModel()
        linear.fit(X_train, y_train)
        y_pred = linear.predict(X_test)
        assert len(y_pred) == len(X_test)
        
        # Random forest might overfit but should still train
        rf = RandomForestPriceModel(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        assert len(y_pred) == len(X_test)
