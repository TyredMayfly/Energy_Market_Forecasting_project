"""
Unit tests for hyperparameter search functionality.

Tests cover:
- Parameter grid generation
- Model-market compatibility validation
- Search result saving and loading
- Model-market combination enumeration
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.hyperparameter_service import (
    get_model_market_combinations,
    get_param_grid,
    list_available_tuned_models,
    load_best_params,
    run_hyperparameter_search,
    save_search_results,
)


class TestGetParamGrid:
    """Tests for get_param_grid function."""

    def test_persistence_raises_error(self):
        """Persistence model should raise ValueError (no hyperparameters to tune)."""
        with pytest.raises(ValueError, match="does not support hyperparameter tuning"):
            get_param_grid("persistence")

    def test_linear_regression_grid(self):
        """Linear regression should return small grid with fit_intercept."""
        grid = get_param_grid("linear_regression")
        assert "fit_intercept" in grid
        assert isinstance(grid["fit_intercept"], list)
        assert True in grid["fit_intercept"]
        assert False in grid["fit_intercept"]

    def test_random_forest_grid(self):
        """Random forest should return comprehensive grid."""
        grid = get_param_grid("random_forest")
        assert "n_estimators" in grid
        assert "max_depth" in grid
        assert "min_samples_split" in grid
        assert "min_samples_leaf" in grid
        assert "max_features" in grid
        assert "bootstrap" in grid

    def test_xgboost_grid(self):
        """XGBoost classifier should return boosting-specific grid."""
        grid = get_param_grid("xgboost_classifier")
        assert "n_estimators" in grid
        assert "max_depth" in grid
        assert "learning_rate" in grid
        assert "subsample" in grid
        assert "colsample_bytree" in grid
        assert "gamma" in grid
        assert "reg_alpha" in grid
        assert "reg_lambda" in grid

    def test_hist_gradient_boosting_grid(self):
        """HistGradientBoosting should return appropriate grid."""
        grid = get_param_grid("hist_gradient_boosting")
        assert "learning_rate" in grid
        assert "max_depth" in grid
        assert "max_leaf_nodes" in grid
        assert "min_samples_leaf" in grid
        assert "max_iter" in grid
        assert "l2_regularization" in grid

    def test_unknown_model_raises_error(self):
        """Unknown model type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            get_param_grid("nonexistent_model")


class TestSaveAndLoadResults:
    """Tests for saving and loading hyperparameter search results."""

    def test_save_and_load_results(self):
        """Test saving results and loading them back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Test data
            best_params = {"n_estimators": 100, "max_depth": 10}
            best_score = 0.85
            metadata = {
                "model_type": "random_forest",
                "market_type": "day_ahead",
                "n_samples_train": 1000,
            }

            # Save results
            filepath = save_search_results(
                market_type="day_ahead",
                model_type="random_forest",
                best_params=best_params,
                best_score=best_score,
                metadata=metadata,
                output_dir=tmpdir,
            )

            # Verify files exist
            assert filepath.exists()
            latest_path = tmpdir / "day_ahead" / "random_forest_latest.json"
            assert latest_path.exists()

            # Load results
            loaded_params = load_best_params(
                market_type="day_ahead",
                model_type="random_forest",
                search_dir=tmpdir,
            )

            # Verify loaded params match
            assert loaded_params == best_params

    def test_load_nonexistent_returns_none(self):
        """Loading non-existent params should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            params = load_best_params(
                market_type="nonexistent",
                model_type="random_forest",
                search_dir=tmpdir,
            )
            assert params is None

    def test_save_creates_directory_structure(self):
        """Saving should create necessary directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            save_search_results(
                market_type="day_ahead",
                model_type="random_forest",
                best_params={},
                best_score=0.8,
                metadata={},
                output_dir=tmpdir,
            )

            # Verify directory was created
            market_dir = tmpdir / "day_ahead"
            assert market_dir.exists()
            assert market_dir.is_dir()


class TestListAvailableTunedModels:
    """Tests for listing available tuned models."""

    def test_list_empty_directory(self):
        """Empty directory should return empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            available = list_available_tuned_models(search_dir=tmpdir)
            assert available == []

    def test_list_with_multiple_models(self):
        """Should find all *_latest.json files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create multiple result files
            markets = ["day_ahead", "imbalance_shortage"]
            models = ["random_forest", "linear_regression"]

            for market in markets:
                for model in models:
                    save_search_results(
                        market_type=market,
                        model_type=model,
                        best_params={},
                        best_score=0.8,
                        metadata={},
                        output_dir=tmpdir,
                    )

            # List available
            available = list_available_tuned_models(search_dir=tmpdir)

            # Should find 4 combinations
            assert len(available) == 4

            # Verify all combinations present
            combos = {(a["market_type"], a["model_type"]) for a in available}
            expected = {(m, mod) for m in markets for mod in models}
            assert combos == expected


class TestGetModelMarketCombinations:
    """Tests for getting valid model-market combinations."""

    def test_returns_valid_combinations(self):
        """Should return list of valid (market, model) tuples."""
        combinations = get_model_market_combinations()

        # Should be a list of tuples
        assert isinstance(combinations, list)
        assert all(isinstance(c, tuple) and len(c) == 2 for c in combinations)

        # Should not include persistence (no hyperparameters)
        models = {c[1] for c in combinations}
        assert "persistence" not in models

    def test_classification_model_only_for_classification_markets(self):
        """XGBoost classifier should only appear with classification markets."""
        combinations = get_model_market_combinations()

        # Get all markets that use xgboost_classifier
        xgboost_markets = {c[0] for c in combinations if c[1] == "xgboost_classifier"}

        # These should all be classification markets
        from app.core.config import MARKET_TYPES

        for market in xgboost_markets:
            assert MARKET_TYPES[market].get("target_type") == "classification"

    def test_regression_models_only_for_regression_markets(self):
        """Regression models should only appear with regression markets."""
        combinations = get_model_market_combinations()

        regression_models = {"linear_regression", "random_forest", "hist_gradient_boosting"}

        for market, model in combinations:
            if model in regression_models:
                from app.core.config import MARKET_TYPES

                target_type = MARKET_TYPES[market].get("target_type", "regression")
                assert target_type == "regression"


class TestRunHyperparameterSearch:
    """Tests for running hyperparameter search (integration tests with mocking)."""

    @patch("app.services.hyperparameter_service.build_features_and_target")
    @patch("app.services.hyperparameter_service.RandomizedSearchCV")
    def test_search_with_mocked_data(self, mock_search_cv, mock_build_features):
        """Test search pipeline with mocked data and search."""
        # Mock feature building
        n_samples = 1000
        n_features = 10
        X = pd.DataFrame(np.random.randn(n_samples, n_features))
        y = pd.Series(np.random.randn(n_samples))
        mock_build_features.return_value = (X, y)

        # Mock search results
        mock_search_instance = MagicMock()
        mock_search_instance.best_params_ = {"n_estimators": 100, "max_depth": 10}
        mock_search_instance.best_score_ = 0.85
        mock_search_instance.cv_results_ = {
            "mean_test_score": np.array([0.8, 0.85, 0.82]),
            "std_test_score": np.array([0.05, 0.04, 0.06]),
            "params": [
                {"n_estimators": 50, "max_depth": 5},
                {"n_estimators": 100, "max_depth": 10},
                {"n_estimators": 150, "max_depth": 15},
            ],
        }

        # Mock best estimator
        mock_estimator = MagicMock()
        mock_estimator.score.return_value = 0.83
        # Mock predict to return proper numpy array for RMSE calculation
        mock_estimator.predict.return_value = np.random.randn(n_samples // 5)  # test set size
        mock_search_instance.best_estimator_ = mock_estimator

        mock_search_cv.return_value = mock_search_instance

        # Run search
        best_params, best_score, metadata = run_hyperparameter_search(
            model_type="random_forest",
            market_type="day_ahead",
            n_iter=10,
            cv_splits=3,
        )

        # Verify results
        assert best_params == {"n_estimators": 100, "max_depth": 10}
        assert best_score == 0.85
        assert metadata["model_type"] == "random_forest"
        assert metadata["market_type"] == "day_ahead"
        assert "best_cv_score" in metadata
        assert "test_score" in metadata

    def test_search_invalid_market_raises_error(self):
        """Search with invalid market should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown market type"):
            run_hyperparameter_search(
                model_type="random_forest",
                market_type="nonexistent_market",
            )

    def test_search_incompatible_model_market_raises_error(self):
        """Search with incompatible model-market combo should raise ValueError."""
        # Try to use classifier on regression market
        with pytest.raises(ValueError, match="cannot be used for regression"):
            run_hyperparameter_search(
                model_type="xgboost_classifier",
                market_type="day_ahead",  # Regression market
            )

        # Try to use regression model on classification market
        with pytest.raises(ValueError, match="cannot be used for classification"):
            run_hyperparameter_search(
                model_type="random_forest",
                market_type="regulation_state",  # Classification market
            )

    @patch("app.services.hyperparameter_service.build_features_and_target")
    def test_search_with_empty_data_raises_error(self, mock_build_features):
        """Search with empty data should raise ValueError."""
        # Return empty data
        mock_build_features.return_value = (pd.DataFrame(), pd.Series())

        with pytest.raises(ValueError, match="No data available"):
            run_hyperparameter_search(
                model_type="random_forest",
                market_type="day_ahead",
            )


class TestIntegration:
    """Integration tests using actual configurations."""

    def test_all_combinations_have_param_grids(self):
        """All valid combinations should have parameter grids defined."""
        combinations = get_model_market_combinations()

        for market, model in combinations:
            # Should not raise error
            grid = get_param_grid(model)
            assert isinstance(grid, dict)
            assert len(grid) > 0

    def test_save_load_roundtrip_preserves_data(self):
        """Saving and loading should preserve all data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create comprehensive test data
            best_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "learning_rate": 0.1,
            }
            best_score = 0.876543
            metadata = {
                "model_type": "random_forest",
                "market_type": "day_ahead",
                "n_samples_train": 1000,
                "n_samples_test": 250,
                "cv_splits": 5,
                "best_cv_score": 0.876543,
                "test_score": 0.865432,
            }

            # Save
            save_search_results(
                market_type="day_ahead",
                model_type="random_forest",
                best_params=best_params,
                best_score=best_score,
                metadata=metadata,
                output_dir=tmpdir,
            )

            # Load and verify
            loaded_params = load_best_params(
                market_type="day_ahead",
                model_type="random_forest",
                search_dir=tmpdir,
            )

            assert loaded_params == best_params

            # Also verify full file content
            latest_path = tmpdir / "day_ahead" / "random_forest_latest.json"
            with open(latest_path, "r") as f:
                data = json.load(f)

            assert data["best_params"] == best_params
            assert data["best_score"] == best_score
            assert data["metadata"]["model_type"] == "random_forest"

    def test_hist_gradient_boosting_in_combinations(self):
        """Test that hist_gradient_boosting is included in valid combinations."""
        combinations = get_model_market_combinations()

        # Extract all model types
        models = {c[1] for c in combinations}

        # hist_gradient_boosting should be present
        assert "hist_gradient_boosting" in models

        # Should appear with regression markets only
        hgb_combos = [c for c in combinations if c[1] == "hist_gradient_boosting"]
        assert len(hgb_combos) > 0

        # Verify all are regression markets
        from app.core.config import MARKET_TYPES

        for market, _ in hgb_combos:
            assert MARKET_TYPES[market].get("target_type", "regression") == "regression"
