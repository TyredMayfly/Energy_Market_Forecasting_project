"""
Tests for the full hyperparameter search orchestrator.

Tests cover:
- Filtering combinations by market and model
- Running searches with minimal configs
- Saving results to correct locations
- Updating summary CSV
- Creating defaults config
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scripts.run_full_hyperparam_search import (
    create_defaults_config,
    get_filtered_combinations,
    run_single_search,
    update_summary_csv,
)


class TestGetFilteredCombinations:
    """Test combination filtering logic."""

    def test_no_filters_returns_all_combinations(self):
        """Test that no filters returns all valid combinations."""
        combinations = get_filtered_combinations()

        # Should have at least the known combinations
        assert len(combinations) > 0
        assert all(isinstance(combo, tuple) and len(combo) == 2 for combo in combinations)

        # Check some expected combinations
        markets = [c[0] for c in combinations]
        models = [c[1] for c in combinations]

        assert "day_ahead" in markets
        assert "random_forest" in models

    def test_market_filter(self):
        """Test filtering by market."""
        combinations = get_filtered_combinations(markets=["day_ahead"])

        assert len(combinations) > 0
        assert all(combo[0] == "day_ahead" for combo in combinations)

    def test_model_filter(self):
        """Test filtering by model."""
        combinations = get_filtered_combinations(models=["random_forest"])

        assert len(combinations) > 0
        assert all(combo[1] == "random_forest" for combo in combinations)

    def test_combined_filters(self):
        """Test filtering by both market and model."""
        combinations = get_filtered_combinations(
            markets=["day_ahead"],
            models=["random_forest"],
        )

        assert len(combinations) == 1
        assert combinations[0] == ("day_ahead", "random_forest")

    def test_empty_filters_returns_empty(self):
        """Test that invalid filters return empty list."""
        combinations = get_filtered_combinations(
            markets=["nonexistent_market"],
        )

        assert len(combinations) == 0


class TestUpdateSummaryCSV:
    """Test summary CSV update functionality."""

    def test_creates_new_summary_csv(self, tmp_path):
        """Test creating a new summary CSV."""
        results = [
            {
                "market": "day_ahead",
                "model": "random_forest",
                "metric": "neg_mean_squared_error",
                "best_cv_score": -100.5,
                "test_score": -105.2,
                "n_samples_train": 1000,
                "n_samples_test": 200,
                "n_features": 15,
                "param_n_estimators": 100,
                "param_max_depth": 10,
            }
        ]

        update_summary_csv(results, tmp_path)

        summary_path = tmp_path / "summary.csv"
        assert summary_path.exists()

        df = pd.read_csv(summary_path)
        assert len(df) == 1
        assert df.iloc[0]["market"] == "day_ahead"
        assert df.iloc[0]["model"] == "random_forest"
        assert df.iloc[0]["param_n_estimators"] == 100

    def test_appends_to_existing_summary_csv(self, tmp_path):
        """Test appending to an existing summary CSV."""
        # Create initial summary
        initial_results = [
            {
                "market": "day_ahead",
                "model": "random_forest",
                "test_score": -100.0,
            }
        ]
        update_summary_csv(initial_results, tmp_path)

        # Append new results
        new_results = [
            {
                "market": "imbalance_shortage",
                "model": "linear_regression",
                "test_score": -50.0,
            }
        ]
        update_summary_csv(new_results, tmp_path)

        summary_path = tmp_path / "summary.csv"
        df = pd.read_csv(summary_path)

        assert len(df) == 2
        assert set(df["market"]) == {"day_ahead", "imbalance_shortage"}

    def test_handles_empty_results(self, tmp_path, caplog):
        """Test handling empty results list."""
        update_summary_csv([], tmp_path)

        summary_path = tmp_path / "summary.csv"
        assert not summary_path.exists()
        assert "No results to save" in caplog.text

    def test_handles_heterogeneous_columns(self, tmp_path):
        """Test handling results with different column sets."""
        results = [
            {"market": "day_ahead", "model": "random_forest", "param_a": 1},
            {"market": "day_ahead", "model": "linear_regression", "param_b": 2},
        ]

        update_summary_csv(results, tmp_path)

        df = pd.read_csv(summary_path := tmp_path / "summary.csv")
        assert len(df) == 2
        assert "param_a" in df.columns
        assert "param_b" in df.columns
        # Check NaN handling
        assert pd.isna(df.iloc[0]["param_b"])
        assert pd.isna(df.iloc[1]["param_a"])


class TestCreateDefaultsConfig:
    """Test defaults configuration creation."""

    def test_creates_defaults_from_results(self, tmp_path):
        """Test creating defaults config from search results."""
        # Create fake search results
        market_dir = tmp_path / "day_ahead"
        market_dir.mkdir()

        result_data = {
            "best_params": {"n_estimators": 100, "max_depth": 10},
            "best_score": -95.5,
            "metadata": {
                "scoring": "neg_mean_squared_error",
                "search_timestamp": "2025-01-01T12:00:00",
            },
        }

        with open(market_dir / "random_forest_latest.json", "w") as f:
            json.dump(result_data, f)

        # Create defaults config
        create_defaults_config(tmp_path)

        defaults_path = tmp_path / "hyperparams_defaults.json"
        assert defaults_path.exists()

        with open(defaults_path) as f:
            defaults = json.load(f)

        assert "day_ahead" in defaults
        assert "random_forest" in defaults["day_ahead"]
        assert defaults["day_ahead"]["random_forest"]["best_params"] == {"n_estimators": 100, "max_depth": 10}
        assert defaults["day_ahead"]["random_forest"]["best_score"] == -95.5

    def test_handles_missing_results(self, tmp_path):
        """Test handling directory with no results."""
        # Create empty directory
        create_defaults_config(tmp_path)

        defaults_path = tmp_path / "hyperparams_defaults.json"
        assert defaults_path.exists()

        with open(defaults_path) as f:
            defaults = json.load(f)

        assert defaults == {}


class TestRunSingleSearch:
    """Test single search execution."""

    @patch("scripts.run_full_hyperparam_search.run_hyperparameter_search")
    @patch("scripts.run_full_hyperparam_search.save_search_results_with_trials")
    def test_successful_search(self, mock_save, mock_search, tmp_path):
        """Test successful search execution."""
        # Mock search results
        mock_search.return_value = (
            {"n_estimators": 100},  # best_params
            -95.5,  # best_score
            {
                "scoring": "neg_mean_squared_error",
                "search_timestamp": "2025-01-01T12:00:00",
                "test_score": -100.0,
                "n_samples_train": 1000,
                "n_samples_test": 200,
                "n_features": 15,
            },
        )

        result = run_single_search(
            market_type="day_ahead",
            model_type="random_forest",
            n_iter=5,
            cv_splits=2,
            test_size=0.2,
            random_state=42,
            n_jobs=1,
            output_dir=tmp_path,
        )

        assert result is not None
        assert result["market"] == "day_ahead"
        assert result["model"] == "random_forest"
        assert result["best_cv_score"] == -95.5
        assert result["test_score"] == -100.0
        assert result["param_n_estimators"] == 100

        # Verify functions were called
        mock_search.assert_called_once()
        mock_save.assert_called_once()

    @patch("scripts.run_full_hyperparam_search.run_hyperparameter_search")
    def test_handles_search_failure(self, mock_search, tmp_path, caplog):
        """Test handling search failure."""
        # Mock search to raise exception
        mock_search.side_effect = ValueError("Test error")

        result = run_single_search(
            market_type="day_ahead",
            model_type="random_forest",
            n_iter=5,
            cv_splits=2,
            test_size=0.2,
            random_state=42,
            n_jobs=1,
            output_dir=tmp_path,
        )

        assert result is None
        assert "Failed" in caplog.text
        assert "Test error" in caplog.text


class TestSaveSearchResultsWithTrials:
    """Test the new trial-level saving functionality."""

    def test_saves_best_and_trials(self, tmp_path):
        """Test saving both best params and trial data."""
        from app.services.hyperparameter_service import save_search_results_with_trials

        metadata = {
            "model_type": "random_forest",
            "market_type": "day_ahead",
            "search_timestamp": "2025-01-01T12:00:00",
            "scoring": "neg_mean_squared_error",
            "best_cv_score": -95.5,
            "test_score": -100.0,
            "n_samples_train": 1000,
            "n_samples_test": 200,
            "n_features": 15,
            "cv_results": {
                "params": [
                    {"n_estimators": 50, "max_depth": 5},
                    {"n_estimators": 100, "max_depth": 10},
                ],
                "mean_test_scores": [-100.0, -95.5],
                "std_test_scores": [5.0, 4.5],
            },
        }

        best_json_path, trials_csv_path = save_search_results_with_trials(
            market_type="day_ahead",
            model_type="random_forest",
            best_params={"n_estimators": 100, "max_depth": 10},
            best_score=-95.5,
            metadata=metadata,
            output_dir=tmp_path,
        )

        # Check best params JSON exists
        assert best_json_path.exists()
        with open(best_json_path) as f:
            data = json.load(f)
        assert data["best_params"]["n_estimators"] == 100

        # Check trials CSV exists
        assert trials_csv_path.exists()
        trials_df = pd.read_csv(trials_csv_path)
        assert len(trials_df) == 2
        assert "trial_id" in trials_df.columns
        assert "mean_cv_score" in trials_df.columns
        assert "n_estimators" in trials_df.columns

    def test_handles_missing_trial_data(self, tmp_path):
        """Test handling metadata without cv_results."""
        from app.services.hyperparameter_service import save_search_results_with_trials

        metadata = {
            "model_type": "random_forest",
            "market_type": "day_ahead",
            "search_timestamp": "2025-01-01T12:00:00",
            "scoring": "neg_mean_squared_error",
            "best_cv_score": -95.5,
            "test_score": -100.0,
        }

        best_json_path, trials_csv_path = save_search_results_with_trials(
            market_type="day_ahead",
            model_type="random_forest",
            best_params={"n_estimators": 100},
            best_score=-95.5,
            metadata=metadata,
            output_dir=tmp_path,
        )

        # Best params should still be saved
        assert best_json_path.exists()

        # Trials CSV should be None
        assert trials_csv_path is None
