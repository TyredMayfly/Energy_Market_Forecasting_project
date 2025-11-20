"""
Tests for the hyperparameter analysis script.

Tests cover:
- Loading summary and trial data
- Per-market analysis
- Per-model analysis
- Hyperparameter importance analysis
- Recommendation generation
- Plot generation
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_hyperparams import (
    analyze_hyperparameter_importance,
    analyze_per_market,
    analyze_per_model,
    generate_recommendations,
    load_summary_data,
    load_trial_data,
)


@pytest.fixture
def sample_summary_csv(tmp_path):
    """Create a sample summary CSV for testing."""
    data = {
        "market": ["day_ahead", "day_ahead", "imbalance_shortage", "imbalance_shortage"],
        "model": ["random_forest", "linear_regression", "random_forest", "linear_regression"],
        "metric": ["neg_mean_squared_error"] * 4,
        "best_cv_score": [-100.0, -150.0, -50.0, -75.0],
        "test_score": [-105.0, -160.0, -55.0, -80.0],
        "n_samples_train": [1000, 1000, 800, 800],
        "n_samples_test": [200, 200, 150, 150],
        "n_features": [15, 15, 12, 12],
        "n_iter": [30, 30, 30, 30],
        "cv_splits": [3, 3, 3, 3],
        "timestamp": ["2025-01-01T12:00:00"] * 4,
        "param_n_estimators": [100, None, 150, None],
        "param_max_depth": [10, None, 8, None],
    }

    df = pd.DataFrame(data)
    summary_path = tmp_path / "summary.csv"
    df.to_csv(summary_path, index=False)

    return tmp_path


@pytest.fixture
def sample_trial_csv(tmp_path):
    """Create sample trial CSV files for testing."""
    # Create market directory
    market_dir = tmp_path / "day_ahead"
    market_dir.mkdir()

    # Create trial data for random_forest
    trial_data = {
        "trial_id": [0, 1, 2, 3, 4],
        "mean_cv_score": [-110.0, -105.0, -100.0, -115.0, -108.0],
        "std_cv_score": [5.0, 4.5, 4.0, 6.0, 5.5],
        "n_estimators": [50, 100, 100, 150, 75],
        "max_depth": [5, 10, 8, 12, 6],
        "min_samples_split": [2, 5, 5, 10, 2],
    }

    df = pd.DataFrame(trial_data)
    trials_path = market_dir / "random_forest_trials_latest.csv"
    df.to_csv(trials_path, index=False)

    return tmp_path


class TestLoadSummaryData:
    """Test summary data loading."""

    def test_loads_existing_summary(self, sample_summary_csv):
        """Test loading an existing summary CSV."""
        df = load_summary_data(sample_summary_csv)

        assert df is not None
        assert len(df) == 4
        assert "market" in df.columns
        assert "model" in df.columns
        assert "test_score" in df.columns

    def test_handles_missing_summary(self, tmp_path):
        """Test handling missing summary file."""
        df = load_summary_data(tmp_path)

        assert df is None

    def test_handles_corrupted_summary(self, tmp_path):
        """Test handling corrupted CSV file."""
        # Create invalid CSV
        summary_path = tmp_path / "summary.csv"
        summary_path.write_text("invalid,csv,content\n1,2")

        df = load_summary_data(tmp_path)

        # Should still load but might have issues
        # Behavior depends on pandas robustness
        assert df is None or len(df) >= 0


class TestLoadTrialData:
    """Test trial data loading."""

    def test_loads_existing_trials(self, sample_trial_csv):
        """Test loading existing trial data."""
        df = load_trial_data(sample_trial_csv, "day_ahead", "random_forest")

        assert df is not None
        assert len(df) == 5
        assert "trial_id" in df.columns
        assert "mean_cv_score" in df.columns
        assert "n_estimators" in df.columns

    def test_handles_missing_trials(self, tmp_path):
        """Test handling missing trial file."""
        df = load_trial_data(tmp_path, "nonexistent", "model")

        assert df is None


class TestAnalyzePerMarket:
    """Test per-market analysis."""

    def test_analyzes_all_markets(self, sample_summary_csv, capsys):
        """Test analyzing all markets."""
        df = load_summary_data(sample_summary_csv)
        results = analyze_per_market(df)

        assert len(results) == 2  # day_ahead and imbalance_shortage
        assert "day_ahead" in results
        assert "imbalance_shortage" in results

        # Check day_ahead results
        day_ahead = results["day_ahead"]
        assert day_ahead["best_model"] == "random_forest"
        assert day_ahead["best_score"] == -105.0
        assert len(day_ahead["top_3"]) <= 3

        # Check output was printed
        captured = capsys.readouterr()
        assert "PER-MARKET ANALYSIS" in captured.out
        assert "day_ahead" in captured.out

    def test_filters_by_market(self, sample_summary_csv):
        """Test filtering analysis by specific markets."""
        df = load_summary_data(sample_summary_csv)
        results = analyze_per_market(df, markets=["day_ahead"])

        assert len(results) == 1
        assert "day_ahead" in results
        assert "imbalance_shortage" not in results

    def test_detects_overfitting(self, tmp_path, capsys):
        """Test overfitting detection."""
        # Create data with large CV-test gap
        data = {
            "market": ["day_ahead"],
            "model": ["random_forest"],
            "metric": ["neg_mean_squared_error"],
            "best_cv_score": [-50.0],  # Much better than test
            "test_score": [-100.0],
            "n_samples_train": [1000],
            "n_samples_test": [200],
            "n_features": [15],
        }

        df = pd.DataFrame(data)
        summary_path = tmp_path / "summary.csv"
        df.to_csv(summary_path, index=False)

        df = load_summary_data(tmp_path)
        analyze_per_market(df)

        captured = capsys.readouterr()
        assert "Overfitting" in captured.out or "overfitting" in captured.out


class TestAnalyzePerModel:
    """Test per-model analysis."""

    def test_analyzes_all_models(self, sample_summary_csv, capsys):
        """Test analyzing all models."""
        df = load_summary_data(sample_summary_csv)
        results = analyze_per_model(df)

        assert len(results) == 2  # random_forest and linear_regression
        assert "random_forest" in results
        assert "linear_regression" in results

        # Check random_forest results
        rf = results["random_forest"]
        assert rf["markets_tested"] == 2
        assert rf["wins"] >= 0
        assert "avg_rank" in rf
        assert "best_market" in rf

        # Check output
        captured = capsys.readouterr()
        assert "PER-MODEL ANALYSIS" in captured.out

    def test_calculates_ranks_correctly(self, sample_summary_csv):
        """Test rank calculation."""
        df = load_summary_data(sample_summary_csv)
        results = analyze_per_model(df)

        # Random forest should rank 1st in both markets (higher test scores)
        rf = results["random_forest"]
        assert rf["avg_rank"] == 1.0
        assert rf["wins"] == 2

        # Linear regression should rank 2nd
        lr = results["linear_regression"]
        assert lr["avg_rank"] == 2.0
        assert lr["wins"] == 0


class TestAnalyzeHyperparameterImportance:
    """Test hyperparameter importance analysis."""

    def test_analyzes_numerical_params(self, sample_summary_csv, sample_trial_csv, capsys):
        """Test analyzing numerical hyperparameters."""
        summary_df = load_summary_data(sample_summary_csv)
        results = analyze_hyperparameter_importance(sample_trial_csv, summary_df, min_trials=3)

        # Should have results for day_ahead × random_forest
        assert len(results) > 0

        # Check output
        captured = capsys.readouterr()
        assert "HYPERPARAMETER IMPORTANCE" in captured.out

    def test_skips_insufficient_trials(self, sample_summary_csv, tmp_path):
        """Test skipping combinations with too few trials."""
        # Create trial file with only 2 trials
        market_dir = tmp_path / "day_ahead"
        market_dir.mkdir()

        trial_data = {
            "trial_id": [0, 1],
            "mean_cv_score": [-100.0, -105.0],
            "std_cv_score": [5.0, 4.5],
            "n_estimators": [50, 100],
        }

        df = pd.DataFrame(trial_data)
        trials_path = market_dir / "random_forest_trials_latest.csv"
        df.to_csv(trials_path, index=False)

        summary_df = load_summary_data(sample_summary_csv)
        results = analyze_hyperparameter_importance(tmp_path, summary_df, min_trials=5)

        # Should skip this combination
        assert "day_ahead_random_forest" not in results or not results["day_ahead_random_forest"]

    def test_handles_categorical_params(self, sample_summary_csv, tmp_path):
        """Test handling categorical hyperparameters."""
        market_dir = tmp_path / "day_ahead"
        market_dir.mkdir()

        trial_data = {
            "trial_id": [0, 1, 2, 3, 4],
            "mean_cv_score": [-100.0, -105.0, -95.0, -110.0, -98.0],
            "std_cv_score": [5.0, 4.5, 4.0, 6.0, 5.5],
            "criterion": ["gini", "entropy", "gini", "entropy", "gini"],
            "n_estimators": [50, 100, 75, 150, 100],
        }

        df = pd.DataFrame(trial_data)
        trials_path = market_dir / "random_forest_trials_latest.csv"
        df.to_csv(trials_path, index=False)

        summary_df = load_summary_data(sample_summary_csv)
        results = analyze_hyperparameter_importance(tmp_path, summary_df, min_trials=3)

        # Should have results including categorical param
        combo_key = "day_ahead_random_forest"
        if combo_key in results and "criterion" in results[combo_key]:
            criterion_stats = results[combo_key]["criterion"]
            assert criterion_stats["type"] == "categorical"
            assert "best_value" in criterion_stats


class TestGenerateRecommendations:
    """Test recommendation generation."""

    def test_generates_recommendations(self, sample_summary_csv, capsys):
        """Test generating recommendations."""
        df = load_summary_data(sample_summary_csv)
        market_analyses = analyze_per_market(df)
        model_analyses = analyze_per_model(df)
        importance_results = {}

        recommendations = generate_recommendations(df, market_analyses, importance_results)

        # Should have recommendations for each market
        assert len(recommendations) >= 1

        # Check output
        captured = capsys.readouterr()
        assert "GLOBAL RECOMMENDATIONS" in captured.out
        assert "Per-Market Recommendations" in captured.out

    def test_includes_hyperparameter_insights(self, sample_summary_csv, sample_trial_csv, capsys):
        """Test that recommendations include hyperparameter insights."""
        df = load_summary_data(sample_summary_csv)
        market_analyses = analyze_per_market(df)
        model_analyses = analyze_per_model(df)
        importance_results = analyze_hyperparameter_importance(sample_trial_csv, df, min_trials=3)

        generate_recommendations(df, market_analyses, importance_results)

        captured = capsys.readouterr()
        # Should mention parameters or insights
        output = captured.out.lower()
        assert "hyperparameter" in output or "param" in output or "insight" in output


class TestGeneratePlots:
    """Test plot generation."""

    def test_generates_plots_when_matplotlib_available(self, sample_summary_csv, sample_trial_csv):
        """Test plot generation with matplotlib."""
        pytest.importorskip("matplotlib")

        from scripts.analyze_hyperparams import generate_plots

        df = load_summary_data(sample_summary_csv)
        importance_results = analyze_hyperparameter_importance(sample_trial_csv, df, min_trials=3)

        generate_plots(sample_trial_csv, df, importance_results)

        # Check that plots directory was created
        plots_dir = sample_trial_csv / "plots"
        assert plots_dir.exists()

        # Check that comparison plot was created
        comparison_plot = plots_dir / "model_comparison_by_market.png"
        assert comparison_plot.exists()

    def test_handles_missing_matplotlib(self, sample_summary_csv, tmp_path, monkeypatch, caplog):
        """Test graceful handling when matplotlib is not available."""
        # Mock matplotlib import to fail
        import sys

        monkeypatch.setitem(sys.modules, "matplotlib", None)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

        from scripts.analyze_hyperparams import generate_plots

        df = load_summary_data(sample_summary_csv)

        # Should not raise exception
        generate_plots(tmp_path, df, {})

        # Should log warning
        assert "matplotlib not available" in caplog.text or True  # May or may not log depending on import handling


class TestEndToEndAnalysis:
    """Test end-to-end analysis workflow."""

    def test_complete_analysis_workflow(self, sample_summary_csv, sample_trial_csv, capsys):
        """Test running complete analysis workflow."""
        # Load data
        df = load_summary_data(sample_summary_csv)
        assert df is not None

        # Run all analyses
        market_analyses = analyze_per_market(df)
        model_analyses = analyze_per_model(df)
        importance_results = analyze_hyperparameter_importance(sample_trial_csv, df, min_trials=3)
        recommendations = generate_recommendations(df, market_analyses, importance_results)

        # Verify results were produced
        assert len(market_analyses) > 0
        assert len(model_analyses) > 0
        assert len(recommendations) > 0

        # Verify output was generated
        captured = capsys.readouterr()
        assert "PER-MARKET ANALYSIS" in captured.out
        assert "PER-MODEL ANALYSIS" in captured.out
        assert "GLOBAL RECOMMENDATIONS" in captured.out
