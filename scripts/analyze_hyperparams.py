#!/usr/bin/env python3
"""
Hyperparameter Search Analysis Script

This script provides in-depth analysis of hyperparameter search results, including:
- Per-market analysis: best models, performance gaps, overfitting detection
- Per-model analysis: average ranks, best/worst markets
- Hyperparameter importance: correlations, patterns, optimal ranges
- Global recommendations: best configurations per market

Usage:
    # Analyze all results
    python -m scripts.analyze_hyperparams

    # Specify custom results directory
    python -m scripts.analyze_hyperparams --results-dir artifacts/hyperparameter_search

    # Generate plots
    python -m scripts.analyze_hyperparams --generate-plots

    # Filter by specific markets
    python -m scripts.analyze_hyperparams --markets day_ahead imbalance_shortage
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import MARKET_TYPES, MODEL_TYPES
from app.core.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze hyperparameter search results and generate insights"
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory containing search results (default: artifacts/hyperparameter_search/)",
    )
    parser.add_argument(
        "--markets",
        nargs="+",
        choices=list(MARKET_TYPES.keys()),
        help="Filter analysis by specific markets (default: all)",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate and save visualization plots",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        help="Save analysis report to file (default: print to stdout)",
    )

    return parser.parse_args()


def load_summary_data(results_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load the summary CSV with all search results.

    Args:
        results_dir: Directory containing results

    Returns:
        DataFrame with summary data, or None if not found
    """
    summary_path = results_dir / "summary.csv"

    if not summary_path.exists():
        logger.error(f"Summary file not found: {summary_path}")
        return None

    try:
        df = pd.read_csv(summary_path)
        logger.info(f"Loaded summary data: {len(df)} results from {summary_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading summary: {e}")
        return None


def load_trial_data(results_dir: Path, market: str, model: str) -> Optional[pd.DataFrame]:
    """
    Load per-trial data for a specific market × model combination.

    Args:
        results_dir: Directory containing results
        market: Market type
        model: Model type

    Returns:
        DataFrame with trial data, or None if not found
    """
    trials_path = results_dir / market / f"{model}_trials_latest.csv"

    if not trials_path.exists():
        logger.debug(f"Trial data not found: {trials_path}")
        return None

    try:
        df = pd.read_csv(trials_path)
        return df
    except Exception as e:
        logger.warning(f"Error loading trial data from {trials_path}: {e}")
        return None


def analyze_per_market(
    summary_df: pd.DataFrame,
    markets: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Analyze results per market to find best models and performance gaps.

    Args:
        summary_df: Summary DataFrame
        markets: List of markets to analyze (None = all)

    Returns:
        Dictionary mapping market to analysis results
    """
    print("\n" + "=" * 80)
    print("PER-MARKET ANALYSIS")
    print("=" * 80 + "\n")

    if markets is None:
        markets = summary_df["market"].unique()

    market_analyses = {}

    for market in markets:
        market_data = summary_df[summary_df["market"] == market].copy()

        if market_data.empty:
            logger.warning(f"No data for market: {market}")
            continue

        # Sort by test score (descending for most metrics, but handle negative metrics)
        # For MSE/error metrics (negative), higher (less negative) is better
        market_data = market_data.sort_values("test_score", ascending=False)

        best_model = market_data.iloc[0]
        market_name = MARKET_TYPES.get(market, {}).get("display_name", market)

        print(f"Market: {market_name} ({market})")
        print(f"  Metric: {best_model['metric']}")
        print(f"  Best Model: {best_model['model']}")
        print(f"    Test Score: {best_model['test_score']:.4f}")
        if 'test_rmse' in best_model and pd.notna(best_model['test_rmse']):
            print(f"    Test RMSE: {best_model['test_rmse']:.4f}")
        print(f"    CV Score: {best_model['best_cv_score']:.4f}")
        print(f"    Training Samples: {int(best_model['n_samples_train'])}")
        print(f"    Features: {int(best_model['n_features'])}")

        # Show top 3 models
        print(f"\n  Top Models:")
        for idx, row in market_data.head(3).iterrows():
            diff = row["test_score"] - best_model["test_score"]
            diff_pct = (diff / abs(best_model["test_score"]) * 100) if best_model["test_score"] != 0 else 0
            print(f"    {idx + 1}. {row['model']:<25} Score: {row['test_score']:>8.4f}  (Δ {diff:+.4f}, {diff_pct:+.1f}%)")

        # Check for overfitting (CV score much better than test score)
        cv_test_gap = best_model["best_cv_score"] - best_model["test_score"]
        if abs(cv_test_gap) > 0.1 * abs(best_model["test_score"]):
            print(f"\n  ⚠ Potential Overfitting Detected:")
            print(f"    CV score ({best_model['best_cv_score']:.4f}) differs significantly from test score ({best_model['test_score']:.4f})")

        print()

        market_analyses[market] = {
            "best_model": best_model["model"],
            "best_score": best_model["test_score"],
            "metric": best_model["metric"],
            "top_3": market_data.head(3)[["model", "test_score"]].to_dict("records"),
            "cv_test_gap": cv_test_gap,
        }

    return market_analyses


def analyze_per_model(summary_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Analyze results per model across all markets.

    Args:
        summary_df: Summary DataFrame

    Returns:
        Dictionary mapping model to analysis results
    """
    print("\n" + "=" * 80)
    print("PER-MODEL ANALYSIS")
    print("=" * 80 + "\n")

    models = summary_df["model"].unique()
    model_analyses = {}

    for model in models:
        model_data = summary_df[summary_df["model"] == model].copy()
        model_name = MODEL_TYPES.get(model, {}).get("display_name", model)

        print(f"Model: {model_name} ({model})")

        # Calculate ranks per market
        ranks = []
        for market in model_data["market"].unique():
            market_subset = summary_df[summary_df["market"] == market].copy()
            market_subset = market_subset.sort_values("test_score", ascending=False).reset_index(drop=True)
            rank = market_subset[market_subset["model"] == model].index[0] + 1
            ranks.append(rank)

        avg_rank = np.mean(ranks)
        avg_score = model_data["test_score"].mean()
        std_score = model_data["test_score"].std()

        print(f"  Average Rank: {avg_rank:.2f}")
        print(f"  Average Test Score: {avg_score:.4f} (±{std_score:.4f})")
        print(f"  Markets Tested: {len(model_data)}")

        # Find best and worst markets
        best_market_row = model_data.loc[model_data["test_score"].idxmax()]
        worst_market_row = model_data.loc[model_data["test_score"].idxmin()]

        print(f"  Best Market: {best_market_row['market']} (score: {best_market_row['test_score']:.4f})")
        print(f"  Worst Market: {worst_market_row['market']} (score: {worst_market_row['test_score']:.4f})")

        # Count wins (1st place finishes)
        wins = sum(1 for r in ranks if r == 1)
        print(f"  1st Place Finishes: {wins}/{len(model_data)}")
        print()

        model_analyses[model] = {
            "avg_rank": avg_rank,
            "avg_score": avg_score,
            "std_score": std_score,
            "best_market": best_market_row["market"],
            "worst_market": worst_market_row["market"],
            "wins": wins,
            "markets_tested": len(model_data),
        }

    return model_analyses


def analyze_hyperparameter_importance(
    results_dir: Path,
    summary_df: pd.DataFrame,
    min_trials: int = 5,
) -> Dict[str, Dict]:
    """
    Analyze which hyperparameters matter most for each model × market.

    Args:
        results_dir: Directory containing trial data
        summary_df: Summary DataFrame
        min_trials: Minimum trials needed for correlation analysis

    Returns:
        Dictionary with hyperparameter importance per combination
    """
    print("\n" + "=" * 80)
    print("HYPERPARAMETER IMPORTANCE ANALYSIS")
    print("=" * 80 + "\n")

    importance_results = {}

    for _, row in summary_df.iterrows():
        market = row["market"]
        model = row["model"]
        combo_key = f"{market}_{model}"

        # Load trial data
        trials_df = load_trial_data(results_dir, market, model)

        if trials_df is None or len(trials_df) < min_trials:
            continue

        print(f"\n{model} × {market}:")
        print(f"  Trials: {len(trials_df)}")

        # Identify hyperparameter columns (exclude trial_id and scores)
        exclude_cols = {"trial_id", "mean_cv_score", "std_cv_score"}
        param_cols = [col for col in trials_df.columns if col not in exclude_cols]

        if not param_cols:
            print("  No hyperparameters found")
            continue

        correlations = {}

        # Analyze numerical hyperparameters
        for param in param_cols:
            param_values = trials_df[param]

            # Skip if all values are the same
            if param_values.nunique() <= 1:
                continue

            # Handle categorical vs numerical
            if param_values.dtype in [np.float64, np.int64]:
                # Numerical: compute Spearman correlation
                try:
                    corr, p_value = spearmanr(param_values, trials_df["mean_cv_score"])
                    if not np.isnan(corr):
                        correlations[param] = {
                            "correlation": corr,
                            "p_value": p_value,
                            "type": "numerical",
                        }
                except Exception as e:
                    logger.debug(f"Could not compute correlation for {param}: {e}")

            else:
                # Categorical: group by value and compute mean score
                grouped = trials_df.groupby(param)["mean_cv_score"].mean()
                best_value = grouped.idxmax()
                best_score = grouped.max()
                worst_score = grouped.min()

                correlations[param] = {
                    "type": "categorical",
                    "best_value": best_value,
                    "best_score": best_score,
                    "score_range": best_score - worst_score,
                }

        # Print top correlations
        if correlations:
            # Sort by absolute correlation for numerical, score_range for categorical
            sorted_params = sorted(
                correlations.items(),
                key=lambda x: abs(x[1].get("correlation", x[1].get("score_range", 0))),
                reverse=True,
            )

            print(f"  Top Influential Hyperparameters:")
            for param, stats in sorted_params[:5]:
                if stats["type"] == "numerical":
                    print(f"    {param}: correlation={stats['correlation']:+.3f} (p={stats['p_value']:.4f})")
                else:
                    print(f"    {param}: best={stats['best_value']}, score_range={stats['score_range']:.4f}")

        importance_results[combo_key] = correlations

    return importance_results


def generate_recommendations(
    summary_df: pd.DataFrame,
    market_analyses: Dict[str, Dict],
    importance_results: Dict[str, Dict],
) -> str:
    """
    Generate global recommendations based on all analyses.

    Args:
        summary_df: Summary DataFrame
        market_analyses: Per-market analysis results
        importance_results: Hyperparameter importance results

    Returns:
        Formatted recommendations string
    """
    print("\n" + "=" * 80)
    print("GLOBAL RECOMMENDATIONS")
    print("=" * 80 + "\n")

    recommendations = []

    # Per-market recommendations
    print("Per-Market Recommendations:")
    print("-" * 80)
    for market, analysis in market_analyses.items():
        market_name = MARKET_TYPES.get(market, {}).get("display_name", market)
        best_model = analysis["best_model"]
        best_score = analysis["best_score"]
        metric = analysis["metric"]

        # Find best params for this combination
        best_row = summary_df[
            (summary_df["market"] == market) & (summary_df["model"] == best_model)
        ].iloc[0]

        param_cols = [col for col in best_row.index if col.startswith("param_")]
        best_params = {col.replace("param_", ""): best_row[col] for col in param_cols if pd.notna(best_row[col])}

        print(f"\n{market_name} ({market}):")
        print(f"  Recommended Model: {MODEL_TYPES.get(best_model, {}).get('display_name', best_model)}")
        print(f"  Expected {metric}: {best_score:.4f}")
        print(f"  Recommended Hyperparameters:")
        for param, value in best_params.items():
            print(f"    {param}: {value}")

        # Add reasoning from importance analysis
        combo_key = f"{market}_{best_model}"
        if combo_key in importance_results:
            importances = importance_results[combo_key]
            if importances:
                top_param = max(
                    importances.items(),
                    key=lambda x: abs(x[1].get("correlation", x[1].get("score_range", 0)))
                )
                param_name, param_stats = top_param

                if param_stats["type"] == "numerical":
                    direction = "higher" if param_stats["correlation"] > 0 else "lower"
                    print(f"  Key Insight: {direction} {param_name} correlates with better performance (r={param_stats['correlation']:+.3f})")
                else:
                    print(f"  Key Insight: {param_name}={param_stats['best_value']} works best for this market")

        recommendations.append({
            "market": market,
            "model": best_model,
            "params": best_params,
            "score": best_score,
        })

    # Global insights
    print("\n" + "-" * 80)
    print("Global Insights:")
    print("-" * 80)

    # Find most successful model overall
    model_wins = summary_df.groupby("model", group_keys=False).apply(
        lambda x: sum(
            x["test_score"].max() == summary_df[summary_df["market"] == market]["test_score"].max()
            for market in x["market"].unique()
        ),
        include_groups=False
    )
    best_overall_model = model_wins.idxmax()
    wins = int(model_wins.loc[best_overall_model])

    print(f"\n• Most Successful Model: {MODEL_TYPES.get(best_overall_model, {}).get('display_name', best_overall_model)}")
    print(f"  Achieves best performance in {wins} out of {summary_df['market'].nunique()} markets")

    # Model type trends
    regression_models = summary_df[summary_df["model"].isin(["linear_regression", "random_forest", "hist_gradient_boosting"])]
    if not regression_models.empty:
        avg_scores = regression_models.groupby("model")["test_score"].mean()
        print(f"\n• Regression Model Rankings (by average test score):")
        for model, score in avg_scores.sort_values(ascending=False).items():
            print(f"  {model}: {score:.4f}")

    # Common hyperparameter patterns
    print(f"\n• Hyperparameter Patterns:")
    # Example: check if lower learning rates tend to work better
    lr_params = summary_df[[col for col in summary_df.columns if "learning_rate" in col]]
    if not lr_params.empty:
        print(f"  Learning rate configurations vary by model - see per-market recommendations above")

    print()

    return recommendations


def generate_plots(
    results_dir: Path,
    summary_df: pd.DataFrame,
    importance_results: Dict[str, Dict],
) -> None:
    """
    Generate and save visualization plots.

    Args:
        results_dir: Directory containing results
        summary_df: Summary DataFrame
        importance_results: Hyperparameter importance results
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        logger.warning("matplotlib not available - skipping plot generation")
        return

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80 + "\n")

    # Plot 1: Model performance comparison per market
    fig, ax = plt.subplots(figsize=(12, 6))
    markets = summary_df["market"].unique()

    x = np.arange(len(markets))
    width = 0.2
    models = summary_df["model"].unique()

    for i, model in enumerate(models):
        scores = [
            summary_df[(summary_df["market"] == market) & (summary_df["model"] == model)]["test_score"].values[0]
            if len(summary_df[(summary_df["market"] == market) & (summary_df["model"] == model)]) > 0
            else 0
            for market in markets
        ]
        ax.bar(x + i * width, scores, width, label=model)

    ax.set_xlabel("Market")
    ax.set_ylabel("Test Score")
    ax.set_title("Model Performance Comparison by Market")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(markets, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = plots_dir / "model_comparison_by_market.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"✓ Saved: {plot_path}")

    # Plot 2: Hyperparameter importance heatmap (for numerical params)
    for combo_key, importances in importance_results.items():
        if not importances:
            continue

        # Filter numerical correlations
        numerical_importances = {
            k: v for k, v in importances.items()
            if v["type"] == "numerical" and abs(v["correlation"]) > 0.1
        }

        if not numerical_importances:
            continue

        market, model = combo_key.split("_", 1)

        fig, ax = plt.subplots(figsize=(10, max(4, len(numerical_importances) * 0.4)))

        params = list(numerical_importances.keys())
        correlations = [numerical_importances[p]["correlation"] for p in params]

        colors = ["green" if c > 0 else "red" for c in correlations]
        y_pos = np.arange(len(params))

        ax.barh(y_pos, correlations, color=colors, alpha=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.set_xlabel("Spearman Correlation with CV Score")
        ax.set_title(f"Hyperparameter Importance: {model} × {market}")
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plot_path = plots_dir / f"{market}_{model}_param_importance.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"✓ Saved: {plot_path}")

    print()


def main():
    """Main analysis function."""
    args = parse_args()

    # Set default results directory
    if args.results_dir is None:
        args.results_dir = Path(__file__).parent.parent / "artifacts" / "hyperparameter_search"

    if not args.results_dir.exists():
        logger.error(f"Results directory not found: {args.results_dir}")
        return 1

    # Load summary data
    summary_df = load_summary_data(args.results_dir)
    if summary_df is None:
        return 1

    # Filter by markets if specified
    if args.markets:
        summary_df = summary_df[summary_df["market"].isin(args.markets)]

    if summary_df.empty:
        logger.error("No data to analyze after filtering")
        return 1

    # Run analyses
    market_analyses = analyze_per_market(summary_df, args.markets)
    model_analyses = analyze_per_model(summary_df)
    importance_results = analyze_hyperparameter_importance(args.results_dir, summary_df)
    recommendations = generate_recommendations(summary_df, market_analyses, importance_results)

    # Generate plots if requested
    if args.generate_plots:
        generate_plots(args.results_dir, summary_df, importance_results)

    # Save report if requested
    if args.output_report:
        logger.info(f"Report generation to file not yet implemented - see stdout output")

    return 0


if __name__ == "__main__":
    sys.exit(main())
