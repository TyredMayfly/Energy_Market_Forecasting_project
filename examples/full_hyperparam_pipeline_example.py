"""
Example: Full Hyperparameter Search and Analysis Pipeline

This example demonstrates:
1. Running a quick hyperparameter search for selected combinations
2. Analyzing the results
3. Loading and using the best hyperparameters in production

For full pipeline documentation, see docs/FULL_HYPERPARAM_PIPELINE.md
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.logging import get_logger
from app.services.hyperparameter_service import (
    get_model_market_combinations,
    load_best_params,
)

logger = get_logger(__name__)


def example_1_list_combinations():
    """Example 1: List all valid model × market combinations."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: List All Valid Combinations")
    print("=" * 80 + "\n")

    combinations = get_model_market_combinations()

    print(f"Total valid combinations: {len(combinations)}\n")

    # Group by market
    from collections import defaultdict

    by_market = defaultdict(list)
    for market, model in combinations:
        by_market[market].append(model)

    for market, models in sorted(by_market.items()):
        print(f"{market}:")
        for model in models:
            print(f"  - {model}")
        print()


def example_2_quick_search():
    """Example 2: Run a quick search for demonstration."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Quick Hyperparameter Search")
    print("=" * 80 + "\n")

    print("To run a quick search for one combination:")
    print()
    print("  python -m scripts.run_full_hyperparam_search \\")
    print("      --markets day_ahead \\")
    print("      --models random_forest \\")
    print("      --n-iter 5 \\")
    print("      --cv-splits 2")
    print()
    print("Expected runtime: ~30 seconds")
    print("Output: artifacts/hyperparameter_search/day_ahead/random_forest_*.json")
    print()


def example_3_full_search():
    """Example 3: Run full search across all combinations."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Full Hyperparameter Search Pipeline")
    print("=" * 80 + "\n")

    print("To run comprehensive search for all 10 combinations:")
    print()
    print("  python -m scripts.run_full_hyperparam_search \\")
    print("      --n-iter 30 \\")
    print("      --cv-splits 3 \\")
    print("      --update-defaults")
    print()
    print("Expected runtime: ~30-50 minutes")
    print("Output:")
    print("  - artifacts/hyperparameter_search/summary.csv")
    print("  - artifacts/hyperparameter_search/<market>/<model>_*.json")
    print("  - artifacts/hyperparameter_search/<market>/<model>_trials_*.csv")
    print("  - artifacts/hyperparameter_search/hyperparams_defaults.json")
    print()


def example_4_analyze_results():
    """Example 4: Analyze hyperparameter search results."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Analyze Search Results")
    print("=" * 80 + "\n")

    print("After running searches, analyze results:")
    print()
    print("  python -m scripts.analyze_hyperparams --generate-plots")
    print()
    print("This will:")
    print("  - Identify best model per market")
    print("  - Calculate model rankings across markets")
    print("  - Analyze hyperparameter importance")
    print("  - Generate recommendations")
    print("  - Create visualization plots")
    print()
    print("Output:")
    print("  - Detailed analysis printed to console")
    print("  - artifacts/hyperparameter_search/plots/*.png")
    print()


def example_5_load_best_params():
    """Example 5: Load and use best hyperparameters."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Load Best Hyperparameters")
    print("=" * 80 + "\n")

    # Try to load tuned parameters
    market = "day_ahead"
    model = "random_forest"

    best_params = load_best_params(market, model)

    if best_params:
        print(f"✓ Found tuned hyperparameters for {model} × {market}:")
        print()
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print()
        print("These parameters can be used to initialize your model:")
        print()
        print(f"  from app.models.random_forest_model import RandomForestPriceModel")
        print(f"  model = RandomForestPriceModel(**best_params)")
    else:
        print(f"✗ No tuned hyperparameters found for {model} × {market}")
        print()
        print("Run hyperparameter search first:")
        print()
        print(f"  python -m scripts.run_full_hyperparam_search \\")
        print(f"      --markets {market} \\")
        print(f"      --models {model}")
    print()


def example_6_workflow():
    """Example 6: Complete workflow."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Complete Workflow")
    print("=" * 80 + "\n")

    print("Recommended workflow for hyperparameter optimization:")
    print()
    print("1. Start with a quick test:")
    print("   python -m scripts.run_full_hyperparam_search \\")
    print("       --markets day_ahead --models random_forest --n-iter 5")
    print()
    print("2. Run moderate search for selected combinations:")
    print("   python -m scripts.run_full_hyperparam_search \\")
    print("       --markets day_ahead imbalance_shortage \\")
    print("       --n-iter 30")
    print()
    print("3. Analyze results:")
    print("   python -m scripts.analyze_hyperparams --generate-plots")
    print()
    print("4. Run full intensive search based on insights:")
    print("   python -m scripts.run_full_hyperparam_search \\")
    print("       --n-iter 100 --cv-splits 5 --update-defaults")
    print()
    print("5. Final analysis and recommendations:")
    print("   python -m scripts.analyze_hyperparams --generate-plots")
    print()
    print("6. Use best hyperparameters in production:")
    print("   # Automatically loaded by forecast_service.py")
    print("   # Or load manually: load_best_params(market, model)")
    print()


def example_7_custom_analysis():
    """Example 7: Custom analysis with pandas."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Custom Analysis")
    print("=" * 80 + "\n")

    print("Load and analyze results programmatically:")
    print()
    print("  import pandas as pd")
    print("  from pathlib import Path")
    print()
    print("  # Load summary")
    print("  summary = pd.read_csv('artifacts/hyperparameter_search/summary.csv')")
    print()
    print("  # Find best model per market")
    print("  best_per_market = summary.loc[")
    print("      summary.groupby('market')['test_score'].idxmax()")
    print("  ]")
    print()
    print("  # Load trial data for detailed analysis")
    print("  trials = pd.read_csv(")
    print("      'artifacts/hyperparameter_search/day_ahead/random_forest_trials_latest.csv'")
    print("  )")
    print()
    print("  # Analyze parameter correlations")
    print("  trials[['n_estimators', 'max_depth', 'mean_cv_score']].corr()")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("FULL HYPERPARAMETER SEARCH AND ANALYSIS PIPELINE - EXAMPLES")
    print("=" * 80)

    example_1_list_combinations()
    example_2_quick_search()
    example_3_full_search()
    example_4_analyze_results()
    example_5_load_best_params()
    example_6_workflow()
    example_7_custom_analysis()

    print("\n" + "=" * 80)
    print("For complete documentation, see:")
    print("  docs/FULL_HYPERPARAM_PIPELINE.md")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
