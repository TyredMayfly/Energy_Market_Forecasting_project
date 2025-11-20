"""
Example: Using the Hyperparameter Search Pipeline

This script demonstrates how to use the hyperparameter search functionality
both programmatically and via the command-line interface.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.hyperparameter_service import (
    get_model_market_combinations,
    get_param_grid,
    list_available_tuned_models,
    load_best_params,
    run_hyperparameter_search,
    save_search_results,
)


def example_1_list_valid_combinations():
    """Example 1: List all valid model × market combinations."""
    print("=" * 80)
    print("Example 1: Valid Model × Market Combinations")
    print("=" * 80)

    combinations = get_model_market_combinations()
    print(f"\nFound {len(combinations)} valid combinations:\n")

    for market, model in sorted(combinations):
        print(f"  - {market:25s} × {model}")

    print()


def example_2_view_parameter_grids():
    """Example 2: View parameter search spaces for each model."""
    print("=" * 80)
    print("Example 2: Parameter Search Spaces")
    print("=" * 80)

    models = ["linear_regression", "random_forest", "xgboost_classifier", "hist_gradient_boosting"]

    for model in models:
        print(f"\n{model}:")
        print("-" * 40)
        grid = get_param_grid(model)
        for param, values in grid.items():
            print(f"  {param:20s}: {values}")

    print()


def example_3_run_single_search():
    """Example 3: Run hyperparameter search for a single model × market."""
    print("=" * 80)
    print("Example 3: Run Single Hyperparameter Search")
    print("=" * 80)

    market = "day_ahead"
    model = "linear_regression"

    print(f"\nSearching: {market} × {model}")
    print("This will:")
    print("  1. Load market data and build features")
    print("  2. Split into train/test sets (80/20 chronological)")
    print("  3. Run randomized search with 5-fold time-series CV")
    print("  4. Evaluate best model on test set")
    print("  5. Save results to artifacts/hyperparameter_search/")

    print("\nNOTE: Uncomment the code below to actually run the search")
    print("(Commented out by default to avoid long execution times)")

    # Uncomment to actually run:
    # best_params, best_score, metadata = run_hyperparameter_search(
    #     model_type=model,
    #     market_type=market,
    #     n_iter=20,  # Reduced for faster execution
    #     cv_splits=3,
    # )
    #
    # print(f"\nBest parameters: {best_params}")
    # print(f"Best CV score: {best_score:.4f}")
    # print(f"Test score: {metadata['test_score']:.4f}")
    #
    # # Save results
    # filepath = save_search_results(
    #     market_type=market,
    #     model_type=model,
    #     best_params=best_params,
    #     best_score=best_score,
    #     metadata=metadata,
    # )
    # print(f"\nResults saved to: {filepath}")

    print()


def example_4_load_tuned_parameters():
    """Example 4: Load previously tuned hyperparameters."""
    print("=" * 80)
    print("Example 4: Load Tuned Hyperparameters")
    print("=" * 80)

    market = "day_ahead"
    model = "random_forest"

    print(f"\nAttempting to load tuned params for {market} × {model}")

    params = load_best_params(market_type=market, model_type=model)

    if params:
        print("✓ Found tuned hyperparameters:")
        for param, value in params.items():
            print(f"  {param:20s}: {value}")
    else:
        print("✗ No tuned hyperparameters found")
        print("  Run hyperparameter search first:")
        print(f"  python scripts/search_hyperparameters.py --market {market} --model {model}")

    print()


def example_5_list_available_tuned():
    """Example 5: List all available tuned models."""
    print("=" * 80)
    print("Example 5: List Available Tuned Models")
    print("=" * 80)

    available = list_available_tuned_models()

    if not available:
        print("\nNo tuned models found yet.")
        print("Run hyperparameter search to create tuned models:")
        print("  python scripts/search_hyperparameters.py")
    else:
        print(f"\nFound {len(available)} tuned models:\n")

        # Group by market
        by_market = {}
        for entry in available:
            market = entry["market_type"]
            model = entry["model_type"]
            if market not in by_market:
                by_market[market] = []
            by_market[market].append(model)

        for market in sorted(by_market.keys()):
            print(f"  {market}:")
            for model in sorted(by_market[market]):
                print(f"    - {model}")

    print()


def example_6_cli_usage():
    """Example 6: Command-line interface usage examples."""
    print("=" * 80)
    print("Example 6: Command-Line Usage")
    print("=" * 80)

    examples = [
        ("Search all combinations", "python scripts/search_hyperparameters.py"),
        (
            "Search specific market",
            "python scripts/search_hyperparameters.py --market day_ahead",
        ),
        (
            "Search specific model",
            "python scripts/search_hyperparameters.py --model random_forest",
        ),
        (
            "Quick search (reduced iterations)",
            "python scripts/search_hyperparameters.py --n-iter 20 --cv-splits 3",
        ),
        (
            "Thorough search",
            "python scripts/search_hyperparameters.py --n-iter 100 --cv-splits 10",
        ),
        ("List available tuned models", "python scripts/search_hyperparameters.py --list"),
        ("Get help", "python scripts/search_hyperparameters.py --help"),
    ]

    print("\nCommand-Line Examples:\n")
    for description, command in examples:
        print(f"{description}:")
        print(f"  {command}\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH PIPELINE EXAMPLES")
    print("=" * 80)
    print()

    example_1_list_valid_combinations()
    example_2_view_parameter_grids()
    example_3_run_single_search()
    example_4_load_tuned_parameters()
    example_5_list_available_tuned()
    example_6_cli_usage()

    print("=" * 80)
    print("For full documentation, see: docs/HYPERPARAMETER_SEARCH.md")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
