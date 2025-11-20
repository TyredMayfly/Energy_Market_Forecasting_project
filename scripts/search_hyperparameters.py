#!/usr/bin/env python3
"""
Hyperparameter Search Script

This script runs hyperparameter search for all or specific model × market combinations.
Results are saved to artifacts/hyperparameter_search/ for later use in training.

Usage:
    # Search all combinations
    python scripts/search_hyperparameters.py

    # Search specific market
    python scripts/search_hyperparameters.py --market day_ahead

    # Search specific model
    python scripts/search_hyperparameters.py --model random_forest

    # Search specific combination
    python scripts/search_hyperparameters.py --market day_ahead --model random_forest

    # Customize search parameters
    python scripts/search_hyperparameters.py --n-iter 100 --cv-splits 10

    # List available tuned models
    python scripts/search_hyperparameters.py --list
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.logging import get_logger
from app.services.hyperparameter_service import (
    get_model_market_combinations,
    list_available_tuned_models,
    load_best_params,
    run_hyperparameter_search,
    save_search_results,
)

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run hyperparameter search for market forecasting models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search all model × market combinations
  python scripts/search_hyperparameters.py

  # Search only day-ahead market
  python scripts/search_hyperparameters.py --market day_ahead

  # Search only random forest models
  python scripts/search_hyperparameters.py --model random_forest

  # Custom search with 100 iterations
  python scripts/search_hyperparameters.py --n-iter 100 --cv-splits 10

  # List all tuned models
  python scripts/search_hyperparameters.py --list
        """,
    )

    parser.add_argument(
        "--market",
        type=str,
        help="Specific market type to search (e.g., 'day_ahead', 'imbalance_shortage')",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Specific model type to search (e.g., 'random_forest', 'xgboost_classifier')",
    )

    parser.add_argument(
        "--n-iter",
        type=int,
        default=50,
        help="Number of parameter settings to sample (default: 50)",
    )

    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation splits (default: 5)",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Ratio of data to use for final testing (default: 0.2)",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs, -1 for all cores (default: -1)",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0=silent, 1=progress, 2=detailed) (default: 1)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available tuned models and exit",
    )

    return parser.parse_args()


def list_tuned_models():
    """List all available tuned models."""
    logger.info("Available tuned models:")
    available = list_available_tuned_models()

    if not available:
        logger.info("  No tuned models found")
        return

    # Group by market
    by_market = {}
    for entry in available:
        market = entry["market_type"]
        model = entry["model_type"]
        if market not in by_market:
            by_market[market] = []
        by_market[market].append(model)

    for market in sorted(by_market.keys()):
        logger.info(f"\n  Market: {market}")
        for model in sorted(by_market[market]):
            # Load params to show summary
            params = load_best_params(market, model)
            if params:
                param_summary = ", ".join(f"{k}={v}" for k, v in list(params.items())[:3])
                if len(params) > 3:
                    param_summary += ", ..."
                logger.info(f"    - {model}: {param_summary}")
            else:
                logger.info(f"    - {model}")


def main():
    """Main entry point for hyperparameter search."""
    args = parse_args()

    # Handle --list option
    if args.list:
        list_tuned_models()
        return 0

    # Get all valid combinations
    all_combinations = get_model_market_combinations()

    # Filter combinations based on arguments
    combinations = []
    for market_type, model_type in all_combinations:
        if args.market and market_type != args.market:
            continue
        if args.model and model_type != args.model:
            continue
        combinations.append((market_type, model_type))

    if not combinations:
        logger.error("No valid model × market combinations found with the given filters")
        logger.error(f"  Market filter: {args.market or 'all'}")
        logger.error(f"  Model filter: {args.model or 'all'}")
        logger.error("\nAvailable combinations:")
        for market_type, model_type in all_combinations:
            logger.error(f"  - {market_type} × {model_type}")
        return 1

    # Log search plan
    logger.info("=" * 80)
    logger.info("HYPERPARAMETER SEARCH")
    logger.info("=" * 80)
    logger.info(f"Combinations to search: {len(combinations)}")
    logger.info(f"Iterations per search: {args.n_iter}")
    logger.info(f"Cross-validation splits: {args.cv_splits}")
    logger.info(f"Test size ratio: {args.test_size}")
    logger.info(f"Random state: {args.random_state}")
    logger.info(f"Parallel jobs: {args.n_jobs}")
    logger.info("=" * 80)

    # Track results
    results = []
    errors = []

    # Run search for each combination
    for idx, (market_type, model_type) in enumerate(combinations, 1):
        logger.info(f"\n[{idx}/{len(combinations)}] Searching: {market_type} × {model_type}")
        logger.info("-" * 80)

        try:
            # Run search
            best_params, best_score, metadata = run_hyperparameter_search(
                model_type=model_type,
                market_type=market_type,
                n_iter=args.n_iter,
                cv_splits=args.cv_splits,
                test_size_ratio=args.test_size,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                verbose=args.verbose,
            )

            # Save results
            filepath = save_search_results(
                market_type=market_type,
                model_type=model_type,
                best_params=best_params,
                best_score=best_score,
                metadata=metadata,
            )

            results.append(
                {
                    "market_type": market_type,
                    "model_type": model_type,
                    "best_score": best_score,
                    "filepath": filepath,
                }
            )

            logger.info(f"✓ Search completed successfully")

        except Exception as e:
            logger.error(f"✗ Error during search: {e}", exc_info=True)
            errors.append(
                {
                    "market_type": market_type,
                    "model_type": model_type,
                    "error": str(e),
                }
            )

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SEARCH SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total combinations: {len(combinations)}")
    logger.info(f"Successful: {len(results)}")
    logger.info(f"Failed: {len(errors)}")

    if results:
        logger.info("\nSuccessful searches:")
        for result in results:
            logger.info(
                f"  ✓ {result['market_type']} × {result['model_type']}: "
                f"score={result['best_score']:.4f}"
            )

    if errors:
        logger.info("\nFailed searches:")
        for error in errors:
            logger.info(f"  ✗ {error['market_type']} × {error['model_type']}: {error['error']}")

    logger.info("=" * 80)

    # Return exit code
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
