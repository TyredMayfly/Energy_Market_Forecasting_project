#!/usr/bin/env python3
"""
Full Hyperparameter Search Orchestrator

This script runs comprehensive hyperparameter search across all or filtered
model × market combinations, storing results in a structured format for
downstream analysis.

Usage:
    # Run all combinations
    python -m scripts.run_full_hyperparam_search

    # Filter by specific markets
    python -m scripts.run_full_hyperparam_search --markets day_ahead imbalance_shortage

    # Filter by specific models
    python -m scripts.run_full_hyperparam_search --models hist_gradient_boosting random_forest

    # Customize search intensity
    python -m scripts.run_full_hyperparam_search --n-iter 50 --cv-splits 5

    # Update default hyperparameters config after search
    python -m scripts.run_full_hyperparam_search --update-defaults
"""

import argparse
import csv
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import MARKET_TYPES, MODEL_TYPES
from app.core.logging import get_logger
from app.services.hyperparameter_service import (
    get_model_market_combinations,
    get_param_grid,
    run_hyperparameter_search,
    save_search_results_with_trials,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive hyperparameter search across all model × market combinations"
    )

    # Filtering options
    parser.add_argument(
        "--markets",
        nargs="+",
        choices=list(MARKET_TYPES.keys()),
        help="Filter by specific markets (default: all markets)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[m for m in MODEL_TYPES.keys() if m != "persistence"],
        help="Filter by specific models (default: all tunable models)",
    )

    # Search configuration
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="Number of parameter settings to sample per combination (default: 30)",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=3,
        help="Number of cross-validation splits (default: 3)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Ratio of data for testing (default: 0.2)",
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

    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results (default: hyperparameters/)",
    )
    parser.add_argument(
        "--update-defaults",
        action="store_true",
        help="Update hyperparams_defaults.json with best configurations",
    )

    return parser.parse_args()


def get_filtered_combinations(
    markets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """
    Get filtered model × market combinations.

    Args:
        markets: List of market types to include (None = all)
        models: List of model types to include (None = all)

    Returns:
        List of (market_type, model_type) tuples
    """
    all_combinations = get_model_market_combinations()

    # Apply filters
    filtered = []
    for market, model in all_combinations:
        if markets is not None and market not in markets:
            continue
        if models is not None and model not in models:
            continue
        filtered.append((market, model))

    return filtered


def run_single_search(
    market_type: str,
    model_type: str,
    n_iter: int,
    cv_splits: int,
    test_size: float,
    random_state: int,
    n_jobs: int,
    output_dir: Path,
) -> Optional[Dict]:
    """
    Run hyperparameter search for a single combination.

    Args:
        market_type: Type of market
        model_type: Type of model
        n_iter: Number of iterations
        cv_splits: Number of CV splits
        test_size: Test set ratio
        random_state: Random seed
        n_jobs: Number of parallel jobs
        output_dir: Output directory

    Returns:
        Dictionary with search results, or None if search failed
    """
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting search: {model_type} × {market_type}")
        logger.info(f"{'='*80}")

        # Run search
        best_params, best_score, metadata = run_hyperparameter_search(
            model_type=model_type,
            market_type=market_type,
            n_iter=n_iter,
            cv_splits=cv_splits,
            test_size_ratio=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=1,
        )

        # Save results with trial-level data
        save_search_results_with_trials(
            market_type=market_type,
            model_type=model_type,
            best_params=best_params,
            best_score=best_score,
            metadata=metadata,
            output_dir=output_dir,
        )

        logger.info(f"✓ Completed: {model_type} × {market_type}")
        logger.info(f"  Best CV score: {best_score:.4f}")
        logger.info(f"  Test score: {metadata['test_score']:.4f}")
        if metadata.get('test_rmse') is not None:
            logger.info(f"  Test RMSE: {metadata['test_rmse']:.4f}")

        # Return summary for aggregation
        return {
            "market": market_type,
            "model": model_type,
            "metric": metadata["scoring"],
            "best_cv_score": best_score,
            "test_score": metadata["test_score"],
            "test_rmse": metadata.get("test_rmse"),
            "n_samples_train": metadata["n_samples_train"],
            "n_samples_test": metadata["n_samples_test"],
            "n_features": metadata["n_features"],
            "n_iter": n_iter,
            "cv_splits": cv_splits,
            "timestamp": metadata["search_timestamp"],
            **{f"param_{k}": v for k, v in best_params.items()},
        }

    except Exception as e:
        logger.error(f"✗ Failed: {model_type} × {market_type}")
        logger.error(f"  Error: {str(e)}")
        logger.error(f"  Traceback:\n{traceback.format_exc()}")
        return None


def update_summary_csv(
    results: List[Dict],
    output_dir: Path,
) -> None:
    """
    Update or create the global summary CSV with all search results.

    Args:
        results: List of result dictionaries from searches
        output_dir: Output directory
    """
    if not results:
        logger.warning("No results to save to summary CSV")
        return

    summary_path = output_dir / "summary.csv"

    # Load existing results if file exists
    all_results = []
    if summary_path.exists():
        import pandas as pd
        try:
            existing_df = pd.read_csv(summary_path)
            all_results = existing_df.to_dict('records')
        except Exception as e:
            logger.warning(f"Could not load existing summary: {e}")
    
    # Add new results
    all_results.extend(results)

    # Define standard column order
    core_columns = [
        'market', 'model', 'metric', 
        'best_cv_score', 'test_score', 'test_rmse',
        'n_samples_train', 'n_samples_test', 'n_features',
        'n_iter', 'cv_splits', 'timestamp'
    ]
    
    # Get all param columns
    param_columns = sorted([k for r in all_results for k in r.keys() if k.startswith('param_')])
    
    # Final column order: core columns first, then params
    columns = [c for c in core_columns if any(c in r for r in all_results)]
    columns.extend(param_columns)

    # Rewrite entire file with consistent columns
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        
        for result in all_results:
            # Fill missing columns with empty string for better CSV formatting
            row = {col: result.get(col, '') for col in columns}
            writer.writerow(row)

    logger.info(f"Updated summary CSV: {summary_path}")
    logger.info(f"  Total results: {len(all_results)} ({len(results)} new)")


def create_defaults_config(
    output_dir: Path,
) -> None:
    """
    Create hyperparams_defaults.json with recommended defaults per market × model.

    Args:
        output_dir: Directory containing search results
    """
    import json

    defaults = {}

    # Scan for all latest results
    for market_dir in output_dir.iterdir():
        if not market_dir.is_dir():
            continue

        market_type = market_dir.name
        defaults[market_type] = {}

        # Load each model's latest results
        for result_file in market_dir.glob("*_latest.json"):
            model_type = result_file.stem.replace("_latest", "")

            try:
                with open(result_file, "r") as f:
                    data = json.load(f)

                defaults[market_type][model_type] = {
                    "best_params": data["best_params"],
                    "best_score": data["best_score"],
                    "scoring": data["metadata"]["scoring"],
                    "last_updated": data["metadata"]["search_timestamp"],
                }

            except Exception as e:
                logger.error(f"Error reading {result_file}: {e}")

    # Save defaults file
    defaults_path = output_dir / "hyperparams_defaults.json"
    with open(defaults_path, "w") as f:
        json.dump(defaults, f, indent=2)

    logger.info(f"Created defaults config: {defaults_path}")


def main():
    """Main orchestrator function."""
    args = parse_args()

    # Set default output directory
    if args.output_dir is None:
        args.output_dir = Path(__file__).parent.parent / "hyperparameters"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get combinations to search
    combinations = get_filtered_combinations(
        markets=args.markets,
        models=args.models,
    )

    if not combinations:
        logger.error("No valid combinations found with given filters")
        return 1

    logger.info(f"\n{'='*80}")
    logger.info(f"FULL HYPERPARAMETER SEARCH PIPELINE")
    logger.info(f"{'='*80}")
    logger.info(f"Total combinations: {len(combinations)}")
    logger.info(f"Iterations per combination: {args.n_iter}")
    logger.info(f"CV splits: {args.cv_splits}")
    logger.info(f"Test size ratio: {args.test_size}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"\nCombinations:")
    for market, model in combinations:
        logger.info(f"  - {model} × {market}")
    logger.info(f"{'='*80}\n")

    # Run searches
    results = []
    successful = 0
    failed = 0

    for market, model in combinations:
        result = run_single_search(
            market_type=market,
            model_type=model,
            n_iter=args.n_iter,
            cv_splits=args.cv_splits,
            test_size=args.test_size,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            output_dir=args.output_dir,
        )

        if result is not None:
            results.append(result)
            successful += 1
        else:
            failed += 1

    # Update summary CSV
    if results:
        update_summary_csv(results, args.output_dir)

    # Update defaults config if requested
    if args.update_defaults:
        create_defaults_config(args.output_dir)

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info(f"SEARCH COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Successful: {successful}/{len(combinations)}")
    logger.info(f"Failed: {failed}/{len(combinations)}")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"{'='*80}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
