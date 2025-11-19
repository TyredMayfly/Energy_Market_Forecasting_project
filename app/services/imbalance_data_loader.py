"""
Imbalance Data Loader Service

This module handles loading and combining imbalance price data from multiple CSV files
in the imbalance_data/ directory into a unified dataset.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_and_combine_imbalance_csvs(source_dir: Path) -> pd.DataFrame:
    """
    Load and combine all imbalance CSV files from the source directory.

    Args:
        source_dir: Path to directory containing imbalance CSV files

    Returns:
        Combined DataFrame with all imbalance data, sorted by timestamp

    Raises:
        ValueError: If no CSV files found or data cannot be loaded
    """
    csv_files = list(source_dir.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {source_dir}")

    logger.info(f"Found {len(csv_files)} CSV files to combine")

    dataframes = []

    for csv_file in csv_files:
        try:
            logger.debug(f"Loading {csv_file.name}")

            # Read CSV with semicolon separator (TenneT format)
            df = pd.read_csv(csv_file, sep=";", parse_dates=["Timeinterval Start Loc"])

            # Standardize column names to snake_case
            df = df.rename(
                columns={
                    "Timeinterval Start Loc": "timestamp_utc",
                    "Price Shortage": "shortage_price",
                    "Price Surplus": "surplus_price",
                    "Regulation State": "regulation_state",
                }
            )

            # Keep only required columns
            required_cols = ["timestamp_utc", "shortage_price", "surplus_price", "regulation_state"]
            df = df[required_cols]

            dataframes.append(df)

        except Exception as e:
            logger.warning(f"Error loading {csv_file.name}: {e}")
            continue

    if not dataframes:
        raise ValueError("Failed to load any CSV files")

    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Remove duplicates based on timestamp
    original_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=["timestamp_utc"], keep="first")
    duplicates_removed = original_count - len(combined_df)

    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate timestamps")

    # Sort by timestamp
    combined_df = combined_df.sort_values("timestamp_utc").reset_index(drop=True)

    # Convert prices to numeric, handling any non-numeric values
    for price_col in ["shortage_price", "surplus_price"]:
        combined_df[price_col] = pd.to_numeric(combined_df[price_col], errors="coerce")

    # Map regulation state to standard codes if needed
    # Expected values: 1 (UP), -1 (DOWN), 2 (UP_AND_DOWN), 0 (BALANCED)
    # Standardize text values to numeric
    state_mapping = {
        "UP": 1,
        "DOWN": -1,
        "UP_AND_DOWN": 2,
        "BALANCED": 0,
        1: 1,
        -1: -1,
        2: 2,
        0: 0,
    }

    combined_df["regulation_state"] = combined_df["regulation_state"].map(
        lambda x: state_mapping.get(x, x) if pd.notna(x) else x
    )

    logger.info(f"Combined {len(combined_df)} records from {len(dataframes)} files")
    logger.info(
        f"Date range: {combined_df['timestamp_utc'].min()} to {combined_df['timestamp_utc'].max()}"
    )

    return combined_df


def build_unified_imbalance_dataset(
    source_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    output_filename: str = "imbalance_unified.csv",
) -> Path:
    """
    Build unified imbalance dataset from source CSVs and save to data directory.

    Args:
        source_dir: Directory containing source CSV files (defaults to imbalance_data/)
        output_dir: Directory to save unified dataset (defaults to data/)
        output_filename: Name of output file

    Returns:
        Path to saved unified dataset

    Raises:
        ValueError: If data cannot be loaded or saved
    """
    # Default paths relative to project root
    if source_dir is None:
        project_root = Path(__file__).parent.parent.parent
        source_dir = project_root / "imbalance_data"

    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "data"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building unified imbalance dataset from {source_dir}")

    # Load and combine data
    combined_df = load_and_combine_imbalance_csvs(source_dir)

    # Save to CSV
    output_path = output_dir / output_filename
    combined_df.to_csv(output_path, index=False)

    logger.info(f"Saved unified dataset to {output_path}")
    logger.info(f"Dataset shape: {combined_df.shape}")
    logger.info(f"Columns: {list(combined_df.columns)}")

    # Log summary statistics
    logger.info("\nDataset Summary:")
    logger.info(
        f"  Shortage price: {combined_df['shortage_price'].min():.2f} to {combined_df['shortage_price'].max():.2f} EUR/MWh"
    )
    logger.info(
        f"  Surplus price: {combined_df['surplus_price'].min():.2f} to {combined_df['surplus_price'].max():.2f} EUR/MWh"
    )
    logger.info(f"  Regulation states: {combined_df['regulation_state'].value_counts().to_dict()}")

    return output_path


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Build dataset
    output_path = build_unified_imbalance_dataset()
    print(f"\n✅ Unified dataset created: {output_path}")
