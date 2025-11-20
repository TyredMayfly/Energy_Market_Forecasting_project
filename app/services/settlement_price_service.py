"""
Settlement Price Service

Provides hybrid local + TenneT API access to settlement prices.
Automatically fetches fresh data from TenneT when local data is stale (>15 minutes old).
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union
from decimal import Decimal

import pandas as pd
import pytz

from app.core.config import settings
from app.services.tennet_client import TennetApiError, get_tennet_client

logger = logging.getLogger(__name__)


class SettlementPriceUnavailableError(Exception):
    """Exception raised when settlement price cannot be obtained from any source."""

    pass


class SettlementPriceService:
    """Service for accessing settlement prices with automatic API refresh."""

    # Freshness threshold: if local data is older than this, fetch from API
    FRESHNESS_THRESHOLD = timedelta(minutes=15)

    def __init__(
        self,
        local_data_path: Optional[Path] = None,
        timezone: Optional[str] = None,
    ):
        """
        Initialize settlement price service.

        Args:
            local_data_path: Path to local settlement price CSV/data file
                           (defaults to data/imbalance_unified.csv)
            timezone: Timezone for datetime operations (defaults to settings.timezone)
        """
        if local_data_path is None:
            self.local_data_path = settings.data_dir / "imbalance_unified.csv"
        else:
            self.local_data_path = Path(local_data_path)

        self.timezone = pytz.timezone(timezone or settings.timezone)
        self.tennet_client = get_tennet_client()

    def _load_local_data(self) -> pd.DataFrame:
        """
        Load settlement price data from local storage.

        Returns:
            DataFrame with local settlement price data, empty if not available

        Note:
            All timestamps are converted to UTC for consistency
        """
        try:
            if not self.local_data_path.exists():
                logger.warning(f"Local settlement data file not found: {self.local_data_path}")
                return pd.DataFrame()

            # Load local CSV
            df = pd.read_csv(self.local_data_path, parse_dates=["timestamp_utc"])

            # Ensure timestamp is timezone-aware UTC
            if df["timestamp_utc"].dt.tz is None:
                df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
            else:
                df["timestamp_utc"] = df["timestamp_utc"].dt.tz_convert("UTC")

            # Ensure required columns exist
            required_cols = ["timestamp_utc", "shortage_price", "surplus_price"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Local data missing required columns: {missing_cols}")
                return pd.DataFrame()

            logger.debug(
                f"Loaded {len(df)} local records "
                f"({df['timestamp_utc'].min()} to {df['timestamp_utc'].max()})"
            )

            return df

        except Exception as e:
            logger.error(f"Error loading local settlement data: {e}")
            return pd.DataFrame()

    def _save_local_data(self, df: pd.DataFrame) -> bool:
        """
        Save settlement price data to local storage.

        Args:
            df: DataFrame to save (will be appended to existing data)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing data
            existing_df = self._load_local_data()

            if not existing_df.empty:
                # Combine with new data
                combined_df = pd.concat([existing_df, df], ignore_index=True)

                # Remove duplicates (keep newest)
                combined_df = combined_df.drop_duplicates(
                    subset=["timestamp_utc"], keep="last"
                )

                # Sort by timestamp
                combined_df = combined_df.sort_values("timestamp_utc").reset_index(drop=True)
            else:
                combined_df = df

            # Ensure directory exists
            self.local_data_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to CSV
            combined_df.to_csv(self.local_data_path, index=False)

            logger.info(f"Saved {len(df)} new settlement price records to local storage")
            logger.info(f"Total local records: {len(combined_df)}")

            return True

        except Exception as e:
            logger.error(f"Error saving settlement data to local storage: {e}")
            return False

    def _get_current_time(self, now: Optional[datetime] = None) -> datetime:
        """
        Get current time as timezone-aware datetime in UTC.

        Args:
            now: Optional current time (for testing), will be converted to UTC

        Returns:
            Timezone-aware datetime in UTC
        """
        if now is None:
            # Get current time in configured timezone, then convert to UTC
            now = datetime.now(self.timezone)

        # Ensure timezone-aware
        if now.tzinfo is None:
            now = self.timezone.localize(now)

        # Convert to UTC
        return now.astimezone(pytz.UTC)

    def get_latest_settlement_price(
        self,
        now: Optional[datetime] = None,
        price_type: str = "shortage",
    ) -> Union[float, Decimal]:
        """
        Get the latest settlement price with 15-minute freshness check.

        Implements hybrid local + API strategy:
        1. Check local data freshness
        2. If fresh (<= 15 min old), return local price
        3. If stale (> 15 min old) or missing, fetch from TenneT API
        4. Update local storage with API data
        5. Fall back to stale local data if API fails

        Args:
            now: Current time (defaults to datetime.now in configured timezone)
            price_type: Type of price to return: "shortage" or "surplus"

        Returns:
            Latest settlement price as float or Decimal

        Raises:
            SettlementPriceUnavailableError: If no price available from any source
        """
        if price_type not in ["shortage", "surplus"]:
            raise ValueError(f"Invalid price_type: {price_type}. Must be 'shortage' or 'surplus'")

        current_time = self._get_current_time(now)
        price_column = f"{price_type}_price"

        logger.info(f"Getting latest {price_type} settlement price (current time: {current_time})")

        # Load local data
        local_df = self._load_local_data()

        # Determine local data status
        latest_local_ts = None
        latest_local_price = None

        if not local_df.empty and price_column in local_df.columns:
            # Get latest non-null price
            valid_prices = local_df[local_df[price_column].notna()]
            if not valid_prices.empty:
                latest_local_ts = valid_prices["timestamp_utc"].max()
                latest_local_price = valid_prices.loc[
                    valid_prices["timestamp_utc"] == latest_local_ts, price_column
                ].iloc[0]

                logger.info(
                    f"Latest local data: {latest_local_ts} "
                    f"({price_type} price: {latest_local_price:.2f} EUR/MWh)"
                )

        # Freshness check
        if latest_local_ts is not None:
            age = current_time - latest_local_ts
            logger.info(f"Local data age: {age.total_seconds() / 60:.1f} minutes")

            if age <= self.FRESHNESS_THRESHOLD:
                logger.info(
                    f"Local data is fresh (age: {age.total_seconds() / 60:.1f} min "
                    f"<= threshold: {self.FRESHNESS_THRESHOLD.total_seconds() / 60} min)"
                )
                return float(latest_local_price)

            logger.info(
                f"Local data is stale (age: {age.total_seconds() / 60:.1f} min "
                f"> threshold: {self.FRESHNESS_THRESHOLD.total_seconds() / 60} min)"
            )
        else:
            logger.info("No local settlement price data available")

        # Fetch from TenneT API
        try:
            logger.info("Fetching fresh settlement prices from TenneT API")
            api_df = self.tennet_client.fetch_latest_settlement_prices(limit=100)

            if api_df.empty:
                logger.warning("TenneT API returned no data")
                # Fall back to local data if available
                if latest_local_price is not None:
                    logger.info("Falling back to stale local price")
                    return float(latest_local_price)
                else:
                    raise SettlementPriceUnavailableError(
                        "No settlement price data available from API or local storage"
                    )

            # Find newest price from API
            if price_column not in api_df.columns:
                logger.error(f"API data missing {price_column} column")
                if latest_local_price is not None:
                    logger.info("Falling back to stale local price")
                    return float(latest_local_price)
                else:
                    raise SettlementPriceUnavailableError(
                        f"API data missing {price_column} column and no local fallback"
                    )

            # Get latest valid price from API
            valid_api_prices = api_df[api_df[price_column].notna()]
            if valid_api_prices.empty:
                logger.warning(f"No valid {price_type} prices in API response")
                if latest_local_price is not None:
                    logger.info("Falling back to stale local price")
                    return float(latest_local_price)
                else:
                    raise SettlementPriceUnavailableError(
                        f"No valid {price_type} prices in API response and no local fallback"
                    )

            latest_api_ts = valid_api_prices["timestamp_utc"].max()
            latest_api_price = valid_api_prices.loc[
                valid_api_prices["timestamp_utc"] == latest_api_ts, price_column
            ].iloc[0]

            logger.info(
                f"Latest API data: {latest_api_ts} "
                f"({price_type} price: {latest_api_price:.2f} EUR/MWh)"
            )

            # Check if API data is actually newer
            if latest_local_ts is not None and latest_api_ts <= latest_local_ts:
                logger.info("API data is not newer than local data")
                return float(latest_local_price)

            # Update local storage with new API data
            # Only save records newer than what we have locally
            if latest_local_ts is not None:
                new_records = api_df[api_df["timestamp_utc"] > latest_local_ts]
            else:
                new_records = api_df

            if not new_records.empty:
                logger.info(f"Updating local storage with {len(new_records)} new records")
                self._save_local_data(new_records)

            return float(latest_api_price)

        except TennetApiError as e:
            logger.warning(f"TenneT API error: {e}")

            # Fall back to local data if available
            if latest_local_price is not None:
                logger.info("Falling back to stale local price due to API error")
                return float(latest_local_price)
            else:
                raise SettlementPriceUnavailableError(
                    f"TenneT API failed ({e}) and no local data available"
                )

        except Exception as e:
            logger.error(f"Unexpected error fetching settlement price: {e}")

            # Fall back to local data if available
            if latest_local_price is not None:
                logger.info("Falling back to stale local price due to unexpected error")
                return float(latest_local_price)
            else:
                raise SettlementPriceUnavailableError(
                    f"Unexpected error ({e}) and no local data available"
                )

    def get_settlement_prices_between(
        self,
        start: datetime,
        end: datetime,
        ensure_fresh: bool = True,
    ) -> pd.DataFrame:
        """
        Get settlement prices for a specific time range.

        Args:
            start: Start datetime (timezone-aware)
            end: End datetime (timezone-aware)
            ensure_fresh: If True, check freshness and update from API if needed

        Returns:
            DataFrame with settlement prices in the specified range
            Columns: timestamp_utc, shortage_price, surplus_price, regulation_state

        Raises:
            ValueError: If start/end are not timezone-aware
        """
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end datetimes must be timezone-aware")

        # Convert to UTC
        start_utc = start.astimezone(pytz.UTC)
        end_utc = end.astimezone(pytz.UTC)

        logger.info(f"Getting settlement prices from {start_utc} to {end_utc}")

        # Load local data
        local_df = self._load_local_data()

        # Filter to requested range
        if not local_df.empty:
            range_df = local_df[
                (local_df["timestamp_utc"] >= start_utc) & (local_df["timestamp_utc"] <= end_utc)
            ].copy()
        else:
            range_df = pd.DataFrame()

        # Check if we need to refresh from API
        if ensure_fresh:
            # Trigger freshness check by calling get_latest_settlement_price
            try:
                self.get_latest_settlement_price(now=end)
                # Reload data after potential update
                local_df = self._load_local_data()
                range_df = local_df[
                    (local_df["timestamp_utc"] >= start_utc)
                    & (local_df["timestamp_utc"] <= end_utc)
                ].copy()
            except Exception as e:
                logger.warning(f"Could not ensure fresh data: {e}")

        logger.info(f"Returning {len(range_df)} settlement price records for requested range")
        return range_df


# Module-level instance for convenience
_service_instance = None


def get_settlement_price_service() -> SettlementPriceService:
    """
    Get singleton settlement price service instance.

    Returns:
        Configured SettlementPriceService instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = SettlementPriceService()
    return _service_instance
