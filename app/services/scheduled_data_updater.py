"""
Scheduled Data Updater Service

This module handles scheduled updates of imbalance settlement prices from TenneT API.
Updates run daily at 02:00 AM to stay within the 25 requests/day rate limit.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.services.tennet_client import TennetClient, TennetApiError

logger = logging.getLogger(__name__)


class ScheduledDataUpdater:
    """Service for scheduled imbalance data updates."""

    def __init__(self):
        """Initialize the scheduled updater."""
        self.scheduler = BackgroundScheduler()
        self.tennet_client = TennetClient()
        self.data_file = Path(__file__).parent.parent.parent / "data" / "imbalance_unified.csv"
        
    def fetch_and_update_imbalance_data(self):
        """
        Fetch latest settlement prices and update the unified dataset.
        
        Fetches data from the last available timestamp up to now,
        up to 24 hours per day.
        """
        try:
            logger.info("Starting scheduled imbalance data update...")
            
            # Load existing data to find the last timestamp
            if not self.data_file.exists():
                logger.warning(f"Data file not found: {self.data_file}")
                return
            
            df_existing = pd.read_csv(self.data_file, parse_dates=["timestamp_utc"])
            
            # Ensure timezone awareness
            if df_existing["timestamp_utc"].dt.tz is None:
                df_existing["timestamp_utc"] = pd.to_datetime(df_existing["timestamp_utc"], utc=True)
            
            last_timestamp = df_existing["timestamp_utc"].max()
            logger.info(f"Last available data: {last_timestamp}")
            
            # Calculate fetch range (last timestamp + 15min to now)
            start = last_timestamp + timedelta(minutes=15)
            end = datetime.now(timezone.utc)
            
            # Limit to 24 hours (full day)
            max_duration = timedelta(hours=24)
            if end - start > max_duration:
                logger.info(f"Limiting fetch to {max_duration.total_seconds() / 3600} hours")
                end = start + max_duration
            
            logger.info(f"Fetching new data from {start} to {end}")
            
            # Fetch new data
            df_new = self.tennet_client.fetch_settlement_prices_from_to(start, end)
            
            if df_new.empty:
                logger.info("No new data available")
                return
            
            logger.info(f"Fetched {len(df_new)} new records")
            
            # Combine with existing data
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=["timestamp_utc"])
            df_combined = df_combined.sort_values("timestamp_utc").reset_index(drop=True)
            
            # Save updated dataset
            df_combined.to_csv(self.data_file, index=False)
            
            logger.info(f"✓ Updated imbalance data: {len(df_combined)} total records")
            logger.info(f"  New date range: {df_combined['timestamp_utc'].min()} to {df_combined['timestamp_utc'].max()}")
            logger.info(f"  Added {len(df_combined) - len(df_existing)} new records")
            
        except TennetApiError as e:
            logger.error(f"TenneT API error during scheduled update: {e}")
        except Exception as e:
            logger.error(f"Error during scheduled imbalance data update: {e}", exc_info=True)
    
    def start(self):
        """
        Start the scheduled updater.
        
        Schedules daily updates at 02:00 AM.
        Also runs an initial update on startup if data is stale.
        """
        # Schedule daily update at 02:00 AM
        self.scheduler.add_job(
            self.fetch_and_update_imbalance_data,
            trigger=CronTrigger(hour=2, minute=0),  # Daily at 02:00
            id="daily_imbalance_update",
            name="Daily Imbalance Data Update",
            replace_existing=True,
        )
        
        logger.info("Scheduled imbalance data updater configured:")
        logger.info("  - Daily updates at 02:00 AM")
        logger.info("  - Fetches up to 24 hours of data per day")
        
        # Check if we should run an initial update
        self._check_and_run_initial_update()
        
        # Start the scheduler
        self.scheduler.start()
        logger.info("✓ Scheduled updater started")
    
    def _check_and_run_initial_update(self):
        """Check if data is stale and run an initial update if needed."""
        try:
            if not self.data_file.exists():
                logger.info("No existing data file - skipping initial update")
                return
            
            df_existing = pd.read_csv(self.data_file, parse_dates=["timestamp_utc"])
            if df_existing["timestamp_utc"].dt.tz is None:
                df_existing["timestamp_utc"] = pd.to_datetime(df_existing["timestamp_utc"], utc=True)
            
            last_timestamp = df_existing["timestamp_utc"].max()
            age = datetime.now(timezone.utc) - last_timestamp
            
            # If data is more than 26 hours old, run an update now
            if age > timedelta(hours=26):
                logger.info(f"Data is {age.total_seconds() / 3600:.1f} hours old - running initial update")
                self.fetch_and_update_imbalance_data()
            else:
                logger.info(f"Data is fresh ({age.total_seconds() / 3600:.1f} hours old) - no initial update needed")
                
        except Exception as e:
            logger.warning(f"Could not check data freshness: {e}")
    
    def stop(self):
        """Stop the scheduled updater."""
        self.scheduler.shutdown()
        logger.info("Scheduled updater stopped")


# Global instance
_updater_instance = None


def get_scheduled_updater() -> ScheduledDataUpdater:
    """
    Get or create the global scheduled updater instance.
    
    Returns:
        ScheduledDataUpdater instance
    """
    global _updater_instance
    if _updater_instance is None:
        _updater_instance = ScheduledDataUpdater()
    return _updater_instance


def start_scheduled_updates():
    """Start the scheduled data updater (convenience function)."""
    updater = get_scheduled_updater()
    updater.start()
    return updater


def stop_scheduled_updates():
    """Stop the scheduled data updater (convenience function)."""
    global _updater_instance
    if _updater_instance is not None:
        _updater_instance.stop()
        _updater_instance = None
