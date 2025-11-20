"""
Script to fetch historical data for Q4 2024 (October, November, December).

This script fetches:
1. ENTSO-E day-ahead market prices
2. ENTSO-E imbalance data  
3. KNMI weather data

for the period October 1, 2024 to December 31, 2024.
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.entsoe_client import EntsoeClient
from app.services.knmi_historical_client import KNMIHistoricalClient
from app.services.data_store import save_market_data, append_market_data, save_weather_data, append_weather_data, load_market_data, load_weather_data
from app.core.logging import get_logger

logger = get_logger(__name__)


def fetch_q4_2024_data():
    """Fetch all Q4 2024 data (October, November, December)."""
    
    logger.info("=" * 80)
    logger.info("Fetching Q4 2024 Historical Data")
    logger.info("=" * 80)
    
    # Define date range for Q4 2024
    start_date = datetime(2024, 10, 1, 0, 0, 0)
    end_date = datetime(2025, 1, 1, 0, 0, 0)  # Up to Jan 1, 2025 00:00
    
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info("")
    
    # ========== ENTSO-E Day-Ahead Prices ==========
    logger.info("--- Fetching ENTSO-E Day-Ahead Prices ---")
    try:
        entsoe_client = EntsoeClient()
        df_day_ahead = entsoe_client.fetch_day_ahead_prices_2025_nl(start_date, end_date)
        
        if not df_day_ahead.empty:
            logger.info(f"✓ Fetched {len(df_day_ahead)} day-ahead records")
            logger.info(f"  Date range: {df_day_ahead['timestamp_utc'].min()} to {df_day_ahead['timestamp_utc'].max()}")
            
            # Load existing data and append
            df_existing = load_market_data("day_ahead")
            if not df_existing.empty:
                logger.info(f"  Existing records: {len(df_existing)}")
                df_combined = pd.concat([df_day_ahead, df_existing], ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset=['timestamp_utc'], keep='first')
                df_combined = df_combined.sort_values('timestamp_utc').reset_index(drop=True)
                save_market_data(df_combined, "day_ahead")
                logger.info(f"  Total after merge: {len(df_combined)} records")
            else:
                save_market_data(df_day_ahead, "day_ahead")
                logger.info(f"  Saved {len(df_day_ahead)} records")
        else:
            logger.warning("✗ No day-ahead data fetched")
            
    except Exception as e:
        logger.error(f"✗ Error fetching day-ahead data: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("")
    
    # ========== ENTSO-E Imbalance Data ==========
    logger.info("--- Fetching ENTSO-E Imbalance Data ---")
    try:
        df_imbalance = entsoe_client.fetch_imbalance_data_2025_nl(start_date, end_date)
        
        if not df_imbalance.empty:
            logger.info(f"✓ Fetched {len(df_imbalance)} imbalance records")
            logger.info(f"  Date range: {df_imbalance['timestamp_utc'].min()} to {df_imbalance['timestamp_utc'].max()}")
            
            # For imbalance data, we need to update the unified CSV
            # Load existing and merge
            df_existing = load_market_data("imbalance_shortage")
            if not df_existing.empty:
                logger.info(f"  Existing records: {len(df_existing)}")
                # Combine with new data
                df_combined = pd.concat([df_imbalance, df_existing], ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset=['timestamp_utc'], keep='first')
                df_combined = df_combined.sort_values('timestamp_utc').reset_index(drop=True)
                
                # Save to all three market types (they share the same file)
                save_market_data(df_combined, "imbalance_shortage")
                logger.info(f"  Total after merge: {len(df_combined)} records")
            else:
                save_market_data(df_imbalance, "imbalance_shortage")
                logger.info(f"  Saved {len(df_imbalance)} records")
        else:
            logger.warning("✗ No imbalance data fetched")
            
    except Exception as e:
        logger.error(f"✗ Error fetching imbalance data: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("")
    
    # ========== KNMI Weather Data ==========
    logger.info("--- Fetching KNMI Historical Weather Data ---")
    try:
        knmi_client = KNMIHistoricalClient()
        
        # Fetch hourly data for Q4 2024
        df_weather = knmi_client.fetch_hourly_data(
            start_date=start_date,
            end_date=end_date,
            variables="T:FH:Q:N:RH"  # Temperature, wind, radiation, cloud cover, precipitation
        )
        
        if not df_weather.empty:
            # Convert to standard format (returns DataFrame with timestamp as index)
            df_weather_standard = knmi_client.convert_to_standard_format(df_weather)
            
            logger.info(f"✓ Fetched {len(df_weather_standard)} hourly weather records")
            logger.info(f"  Date range: {df_weather_standard.index.min()} to {df_weather_standard.index.max()}")
            
            # Reset index to get timestamp as column
            # The index name is 'timestamp', so reset_index will create a 'timestamp' column
            df_weather_standard = df_weather_standard.reset_index()
            
            # Load existing and merge
            df_existing = load_weather_data()
            if not df_existing.empty:
                logger.info(f"  Existing records: {len(df_existing)}")
                df_combined = pd.concat([df_weather_standard, df_existing], ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='first')
                df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
                save_weather_data(df_combined)
                logger.info(f"  Total after merge: {len(df_combined)} records")
            else:
                save_weather_data(df_weather_standard)
                logger.info(f"  Saved {len(df_weather_standard)} records")
        else:
            logger.warning("✗ No weather data fetched")
            
    except Exception as e:
        logger.error(f"✗ Error fetching weather data: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Q4 2024 Data Fetch Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    fetch_q4_2024_data()
