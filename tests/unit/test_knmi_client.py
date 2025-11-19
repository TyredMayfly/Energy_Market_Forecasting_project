"""
Unit tests for KNMI Historical Weather Client
"""

import pytest
import pandas as pd
from datetime import datetime
from app.services.knmi_historical_client import KNMIHistoricalClient


def test_knmi_client_initialization():
    """Test client initialization."""
    client = KNMIHistoricalClient()
    assert client.station == "240"  # Default Schiphol
    
    client_custom = KNMIHistoricalClient(station="260")  # De Bilt
    assert client_custom.station == "260"


def test_fetch_hourly_data():
    """Test fetching hourly data for a small period."""
    client = KNMIHistoricalClient()
    
    # Fetch just one day
    start = datetime(2025, 1, 1, 1)
    end = datetime(2025, 1, 2, 0)  # Next day midnight
    
    df = client.fetch_hourly_data(start, end, variables="T:FH:Q")
    
    # Should have data
    assert len(df) > 0
    # Should have timestamp column
    assert 'timestamp' in df.columns
    # Should have requested variables
    assert 'T' in df.columns
    assert 'FH' in df.columns
    assert 'Q' in df.columns


def test_convert_to_standard_format():
    """Test unit conversion to standard format."""
    client = KNMIHistoricalClient()
    
    # Create sample KNMI data (without U and P)
    df = pd.DataFrame({
        'STN': ['240', '240'],
        'YYYYMMDD': [20250101, 20250101],
        'HH': [1, 2],
        'T': [89, 88],  # 8.9°C, 8.8°C
        'FH': [130, 120],  # 13.0 m/s, 12.0 m/s
        'Q': [100, 200],  # Global radiation
        'N': [8, 4],  # 100%, 50% cloud cover
        'RH': [5, -1],  # 0.5mm, <0.05mm
    })
    
    # Add timestamps
    df['timestamp'] = pd.to_datetime(['2025-01-01 01:00', '2025-01-01 02:00'])
    
    # Convert
    result = client.convert_to_standard_format(df)
    
    # Check conversions
    assert result.loc[result.index[0], 'temperature_deg_c'] == 8.9
    assert result.loc[result.index[0], 'wind_speed_m_per_s'] == 13.0
    assert result.loc[result.index[0], 'cloud_cover_pct'] == 100.0
    assert result.loc[result.index[0], 'precipitation_mm'] == 0.5
    
    # Second row
    assert result.loc[result.index[1], 'cloud_cover_pct'] == 50.0
    assert result.loc[result.index[1], 'precipitation_mm'] == 0.0  # -1 → 0
    
    # Check metadata
    assert all(result['data_type'] == 'historical')
    assert 'fetch_timestamp' in result.columns


def test_fetch_historical_for_2025():
    """Test fetching historical data for 2025."""
    client = KNMIHistoricalClient()
    
    # Fetch first 3 days of 2025
    end_date = datetime(2025, 1, 3, 23)
    df = client.fetch_historical_for_2025(end_date=end_date)
    
    # Should have ~69-72 hours (3 days × 24 hours, minus hour 0)
    assert len(df) >= 69
    assert len(df) <= 75
    
    # Check columns
    expected_cols = [
        'temperature_deg_c', 'wind_speed_m_per_s', 'global_radiation_w_per_m2',
        'cloud_cover_pct', 'precipitation_mm',
        'data_type', 'fetch_timestamp'
    ]
    for col in expected_cols:
        assert col in df.columns
    
    # Check date range
    assert df.index.min() >= datetime(2025, 1, 1, 1)
    assert df.index.max() <= datetime(2025, 1, 4, 0)
    
    # All should be historical
    assert all(df['data_type'] == 'historical')


def test_radiation_conversion():
    """Test global radiation conversion (J/cm²/h → W/m²)."""
    client = KNMIHistoricalClient()
    
    # J/cm²/h to W/m²: multiply by 10000/3600
    # Example: 360 J/cm²/h = 360 × 10000/3600 = 1000 W/m²
    
    df = pd.DataFrame({
        'Q': [0, 360, 720],  # 0, 1000, 2000 W/m²
        'timestamp': pd.to_datetime(['2025-01-01 00:00', '2025-01-01 12:00', '2025-01-01 13:00'])
    })
    
    result = client.convert_to_standard_format(df)
    
    assert abs(result.loc[result.index[0], 'global_radiation_w_per_m2'] - 0) < 0.1
    assert abs(result.loc[result.index[1], 'global_radiation_w_per_m2'] - 1000) < 0.1
    assert abs(result.loc[result.index[2], 'global_radiation_w_per_m2'] - 2000) < 0.1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
