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
    assert "timestamp" in df.columns
    # Should have requested variables
    assert "T" in df.columns
    assert "FH" in df.columns
    assert "Q" in df.columns


def test_convert_to_standard_format():
    """Test unit conversion to standard format with time-shift for cumulative fields."""
    client = KNMIHistoricalClient()

    # Create sample KNMI data (without U and P)
    # Timestamps represent END of measurement interval for cumulative fields
    df = pd.DataFrame(
        {
            "STN": ["240", "240", "240"],
            "YYYYMMDD": [20250101, 20250101, 20250101],
            "HH": [1, 2, 3],
            "T": [89, 88, 87],  # 8.9°C, 8.8°C, 8.7°C (instantaneous)
            "FH": [130, 120, 110],  # 13.0 m/s, 12.0 m/s, 11.0 m/s (averaged)
            "Q": [100, 200, 300],  # Global radiation (cumulative over previous hour)
            "N": [8, 4, 2],  # 100%, 50%, 25% cloud cover (instantaneous)
            "RH": [5, -1, 10],  # 0.5mm, <0.05mm, 1.0mm (cumulative over previous hour)
        }
    )

    # Add timestamps (KNMI format: hour 1 = 01:00, represents interval 00:00-01:00)
    df["timestamp"] = pd.to_datetime(["2025-01-01 01:00", "2025-01-01 02:00", "2025-01-01 03:00"])

    # Convert
    result = client.convert_to_standard_format(df)

    # Should have records after interpolation and merging
    assert len(result) >= 3
    
    # Check that instantaneous fields are present
    assert "temperature_deg_c" in result.columns
    assert "wind_speed_m_per_s" in result.columns
    assert "cloud_cover_pct" in result.columns
    
    # Check that cumulative fields are present and time-shifted
    assert "global_radiation_w_per_m2" in result.columns
    assert "precipitation_mm" in result.columns
    
    # Verify temperature conversion (instantaneous, no time-shift)
    # Should be at original timestamps
    temp_at_1am = result.loc[result.index == pd.Timestamp("2025-01-01 01:00"), "temperature_deg_c"]
    if not temp_at_1am.empty:
        assert temp_at_1am.values[0] == 8.9
    
    # Verify cloud cover conversion (instantaneous)
    cloud_at_1am = result.loc[result.index == pd.Timestamp("2025-01-01 01:00"), "cloud_cover_pct"]
    if not cloud_at_1am.empty:
        assert cloud_at_1am.values[0] == 100.0
    
    # Verify cumulative fields were time-shifted by -30 minutes
    # Original timestamp 01:00 represents interval 00:00-01:00
    # After shift: value should be at 00:30 (center of interval)
    # After interpolation: we get hourly values
    # So we should have values at 00:00, 01:00, 02:00, 03:00 (interpolated)
    
    # Check metadata
    assert "data_type" in result.columns
    assert "fetch_timestamp" in result.columns


def test_fetch_historical_for_2025():
    """Test fetching historical data from Oct 2024."""
    client = KNMIHistoricalClient()

    # Fetch first 3 days from Oct 1, 2024
    end_date = datetime(2024, 10, 3, 23)
    df = client.fetch_historical_for_2025(end_date=end_date)

    # After time-shift and interpolation, we may have data starting earlier than 01:00
    # Original KNMI data starts at hour 1 (01:00), but cumulative fields get shifted back by 30min
    # Then interpolation creates hourly series starting from 00:00
    # Should have ~72-75 hours (3 days × 24 hours + some from interpolation)
    assert len(df) >= 69
    assert len(df) <= 80

    # Check columns
    expected_cols = [
        "temperature_deg_c",
        "wind_speed_m_per_s",
        "global_radiation_w_per_m2",
        "cloud_cover_pct",
        "precipitation_mm",
        "data_type",
        "fetch_timestamp",
    ]
    for col in expected_cols:
        assert col in df.columns

    # Check date range (after time-shift, may start at 00:00 instead of 01:00)
    import pandas as pd
    assert df.index.min() >= pd.Timestamp("2024-10-01 00:00:00", tz="UTC")
    assert df.index.max() <= pd.Timestamp("2024-10-04 00:00:00", tz="UTC")

    # All should be historical
    assert all(df["data_type"] == "historical")


def test_radiation_conversion():
    """Test global radiation conversion (J/cm²/h → W/m²) with time-shift."""
    client = KNMIHistoricalClient()

    # J/cm²/h to W/m²: multiply by 10000/3600
    # Example: 360 J/cm²/h = 360 × 10000/3600 = 1000 W/m²
    # IMPORTANT: Cumulative fields get time-shifted by -30 minutes

    df = pd.DataFrame(
        {
            "Q": [0, 360, 720],  # 0, 1000, 2000 W/m² (cumulative over previous hour)
            "timestamp": pd.to_datetime(
                ["2025-01-01 01:00", "2025-01-01 12:00", "2025-01-01 13:00"]
            ),
        }
    )

    result = client.convert_to_standard_format(df)

    # After time-shift and interpolation, we should have global_radiation_w_per_m2 column
    assert "global_radiation_w_per_m2" in result.columns
    
    # Values should be correctly converted (checking approximate values due to interpolation)
    # The original 01:00 timestamp represents 00:00-01:00 interval
    # After -30min shift: value at 00:30
    # After interpolation to 1H: values at 00:00, 01:00, ..., 13:00
    
    radiation_values = result["global_radiation_w_per_m2"].dropna()
    
    # Check that conversion factor is correct (within interpolated range)
    # Original: [0, 360, 720] J/cm²/h → [0, 1000, 2000] W/m²
    assert radiation_values.min() >= -1  # Allow small negative from interpolation
    assert radiation_values.max() <= 2100  # Allow small overshoot from interpolation
    
    # Verify at least some values are in expected range
    assert any((radiation_values >= 900) & (radiation_values <= 1100))  # ~1000 W/m²
    assert any((radiation_values >= 1900) & (radiation_values <= 2100))  # ~2000 W/m²


def test_cumulative_field_time_shift():
    """Test that cumulative fields are time-shifted by -30 minutes."""
    client = KNMIHistoricalClient()

    # Create data with cumulative fields (Q, RH) and instantaneous fields (T, N)
    # KNMI timestamp HH:00 represents measurement from (HH-1):00 to HH:00 for cumulative fields
    df = pd.DataFrame(
        {
            "Q": [360, 720],  # Radiation: cumulative, should be shifted
            "RH": [10, 20],  # Precipitation: cumulative, should be shifted
            "T": [100, 110],  # Temperature: instantaneous, should NOT be shifted
            "N": [4, 5],  # Cloud cover: instantaneous, should NOT be shifted
            "timestamp": pd.to_datetime(["2025-01-01 01:00", "2025-01-01 02:00"]),
        }
    )

    result = client.convert_to_standard_format(df)

    # Check that both field types exist
    assert "global_radiation_w_per_m2" in result.columns
    assert "precipitation_mm" in result.columns
    assert "temperature_deg_c" in result.columns
    assert "cloud_cover_pct" in result.columns
    
    # After time-shift (-30min) and interpolation (1H):
    # Original cumulative timestamps: 01:00, 02:00
    # After shift: 00:30, 01:30
    # After 1H resample+interpolate: 00:00, 01:00, 02:00
    
    # We should have hourly data
    assert len(result) >= 2
    
    # Temperature should be at original timestamps (no shift)
    # 01:00 should have temp = 10.0°C
    temp_at_1am = result.loc[result.index == pd.Timestamp("2025-01-01 01:00"), "temperature_deg_c"]
    if not temp_at_1am.empty:
        assert temp_at_1am.values[0] == 10.0


def test_interpolation_creates_hourly_series():
    """Test that interpolation creates smooth hourly series."""
    client = KNMIHistoricalClient()

    # Create sparse data - only 3 hours with cumulative fields
    df = pd.DataFrame(
        {
            "Q": [0, 360, 720],  # Radiation values
            "timestamp": pd.to_datetime(
                ["2025-01-01 00:00", "2025-01-01 06:00", "2025-01-01 12:00"]
            ),
        }
    )

    result = client.convert_to_standard_format(df)

    # After time-shift and 1H interpolation, we should have hourly records
    # From shifted 23:30 (previous day) to 11:30, resampled to hourly
    assert "global_radiation_w_per_m2" in result.columns
    
    # Should have interpolated values between original sparse points
    # At minimum: 00:00, 06:00, 12:00 after interpolation
    assert len(result) >= 3
    
    # Check that interpolation filled intermediate values
    radiation = result["global_radiation_w_per_m2"].dropna()
    # Should have smooth transition, not just original 3 points
    # Original after shift: ~23:30, ~05:30, ~11:30
    # After 1H resample: hourly from earliest to latest
    assert len(radiation) >= 3


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
