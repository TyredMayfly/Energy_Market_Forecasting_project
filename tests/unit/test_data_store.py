"""
Tests for data persistence layer.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from app.services.data_store import (
    save_market_data,
    load_market_data,
    append_market_data,
    save_weather_data,
    load_weather_data,
    append_weather_data,
    get_data_summary,
)


class TestMarketDataSave:
    """Test suite for saving market data."""

    def test_save_market_data_success(self, mock_env_vars, sample_market_data):
        """Test successful market data save."""
        result = save_market_data(sample_market_data, "day_ahead")

        assert result is True

    def test_save_market_data_invalid_type(self, mock_env_vars, sample_market_data):
        """Test saving with invalid market type."""
        result = save_market_data(sample_market_data, "invalid_type")

        assert result is False

    def test_save_market_data_creates_file(self, mock_env_vars, test_data_dir, sample_market_data):
        """Test that save creates the data file."""
        save_market_data(sample_market_data, "day_ahead")

        # Check file exists
        expected_file = test_data_dir / "entsoe_day_ahead_prices_2025.csv"
        assert expected_file.exists()

    def test_save_market_data_preserves_timestamps(self, mock_env_vars, sample_market_data):
        """Test that timestamps are properly formatted on save with UTC timezone."""
        save_market_data(sample_market_data, "day_ahead")

        # Load and verify
        loaded = load_market_data("day_ahead")
        assert loaded["timestamp_utc"].dt.tz is not None
        assert str(loaded["timestamp_utc"].dt.tz) == "UTC"

    def test_save_all_market_types(self, mock_env_vars, sample_market_data):
        """Test saving all supported market types."""
        for market_type in ["day_ahead"]:
            result = save_market_data(sample_market_data, market_type)
            assert result is True


class TestMarketDataLoad:
    """Test suite for loading market data."""

    def test_load_market_data_success(self, mock_env_vars, sample_market_data):
        """Test successful market data load."""
        save_market_data(sample_market_data, "day_ahead")

        loaded = load_market_data("day_ahead")

        assert loaded is not None
        assert not loaded.empty
        assert len(loaded) == len(sample_market_data)

    def test_load_market_data_invalid_type(self, mock_env_vars):
        """Test loading with invalid market type."""
        loaded = load_market_data("invalid_type")

        assert loaded is None

    def test_load_market_data_nonexistent_file(self, mock_env_vars, test_data_dir):
        """Test loading when file doesn't exist."""
        # Ensure file doesn't exist
        file_path = test_data_dir / "entsoe_day_ahead_prices_2025.csv"
        if file_path.exists():
            file_path.unlink()

        loaded = load_market_data("day_ahead")

        assert loaded is None

    def test_load_market_data_parses_timestamps(self, mock_env_vars, sample_market_data):
        """Test that timestamps are properly parsed on load."""
        save_market_data(sample_market_data, "day_ahead")

        loaded = load_market_data("day_ahead")

        assert "timestamp_utc" in loaded.columns
        assert pd.api.types.is_datetime64_any_dtype(loaded["timestamp_utc"])

    def test_load_market_data_preserves_columns(self, mock_env_vars, sample_market_data):
        """Test that all columns are preserved."""
        save_market_data(sample_market_data, "day_ahead")

        loaded = load_market_data("day_ahead")

        assert set(loaded.columns) == set(sample_market_data.columns)


class TestMarketDataAppend:
    """Test suite for appending market data."""

    def test_append_to_empty_file(self, mock_env_vars, sample_market_data):
        """Test appending when no existing data."""
        result = append_market_data(sample_market_data, "day_ahead")

        assert result is True
        loaded = load_market_data("day_ahead")
        assert len(loaded) == len(sample_market_data)

    def test_append_new_data(self, mock_env_vars, sample_timestamps):
        """Test appending new data to existing data."""
        # Create initial data
        df1 = pd.DataFrame(
            {
                "timestamp_utc": sample_timestamps[:100],
                "price_eur_per_mwh": range(100),
                "market_type": "day_ahead",
            }
        )
        save_market_data(df1, "day_ahead")

        # Append new data
        df2 = pd.DataFrame(
            {
                "timestamp_utc": sample_timestamps[100:],
                "price_eur_per_mwh": range(100, len(sample_timestamps)),
                "market_type": "day_ahead",
            }
        )
        result = append_market_data(df2, "day_ahead")

        assert result is True
        loaded = load_market_data("day_ahead")
        assert len(loaded) == len(sample_timestamps)

    def test_append_removes_duplicates(self, mock_env_vars, sample_timestamps):
        """Test that append removes duplicate timestamps."""
        # Create initial data
        df1 = pd.DataFrame(
            {
                "timestamp_utc": sample_timestamps[:100],
                "price_eur_per_mwh": range(100),
                "market_type": "day_ahead",
            }
        )
        save_market_data(df1, "day_ahead")

        # Append with overlapping data
        df2 = pd.DataFrame(
            {
                "timestamp_utc": sample_timestamps[50:150],  # Overlap from 50-100
                "price_eur_per_mwh": range(50, 150),
                "market_type": "day_ahead",
            }
        )
        result = append_market_data(df2, "day_ahead")

        assert result is True
        loaded = load_market_data("day_ahead")
        assert len(loaded) == 150  # Should have 150 unique timestamps

    def test_append_sorts_by_timestamp(self, mock_env_vars, sample_timestamps):
        """Test that appended data is sorted by timestamp."""
        # Create data in reverse order
        df1 = pd.DataFrame(
            {
                "timestamp_utc": sample_timestamps[100:],
                "price_eur_per_mwh": range(100, len(sample_timestamps)),
                "market_type": "day_ahead",
            }
        )
        save_market_data(df1, "day_ahead")

        # Append earlier data
        df2 = pd.DataFrame(
            {
                "timestamp_utc": sample_timestamps[:100],
                "price_eur_per_mwh": range(100),
                "market_type": "day_ahead",
            }
        )
        result = append_market_data(df2, "day_ahead")

        assert result is True
        loaded = load_market_data("day_ahead")

        # Verify sorted
        assert loaded["timestamp_utc"].is_monotonic_increasing


class TestWeatherDataSave:
    """Test suite for saving weather data."""

    def test_save_weather_data_success(self, mock_env_vars, sample_weather_data):
        """Test successful weather data save."""
        result = save_weather_data(sample_weather_data)

        assert result is True

    def test_save_weather_data_creates_file(
        self, mock_env_vars, test_data_dir, sample_weather_data
    ):
        """Test that save creates the data file."""
        save_weather_data(sample_weather_data)

        # Check file exists
        expected_file = test_data_dir / "weather_data_2025.csv"
        assert expected_file.exists()

    def test_save_weather_data_preserves_timestamps(self, mock_env_vars, sample_weather_data):
        """Test that timestamps are properly formatted on save with UTC timezone."""
        save_weather_data(sample_weather_data)

        # Load and verify
        loaded = load_weather_data()
        assert loaded.index.tz is not None
        assert str(loaded.index.tz) == "UTC"


class TestWeatherDataLoad:
    """Test suite for loading weather data."""

    def test_load_weather_data_success(self, mock_env_vars, sample_weather_data):
        """Test successful weather data load."""
        save_weather_data(sample_weather_data)

        loaded = load_weather_data()

        assert loaded is not None
        assert not loaded.empty
        assert len(loaded) == len(sample_weather_data)

    def test_load_weather_data_nonexistent_file(self, mock_env_vars, test_data_dir):
        """Test loading when file doesn't exist."""
        # Ensure file doesn't exist
        file_path = test_data_dir / "knmi_weather_2025.csv"
        if file_path.exists():
            file_path.unlink()

        loaded = load_weather_data()

        assert loaded is None

    def test_load_weather_data_parses_timestamps(self, mock_env_vars, sample_weather_data):
        """Test that timestamps are properly parsed on load."""
        save_weather_data(sample_weather_data)

        loaded = load_weather_data()

        assert loaded.index.name == "timestamp"
        assert pd.api.types.is_datetime64_any_dtype(loaded.index)

    def test_load_weather_data_preserves_columns(self, mock_env_vars, sample_weather_data):
        """Test that all columns are preserved."""
        save_weather_data(sample_weather_data)

        loaded = load_weather_data()

        assert set(loaded.columns) == set(sample_weather_data.columns)


class TestWeatherDataAppend:
    """Test suite for appending weather data."""

    def test_append_weather_to_empty_file(self, mock_env_vars, sample_weather_data):
        """Test appending when no existing data."""
        result = append_weather_data(sample_weather_data)

        assert result is True
        loaded = load_weather_data()
        assert len(loaded) == len(sample_weather_data)

    def test_append_weather_new_data(self, mock_env_vars, sample_timestamps):
        """Test appending new weather data."""
        # Create initial data
        df1 = pd.DataFrame(
            {
                "timestamp_utc": sample_timestamps[:100],
                "temperature_deg_c": range(100),
            }
        )
        save_weather_data(df1)

        # Append new data
        df2 = pd.DataFrame(
            {
                "timestamp_utc": sample_timestamps[100:],
                "temperature_deg_c": range(100, len(sample_timestamps)),
            }
        )
        result = append_weather_data(df2)

        assert result is True
        loaded = load_weather_data()
        assert len(loaded) == len(sample_timestamps)

    def test_append_weather_removes_duplicates(self, mock_env_vars, sample_timestamps):
        """Test that append removes duplicate timestamps."""
        # Create initial data
        df1 = pd.DataFrame(
            {
                "timestamp_utc": sample_timestamps[:100],
                "temperature_deg_c": range(100),
            }
        )
        save_weather_data(df1)

        # Append with overlapping data
        df2 = pd.DataFrame(
            {
                "timestamp_utc": sample_timestamps[50:150],
                "temperature_deg_c": range(50, 150),
            }
        )
        result = append_weather_data(df2)

        assert result is True
        loaded = load_weather_data()
        assert len(loaded) == 150


class TestDataSummary:
    """Test suite for data summary functionality."""

    def test_summary_with_no_data(self, mock_env_vars, test_data_dir):
        """Test summary when no data exists."""
        # Clear all data files
        for f in test_data_dir.glob("*.csv"):
            f.unlink()

        summary = get_data_summary()

        assert "markets" in summary
        assert "weather" in summary
        assert summary["weather"]["records"] == 0

    def test_summary_with_market_data(self, mock_env_vars, sample_market_data):
        """Test summary with market data."""
        save_market_data(sample_market_data, "day_ahead")

        summary = get_data_summary()

        assert summary["markets"]["day_ahead"]["records"] == len(sample_market_data)
        assert "start_date" in summary["markets"]["day_ahead"]
        assert "end_date" in summary["markets"]["day_ahead"]

    def test_summary_with_weather_data(self, mock_env_vars, sample_weather_data):
        """Test summary with weather data."""
        save_weather_data(sample_weather_data)

        summary = get_data_summary()

        assert summary["weather"]["records"] == len(sample_weather_data)
        assert "start_date" in summary["weather"]
        assert "end_date" in summary["weather"]

    def test_summary_with_all_data(self, mock_env_vars, sample_market_data, sample_weather_data):
        """Test summary with all data types."""
        # Save day-ahead market type
        save_market_data(sample_market_data, "day_ahead")

        # Save weather
        save_weather_data(sample_weather_data)

        summary = get_data_summary()

        # Check day-ahead market has data
        assert summary["markets"]["day_ahead"]["records"] > 0

        # Check weather has data
        assert summary["weather"]["records"] > 0

    def test_summary_date_format(self, mock_env_vars, sample_market_data):
        """Test that summary dates are in ISO format."""
        save_market_data(sample_market_data, "day_ahead")

        summary = get_data_summary()

        start_date = summary["markets"]["day_ahead"]["start_date"]
        end_date = summary["markets"]["day_ahead"]["end_date"]

        # Should be parseable as ISO format
        pd.to_datetime(start_date)
        pd.to_datetime(end_date)
