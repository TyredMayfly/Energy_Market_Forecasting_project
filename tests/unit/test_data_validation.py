"""
Unit tests for data validation utilities.

Tests timezone validation, date range validation, and timezone conversion helpers.
"""

import pandas as pd
import pytest
from datetime import datetime

from app.services.data_validation import (
    DateRangeValidationError,
    TimezoneValidationError,
    ensure_local_timestamps,
    ensure_utc_timestamps,
    validate_and_align_market_weather_data,
    validate_start_end_dates,
    validate_timezones,
)


class TestValidateTimezones:
    """Tests for timezone validation."""

    def test_validate_timezones_happy_path_utc(self):
        """Test validation passes when all DataFrames have UTC timezone."""
        # Create three DataFrames with UTC timestamps
        df1 = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01", periods=10, freq="h", tz="UTC"),
                "price": [100.0] * 10,
            }
        )

        df2 = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01", periods=10, freq="h", tz="UTC"),
                "price": [200.0] * 10,
            }
        )

        df3 = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01", periods=10, freq="h", tz="UTC"),
                "weather": [15.0] * 10,
            }
        )

        # Should not raise
        validate_timezones(df1, df2, df3, expected_tz="UTC", timestamp_cols=["timestamp_utc"])

    def test_validate_timezones_happy_path_index(self):
        """Test validation passes when all DataFrames have UTC timezone in index."""
        df1 = pd.DataFrame(
            {"price": [100.0] * 10},
            index=pd.date_range("2024-10-01", periods=10, freq="h", tz="UTC"),
        )

        df2 = pd.DataFrame(
            {"price": [200.0] * 10},
            index=pd.date_range("2024-10-01", periods=10, freq="h", tz="UTC"),
        )

        # Should not raise (checking index by default)
        validate_timezones(df1, df2, expected_tz="UTC")

    def test_validate_timezones_different_timezone_raises(self):
        """Test validation fails when DataFrames have different timezones."""
        df1 = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01", periods=10, freq="h", tz="UTC"),
                "price": [100.0] * 10,
            }
        )

        df2 = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range(
                    "2024-10-01", periods=10, freq="h", tz="Europe/Amsterdam"
                ),
                "price": [200.0] * 10,
            }
        )

        with pytest.raises(TimezoneValidationError) as exc_info:
            validate_timezones(df1, df2, expected_tz="UTC", timestamp_cols=["timestamp_utc"])

        assert "Expected timezone 'UTC'" in str(exc_info.value)
        assert "Europe/Amsterdam" in str(exc_info.value)

    def test_validate_timezones_naive_timestamps_raises(self):
        """Test validation fails when timestamps are timezone-naive."""
        df = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01", periods=10, freq="h"),  # No tz
                "price": [100.0] * 10,
            }
        )

        with pytest.raises(TimezoneValidationError) as exc_info:
            validate_timezones(df, expected_tz="UTC", timestamp_cols=["timestamp_utc"])

        assert "timezone-naive" in str(exc_info.value)
        assert "tz_localize" in str(exc_info.value)

    def test_validate_timezones_naive_index_raises(self):
        """Test validation fails when index is timezone-naive."""
        df = pd.DataFrame(
            {"price": [100.0] * 10},
            index=pd.date_range("2024-10-01", periods=10, freq="h"),  # No tz
        )

        with pytest.raises(TimezoneValidationError) as exc_info:
            validate_timezones(df, expected_tz="UTC")

        assert "timezone-naive" in str(exc_info.value)

    def test_validate_timezones_non_datetime_index_raises(self):
        """Test validation fails when index is not DatetimeIndex."""
        df = pd.DataFrame({"price": [100.0] * 10}, index=range(10))

        with pytest.raises(TimezoneValidationError) as exc_info:
            validate_timezones(df, expected_tz="UTC")

        assert "not a DatetimeIndex" in str(exc_info.value)

    def test_validate_timezones_missing_column_raises(self):
        """Test validation fails when timestamp column is missing."""
        df = pd.DataFrame({"price": [100.0] * 10})

        with pytest.raises(TimezoneValidationError) as exc_info:
            validate_timezones(df, expected_tz="UTC", timestamp_cols=["timestamp_utc"])

        assert "Column 'timestamp_utc' not found" in str(exc_info.value)

    def test_validate_timezones_empty_dataframes_passes(self):
        """Test validation passes with empty or None DataFrames."""
        df1 = pd.DataFrame()
        df2 = None

        # Should not raise
        validate_timezones(df1, df2, expected_tz="UTC")

    def test_validate_timezones_amsterdam_timezone(self):
        """Test validation with Europe/Amsterdam timezone."""
        df = pd.DataFrame(
            {"price": [100.0] * 10},
            index=pd.date_range("2024-10-01", periods=10, freq="h", tz="Europe/Amsterdam"),
        )

        # Should not raise
        validate_timezones(df, expected_tz="Europe/Amsterdam")


class TestValidateStartEndDates:
    """Tests for date range validation."""

    def test_validate_start_end_dates_happy_path(self):
        """Test validation passes when all DataFrames have same date range."""
        df1 = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01", periods=100, freq="15min", tz="UTC"),
                "price": [100.0] * 100,
            }
        )

        df2 = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01", periods=100, freq="15min", tz="UTC"),
                "price": [200.0] * 100,
            }
        )

        # Should not raise
        validate_start_end_dates(df1, df2, timestamp_cols=["timestamp_utc"])

    def test_validate_start_end_dates_index_happy_path(self):
        """Test validation with DatetimeIndex."""
        df1 = pd.DataFrame(
            {"price": [100.0] * 100},
            index=pd.date_range("2024-10-01", periods=100, freq="15min", tz="UTC"),
        )

        df2 = pd.DataFrame(
            {"price": [200.0] * 100},
            index=pd.date_range("2024-10-01", periods=100, freq="15min", tz="UTC"),
        )

        # Should not raise
        validate_start_end_dates(df1, df2)

    def test_validate_start_end_dates_different_start_raises(self):
        """Test validation fails when DataFrames have different start dates."""
        df1 = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01", periods=100, freq="h", tz="UTC"),
                "price": [100.0] * 100,
            }
        )

        df2 = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-02", periods=100, freq="h", tz="UTC"),
                "price": [200.0] * 100,
            }
        )

        with pytest.raises(DateRangeValidationError) as exc_info:
            validate_start_end_dates(df1, df2, timestamp_cols=["timestamp_utc"])

        assert "Start date mismatch" in str(exc_info.value)
        assert "2024-10-01" in str(exc_info.value)
        assert "2024-10-02" in str(exc_info.value)

    def test_validate_start_end_dates_different_end_raises(self):
        """Test validation fails when DataFrames have different end dates."""
        df1 = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01", periods=100, freq="h", tz="UTC"),
                "price": [100.0] * 100,
            }
        )

        df2 = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01", periods=50, freq="h", tz="UTC"),
                "price": [200.0] * 50,
            }
        )

        with pytest.raises(DateRangeValidationError) as exc_info:
            validate_start_end_dates(df1, df2, timestamp_cols=["timestamp_utc"])

        assert "End date mismatch" in str(exc_info.value)

    def test_validate_start_end_dates_with_tolerance(self):
        """Test validation passes with tolerance for small differences."""
        # df1 starts at 00:00:00
        df1 = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01 00:00:00", periods=10, freq="h", tz="UTC"),
                "price": [100.0] * 10,
            }
        )

        # df2 starts at 00:00:30 (30 seconds later)
        df2 = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01 00:00:30", periods=10, freq="h", tz="UTC"),
                "price": [200.0] * 10,
            }
        )

        # Should raise with 0 tolerance
        with pytest.raises(DateRangeValidationError):
            validate_start_end_dates(df1, df2, timestamp_cols=["timestamp_utc"], tolerance_seconds=0)

        # Should not raise with 60 second tolerance
        validate_start_end_dates(df1, df2, timestamp_cols=["timestamp_utc"], tolerance_seconds=60)

    def test_validate_start_end_dates_naive_timestamps_raises(self):
        """Test validation fails with naive timestamps."""
        df1 = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01", periods=10, freq="h"),  # No tz
                "price": [100.0] * 10,
            }
        )

        df2 = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01", periods=10, freq="h"),  # No tz
                "price": [200.0] * 10,
            }
        )

        with pytest.raises(DateRangeValidationError) as exc_info:
            validate_start_end_dates(df1, df2, timestamp_cols=["timestamp_utc"])

        assert "timezone-naive" in str(exc_info.value)

    def test_validate_start_end_dates_single_dataframe_passes(self):
        """Test validation passes with single DataFrame."""
        df = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01", periods=10, freq="h", tz="UTC"),
                "price": [100.0] * 10,
            }
        )

        # Should not raise (nothing to compare)
        validate_start_end_dates(df, timestamp_cols=["timestamp_utc"])

    def test_validate_start_end_dates_empty_dataframes_passes(self):
        """Test validation passes with empty DataFrames."""
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()

        # Should not raise
        validate_start_end_dates(df1, df2)


class TestEnsureUtcTimestamps:
    """Tests for UTC timestamp conversion."""

    def test_ensure_utc_timestamps_column(self):
        """Test converting column timestamps to UTC."""
        df = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01", periods=10, freq="h"),
                "price": [100.0] * 10,
            }
        )

        result = ensure_utc_timestamps(df, "timestamp_utc")

        assert result["timestamp_utc"].dt.tz is not None
        assert str(result["timestamp_utc"].dt.tz) == "UTC"

    def test_ensure_utc_timestamps_already_utc(self):
        """Test that UTC timestamps remain unchanged."""
        df = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01", periods=10, freq="h", tz="UTC"),
                "price": [100.0] * 10,
            }
        )

        result = ensure_utc_timestamps(df, "timestamp_utc")

        assert str(result["timestamp_utc"].dt.tz) == "UTC"
        pd.testing.assert_frame_equal(result, df)

    def test_ensure_utc_timestamps_convert_from_amsterdam(self):
        """Test converting from Europe/Amsterdam to UTC."""
        df = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range(
                    "2024-10-01", periods=10, freq="h", tz="Europe/Amsterdam"
                ),
                "price": [100.0] * 10,
            }
        )

        result = ensure_utc_timestamps(df, "timestamp_utc")

        assert str(result["timestamp_utc"].dt.tz) == "UTC"
        # Values should represent the same instant, but with different timezone strings
        # Amsterdam time 00:00 in October is UTC 22:00 (previous day) due to UTC+2 offset
        assert result["timestamp_utc"].iloc[0].hour != df["timestamp_utc"].iloc[0].hour

    def test_ensure_utc_timestamps_index(self):
        """Test converting index timestamps to UTC."""
        df = pd.DataFrame(
            {"price": [100.0] * 10},
            index=pd.date_range("2024-10-01", periods=10, freq="h"),
        )
        df.index.name = "timestamp_utc"

        result = ensure_utc_timestamps(df, "timestamp_utc")

        assert result.index.tz is not None
        assert str(result.index.tz) == "UTC"

    def test_ensure_utc_timestamps_missing_column_raises(self):
        """Test error when column doesn't exist."""
        df = pd.DataFrame({"price": [100.0] * 10})

        with pytest.raises(ValueError) as exc_info:
            ensure_utc_timestamps(df, "timestamp_utc")

        assert "not found" in str(exc_info.value)

    def test_ensure_utc_timestamps_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()

        result = ensure_utc_timestamps(df, "timestamp_utc")

        assert result.empty


class TestEnsureLocalTimestamps:
    """Tests for local timestamp conversion."""

    def test_ensure_local_timestamps_default_timezone(self):
        """Test converting to default timezone (Europe/Amsterdam)."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-10-01", periods=10, freq="h", tz="UTC"),
                "price": [100.0] * 10,
            }
        )

        result = ensure_local_timestamps(df, "timestamp")

        assert result["timestamp"].dt.tz is not None
        assert "Europe/Amsterdam" in str(result["timestamp"].dt.tz)

    def test_ensure_local_timestamps_custom_timezone(self):
        """Test converting to custom timezone."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-10-01", periods=10, freq="h"),
                "price": [100.0] * 10,
            }
        )

        result = ensure_local_timestamps(df, "timestamp", tz="US/Eastern")

        assert str(result["timestamp"].dt.tz) == "US/Eastern"

    def test_ensure_local_timestamps_naive_localize(self):
        """Test localizing naive timestamps."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-10-01", periods=10, freq="h"),
                "price": [100.0] * 10,
            }
        )

        result = ensure_local_timestamps(df, "timestamp", tz="Europe/Amsterdam")

        assert result["timestamp"].dt.tz is not None


class TestValidateAndAlignMarketWeatherData:
    """Tests for market and weather data validation and alignment."""

    def test_validate_and_align_happy_path(self):
        """Test successful validation and alignment."""
        df_market = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01", periods=100, freq="15min", tz="UTC"),
                "price": [100.0] * 100,
            }
        )

        df_weather = pd.DataFrame(
            {"temperature": [15.0] * 25},
            index=pd.date_range("2024-10-01", periods=25, freq="h", tz="UTC"),
        )

        # Should not raise
        market, weather = validate_and_align_market_weather_data(df_market, df_weather)

        assert market is not None
        assert weather is not None
        assert str(market["timestamp_utc"].dt.tz) == "UTC"
        assert str(weather.index.tz) == "UTC"

    def test_validate_and_align_naive_market_timestamps(self):
        """Test that naive market timestamps are converted to UTC."""
        df_market = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01", periods=100, freq="15min"),
                "price": [100.0] * 100,
            }
        )

        market, _ = validate_and_align_market_weather_data(df_market, None)

        assert str(market["timestamp_utc"].dt.tz) == "UTC"

    def test_validate_and_align_weather_coverage_warning(self):
        """Test warning when weather data doesn't cover full market range."""
        df_market = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01", periods=100, freq="h", tz="UTC"),
                "price": [100.0] * 100,
            }
        )

        # Weather starts later
        df_weather = pd.DataFrame(
            {"temperature": [15.0] * 50},
            index=pd.date_range("2024-10-03", periods=50, freq="h", tz="UTC"),
        )

        # Should still work but log warning
        market, weather = validate_and_align_market_weather_data(df_market, df_weather)

        assert market is not None
        assert weather is not None

    def test_validate_and_align_no_weather_data(self):
        """Test with no weather data."""
        df_market = pd.DataFrame(
            {
                "timestamp_utc": pd.date_range("2024-10-01", periods=100, freq="15min", tz="UTC"),
                "price": [100.0] * 100,
            }
        )

        market, weather = validate_and_align_market_weather_data(df_market, None)

        assert market is not None
        assert weather is None

    def test_validate_and_align_empty_market_data(self):
        """Test with empty market data."""
        df_market = pd.DataFrame()
        df_weather = pd.DataFrame(
            {"temperature": [15.0] * 10},
            index=pd.date_range("2024-10-01", periods=10, freq="h", tz="UTC"),
        )

        market, weather = validate_and_align_market_weather_data(df_market, df_weather)

        assert market.empty
