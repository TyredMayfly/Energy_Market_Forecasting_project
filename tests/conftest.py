"""
Pytest configuration and fixtures.

Shared fixtures used across all test modules.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def test_data_dir(tmp_path):
    """
    Create a temporary data directory for tests.
    
    Returns:
        Path to temporary data directory
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def mock_env_vars(monkeypatch, test_data_dir):
    """
    Set up mock environment variables for testing.
    
    Args:
        monkeypatch: pytest monkeypatch fixture
        test_data_dir: temporary data directory
    """
    monkeypatch.setenv("ENTSOE_API_KEY", "test_entsoe_key")
    monkeypatch.setenv("KNMI_API_KEY", "test_knmi_key")
    monkeypatch.setenv("DATA_DIR", str(test_data_dir))
    
    # Reload all relevant modules to pick up new environment variables
    from app.core import config
    from app.services import entsoe_client, knmi_client, data_store
    import importlib
    
    importlib.reload(config)
    importlib.reload(entsoe_client)
    importlib.reload(knmi_client)
    importlib.reload(data_store)
    
    # Also set directly on settings object
    config.settings.data_dir = test_data_dir
    config.settings.entsoe_api_key = "test_entsoe_key"
    config.settings.knmi_api_key = "test_knmi_key"


@pytest.fixture
def sample_timestamps():
    """
    Generate sample 15-minute resolution timestamps starting from 2025-01-01.
    
    Provides data from 2025-01-01 until 2025-11-18 (current date).
    
    Returns:
        List of datetime objects at 15-minute intervals
    """
    start = datetime(2025, 1, 1, 0, 0, 0)
    end = datetime(2025, 11, 18, 23, 45, 0)  # Until end of November 18, 2025
    # Calculate number of 15-minute intervals between start and end
    num_intervals = int((end - start).total_seconds() / (15 * 60)) + 1
    return [start + timedelta(minutes=15*i) for i in range(num_intervals)]


@pytest.fixture
def sample_market_data(sample_timestamps):
    """
    Create sample market price data.
    
    Returns:
        DataFrame with market prices
    """
    return pd.DataFrame({
        'timestamp_utc': sample_timestamps,
        'price_eur_per_mwh': np.random.uniform(30, 80, len(sample_timestamps)),
        'market_type': 'day_ahead'
    })


@pytest.fixture
def sample_weather_data(sample_timestamps):
    """
    Create sample weather data.
    
    Returns:
        DataFrame with weather observations
    """
    return pd.DataFrame({
        'timestamp_utc': sample_timestamps,
        'temperature_deg_c': np.random.uniform(5, 20, len(sample_timestamps)),
        'wind_speed_m_per_s': np.random.uniform(2, 15, len(sample_timestamps)),
        'global_radiation_w_per_m2': np.random.uniform(0, 500, len(sample_timestamps))
    })


@pytest.fixture
def sample_entsoe_xml():
    """
    Provide valid ENTSO-E XML response for testing.
    
    Returns:
        XML string representing ENTSO-E price document
    """
    return """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0">
    <mRID>test-document-id</mRID>
    <revisionNumber>1</revisionNumber>
    <type>A44</type>
    <TimeSeries>
        <mRID>1</mRID>
        <businessType>A62</businessType>
        <in_Domain.mRID codingScheme="A01">10YNL----------L</in_Domain.mRID>
        <out_Domain.mRID codingScheme="A01">10YNL----------L</out_Domain.mRID>
        <Period>
            <timeInterval>
                <start>2025-01-01T00:00:00Z</start>
                <end>2025-01-01T04:00:00Z</end>
            </timeInterval>
            <resolution>PT60M</resolution>
            <Point>
                <position>1</position>
                <price.amount>45.50</price.amount>
            </Point>
            <Point>
                <position>2</position>
                <price.amount>46.75</price.amount>
            </Point>
            <Point>
                <position>3</position>
                <price.amount>44.20</price.amount>
            </Point>
            <Point>
                <position>4</position>
                <price.amount>43.80</price.amount>
            </Point>
        </Period>
    </TimeSeries>
</Publication_MarketDocument>
"""


@pytest.fixture
def sample_knmi_files_response():
    """
    Provide sample KNMI file listing response.
    
    Returns:
        Dictionary representing KNMI API response
    """
    return {
        "files": [
            {
                "filename": "KMDS__OPER_P___10M_OBS_L2_202501010000.nc",
                "size": 12345,
                "lastModified": "2025-01-01T01:00:00Z"
            },
            {
                "filename": "KMDS__OPER_P___10M_OBS_L2_202501020000.nc",
                "size": 12346,
                "lastModified": "2025-01-02T01:00:00Z"
            },
            {
                "filename": "KMDS__OPER_P___10M_OBS_L2_202412310000.nc",
                "size": 12344,
                "lastModified": "2024-12-31T01:00:00Z"
            }
        ],
        "isTruncated": False
    }


@pytest.fixture
def sample_knmi_download_url_response():
    """
    Provide sample KNMI download URL response.
    
    Returns:
        Dictionary with temporary download URL
    """
    return {
        "temporaryDownloadUrl": "https://example.com/download/file.nc?token=abc123"
    }
