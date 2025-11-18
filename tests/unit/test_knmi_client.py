"""
Tests for KNMI client.
"""

from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

import pandas as pd
import pytest
import requests
from app.services.knmi_client import KnmiClient


class TestKnmiClientInitialization:
    """Test suite for KnmiClient initialization."""

    def test_client_initialization_with_env_key(self, mock_env_vars):
        """Test that client initializes with API key from environment."""
        client = KnmiClient()
        
        assert client.api_key == "test_knmi_key"
        assert client.base_url is not None
        assert client.dataset_name is not None
        assert client.dataset_version is not None
        assert "Authorization" in client.headers

    def test_client_initialization_with_explicit_key(self, mock_env_vars):
        """Test initialization with explicitly provided API key."""
        client = KnmiClient(api_key="explicit_key")
        
        assert client.api_key == "explicit_key"
        assert client.headers["Authorization"] == "explicit_key"

    def test_client_initialization_without_key(self):
        """Test that client raises error without API key."""
        # Pass empty string explicitly (not None which would use settings)
        with pytest.raises(ValueError, match="API key is required"):
            KnmiClient(api_key="")


class TestKnmiFileListng:
    """Test suite for KNMI file listing."""

    @patch("app.services.knmi_client.requests.get")
    def test_list_files_success(self, mock_get, mock_env_vars, sample_knmi_files_response):
        """Test successful file listing."""
        client = KnmiClient()
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_knmi_files_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        files = client._list_files()
        
        assert len(files) == 3
        assert files[0]["filename"] == "KMDS__OPER_P___10M_OBS_L2_202501010000.nc"
        assert mock_get.called

    @patch("app.services.knmi_client.requests.get")
    def test_list_files_with_max_keys(self, mock_get, mock_env_vars, sample_knmi_files_response):
        """Test file listing with max_keys parameter."""
        client = KnmiClient()
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_knmi_files_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        files = client._list_files(max_keys=10)
        
        # Check that max_keys was passed in request
        call_args = mock_get.call_args
        assert call_args[1]["params"]["maxKeys"] == 10

    @patch("app.services.knmi_client.requests.get")
    def test_list_files_with_pagination(self, mock_get, mock_env_vars, sample_knmi_files_response):
        """Test file listing with pagination."""
        client = KnmiClient()
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_knmi_files_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        files = client._list_files(start_after_filename="file1.nc")
        
        # Check that pagination parameter was passed
        call_args = mock_get.call_args
        assert call_args[1]["params"]["startAfterFilename"] == "file1.nc"

    @patch("app.services.knmi_client.requests.get")
    def test_list_files_http_error(self, mock_get, mock_env_vars):
        """Test handling of HTTP errors during file listing."""
        client = KnmiClient()
        
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Not found")
        mock_get.return_value = mock_response
        
        files = client._list_files()
        
        # Should return empty list on error
        assert files == []

    @patch("app.services.knmi_client.requests.get")
    def test_list_files_empty_response(self, mock_get, mock_env_vars):
        """Test handling of empty file list response."""
        client = KnmiClient()
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"files": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        files = client._list_files()
        
        assert files == []


class TestKnmiDownloadUrl:
    """Test suite for KNMI download URL retrieval."""

    @patch("app.services.knmi_client.requests.get")
    def test_get_download_url_success(self, mock_get, mock_env_vars, sample_knmi_download_url_response):
        """Test successful download URL retrieval."""
        client = KnmiClient()
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_knmi_download_url_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        url = client._get_download_url("test_file.nc")
        
        assert url == "https://example.com/download/file.nc?token=abc123"
        assert mock_get.called

    @patch("app.services.knmi_client.requests.get")
    def test_get_download_url_builds_correct_path(self, mock_get, mock_env_vars, sample_knmi_download_url_response):
        """Test that download URL request uses correct path."""
        client = KnmiClient()
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_knmi_download_url_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        filename = "test_file.nc"
        client._get_download_url(filename)
        
        # Verify URL construction
        call_url = mock_get.call_args[0][0]
        assert filename in call_url
        assert "files" in call_url
        assert "url" in call_url

    @patch("app.services.knmi_client.requests.get")
    def test_get_download_url_http_error(self, mock_get, mock_env_vars):
        """Test handling of HTTP errors during download URL retrieval."""
        client = KnmiClient()
        
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Forbidden")
        mock_get.return_value = mock_response
        
        url = client._get_download_url("test_file.nc")
        
        assert url is None


class TestKnmiFileFiltering:
    """Test suite for KNMI file filtering."""

    def test_filter_2025_files(self, mock_env_vars, sample_knmi_files_response):
        """Test filtering for 2025 files."""
        client = KnmiClient()
        
        files = sample_knmi_files_response["files"]
        filtered = client._filter_2025_files(files)
        
        # Should only include files with "2025" in filename
        assert len(filtered) == 2
        assert all("2025" in f["filename"] for f in filtered)

    def test_filter_2025_files_empty_list(self, mock_env_vars):
        """Test filtering with empty file list."""
        client = KnmiClient()
        
        filtered = client._filter_2025_files([])
        
        assert filtered == []

    def test_filter_2025_files_no_matches(self, mock_env_vars):
        """Test filtering when no files match."""
        client = KnmiClient()
        
        files = [
            {"filename": "file_2024.nc"},
            {"filename": "file_2023.nc"}
        ]
        
        filtered = client._filter_2025_files(files)
        
        assert filtered == []

    @patch("app.services.knmi_client.requests.get")
    def test_list_files_for_2025(self, mock_get, mock_env_vars, sample_knmi_files_response):
        """Test listing files specifically for 2025."""
        client = KnmiClient()
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_knmi_files_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        files = client.list_files_for_2025()
        
        # Should only return 2025 files
        assert len(files) == 2
        assert all("2025" in f["filename"] for f in files)


class TestKnmiFileDownload:
    """Test suite for KNMI file download."""

    @patch("app.services.knmi_client.requests.get")
    def test_download_file_success(self, mock_get, mock_env_vars, tmp_path):
        """Test successful file download."""
        client = KnmiClient()
        
        # Mock successful download
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[b"test data chunk"])
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        download_path = tmp_path / "test_file.nc"
        success = client._download_file("https://example.com/file.nc", download_path)
        
        assert success
        assert download_path.exists()

    @patch("app.services.knmi_client.requests.get")
    def test_download_file_creates_directory(self, mock_get, mock_env_vars, tmp_path):
        """Test that download creates parent directories."""
        client = KnmiClient()
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = Mock(return_value=[b"data"])
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Path with non-existent parent directory
        download_path = tmp_path / "subdir" / "nested" / "file.nc"
        success = client._download_file("https://example.com/file.nc", download_path)
        
        assert success
        assert download_path.parent.exists()

    @patch("app.services.knmi_client.requests.get")
    def test_download_file_http_error(self, mock_get, mock_env_vars, tmp_path):
        """Test handling of HTTP errors during download."""
        client = KnmiClient()
        
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server error")
        mock_get.return_value = mock_response
        
        download_path = tmp_path / "test_file.nc"
        success = client._download_file("https://example.com/file.nc", download_path)
        
        assert not success


class TestKnmiNetcdfParsing:
    """Test suite for NetCDF parsing."""

    @patch("app.services.knmi_client.xr.open_dataset")
    def test_parse_netcdf_with_valid_data(self, mock_open_dataset, mock_env_vars, tmp_path):
        """Test parsing valid NetCDF file."""
        client = KnmiClient()
        
        # Create mock xarray dataset
        import xarray as xr
        import numpy as np
        
        times = pd.date_range('2025-01-01', periods=24, freq='h')
        
        mock_dataset = xr.Dataset({
            'T': xr.DataArray(
                np.random.uniform(5, 15, 24),
                coords={'time': times},
                dims=['time']
            ),
            'FF': xr.DataArray(
                np.random.uniform(2, 10, 24),
                coords={'time': times},
                dims=['time']
            ),
            'Q': xr.DataArray(
                np.random.uniform(0, 500, 24),
                coords={'time': times},
                dims=['time']
            )
        })
        
        mock_open_dataset.return_value = mock_dataset
        
        test_file = tmp_path / "test.nc"
        test_file.touch()
        
        df = client._parse_netcdf_to_dataframe(test_file)
        
        assert not df.empty
        assert "timestamp_utc" in df.columns
        assert "variable" in df.columns
        assert "value" in df.columns

    @patch("app.services.knmi_client.xr.open_dataset")
    def test_parse_netcdf_handles_errors(self, mock_open_dataset, mock_env_vars, tmp_path):
        """Test handling of NetCDF parsing errors."""
        client = KnmiClient()
        
        # Mock error during parsing
        mock_open_dataset.side_effect = Exception("Invalid NetCDF file")
        
        test_file = tmp_path / "bad_file.nc"
        test_file.touch()
        
        df = client._parse_netcdf_to_dataframe(test_file)
        
        # Should return empty DataFrame, not crash
        assert df.empty
        assert list(df.columns) == ["timestamp_utc", "station_id", "variable", "value"]

    @patch("app.services.knmi_client.xr.open_dataset")
    def test_parse_netcdf_empty_file(self, mock_open_dataset, mock_env_vars, tmp_path):
        """Test parsing empty NetCDF file."""
        client = KnmiClient()
        
        # Mock empty dataset
        import xarray as xr
        mock_dataset = xr.Dataset({})
        mock_open_dataset.return_value = mock_dataset
        
        test_file = tmp_path / "empty.nc"
        test_file.touch()
        
        df = client._parse_netcdf_to_dataframe(test_file)
        
        assert df.empty


class TestKnmiEndToEnd:
    """Test suite for end-to-end KNMI operations."""

    @patch("app.services.knmi_client.requests.get")
    @patch("app.services.knmi_client.xr.open_dataset")
    def test_download_and_process_files_for_2025(
        self, mock_open_dataset, mock_get, mock_env_vars, 
        sample_knmi_files_response, sample_knmi_download_url_response
    ):
        """Test downloading and processing files for 2025."""
        client = KnmiClient()
        
        # Mock file listing
        files_response = Mock()
        files_response.status_code = 200
        files_response.json.return_value = sample_knmi_files_response
        files_response.raise_for_status = Mock()
        
        # Mock download URL
        url_response = Mock()
        url_response.status_code = 200
        url_response.json.return_value = sample_knmi_download_url_response
        url_response.raise_for_status = Mock()
        
        # Mock file download
        download_response = Mock()
        download_response.status_code = 200
        download_response.iter_content = Mock(return_value=[b"data"])
        download_response.raise_for_status = Mock()
        
        # Set up sequence of responses
        mock_get.side_effect = [files_response, url_response, download_response, url_response, download_response]
        
        # Mock NetCDF parsing
        import xarray as xr
        import numpy as np
        times = pd.date_range('2025-01-01', periods=10, freq='h')
        mock_dataset = xr.Dataset({
            'T': xr.DataArray(np.random.uniform(5, 15, 10), coords={'time': times}, dims=['time']),
            'FF': xr.DataArray(np.random.uniform(2, 10, 10), coords={'time': times}, dims=['time']),
            'Q': xr.DataArray(np.random.uniform(0, 500, 10), coords={'time': times}, dims=['time'])
        })
        mock_open_dataset.return_value = mock_dataset
        
        df = client.download_and_process_files_for_2025(max_files=2)
        
        # Should have processed data
        assert "timestamp_utc" in df.columns
