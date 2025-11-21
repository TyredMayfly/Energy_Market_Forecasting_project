"""
Tests for forecast API endpoints.
"""

import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, call
import pandas as pd
import pytz


@pytest.fixture
def client():
    """Create test client with mocked lifespan."""
    with patch('app.api.main.lifespan'):
        from app.api.main import create_app
        app = create_app()
        return TestClient(app)


@pytest.fixture
def mock_forecast_df():
    """Create a mock forecast DataFrame."""
    timestamps = [datetime.now(pytz.UTC) + timedelta(hours=i) for i in range(24)]
    return pd.DataFrame({
        'timestamp_utc': timestamps,
        'forecast_price_eur_per_mwh': [50.0 + i for i in range(24)]
    })


class TestHealthCheck:
    """Tests for health check endpoint."""
    
    def test_health_check_success(self, client):
        """Test health check returns healthy status."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        
        # Verify timestamp is valid ISO format
        datetime.fromisoformat(data["timestamp"])


class TestDataSummary:
    """Tests for data summary endpoint."""
    
    @patch('app.api.forecast.get_data_summary')
    def test_get_data_summary_success(self, mock_summary, client):
        """Test successful data summary retrieval."""
        mock_summary.return_value = {
            "markets": {
                "day_ahead": {"start": "2024-01-01", "end": "2024-12-31", "count": 8760}
            },
            "weather": {
                "historical": {"start": "2024-01-01", "end": "2024-12-31", "count": 8760}
            }
        }
        
        response = client.get("/api/data/summary")
        
        assert response.status_code == 200
        data = response.json()
        assert "markets" in data
        assert "weather" in data
        assert "day_ahead" in data["markets"]
        mock_summary.assert_called_once()
    
    @patch('app.api.forecast.get_data_summary', side_effect=Exception("Database error"))
    def test_get_data_summary_error(self, mock_summary, client):
        """Test data summary handles errors."""
        response = client.get("/api/data/summary")
        
        assert response.status_code == 500
        assert "Database error" in response.json()["detail"]


class TestDataRefreshIntegration:
    """Tests for data refresh integration in forecast endpoints."""
    
    @patch('app.api.forecast.ensure_fresh_data')
    @patch('app.api.forecast.get_forecast_service')
    def test_forecast_endpoint_calls_refresh(self, mock_service, mock_refresh, client, mock_forecast_df):
        """Test that forecast endpoint calls ensure_fresh_data."""
        # Setup mocks
        mock_forecast_service = MagicMock()
        mock_service.return_value = mock_forecast_service
        mock_forecast_service.generate_forecast.return_value = mock_forecast_df
        
        # Make request
        response = client.post("/api/forecast", json={
            "market_type": "day_ahead",
            "model_type": "linear_regression",
            "horizon_hours": 24
        })
        
        # Assertions
        assert response.status_code == 200
        mock_refresh.assert_called_once_with(freshness_threshold_minutes=60)
    
    @patch('app.api.forecast.ensure_fresh_data', side_effect=Exception("Refresh failed"))
    @patch('app.api.forecast.get_forecast_service')
    def test_forecast_continues_if_refresh_fails(self, mock_service, mock_refresh, client, mock_forecast_df):
        """Test that forecast continues even if refresh fails."""
        # Setup mocks
        mock_forecast_service = MagicMock()
        mock_service.return_value = mock_forecast_service
        mock_forecast_service.generate_forecast.return_value = mock_forecast_df
        
        # Make request - should succeed despite refresh failure
        response = client.post("/api/forecast", json={
            "market_type": "day_ahead",
            "model_type": "linear_regression",
            "horizon_hours": 24
        })
        
        # Should still succeed with a warning logged
        assert response.status_code == 200
        mock_refresh.assert_called_once()
    
    @patch('app.api.forecast.ensure_fresh_data')
    @patch('app.api.forecast.get_forecast_service')
    def test_compare_endpoint_calls_refresh(self, mock_service, mock_refresh, client):
        """Test that compare endpoint calls ensure_fresh_data."""
        # Setup mocks
        mock_forecast_service = MagicMock()
        mock_service.return_value = mock_forecast_service
        
        # Create mock comparison DataFrame
        timestamps = [datetime.now(pytz.UTC) + timedelta(hours=i) for i in range(24)]
        mock_comparison_df = pd.DataFrame({
            'timestamp_utc': timestamps,
            'model_type': ['persistence'] * 24,
            'forecast_price_eur_per_mwh': [50.0] * 24
        })
        mock_forecast_service.compare_models.return_value = mock_comparison_df
        
        # Make request
        response = client.post(
            "/api/forecast/compare",
            params={
                "market_type": "day_ahead",
                "model_types": ["persistence", "linear_regression"],
                "horizon_hours": 24
            }
        )
        
        # Assertions
        assert response.status_code == 200
        mock_refresh.assert_called_once_with(freshness_threshold_minutes=60)


class TestListMarkets:
    """Tests for list markets endpoint."""
    
    def test_list_markets_success(self, client):
        """Test listing all available markets."""
        response = client.get("/api/markets")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify common market types are present
        assert "day_ahead" in data
        assert "imbalance_shortage" in data
        assert "imbalance_surplus" in data
        
        # Verify structure
        for market in data.values():
            assert "display_name" in market


class TestListModels:
    """Tests for list models endpoint."""
    
    def test_list_models_success(self, client):
        """Test listing all available models."""
        response = client.get("/api/models")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify common model types are present
        assert "persistence" in data
        assert "linear_regression" in data
        assert "random_forest" in data
        
        # Verify structure
        for model in data.values():
            assert "display_name" in model
            assert "description" in model


class TestGenerateForecast:
    """Tests for forecast generation endpoint."""
    
    @patch('app.api.forecast.get_forecast_service')
    def test_generate_forecast_success(self, mock_service, client, mock_forecast_df):
        """Test successful forecast generation."""
        # Mock the service
        mock_instance = MagicMock()
        mock_instance.generate_forecast.return_value = mock_forecast_df
        mock_service.return_value = mock_instance
        
        request_data = {
            "market_type": "day_ahead",
            "model_type": "linear_regression",
            "horizon_hours": 24
        }
        
        response = client.post("/api/forecast", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "timestamps" in data
        assert "forecasts" in data
        assert data["market_type"] == "day_ahead"
        assert data["model_type"] == "linear_regression"
        assert data["horizon_hours"] == 24
        assert len(data["timestamps"]) == 24
        assert len(data["forecasts"]) == 24
        
        # Verify service was called correctly
        mock_instance.generate_forecast.assert_called_once()
        call_kwargs = mock_instance.generate_forecast.call_args[1]
        assert call_kwargs["market_type"] == "day_ahead"
        assert call_kwargs["model_type"] == "linear_regression"
        assert call_kwargs["horizon_hours"] == 24
    
    @patch('app.api.forecast.get_forecast_service')
    def test_generate_forecast_with_start_time(self, mock_service, client, mock_forecast_df):
        """Test forecast generation with custom start time."""
        mock_instance = MagicMock()
        mock_instance.generate_forecast.return_value = mock_forecast_df
        mock_service.return_value = mock_instance
        
        request_data = {
            "market_type": "day_ahead",
            "model_type": "persistence",
            "horizon_hours": 12,
            "forecast_start": "2024-01-15T10:00:00Z"
        }
        
        response = client.post("/api/forecast", json=request_data)
        
        assert response.status_code == 200
        
        # Verify start time was parsed and passed
        call_kwargs = mock_instance.generate_forecast.call_args[1]
        assert call_kwargs["forecast_start"] is not None
        assert isinstance(call_kwargs["forecast_start"], datetime)
    
    def test_generate_forecast_invalid_market_type(self, client):
        """Test forecast with invalid market type."""
        request_data = {
            "market_type": "invalid_market",
            "model_type": "linear_regression",
            "horizon_hours": 24
        }
        
        response = client.post("/api/forecast", json=request_data)
        
        assert response.status_code == 400
        assert "Invalid market type" in response.json()["detail"]
    
    def test_generate_forecast_invalid_model_type(self, client):
        """Test forecast with invalid model type."""
        request_data = {
            "market_type": "day_ahead",
            "model_type": "invalid_model",
            "horizon_hours": 24
        }
        
        response = client.post("/api/forecast", json=request_data)
        
        assert response.status_code == 400
        assert "Invalid model type" in response.json()["detail"]
    
    def test_generate_forecast_invalid_horizon(self, client):
        """Test forecast with invalid horizon."""
        request_data = {
            "market_type": "day_ahead",
            "model_type": "linear_regression",
            "horizon_hours": 100  # Exceeds maximum of 36
        }
        
        response = client.post("/api/forecast", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_generate_forecast_invalid_timestamp_format(self, client):
        """Test forecast with invalid timestamp format."""
        request_data = {
            "market_type": "day_ahead",
            "model_type": "linear_regression",
            "horizon_hours": 24,
            "forecast_start": "not-a-timestamp"
        }
        
        response = client.post("/api/forecast", json=request_data)
        
        assert response.status_code == 400
        assert "Invalid timestamp format" in response.json()["detail"]
    
    @patch('app.api.forecast.get_forecast_service')
    def test_generate_forecast_empty_result(self, mock_service, client):
        """Test forecast when service returns empty DataFrame."""
        mock_instance = MagicMock()
        mock_instance.generate_forecast.return_value = pd.DataFrame()
        mock_service.return_value = mock_instance
        
        request_data = {
            "market_type": "day_ahead",
            "model_type": "linear_regression",
            "horizon_hours": 24
        }
        
        response = client.post("/api/forecast", json=request_data)
        
        assert response.status_code == 500
        assert "Failed to generate forecast" in response.json()["detail"]
    
    @patch('app.api.forecast.get_forecast_service')
    def test_generate_forecast_service_exception(self, mock_service, client):
        """Test forecast when service raises exception."""
        mock_instance = MagicMock()
        mock_instance.generate_forecast.side_effect = ValueError("Insufficient data")
        mock_service.return_value = mock_instance
        
        request_data = {
            "market_type": "day_ahead",
            "model_type": "linear_regression",
            "horizon_hours": 24
        }
        
        response = client.post("/api/forecast", json=request_data)
        
        assert response.status_code == 500
        assert "Insufficient data" in response.json()["detail"]
    
    def test_generate_forecast_missing_required_fields(self, client):
        """Test forecast with missing required fields."""
        request_data = {
            "market_type": "day_ahead",
            # Missing model_type and horizon_hours
        }
        
        response = client.post("/api/forecast", json=request_data)
        
        assert response.status_code == 422  # Validation error


class TestForecastRequestValidation:
    """Tests for request validation."""
    
    def test_horizon_minimum_validation(self, client):
        """Test that horizon must be at least 1."""
        request_data = {
            "market_type": "day_ahead",
            "model_type": "linear_regression",
            "horizon_hours": 0
        }
        
        response = client.post("/api/forecast", json=request_data)
        assert response.status_code == 422
    
    def test_horizon_maximum_validation(self, client):
        """Test that horizon cannot exceed 36."""
        request_data = {
            "market_type": "day_ahead",
            "model_type": "linear_regression",
            "horizon_hours": 37
        }
        
        response = client.post("/api/forecast", json=request_data)
        assert response.status_code == 422
