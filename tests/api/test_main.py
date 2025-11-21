"""
Tests for FastAPI main application.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


def test_create_app_structure():
    """Test that app is created with correct structure."""
    from app.api.main import create_app
    
    app = create_app()
    
    assert app.title == "Market Forecasting API"
    assert app.version == "0.1.0"
    assert "forecasting" in [tag for route in app.routes for tag in getattr(route, 'tags', [])]


def test_root_endpoint():
    """Test root endpoint returns correct information."""
    from app.api.main import create_app
    
    # Create app without lifespan to avoid scheduler
    with patch('app.api.main.lifespan'):
        app = create_app()
        client = TestClient(app)
        
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Market Forecasting API"
        assert data["version"] == "0.1.0"
        assert data["status"] == "running"


def test_cors_configured():
    """Test that CORS middleware is configured."""
    from app.api.main import create_app
    
    with patch('app.api.main.lifespan'):
        app = create_app()
        
        # Check that middleware is configured
        # CORS adds to user_middleware
        assert len(app.user_middleware) > 0


@patch('app.api.main.start_scheduler')
@patch('app.api.main.stop_scheduler')
def test_lifespan_startup_shutdown(mock_stop, mock_start):
    """Test lifespan events call scheduler functions."""
    from app.api.main import create_app
    
    app = create_app()
    
    with TestClient(app) as client:
        # Verify startup was called
        mock_start.assert_called_once()
        
        # Make a request to ensure app is running
        response = client.get("/")
        assert response.status_code == 200
    
    # Verify shutdown was called
    mock_stop.assert_called_once()


@patch('app.api.main.start_scheduler', side_effect=Exception("Scheduler error"))
def test_lifespan_startup_error_handling(mock_start):
    """Test that startup errors are handled gracefully."""
    from app.api.main import create_app
    
    # App should still start even if scheduler fails
    app = create_app()
    
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
    
    mock_start.assert_called_once()


@patch('app.api.main.stop_scheduler', side_effect=Exception("Stop error"))
@patch('app.api.main.start_scheduler')
def test_lifespan_shutdown_error_handling(mock_start, mock_stop):
    """Test that shutdown errors are handled gracefully."""
    from app.api.main import create_app
    
    app = create_app()
    
    # Should not raise exception even if stop fails
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
    
    mock_stop.assert_called_once()
