"""
Integration test fixtures.

Fixtures specific to integration tests.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client():
    """
    Create a FastAPI test client.
    
    Returns:
        TestClient for the FastAPI app
    """
    from app.api.main import app
    return TestClient(app)
