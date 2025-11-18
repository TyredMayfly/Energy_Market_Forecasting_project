"""
Unit test fixtures.

Fixtures specific to unit tests.
"""

from unittest.mock import Mock, MagicMock
import pytest


@pytest.fixture
def mock_requests_get():
    """
    Create a mock for requests.get.
    
    Returns:
        Mock object for requests.get
    """
    return Mock()


@pytest.fixture
def mock_successful_http_response():
    """
    Create a mock successful HTTP response.
    
    Returns:
        Mock response object
    """
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    return response


@pytest.fixture
def mock_failed_http_response():
    """
    Create a mock failed HTTP response.
    
    Returns:
        Mock response object that raises HTTPError
    """
    response = Mock()
    response.status_code = 404
    response.raise_for_status = Mock(side_effect=Exception("Not found"))
    return response
