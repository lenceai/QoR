"""
Basic tests for Phase 1 - Foundation and Core Infrastructure.
"""

import pytest
from fastapi.testclient import TestClient

# Import from src
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.main import app
from src.core.config import settings
from src.utils.helpers import generate_uuid, normalize_string, validate_email


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


def test_system_info(client):
    """Test the system info endpoint."""
    response = client.get("/api/v1/system/info")
    assert response.status_code == 200
    
    data = response.json()
    assert data["name"] == settings.PROJECT_NAME
    assert data["version"] == settings.VERSION
    assert "api_version" in data


def test_health_check_basic(client):
    """Test the basic health check endpoint."""
    response = client.get("/api/v1/system/health")
    assert response.status_code == 200


def test_papers_placeholder(client):
    """Test papers endpoints return placeholders."""
    response = client.get("/api/v1/papers/")
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert "Phase 2" in data["message"]


def test_utility_functions():
    """Test utility functions."""
    # Test UUID generation
    uuid1 = generate_uuid()
    uuid2 = generate_uuid()
    assert uuid1 != uuid2
    assert len(uuid1) == 36  # Standard UUID length
    
    # Test string normalization
    assert normalize_string("  Hello   World  ") == "hello world"
    assert normalize_string("") == ""
    
    # Test email validation
    assert validate_email("test@example.com") == True
    assert validate_email("invalid-email") == False
    assert validate_email("test@.com") == False


def test_config_loading():
    """Test configuration loading."""
    assert settings.PROJECT_NAME == "CERN Knowledge Explorer"
    assert settings.VERSION == "1.0.0"
    assert settings.API_V1_STR == "/api/v1" 