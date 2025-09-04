import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_api_docs_available():
    """Test that API documentation is available."""
    response = client.get("/docs")
    assert response.status_code == 200

def test_openapi_schema():
    """Test that OpenAPI schema is available."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "Food Vision Pro API"
    assert "paths" in schema

def test_cors_headers():
    """Test that CORS headers are properly set."""
    response = client.options("/health")
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers

def test_app_info():
    """Test that app information is correct."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "Food Vision Pro"
    assert data["version"] == "1.0.0"
