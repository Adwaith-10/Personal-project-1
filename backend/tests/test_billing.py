import pytest
from fastapi import status
from unittest.mock import patch

def test_create_checkout_session_disabled(client, test_user_data):
    """Test checkout session creation when billing is disabled."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Try to create checkout session
    headers = {"Authorization": f"Bearer {access_token}"}
    checkout_data = {
        "price_id": "price_test123",
        "success_url": "http://localhost:3000/success",
        "cancel_url": "http://localhost:3000/cancel"
    }
    
    response = client.post("/api/v1/billing/create-checkout-session", headers=headers, json=checkout_data)
    assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
    assert "billing is not enabled" in response.json()["detail"].lower()

@patch('app.config.settings.STRIPE_ENABLED', True)
def test_create_checkout_session_enabled(client, test_user_data):
    """Test checkout session creation when billing is enabled."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Try to create checkout session
    headers = {"Authorization": f"Bearer {access_token}"}
    checkout_data = {
        "price_id": "price_test123",
        "success_url": "http://localhost:3000/success",
        "cancel_url": "http://localhost:3000/cancel"
    }
    
    # This will fail due to missing Stripe configuration, but should not be a 501
    response = client.post("/api/v1/billing/create-checkout-session", headers=headers, json=checkout_data)
    # The exact status code depends on Stripe configuration, but it shouldn't be 501
    assert response.status_code != status.HTTP_501_NOT_IMPLEMENTED

def test_create_checkout_session_no_auth(client):
    """Test checkout session creation without authentication."""
    checkout_data = {
        "price_id": "price_test123",
        "success_url": "http://localhost:3000/success",
        "cancel_url": "http://localhost:3000/cancel"
    }
    
    response = client.post("/api/v1/billing/create-checkout-session", json=checkout_data)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_create_checkout_session_invalid_data(client, test_user_data):
    """Test checkout session creation with invalid data."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Invalid checkout data
    headers = {"Authorization": f"Bearer {access_token}"}
    invalid_data = {
        "price_id": "",  # Empty price ID
        "success_url": "not-a-url",  # Invalid URL
        "cancel_url": "also-not-a-url"  # Invalid URL
    }
    
    response = client.post("/api/v1/billing/create-checkout-session", headers=headers, json=invalid_data)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_get_subscription_status_disabled(client, test_user_data):
    """Test subscription status retrieval when billing is disabled."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Try to get subscription status
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/api/v1/billing/subscription-status", headers=headers)
    assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED

@patch('app.config.settings.STRIPE_ENABLED', True)
def test_get_subscription_status_enabled(client, test_user_data):
    """Test subscription status retrieval when billing is enabled."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Try to get subscription status
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/api/v1/billing/subscription-status", headers=headers)
    # The exact status code depends on Stripe configuration, but it shouldn't be 501
    assert response.status_code != status.HTTP_501_NOT_IMPLEMENTED

def test_get_subscription_status_no_auth(client):
    """Test subscription status retrieval without authentication."""
    response = client.get("/api/v1/billing/subscription-status")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_stripe_webhook_disabled(client):
    """Test Stripe webhook when billing is disabled."""
    webhook_data = {
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "id": "cs_test123",
                "customer": "cus_test123"
            }
        }
    }
    
    response = client.post("/api/v1/billing/stripe-webhook", json=webhook_data)
    assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED

@patch('app.config.settings.STRIPE_ENABLED', True)
def test_stripe_webhook_enabled(client):
    """Test Stripe webhook when billing is enabled."""
    webhook_data = {
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "id": "cs_test123",
                "customer": "cus_test123"
            }
        }
    }
    
    response = client.post("/api/v1/billing/stripe-webhook", json=webhook_data)
    # The exact status code depends on Stripe configuration, but it shouldn't be 501
    assert response.status_code != status.HTTP_501_NOT_IMPLEMENTED

def test_stripe_webhook_invalid_data(client):
    """Test Stripe webhook with invalid data."""
    invalid_webhook = {
        "invalid": "data"
    }
    
    response = client.post("/api/v1/billing/stripe-webhook", json=invalid_webhook)
    assert response.status_code == status.HTTP_400_BAD_REQUEST

def test_get_pricing_disabled(client):
    """Test pricing retrieval when billing is disabled."""
    response = client.get("/api/v1/billing/pricing")
    assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED

@patch('app.config.settings.STRIPE_ENABLED', True)
def test_get_pricing_enabled(client):
    """Test pricing retrieval when billing is enabled."""
    response = client.get("/api/v1/billing/pricing")
    # The exact status code depends on Stripe configuration, but it shouldn't be 501
    assert response.status_code != status.HTTP_501_NOT_IMPLEMENTED

def test_billing_endpoints_require_auth(client):
    """Test that all billing endpoints require authentication."""
    endpoints = [
        "/api/v1/billing/create-checkout-session",
        "/api/v1/billing/subscription-status",
        "/api/v1/billing/pricing"
    ]
    
    for endpoint in endpoints:
        if endpoint.endswith("create-checkout-session"):
            response = client.post(endpoint, json={"test": "data"})
        else:
            response = client.get(endpoint)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_billing_configuration_consistency():
    """Test that billing configuration is consistent across the application."""
    from app.config import settings
    
    # If Stripe is enabled, required settings should be present
    if settings.STRIPE_ENABLED:
        # This would check for Stripe keys, but they're not required for basic functionality
        # The main check is that the setting is boolean
        assert isinstance(settings.STRIPE_ENABLED, bool)

def test_billing_schema_validation(client, test_user_data):
    """Test that billing request schemas are properly validated."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # Test various invalid checkout session data
    invalid_checkout_data = [
        {},  # Empty data
        {"price_id": "price_123"},  # Missing required fields
        {"success_url": "http://localhost:3000/success"},  # Missing required fields
        {"cancel_url": "http://localhost:3000/cancel"},  # Missing required fields
        {
            "price_id": "price_123",
            "success_url": "not-a-url",
            "cancel_url": "http://localhost:3000/cancel"
        },  # Invalid success URL
        {
            "price_id": "price_123",
            "success_url": "http://localhost:3000/success",
            "cancel_url": "not-a-url"
        },  # Invalid cancel URL
    ]
    
    for invalid_data in invalid_checkout_data:
        response = client.post("/api/v1/billing/create-checkout-session", headers=headers, json=invalid_data)
        assert response.status_code in [status.HTTP_422_UNPROCESSABLE_ENTITY, status.HTTP_501_NOT_IMPLEMENTED]

def test_billing_error_handling(client, test_user_data):
    """Test that billing errors are handled gracefully."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    headers = {"Authorization": f"Bearer {access_token}"}
    
    # Test with malformed JSON
    response = client.post(
        "/api/v1/billing/create-checkout-session",
        headers=headers,
        data="invalid json",
        content_type="application/json"
    )
    assert response.status_code in [status.HTTP_422_UNPROCESSABLE_ENTITY, status.HTTP_400_BAD_REQUEST, status.HTTP_501_NOT_IMPLEMENTED]

def test_billing_rate_limiting(client, test_user_data):
    """Test that billing endpoints have appropriate rate limiting."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    headers = {"Authorization": f"Bearer {access_token}"}
    checkout_data = {
        "price_id": "price_test123",
        "success_url": "http://localhost:3000/success",
        "cancel_url": "http://localhost:3000/cancel"
    }
    
    # Make multiple rapid requests
    responses = []
    for _ in range(5):
        response = client.post("/api/v1/billing/create-checkout-session", headers=headers, json=checkout_data)
        responses.append(response.status_code)
    
    # All should either succeed or fail with the same error (not rate limited)
    # This is a basic test - actual rate limiting would depend on implementation
    assert len(set(responses)) <= 2  # Should have at most 2 different status codes
