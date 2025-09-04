import pytest
from fastapi import status
from app.models import User

def test_signup_success(client, test_user_data):
    """Test successful user signup."""
    response = client.post("/api/v1/auth/signup", json=test_user_data)
    assert response.status_code == status.HTTP_201_CREATED
    
    data = response.json()
    assert "id" in data
    assert data["email"] == test_user_data["email"]
    assert "hashed_password" not in data
    assert "created_at" in data

def test_signup_duplicate_email(client, test_user_data):
    """Test signup with duplicate email."""
    # First signup
    client.post("/api/v1/auth/signup", json=test_user_data)
    
    # Second signup with same email
    response = client.post("/api/v1/auth/signup", json=test_user_data)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "email already registered" in response.json()["detail"].lower()

def test_signup_invalid_email(client):
    """Test signup with invalid email."""
    invalid_data = {"email": "invalid-email", "password": "testpass123"}
    response = client.post("/api/v1/auth/signup", json=invalid_data)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_signup_short_password(client):
    """Test signup with short password."""
    invalid_data = {"email": "test@example.com", "password": "123"}
    response = client.post("/api/v1/auth/signup", json=invalid_data)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_login_success(client, test_user_data):
    """Test successful user login."""
    # First signup
    client.post("/api/v1/auth/signup", json=test_user_data)
    
    # Then login
    response = client.post("/api/v1/auth/login", json=test_user_data)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert "token_type" in data
    assert data["token_type"] == "bearer"

def test_login_invalid_credentials(client, test_user_data):
    """Test login with invalid credentials."""
    # First signup
    client.post("/api/v1/auth/signup", json=test_user_data)
    
    # Login with wrong password
    wrong_password_data = {
        "email": test_user_data["email"],
        "password": "wrongpassword"
    }
    response = client.post("/api/v1/auth/login", json=wrong_password_data)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "invalid credentials" in response.json()["detail"].lower()

def test_login_nonexistent_user(client):
    """Test login with nonexistent user."""
    nonexistent_data = {
        "email": "nonexistent@example.com",
        "password": "testpass123"
    }
    response = client.post("/api/v1/auth/login", json=nonexistent_data)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_refresh_token_success(client, test_user_data):
    """Test successful token refresh."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    refresh_token = login_response.json()["refresh_token"]
    
    # Refresh token
    response = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data

def test_refresh_token_invalid(client):
    """Test token refresh with invalid token."""
    response = client.post("/api/v1/auth/refresh", json={"refresh_token": "invalid_token"})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_get_current_user_success(client, test_user_data):
    """Test getting current user with valid token."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Get current user
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/api/v1/auth/me", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert data["email"] == test_user_data["email"]
    assert "id" in data
    assert "created_at" in data

def test_get_current_user_invalid_token(client):
    """Test getting current user with invalid token."""
    headers = {"Authorization": "Bearer invalid_token"}
    response = client.get("/api/v1/auth/me", headers=headers)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_get_current_user_no_token(client):
    """Test getting current user without token."""
    response = client.get("/api/v1/auth/me")
    assert response.status_code == status.HTTP_403_FORBIDDEN

def test_logout_success(client, test_user_data):
    """Test successful logout."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    refresh_token = login_response.json()["refresh_token"]
    
    # Logout
    response = client.post("/api/v1/auth/logout", json={"refresh_token": refresh_token})
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["message"] == "Successfully logged out"

def test_logout_invalid_token(client):
    """Test logout with invalid token."""
    response = client.post("/api/v1/auth/logout", json={"refresh_token": "invalid_token"})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_password_hashing():
    """Test that passwords are properly hashed."""
    from app.security import get_password_hash, verify_password
    
    password = "testpassword123"
    hashed = get_password_hash(password)
    
    # Hash should be different from original password
    assert hashed != password
    
    # Verification should work
    assert verify_password(password, hashed) is True
    
    # Wrong password should fail
    assert verify_password("wrongpassword", hashed) is False

def test_jwt_token_creation():
    """Test JWT token creation and verification."""
    from app.security import create_access_token, verify_token
    
    user_data = {"sub": "test@example.com", "user_id": "123"}
    token = create_access_token(user_data)
    
    # Token should be a string
    assert isinstance(token, str)
    assert len(token) > 0
    
    # Token should be verifiable
    payload = verify_token(token, "access")
    assert payload is not None
    assert payload["sub"] == "test@example.com"
    assert payload["user_id"] == "123"
