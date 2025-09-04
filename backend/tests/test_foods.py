import pytest
from fastapi import status

def test_search_foods_success(client, test_user_data):
    """Test successful food search."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Search for foods
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/api/v1/foods/search?q=chicken", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert "foods" in data
    assert "total" in data
    assert isinstance(data["foods"], list)

def test_search_foods_no_auth(client):
    """Test food search without authentication."""
    response = client.get("/api/v1/foods/search?q=chicken")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_search_foods_empty_query(client, test_user_data):
    """Test food search with empty query."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Search with empty query
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/api/v1/foods/search?q=", headers=headers)
    assert response.status_code == status.HTTP_400_BAD_REQUEST

def test_search_foods_no_results(client, test_user_data):
    """Test food search with no results."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Search for non-existent food
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/api/v1/foods/search?q=nonexistentfood", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert len(data["foods"]) == 0
    assert data["total"] == 0

def test_get_food_by_barcode_success(client, test_user_data):
    """Test successful food retrieval by barcode."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Get food by barcode
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/api/v1/foods/barcode/1234567890123", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert "barcode" in data
    assert data["barcode"] == "1234567890123"

def test_get_food_by_barcode_not_found(client, test_user_data):
    """Test food retrieval by non-existent barcode."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Try to get non-existent barcode
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/api/v1/foods/barcode/0000000000000", headers=headers)
    assert response.status_code == status.HTTP_404_NOT_FOUND

def test_get_food_by_barcode_no_auth(client):
    """Test food retrieval by barcode without authentication."""
    response = client.get("/api/v1/foods/barcode/1234567890123")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_get_popular_foods_success(client, test_user_data):
    """Test successful popular foods retrieval."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Get popular foods
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/api/v1/foods/popular", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert "foods" in data
    assert isinstance(data["foods"], list)

def test_get_popular_foods_no_auth(client):
    """Test popular foods retrieval without authentication."""
    response = client.get("/api/v1/foods/popular")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_search_foods_with_pagination(client, test_user_data):
    """Test food search with pagination."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Search with pagination
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/api/v1/foods/search?q=chicken&page=1&size=10", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert "page" in data
    assert "size" in data
    assert "total" in data

def test_search_foods_response_structure(client, test_user_data):
    """Test that food search response has correct structure."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Search for foods
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/api/v1/foods/search?q=chicken", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    
    # Check response structure
    assert "foods" in data
    assert "total" in data
    
    # Check foods structure if any exist
    if len(data["foods"]) > 0:
        food = data["foods"][0]
        required_fields = ["id", "name", "barcode", "calories_per_100g", "protein_per_100g", "carb_per_100g", "fat_per_100g"]
        for field in required_fields:
            assert field in food

def test_food_barcode_response_structure(client, test_user_data):
    """Test that food barcode response has correct structure."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Get food by barcode
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/api/v1/foods/barcode/1234567890123", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    
    # Check response structure
    required_fields = ["id", "name", "barcode", "calories_per_100g", "protein_per_100g", "carb_per_100g", "fat_per_100g"]
    for field in required_fields:
        assert field in data
    
    # Check data types
    assert isinstance(data["calories_per_100g"], (int, float))
    assert isinstance(data["protein_per_100g"], (int, float))
    assert isinstance(data["carb_per_100g"], (int, float))
    assert isinstance(data["fat_per_100g"], (int, float))

def test_search_foods_special_characters(client, test_user_data):
    """Test food search with special characters."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Search with special characters
    headers = {"Authorization": f"Bearer {access_token}"}
    special_queries = ["chicken & rice", "pasta's", "fish & chips", "coffee+cream"]
    
    for query in special_queries:
        response = client.get(f"/api/v1/foods/search?q={query}", headers=headers)
        assert response.status_code == status.HTTP_200_OK

def test_search_foods_long_query(client, test_user_data):
    """Test food search with very long query."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Search with long query
    long_query = "a" * 1000  # 1000 character query
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get(f"/api/v1/foods/search?q={long_query}", headers=headers)
    assert response.status_code == status.HTTP_200_OK

def test_food_cache_behavior(client, test_user_data):
    """Test that food search results are cached appropriately."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # First search
    headers = {"Authorization": f"Bearer {access_token}"}
    response1 = client.get("/api/v1/foods/search?q=chicken", headers=headers)
    assert response1.status_code == status.HTTP_200_OK
    
    # Second search (should be faster due to caching)
    response2 = client.get("/api/v1/foods/search?q=chicken", headers=headers)
    assert response2.status_code == status.HTTP_200_OK
    
    # Both responses should be identical
    assert response1.json() == response2.json()

def test_food_search_performance(client, test_user_data):
    """Test that food search responds within reasonable time."""
    import time
    
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Measure search time
    headers = {"Authorization": f"Bearer {access_token}"}
    start_time = time.time()
    response = client.get("/api/v1/foods/search?q=chicken", headers=headers)
    end_time = time.time()
    
    assert response.status_code == status.HTTP_200_OK
    assert (end_time - start_time) < 5.0  # Should respond within 5 seconds
