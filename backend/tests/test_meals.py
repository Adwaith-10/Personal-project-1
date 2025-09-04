import pytest
from fastapi import status
from datetime import datetime, timedelta

def test_create_meal_success(client, test_user_data, test_meal_data):
    """Test successful meal creation."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Create meal
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.post("/api/v1/meals/", headers=headers, json=test_meal_data)
    assert response.status_code == status.HTTP_201_CREATED
    
    data = response.json()
    assert "id" in data
    assert data["name"] == test_meal_data["name"]
    assert data["user_id"] is not None
    assert "created_at" in data
    assert len(data["items"]) == len(test_meal_data["items"])

def test_create_meal_no_auth(client, test_meal_data):
    """Test meal creation without authentication."""
    response = client.post("/api/v1/meals/", json=test_meal_data)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_create_meal_invalid_data(client, test_user_data):
    """Test meal creation with invalid data."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Invalid meal data
    invalid_meal = {
        "name": "",  # Empty name
        "items": []  # Empty items
    }
    
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.post("/api/v1/meals/", headers=headers, json=invalid_meal)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_get_meals_success(client, test_user_data, test_meal_data):
    """Test successful meal retrieval."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Create a meal first
    headers = {"Authorization": f"Bearer {access_token}"}
    client.post("/api/v1/meals/", headers=headers, json=test_meal_data)
    
    # Get meals
    response = client.get("/api/v1/meals/", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "size" in data
    assert len(data["items"]) > 0

def test_get_meals_with_date_filter(client, test_user_data, test_meal_data):
    """Test meal retrieval with date filtering."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Create a meal first
    headers = {"Authorization": f"Bearer {access_token}"}
    client.post("/api/v1/meals/", headers=headers, json=test_meal_data)
    
    # Get meals with date filter
    today = datetime.now().date()
    from_date = (today - timedelta(days=7)).isoformat()
    to_date = (today + timedelta(days=7)).isoformat()
    
    response = client.get(
        f"/api/v1/meals/?from_date={from_date}&to_date={to_date}",
        headers=headers
    )
    assert response.status_code == status.HTTP_200_OK

def test_get_meals_with_pagination(client, test_user_data, test_meal_data):
    """Test meal retrieval with pagination."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Create a meal first
    headers = {"Authorization": f"Bearer {access_token}"}
    client.post("/api/v1/meals/", headers=headers, json=test_meal_data)
    
    # Get meals with pagination
    response = client.get("/api/v1/meals/?page=1&size=10", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert data["page"] == 1
    assert data["size"] == 10

def test_get_meal_by_id_success(client, test_user_data, test_meal_data):
    """Test successful meal retrieval by ID."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Create a meal first
    headers = {"Authorization": f"Bearer {access_token}"}
    create_response = client.post("/api/v1/meals/", headers=headers, json=test_meal_data)
    meal_id = create_response.json()["id"]
    
    # Get meal by ID
    response = client.get(f"/api/v1/meals/{meal_id}", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert data["id"] == meal_id
    assert data["name"] == test_meal_data["name"]

def test_get_meal_by_id_not_found(client, test_user_data):
    """Test meal retrieval by non-existent ID."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Try to get non-existent meal
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/api/v1/meals/nonexistent-id", headers=headers)
    assert response.status_code == status.HTTP_404_NOT_FOUND

def test_update_meal_success(client, test_user_data, test_meal_data):
    """Test successful meal update."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Create a meal first
    headers = {"Authorization": f"Bearer {access_token}"}
    create_response = client.post("/api/v1/meals/", headers=headers, json=test_meal_data)
    meal_id = create_response.json()["id"]
    
    # Update meal
    update_data = {
        "name": "Updated Meal Name",
        "items": [
            {
                "label": "updated_food",
                "grams": 200.0,
                "calories": 300.0,
                "protein_g": 35.0,
                "carb_g": 10.0,
                "fat_g": 12.0
            }
        ]
    }
    
    response = client.patch(f"/api/v1/meals/{meal_id}", headers=headers, json=update_data)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert data["name"] == update_data["name"]
    assert len(data["items"]) == len(update_data["items"])

def test_update_meal_not_found(client, test_user_data):
    """Test meal update with non-existent ID."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Try to update non-existent meal
    headers = {"Authorization": f"Bearer {access_token}"}
    update_data = {"name": "Updated Name"}
    
    response = client.patch("/api/v1/meals/nonexistent-id", headers=headers, json=update_data)
    assert response.status_code == status.HTTP_404_NOT_FOUND

def test_delete_meal_success(client, test_user_data, test_meal_data):
    """Test successful meal deletion."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Create a meal first
    headers = {"Authorization": f"Bearer {access_token}"}
    create_response = client.post("/api/v1/meals/", headers=headers, json=test_meal_data)
    meal_id = create_response.json()["id"]
    
    # Delete meal
    response = client.delete(f"/api/v1/meals/{meal_id}", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    
    # Verify meal is deleted
    get_response = client.get(f"/api/v1/meals/{meal_id}", headers=headers)
    assert get_response.status_code == status.HTTP_404_NOT_FOUND

def test_delete_meal_not_found(client, test_user_data):
    """Test meal deletion with non-existent ID."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Try to delete non-existent meal
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.delete("/api/v1/meals/nonexistent-id", headers=headers)
    assert response.status_code == status.HTTP_404_NOT_FOUND

def test_get_daily_totals_success(client, test_user_data, test_meal_data):
    """Test successful daily totals retrieval."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Create a meal first
    headers = {"Authorization": f"Bearer {access_token}"}
    client.post("/api/v1/meals/", headers=headers, json=test_meal_data)
    
    # Get daily totals
    today = datetime.now().date().isoformat()
    response = client.get(f"/api/v1/meals/daily-totals/{today}", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert "date" in data
    assert "total_calories" in data
    assert "total_protein" in data
    assert "total_carbs" in data
    assert "total_fat" in data
    assert "meal_count" in data

def test_get_daily_totals_no_meals(client, test_user_data):
    """Test daily totals retrieval when no meals exist."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Get daily totals for a date with no meals
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/api/v1/meals/daily-totals/2023-01-01", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert data["total_calories"] == 0
    assert data["total_protein"] == 0
    assert data["total_carbs"] == 0
    assert data["total_fat"] == 0
    assert data["meal_count"] == 0

def test_meal_items_structure(client, test_user_data, test_meal_data):
    """Test that meal items have correct structure."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Create a meal
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.post("/api/v1/meals/", headers=headers, json=test_meal_data)
    assert response.status_code == status.HTTP_201_CREATED
    
    data = response.json()
    assert len(data["items"]) > 0
    
    # Check item structure
    item = data["items"][0]
    required_fields = ["id", "label", "grams", "calories", "protein_g", "carb_g", "fat_g"]
    for field in required_fields:
        assert field in item
    
    # Check data types
    assert isinstance(item["grams"], (int, float))
    assert isinstance(item["calories"], (int, float))
    assert isinstance(item["protein_g"], (int, float))
    assert isinstance(item["carb_g"], (int, float))
    assert isinstance(item["fat_g"], (int, float))
