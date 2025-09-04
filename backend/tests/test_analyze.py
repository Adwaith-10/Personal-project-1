import pytest
from fastapi import status
import io

def test_analyze_food_image_success(
    client, 
    test_user_data, 
    mock_image_data,
    mock_segmentation_service,
    mock_classification_service,
    mock_portion_service,
    mock_nutrition_service
):
    """Test successful food image analysis."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Prepare image file
    image_file = io.BytesIO(mock_image_data)
    image_file.name = "test_image.jpg"
    
    # Analyze image
    headers = {"Authorization": f"Bearer {access_token}"}
    files = {"image": ("test_image.jpg", image_file, "image/jpeg")}
    data = {"plate_diameter_cm": "26.0"}
    
    response = client.post("/api/v1/analyze", headers=headers, files=files, data=data)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert "items" in data
    assert "totals" in data
    assert "processing_time_ms" in data
    assert len(data["items"]) > 0

def test_analyze_food_image_no_auth(client, mock_image_data):
    """Test food image analysis without authentication."""
    image_file = io.BytesIO(mock_image_data)
    image_file.name = "test_image.jpg"
    
    files = {"image": ("test_image.jpg", image_file, "image/jpeg")}
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

def test_analyze_food_image_invalid_file_type(client, test_user_data):
    """Test food image analysis with invalid file type."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Prepare invalid file
    invalid_file = io.BytesIO(b"not an image")
    invalid_file.name = "test.txt"
    
    headers = {"Authorization": f"Bearer {access_token}"}
    files = {"image": ("test.txt", invalid_file, "text/plain")}
    
    response = client.post("/api/v1/analyze", headers=headers, files=files)
    assert response.status_code == status.HTTP_400_BAD_REQUEST

def test_analyze_food_image_with_barcode(
    client, 
    test_user_data, 
    mock_image_data,
    mock_segmentation_service,
    mock_classification_service,
    mock_portion_service,
    mock_nutrition_service
):
    """Test food image analysis with barcode."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Prepare image file
    image_file = io.BytesIO(mock_image_data)
    image_file.name = "test_image.jpg"
    
    # Analyze image with barcode
    headers = {"Authorization": f"Bearer {access_token}"}
    files = {"image": ("test_image.jpg", image_file, "image/jpeg")}
    data = {"plate_diameter_cm": "26.0", "barcode": "1234567890123"}
    
    response = client.post("/api/v1/analyze", headers=headers, files=files, data=data)
    assert response.status_code == status.HTTP_200_OK

def test_analyze_food_image_with_overrides(
    client, 
    test_user_data, 
    mock_image_data,
    mock_segmentation_service,
    mock_classification_service,
    mock_portion_service,
    mock_nutrition_service
):
    """Test food image analysis with manual overrides."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Prepare image file
    image_file = io.BytesIO(mock_image_data)
    image_file.name = "test_image.jpg"
    
    # Prepare overrides
    overrides = {
        "m0": {
            "label": "custom_food",
            "grams": 200.0
        }
    }
    
    # Analyze image with overrides
    headers = {"Authorization": f"Bearer {access_token}"}
    files = {"image": ("test_image.jpg", image_file, "image/jpeg")}
    data = {
        "plate_diameter_cm": "26.0",
        "overrides": str(overrides)
    }
    
    response = client.post("/api/v1/analyze", headers=headers, files=files, data=data)
    assert response.status_code == status.HTTP_200_OK

def test_analyze_food_image_large_file(client, test_user_data):
    """Test food image analysis with file too large."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Prepare large file (11MB)
    large_file = io.BytesIO(b"x" * (11 * 1024 * 1024))
    large_file.name = "large_image.jpg"
    
    headers = {"Authorization": f"Bearer {access_token}"}
    files = {"image": ("large_image.jpg", large_file, "image/jpeg")}
    
    response = client.post("/api/v1/analyze", headers=headers, files=files)
    assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE

def test_analyze_food_image_invalid_plate_diameter(client, test_user_data, mock_image_data):
    """Test food image analysis with invalid plate diameter."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Prepare image file
    image_file = io.BytesIO(mock_image_data)
    image_file.name = "test_image.jpg"
    
    headers = {"Authorization": f"Bearer {access_token}"}
    files = {"image": ("test_image.jpg", image_file, "image/jpeg")}
    data = {"plate_diameter_cm": "100.0"}  # Too large
    
    response = client.post("/api/v1/analyze", headers=headers, files=files, data=data)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_analyze_food_image_missing_image(client, test_user_data):
    """Test food image analysis without image file."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    headers = {"Authorization": f"Bearer {access_token}"}
    data = {"plate_diameter_cm": "26.0"}
    
    response = client.post("/api/v1/analyze", headers=headers, data=data)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_analyze_food_image_processing_error(
    client, 
    test_user_data, 
    mock_image_data,
    mock_segmentation_service
):
    """Test food image analysis when processing fails."""
    # Mock segmentation service to raise an error
    mock_segmentation_service.detect_foods.side_effect = Exception("Processing error")
    
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Prepare image file
    image_file = io.BytesIO(mock_image_data)
    image_file.name = "test_image.jpg"
    
    headers = {"Authorization": f"Bearer {access_token}"}
    files = {"image": ("test_image.jpg", image_file, "image/jpeg")}
    
    response = client.post("/api/v1/analyze", headers=headers, files=files)
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

def test_analyze_food_image_response_structure(
    client, 
    test_user_data, 
    mock_image_data,
    mock_segmentation_service,
    mock_classification_service,
    mock_portion_service,
    mock_nutrition_service
):
    """Test that analysis response has correct structure."""
    # Signup and login
    client.post("/api/v1/auth/signup", json=test_user_data)
    login_response = client.post("/api/v1/auth/login", json=test_user_data)
    access_token = login_response.json()["access_token"]
    
    # Prepare image file
    image_file = io.BytesIO(mock_image_data)
    image_file.name = "test_image.jpg"
    
    headers = {"Authorization": f"Bearer {access_token}"}
    files = {"image": ("test_image.jpg", image_file, "image/jpeg")}
    
    response = client.post("/api/v1/analyze", headers=headers, files=files)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    
    # Check response structure
    assert "items" in data
    assert "totals" in data
    assert "processing_time_ms" in data
    
    # Check items structure
    if len(data["items"]) > 0:
        item = data["items"][0]
        required_fields = ["mask_id", "label", "candidates", "grams_est", "kcal", "protein_g", "carb_g", "fat_g"]
        for field in required_fields:
            assert field in item
    
    # Check totals structure
    required_totals = ["kcal", "protein_g", "carb_g", "fat_g"]
    for total in required_totals:
        assert total in data["totals"]
    
    # Check processing time
    assert isinstance(data["processing_time_ms"], int)
    assert data["processing_time_ms"] >= 0
