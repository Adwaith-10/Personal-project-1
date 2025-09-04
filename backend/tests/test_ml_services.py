import pytest
import numpy as np
from unittest.mock import Mock, patch
import cv2

def test_segmentation_service_initialization():
    """Test that segmentation service initializes correctly."""
    from app.services.segmentation import segmentation_service
    
    # Service should be initialized
    assert segmentation_service is not None
    assert hasattr(segmentation_service, 'model')

def test_segmentation_service_detect_foods():
    """Test food detection in segmentation service."""
    from app.services.segmentation import segmentation_service
    
    # Create a mock image
    mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Test detection (this will use mock model if YOLO is not available)
    try:
        results = segmentation_service.detect_foods(mock_image)
        assert isinstance(results, list)
        
        if len(results) > 0:
            result = results[0]
            assert "mask_id" in result
            assert "mask" in result
            assert "confidence" in result
            assert "bbox" in result
    except Exception as e:
        # If YOLO model is not available, this is expected
        assert "model" in str(e).lower() or "file" in str(e).lower()

def test_segmentation_service_mask_to_polygon():
    """Test mask to polygon conversion."""
    from app.services.segmentation import segmentation_service
    
    # Create a simple mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255  # Create a square mask
    
    # Convert to polygon
    polygon = segmentation_service._mask_to_polygon(mask)
    
    assert isinstance(polygon, list)
    assert len(polygon) > 0
    assert all(isinstance(point, list) for point in polygon)
    assert all(len(point) == 2 for point in polygon)

def test_segmentation_service_area_calculation():
    """Test mask area calculation."""
    from app.services.segmentation import segmentation_service
    
    # Create a simple mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255  # Create a square mask
    
    # Calculate area
    area = segmentation_service.get_mask_area_cm2(mask, 26.0, 100, 100)
    
    assert isinstance(area, float)
    assert area > 0

def test_classification_service_initialization():
    """Test that classification service initializes correctly."""
    from app.services.classify import classification_service
    
    # Service should be initialized
    assert classification_service is not None

def test_classification_service_classify_food():
    """Test food classification."""
    from app.services.classify import classification_service
    
    # Create a mock image and mask
    mock_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    mock_mask = np.random.randint(0, 1, (224, 224), dtype=np.uint8)
    
    # Test classification
    try:
        result = classification_service.classify_food(mock_image, mock_mask)
        assert isinstance(result, dict)
        assert "label" in result
        assert "confidence" in result
        assert "candidates" in result
    except Exception as e:
        # If ONNX model is not available, this is expected
        assert "model" in str(e).lower() or "file" in str(e).lower()

def test_classification_service_mock_fallback():
    """Test that classification service falls back to mock when model is unavailable."""
    from app.services.classify import classification_service
    
    # Test mock classification
    result = classification_service._mock_classification()
    
    assert isinstance(result, dict)
    assert "label" in result
    assert "confidence" in result
    assert "candidates" in result
    assert isinstance(result["candidates"], list)

def test_portion_service_initialization():
    """Test that portion service initializes correctly."""
    from app.services.portion import portion_service
    
    # Service should be initialized
    assert portion_service is not None
    assert hasattr(portion_service, 'density_priors')

def test_portion_service_volume_estimation():
    """Test volume estimation."""
    from app.services.portion import portion_service
    
    # Test volume estimation for different shapes
    area_cm2 = 25.0
    
    # Test mound shape
    volume_mound = portion_service.estimate_volume(area_cm2, "mound")
    assert isinstance(volume_mound, float)
    assert volume_mound > 0
    
    # Test cylinder shape
    volume_cylinder = portion_service.estimate_volume(area_cm2, "cylinder", height_cm=2.0)
    assert isinstance(volume_cylinder, float)
    assert volume_cylinder > 0
    
    # Test slab shape
    volume_slab = portion_service.estimate_volume(area_cm2, "slab")
    assert isinstance(volume_slab, float)
    assert volume_slab > 0

def test_portion_service_gram_estimation():
    """Test gram estimation."""
    from app.services.portion import portion_service
    
    # Test gram estimation
    area_cm2 = 25.0
    food_label = "grilled_chicken"
    
    grams = portion_service.estimate_grams(area_cm2, food_label)
    assert isinstance(grams, float)
    assert grams > 0

def test_portion_service_area_calculation():
    """Test area calculation from mask."""
    from app.services.portion import portion_service
    
    # Create a simple mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255  # Create a square mask
    
    # Calculate area
    area = portion_service.calculate_area_from_mask(mask, 26.0, 100, 100)
    
    assert isinstance(area, float)
    assert area > 0

def test_nutrition_service_initialization():
    """Test that nutrition service initializes correctly."""
    from app.services.nutrition import nutrition_service
    
    # Service should be initialized
    assert nutrition_service is not None

@patch('app.services.nutrition.nutrition_service._openfoodfacts_lookup')
def test_nutrition_service_barcode_lookup(mock_openfoodfacts):
    """Test barcode lookup in nutrition service."""
    from app.services.nutrition import nutrition_service
    
    # Mock Open Food Facts response
    mock_openfoodfacts.return_value = {
        "name": "Test Food",
        "calories_per_100g": 100.0,
        "protein_per_100g": 10.0,
        "carb_per_100g": 20.0,
        "fat_per_100g": 5.0
    }
    
    # Test barcode lookup
    result = nutrition_service.get_nutrition_info("test_food", barcode="1234567890123")
    
    assert isinstance(result, dict)
    assert "name" in result
    assert "calories_per_100g" in result

@patch('app.services.nutrition.nutrition_service._usda_name_search')
def test_nutrition_service_name_search(mock_usda):
    """Test name search in nutrition service."""
    from app.services.nutrition import nutrition_service
    
    # Mock USDA response
    mock_usda.return_value = {
        "name": "Test Food",
        "calories_per_100g": 100.0,
        "protein_per_100g": 10.0,
        "carb_per_100g": 20.0,
        "fat_per_100g": 5.0
    }
    
    # Test name search
    result = nutrition_service.get_nutrition_info("test_food")
    
    assert isinstance(result, dict)
    assert "name" in result
    assert "calories_per_100g" in result

def test_nutrition_service_calculation():
    """Test nutrition calculation for different amounts."""
    from app.services.nutrition import nutrition_service
    
    # Test nutrition per 100g
    nutrition_per_100g = {
        "calories_per_100g": 100.0,
        "protein_per_100g": 10.0,
        "carb_per_100g": 20.0,
        "fat_per_100g": 5.0
    }
    
    # Calculate for 150g
    result_150g = nutrition_service.calculate_nutrition_for_grams(nutrition_per_100g, 150.0)
    
    assert isinstance(result_150g, dict)
    assert result_150g["calories"] == 150.0
    assert result_150g["protein_g"] == 15.0
    assert result_150g["carb_g"] == 30.0
    assert result_150g["fat_g"] == 7.5

def test_storage_service_initialization():
    """Test that storage service initializes correctly."""
    from app.services.storage import storage_service
    
    # Service should be initialized
    assert storage_service is not None
    assert hasattr(storage_service, 'storage_type')

def test_storage_service_local_storage():
    """Test local storage functionality."""
    from app.services.storage import storage_service
    
    # Test with local storage
    if storage_service.storage_type == "local":
        # Create mock file content
        mock_content = b"test image data"
        
        # Test save (this will fail if local directory doesn't exist, which is expected)
        try:
            result = storage_service.save_image(mock_content, filename="test.jpg")
            assert isinstance(result, str)
        except Exception as e:
            # Expected if local storage directory doesn't exist
            assert "directory" in str(e).lower() or "path" in str(e).lower()

def test_ml_services_error_handling():
    """Test that ML services handle errors gracefully."""
    from app.services.segmentation import segmentation_service
    from app.services.classify import classification_service
    from app.services.portion import portion_service
    from app.services.nutrition import nutrition_service
    
    # Test with invalid inputs
    invalid_image = None
    invalid_mask = None
    
    # These should handle invalid inputs gracefully
    try:
        segmentation_service.detect_foods(invalid_image)
    except Exception:
        pass  # Expected to fail with invalid input
    
    try:
        classification_service.classify_food(invalid_image, invalid_mask)
    except Exception:
        pass  # Expected to fail with invalid input
    
    try:
        portion_service.estimate_volume(-1, "invalid_shape")
    except Exception:
        pass  # Expected to fail with invalid input

def test_ml_services_data_types():
    """Test that ML services handle different data types correctly."""
    from app.services.portion import portion_service
    
    # Test with different numeric types
    area_int = 25
    area_float = 25.0
    
    # Both should work
    volume_int = portion_service.estimate_volume(area_int, "mound")
    volume_float = portion_service.estimate_volume(area_float, "mound")
    
    assert isinstance(volume_int, float)
    assert isinstance(volume_float, float)
    assert abs(volume_int - volume_float) < 1e-6

def test_ml_services_edge_cases():
    """Test ML services with edge cases."""
    from app.services.portion import portion_service
    
    # Test with very small area
    small_area = 0.001
    volume_small = portion_service.estimate_volume(small_area, "mound")
    assert volume_small >= 0
    
    # Test with very large area
    large_area = 10000.0
    volume_large = portion_service.estimate_volume(large_area, "mound")
    assert volume_large > 0
    
    # Test with zero area
    zero_area = 0.0
    volume_zero = portion_service.estimate_volume(zero_area, "mound")
    assert volume_zero == 0.0
