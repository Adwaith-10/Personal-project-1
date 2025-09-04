"""
Pytest configuration and fixtures for Health AI Twin tests
"""

import pytest
import asyncio
import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import uuid

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from fastapi.testclient import TestClient
from motor.motor_asyncio import AsyncIOMotorClient
import pytest_asyncio

# Import the FastAPI app
from app.main import app
from app.services.database import get_database, connect_to_mongo, close_mongo_connection

# Test database configuration
TEST_MONGODB_URL = "mongodb://localhost:27017"
TEST_DATABASE_NAME = "health_aitwin_test"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_database():
    """Create a test database connection."""
    # Connect to test database
    client = AsyncIOMotorClient(TEST_MONGODB_URL)
    db = client[TEST_DATABASE_NAME]
    
    # Clear test database
    await db.client.drop_database(TEST_DATABASE_NAME)
    
    yield db
    
    # Cleanup
    await db.client.drop_database(TEST_DATABASE_NAME)
    client.close()

@pytest.fixture
async def test_client(test_database):
    """Create a test client with test database."""
    # Override the database dependency
    async def override_get_database():
        return test_database
    
    app.dependency_overrides[get_database] = override_get_database
    
    with TestClient(app) as client:
        yield client
    
    # Cleanup
    app.dependency_overrides.clear()

@pytest.fixture
def sample_users():
    """Sample user data for testing."""
    return [
        {
            "email": "john.doe@test.com",
            "password": "testpassword123",
            "first_name": "John",
            "last_name": "Doe",
            "date_of_birth": "1985-03-15T00:00:00",
            "gender": "male",
            "phone": "+1234567890",
            "role": "patient",
            "emergency_contact": {
                "name": "Jane Doe",
                "phone": "+1234567891",
                "relationship": "spouse"
            }
        },
        {
            "email": "sarah.smith@test.com",
            "password": "testpassword123",
            "first_name": "Sarah",
            "last_name": "Smith",
            "date_of_birth": "1990-07-22T00:00:00",
            "gender": "female",
            "phone": "+1987654321",
            "role": "patient",
            "emergency_contact": {
                "name": "Mike Smith",
                "phone": "+1987654322",
                "relationship": "spouse"
            }
        }
    ]

@pytest.fixture
def sample_lab_reports():
    """Sample lab report data for testing."""
    return [
        {
            "patient_id": "patient_001",
            "report_date": "2024-01-15T00:00:00",
            "lab_name": "LabCorp",
            "report_type": "Comprehensive Metabolic Panel",
            "biomarkers": [
                {
                    "name": "LDL",
                    "value": 120.0,
                    "unit": "mg/dL",
                    "reference_range": "<100",
                    "status": "high",
                    "extracted_confidence": 0.95
                },
                {
                    "name": "glucose",
                    "value": 95.0,
                    "unit": "mg/dL",
                    "reference_range": "70-100",
                    "status": "normal",
                    "extracted_confidence": 0.92
                },
                {
                    "name": "hemoglobin",
                    "value": 14.5,
                    "unit": "g/dL",
                    "reference_range": "13.5-17.5",
                    "status": "normal",
                    "extracted_confidence": 0.88
                }
            ],
            "overall_health_status": "good",
            "recommendations": [
                "Consider reducing saturated fat intake",
                "Maintain regular exercise routine"
            ],
            "notes": "Annual checkup lab report"
        },
        {
            "patient_id": "patient_002",
            "report_date": "2024-01-20T00:00:00",
            "lab_name": "Quest Diagnostics",
            "report_type": "Lipid Panel",
            "biomarkers": [
                {
                    "name": "LDL",
                    "value": 85.0,
                    "unit": "mg/dL",
                    "reference_range": "<100",
                    "status": "normal",
                    "extracted_confidence": 0.94
                },
                {
                    "name": "HDL",
                    "value": 65.0,
                    "unit": "mg/dL",
                    "reference_range": ">40",
                    "status": "normal",
                    "extracted_confidence": 0.91
                },
                {
                    "name": "glucose",
                    "value": 88.0,
                    "unit": "mg/dL",
                    "reference_range": "70-100",
                    "status": "normal",
                    "extracted_confidence": 0.93
                }
            ],
            "overall_health_status": "excellent",
            "recommendations": [
                "Continue current healthy lifestyle",
                "Maintain regular checkups"
            ],
            "notes": "Routine lipid screening"
        }
    ]

@pytest.fixture
def sample_wearable_data():
    """Generate 10 days of sample wearable data for 2 users."""
    data = []
    base_date = datetime(2024, 1, 15)
    
    for user_idx in range(2):
        user_id = f"user_{user_idx + 1:03d}"
        patient_id = f"patient_{user_idx + 1:03d}"
        
        for day in range(10):
            current_date = base_date + timedelta(days=day)
            
            # Generate realistic wearable data with some variation
            heart_rate_avg = 70 + (user_idx * 5) + (day % 3) * 2  # Varies by user and day
            sleep_hours = 7.5 + (user_idx * 0.5) + (day % 2) * 0.5  # Varies by user and day
            steps = 8000 + (user_idx * 1000) + (day % 3) * 500  # Varies by user and day
            
            daily_log = {
                "user_id": user_id,
                "patient_id": patient_id,
                "date": current_date.isoformat(),
                "device_id": f"apple_watch_{user_idx + 1}",
                "device_type": "apple_watch",
                "total_steps": steps,
                "total_calories_burned": steps * 0.04,
                "total_sleep_minutes": int(sleep_hours * 60),
                "avg_heart_rate": heart_rate_avg,
                "avg_spo2": 98.0 + (day % 2) * 0.5,
                "heart_rate_data": [
                    {
                        "heart_rate": heart_rate_avg + (hour % 3) * 2,
                        "hrv_ms": 45 + (hour % 2) * 5,
                        "zone": "rest" if hour < 6 or hour > 22 else "active",
                        "confidence": 0.95,
                        "source": "apple_watch",
                        "timestamp": (current_date + timedelta(hours=hour)).isoformat()
                    }
                    for hour in range(24)
                ],
                "sleep_data": [
                    {
                        "stage": "deep_sleep",
                        "duration_minutes": int(sleep_hours * 60 * 0.2),
                        "start_time": (current_date + timedelta(hours=23)).isoformat(),
                        "efficiency_percentage": 85.0 + (day % 2) * 5,
                        "source": "apple_watch"
                    },
                    {
                        "stage": "rem_sleep",
                        "duration_minutes": int(sleep_hours * 60 * 0.25),
                        "start_time": (current_date + timedelta(hours=1)).isoformat(),
                        "efficiency_percentage": 80.0 + (day % 2) * 5,
                        "source": "apple_watch"
                    },
                    {
                        "stage": "light_sleep",
                        "duration_minutes": int(sleep_hours * 60 * 0.55),
                        "start_time": (current_date + timedelta(hours=0)).isoformat(),
                        "efficiency_percentage": 75.0 + (day % 2) * 5,
                        "source": "apple_watch"
                    }
                ],
                "activity_data": [
                    {
                        "activity_type": "walking",
                        "duration_minutes": 30 + (day % 3) * 10,
                        "calories_burned": 150 + (day % 3) * 25,
                        "intensity": "moderate",
                        "source": "apple_watch",
                        "timestamp": (current_date + timedelta(hours=8)).isoformat()
                    }
                ],
                "created_at": current_date.isoformat(),
                "updated_at": current_date.isoformat(),
                "data_quality_score": 0.95 - (day % 3) * 0.02
            }
            
            data.append(daily_log)
    
    return data

@pytest.fixture
def sample_food_logs():
    """Generate sample food logs for 2 users over 10 days."""
    data = []
    base_date = datetime(2024, 1, 15)
    
    # Sample foods with nutrition data
    foods = [
        {
            "name": "apple",
            "category": "fruits",
            "calories": 95,
            "protein": 0.5,
            "carbs": 25,
            "fat": 0.3,
            "fiber": 4.4
        },
        {
            "name": "chicken_breast",
            "category": "protein",
            "calories": 165,
            "protein": 31,
            "carbs": 0,
            "fat": 3.6,
            "fiber": 0
        },
        {
            "name": "brown_rice",
            "category": "grains",
            "calories": 216,
            "protein": 4.5,
            "carbs": 45,
            "fat": 1.8,
            "fiber": 3.5
        },
        {
            "name": "salmon",
            "category": "protein",
            "calories": 208,
            "protein": 25,
            "carbs": 0,
            "fat": 12,
            "fiber": 0
        },
        {
            "name": "spinach",
            "category": "vegetables",
            "calories": 23,
            "protein": 2.9,
            "carbs": 3.6,
            "fat": 0.4,
            "fiber": 2.2
        }
    ]
    
    for user_idx in range(2):
        user_id = f"user_{user_idx + 1:03d}"
        patient_id = f"patient_{user_idx + 1:03d}"
        
        for day in range(10):
            current_date = base_date + timedelta(days=day)
            
            # Generate 3 meals per day
            for meal_idx, meal_type in enumerate(["breakfast", "lunch", "dinner"]):
                # Select 2-3 foods per meal
                meal_foods = foods[meal_idx % len(foods):(meal_idx + 2) % len(foods)]
                if len(meal_foods) < 2:
                    meal_foods.extend(foods[:2])
                
                total_calories = sum(food["calories"] for food in meal_foods)
                total_protein = sum(food["protein"] for food in meal_foods)
                total_carbs = sum(food["carbs"] for food in meal_foods)
                total_fat = sum(food["fat"] for food in meal_foods)
                total_fiber = sum(food["fiber"] for food in meal_foods)
                
                food_log = {
                    "user_id": user_id,
                    "patient_id": patient_id,
                    "timestamp": (current_date + timedelta(hours=6 + meal_idx * 6)).isoformat(),
                    "meal_type": meal_type,
                    "image_filename": f"food_{user_idx}_{day}_{meal_idx}.jpg",
                    "image_size": 1024000,
                    "predictions": [
                        {
                            "food_name": food["name"],
                            "confidence": 0.9 + (day % 3) * 0.02,
                            "category": food["category"],
                            "nutrition": food
                        }
                        for food in meal_foods
                    ],
                    "top_prediction": {
                        "food_name": meal_foods[0]["name"],
                        "confidence": 0.95,
                        "category": meal_foods[0]["category"],
                        "nutrition": meal_foods[0]
                    },
                    "total_calories": total_calories,
                    "total_protein": total_protein,
                    "total_carbs": total_carbs,
                    "total_fat": total_fat,
                    "total_fiber": total_fiber,
                    "portion_size": 1.0,
                    "estimated_weight": 200 + (day % 3) * 50,
                    "processing_time": 2.5,
                    "model_version": "resnet18_v1",
                    "confidence_threshold": 0.5,
                    "created_at": current_date.isoformat(),
                    "updated_at": current_date.isoformat()
                }
                
                data.append(food_log)
    
    return data

@pytest.fixture
def sample_health_predictions():
    """Generate sample health predictions for 2 users over 10 days."""
    data = []
    base_date = datetime(2024, 1, 15)
    
    for user_idx in range(2):
        user_id = f"user_{user_idx + 1:03d}"
        patient_id = f"patient_{user_idx + 1:03d}"
        
        for day in range(10):
            current_date = base_date + timedelta(days=day)
            
            # Generate realistic health predictions with trends
            base_ldl = 100 + (user_idx * 20) + (day % 3) * 5
            base_glucose = 90 + (user_idx * 5) + (day % 2) * 3
            base_hemoglobin = 14.0 + (user_idx * 0.5) + (day % 2) * 0.2
            
            prediction_log = {
                "user_id": user_id,
                "patient_id": patient_id,
                "timestamp": current_date.isoformat(),
                "model_version": "1.0.0",
                "model_type": "xgboost",
                "predictions": [
                    {
                        "metric": "ldl",
                        "predicted_value": base_ldl,
                        "confidence": 0.85 + (day % 3) * 0.02,
                        "unit": "mg/dL",
                        "normal_range": {"min": 0, "max": 100},
                        "status": "normal" if base_ldl <= 100 else "high",
                        "risk_level": "low" if base_ldl <= 100 else "medium",
                        "recommendations": [
                            "Maintain healthy diet" if base_ldl <= 100 else "Consider reducing saturated fat"
                        ]
                    },
                    {
                        "metric": "glucose",
                        "predicted_value": base_glucose,
                        "confidence": 0.88 + (day % 3) * 0.02,
                        "unit": "mg/dL",
                        "normal_range": {"min": 70, "max": 100},
                        "status": "normal" if base_glucose <= 100 else "elevated",
                        "risk_level": "low" if base_glucose <= 100 else "medium",
                        "recommendations": [
                            "Maintain regular exercise" if base_glucose <= 100 else "Monitor carbohydrate intake"
                        ]
                    },
                    {
                        "metric": "hemoglobin",
                        "predicted_value": base_hemoglobin,
                        "confidence": 0.82 + (day % 3) * 0.02,
                        "unit": "g/dL",
                        "normal_range": {"min": 13.5, "max": 17.5},
                        "status": "normal",
                        "risk_level": "low",
                        "recommendations": [
                            "Continue healthy lifestyle"
                        ]
                    }
                ],
                "input_features": {
                    "age": 30 + user_idx * 5,
                    "gender": "male" if user_idx == 0 else "female",
                    "bmi": 25.0 + user_idx * 2,
                    "avg_heart_rate": 70 + user_idx * 5,
                    "avg_sleep_hours": 7.5 + user_idx * 0.5,
                    "avg_steps": 8000 + user_idx * 1000
                },
                "confidence_scores": {
                    "ldl": 0.85 + (day % 3) * 0.02,
                    "glucose": 0.88 + (day % 3) * 0.02,
                    "hemoglobin": 0.82 + (day % 3) * 0.02
                },
                "processing_time": 1.5 + (day % 3) * 0.2,
                "data_quality_score": 0.95 - (day % 3) * 0.02,
                "missing_features": [],
                "overall_health_score": 85 + user_idx * 5 - (day % 3) * 2,
                "risk_factors": [
                    "Elevated LDL" if base_ldl > 100 else "Good cholesterol levels"
                ],
                "recommendations": [
                    "Maintain regular exercise routine",
                    "Follow balanced diet",
                    "Get adequate sleep"
                ],
                "created_at": current_date.isoformat(),
                "updated_at": current_date.isoformat()
            }
            
            data.append(prediction_log)
    
    return data

@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_pdf_content():
    """Mock PDF content for testing lab report extraction."""
    return """
    LABORATORY REPORT
    Patient: John Doe
    Date: 2024-01-15
    Lab: LabCorp
    
    RESULTS:
    LDL Cholesterol: 120 mg/dL (Reference: <100)
    Glucose: 95 mg/dL (Reference: 70-100)
    Hemoglobin: 14.5 g/dL (Reference: 13.5-17.5)
    HDL Cholesterol: 55 mg/dL (Reference: >40)
    Triglycerides: 150 mg/dL (Reference: <150)
    
    INTERPRETATION:
    LDL is elevated. Consider lifestyle modifications.
    Other values are within normal range.
    """

@pytest.fixture
def mock_image_file():
    """Create a mock image file for testing."""
    # Create a simple test image (1x1 pixel PNG)
    import io
    from PIL import Image
    
    img = Image.new('RGB', (1, 1), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

@pytest.fixture
def expected_ml_metrics():
    """Expected ML model performance metrics."""
    return {
        "ldl": {
            "rmse": 15.0,  # mg/dL
            "mae": 12.0,   # mg/dL
            "r2": 0.75,    # R-squared
            "accuracy": 0.85  # Classification accuracy
        },
        "glucose": {
            "rmse": 8.0,   # mg/dL
            "mae": 6.5,    # mg/dL
            "r2": 0.80,    # R-squared
            "accuracy": 0.90  # Classification accuracy
        },
        "hemoglobin": {
            "rmse": 0.8,   # g/dL
            "mae": 0.6,    # g/dL
            "r2": 0.70,    # R-squared
            "accuracy": 0.88  # Classification accuracy
        }
    }

@pytest.fixture
def test_data_quality_thresholds():
    """Quality thresholds for test data validation."""
    return {
        "wearable_data": {
            "min_heart_rate": 40,
            "max_heart_rate": 200,
            "min_sleep_hours": 4,
            "max_sleep_hours": 12,
            "min_steps": 100,
            "max_steps": 50000
        },
        "lab_reports": {
            "min_ldl": 50,
            "max_ldl": 300,
            "min_glucose": 40,
            "max_glucose": 400,
            "min_hemoglobin": 8,
            "max_hemoglobin": 20
        },
        "food_logs": {
            "min_calories": 10,
            "max_calories": 2000,
            "min_protein": 0,
            "max_protein": 100,
            "min_carbs": 0,
            "max_carbs": 300
        }
    }
