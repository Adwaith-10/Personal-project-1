import pytest
import asyncio
from typing import AsyncGenerator, Generator
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db import get_db, init_db
from app.models import Base
from app.config import settings

# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Create test engine
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_db():
    """Create test database and tables."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield test_engine
    await test_engine.dispose()

@pytest.fixture
async def db_session(test_db) -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database session for a test."""
    async with TestingSessionLocal() as session:
        yield session
        await session.rollback()
        await session.close()

@pytest.fixture
def client(db_session) -> Generator[TestClient, None, None]:
    """Create a test client with test database."""
    async def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def test_user_data():
    """Sample user data for testing."""
    return {
        "email": "test@example.com",
        "password": "testpassword123"
    }

@pytest.fixture
def test_meal_data():
    """Sample meal data for testing."""
    return {
        "name": "Test Meal",
        "items": [
            {
                "label": "grilled_chicken",
                "grams": 150.0,
                "calories": 250.0,
                "protein_g": 30.0,
                "carb_g": 0.0,
                "fat_g": 8.0
            }
        ]
    }

@pytest.fixture
def mock_image_data():
    """Mock image data for testing."""
    return b"fake_image_data"

@pytest.fixture
def mock_analysis_response():
    """Mock analysis response for testing."""
    return {
        "items": [
            {
                "mask_id": "m0",
                "label": "grilled_chicken",
                "candidates": ["grilled_chicken", "roasted_chicken"],
                "grams_est": 135.0,
                "kcal": 220.5,
                "protein_g": 28.5,
                "carb_g": 0.0,
                "fat_g": 7.5,
                "polygon": [[100, 100], [200, 100], [200, 200], [100, 200]]
            }
        ],
        "totals": {
            "kcal": 220.5,
            "protein_g": 28.5,
            "carb_g": 0.0,
            "fat_g": 7.5
        },
        "processing_time_ms": 1500
    }

# Mock external services
@pytest.fixture
def mock_nutrition_service(mocker):
    """Mock nutrition service responses."""
    mock_service = mocker.patch("app.services.nutrition.nutrition_service")
    mock_service.get_nutrition_info.return_value = {
        "name": "grilled_chicken",
        "calories_per_100g": 165.0,
        "protein_per_100g": 31.0,
        "carb_per_100g": 0.0,
        "fat_per_100g": 3.6
    }
    return mock_service

@pytest.fixture
def mock_segmentation_service(mocker):
    """Mock segmentation service responses."""
    mock_service = mocker.patch("app.services.segmentation.segmentation_service")
    mock_service.detect_foods.return_value = [
        {
            "mask_id": "m0",
            "mask": "mock_mask_data",
            "confidence": 0.95,
            "bbox": [100, 100, 200, 200]
        }
    ]
    return mock_service

@pytest.fixture
def mock_classification_service(mocker):
    """Mock classification service responses."""
    mock_service = mocker.patch("app.services.classify.classification_service")
    mock_service.classify_food.return_value = {
        "label": "grilled_chicken",
        "confidence": 0.92,
        "candidates": ["grilled_chicken", "roasted_chicken", "tofu"]
    }
    return mock_service

@pytest.fixture
def mock_portion_service(mocker):
    """Mock portion service responses."""
    mock_service = mocker.patch("app.services.portion.portion_service")
    mock_service.estimate_grams.return_value = 135.0
    mock_service.calculate_area_from_mask.return_value = 25.0
    return mock_service

# Test utilities
def create_test_user(client, user_data):
    """Helper function to create a test user."""
    response = client.post("/api/v1/auth/signup", json=user_data)
    return response.json()

def login_test_user(client, user_data):
    """Helper function to login a test user."""
    response = client.post("/api/v1/auth/login", json=user_data)
    return response.json()

def get_auth_headers(token):
    """Helper function to get authenticated headers."""
    return {"Authorization": f"Bearer {token}"}
