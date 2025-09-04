from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "Food Vision Pro"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Security
    JWT_SECRET: str = "your-super-secret-jwt-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/foodvision"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # Storage
    STORAGE_TYPE: str = "local"  # local, s3, minio
    S3_BUCKET: str = "foodvision-images"
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""
    S3_ENDPOINT_URL: str = ""
    LOCAL_STORAGE_PATH: str = "./uploads"
    
    # External APIs
    USDA_FDC_API_KEY: str = ""
    OPENFOODFACTS_BASE_URL: str = "https://world.openfoodfacts.org"
    
    # Stripe (optional)
    STRIPE_ENABLED: bool = False
    STRIPE_SECRET_KEY: str = ""
    STRIPE_PUBLISHABLE_KEY: str = ""
    STRIPE_WEBHOOK_SECRET: str = ""
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://localhost:19006",  # Expo
        "exp://localhost:19000"   # Expo
    ]
    
    # Hosts
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1", "0.0.0.0"]
    
    # ML Models
    YOLO_MODEL_PATH: str = "./ml_models/yolov8n-seg.pt"
    CLASSIFIER_MODEL_PATH: str = "./ml_models/food_classifier.onnx"
    CLASS_INDEX_PATH: str = "./ml_models/class_index.json"
    DENSITY_PRIORS_PATH: str = "./ml_models/density_priors.yaml"
    
    # File upload limits
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.LOCAL_STORAGE_PATH, exist_ok=True)
