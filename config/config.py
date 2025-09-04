import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # MongoDB Configuration
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "health_ai_twin")
    
    # FastAPI Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_DEBUG = os.getenv("API_DEBUG", "True").lower() == "true"
    
    # Streamlit Configuration
    STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "localhost")
    STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))
    
    # ML Model Configuration
    MODEL_PATH = os.getenv("MODEL_PATH", "ml_models/models/")
    PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "ml_models/preprocessing/")
    
    # Security Configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
    
    # CORS Configuration
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
    
    # Data Configuration
    DATA_PATH = os.getenv("DATA_PATH", "data/")
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB
    
    # Health Check Configuration
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", 300))  # 5 minutes
    
    @classmethod
    def get_database_url(cls):
        """Get the complete database URL"""
        return f"{cls.MONGODB_URL}/{cls.MONGODB_DATABASE}"
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        errors = []
        
        # Check required environment variables
        if not cls.MONGODB_URL:
            errors.append("MONGODB_URL is required")
        
        if not cls.SECRET_KEY or cls.SECRET_KEY == "your-secret-key-here":
            errors.append("SECRET_KEY must be set to a secure value")
        
        # Validate port numbers
        if not (1 <= cls.API_PORT <= 65535):
            errors.append("API_PORT must be between 1 and 65535")
        
        if not (1 <= cls.STREAMLIT_PORT <= 65535):
            errors.append("STREAMLIT_PORT must be between 1 and 65535")
        
        return errors

class DevelopmentConfig(Config):
    """Development configuration"""
    API_DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    """Production configuration"""
    API_DEBUG = False
    LOG_LEVEL = "WARNING"
    CORS_ORIGINS = ["https://yourdomain.com"]  # Set to your actual domain

class TestingConfig(Config):
    """Testing configuration"""
    MONGODB_DATABASE = "health_ai_twin_test"
    API_DEBUG = True
    LOG_LEVEL = "DEBUG"

# Configuration mapping
config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig
}

def get_config(environment=None):
    """Get configuration based on environment"""
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")
    
    config_class = config_map.get(environment, DevelopmentConfig)
    return config_class()
