from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum

class SleepStage(str, Enum):
    """Sleep stage types"""
    AWAKE = "awake"
    LIGHT_SLEEP = "light_sleep"
    DEEP_SLEEP = "deep_sleep"
    REM_SLEEP = "rem_sleep"
    UNKNOWN = "unknown"

class ActivityType(str, Enum):
    """Activity types from wearable devices"""
    WALKING = "walking"
    RUNNING = "running"
    CYCLING = "cycling"
    SWIMMING = "swimming"
    SLEEPING = "sleeping"
    SEDENTARY = "sedentary"
    EXERCISE = "exercise"
    UNKNOWN = "unknown"

class HeartRateZone(str, Enum):
    """Heart rate zones"""
    REST = "rest"
    FAT_BURN = "fat_burn"
    CARDIO = "cardio"
    PEAK = "peak"
    UNKNOWN = "unknown"

class WearableDataPoint(BaseModel):
    """Individual data point from wearable device"""
    timestamp: datetime
    value: Union[float, int]
    unit: str
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    source: Optional[str] = None  # Device name/model

class HeartRateData(BaseModel):
    """Heart rate data structure"""
    heart_rate: int = Field(..., ge=30, le=220)
    hrv_ms: Optional[int] = Field(None, ge=0, le=200)  # Heart Rate Variability in milliseconds
    zone: Optional[HeartRateZone] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    source: Optional[str] = None

class SpO2Data(BaseModel):
    """Blood oxygen saturation data"""
    spo2_percentage: float = Field(..., ge=70.0, le=100.0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    source: Optional[str] = None

class SleepData(BaseModel):
    """Sleep data structure"""
    stage: SleepStage
    duration_minutes: int = Field(..., ge=0)
    start_time: datetime
    end_time: Optional[datetime] = None
    efficiency_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    source: Optional[str] = None

class ActivityData(BaseModel):
    """Activity data structure"""
    activity_type: ActivityType
    duration_minutes: int = Field(..., ge=0)
    calories_burned: Optional[float] = Field(None, ge=0.0)
    distance_meters: Optional[float] = Field(None, ge=0.0)
    steps: Optional[int] = Field(None, ge=0)
    start_time: datetime
    end_time: Optional[datetime] = None
    source: Optional[str] = None

class StepsData(BaseModel):
    """Steps data structure"""
    steps_count: int = Field(..., ge=0)
    distance_meters: Optional[float] = Field(None, ge=0.0)
    calories_burned: Optional[float] = Field(None, ge=0.0)
    source: Optional[str] = None

class CaloriesData(BaseModel):
    """Calories data structure"""
    calories_burned: float = Field(..., ge=0.0)
    calories_consumed: Optional[float] = Field(None, ge=0.0)
    net_calories: Optional[float] = None
    source: Optional[str] = None

class TemperatureData(BaseModel):
    """Body temperature data"""
    temperature_celsius: float = Field(..., ge=35.0, le=42.0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    source: Optional[str] = None

class WearableDataUpload(BaseModel):
    """Model for wearable data upload request"""
    patient_id: str
    device_id: Optional[str] = None
    device_type: Optional[str] = None  # apple_watch, fitbit, garmin, etc.
    date: Optional[datetime] = None  # Date for the data (defaults to current date)
    
    # Data arrays
    heart_rate_data: Optional[List[HeartRateData]] = []
    spo2_data: Optional[List[SpO2Data]] = []
    sleep_data: Optional[List[SleepData]] = []
    activity_data: Optional[List[ActivityData]] = []
    steps_data: Optional[List[StepsData]] = []
    calories_data: Optional[List[CaloriesData]] = []
    temperature_data: Optional[List[TemperatureData]] = []
    
    # Raw data for flexibility
    raw_data: Optional[Dict[str, Any]] = None
    
    @validator('date', pre=True, always=True)
    def set_date_if_none(cls, v):
        """Set date to current date if not provided"""
        if v is None:
            return datetime.now()
        return v

class DailyLog(BaseModel):
    """Complete daily log model for MongoDB storage"""
    id: Optional[str] = Field(alias="_id")
    user_id: str = Field(..., description="User ID")
    patient_id: str
    date: datetime
    device_id: Optional[str] = None
    device_type: Optional[str] = None
    
    # Summary statistics
    total_steps: Optional[int] = Field(None, ge=0)
    total_calories_burned: Optional[float] = Field(None, ge=0.0)
    total_sleep_minutes: Optional[int] = Field(None, ge=0)
    avg_heart_rate: Optional[float] = Field(None, ge=0.0)
    avg_spo2: Optional[float] = Field(None, ge=0.0, le=100.0)
    
    # Detailed data arrays
    heart_rate_data: List[HeartRateData] = []
    spo2_data: List[SpO2Data] = []
    sleep_data: List[SleepData] = []
    activity_data: List[ActivityData] = []
    steps_data: List[StepsData] = []
    calories_data: List[CaloriesData] = []
    temperature_data: List[TemperatureData] = []
    
    # Raw data
    raw_data: Optional[Dict[str, Any]] = None
    
    # Metadata
    created_at: datetime
    updated_at: datetime
    data_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    class Config:
        validate_by_name = True
        json_schema_extra = {
            "example": {
                "patient_id": "507f1f77bcf86cd799439011",
                "date": "2024-01-15T00:00:00",
                "device_id": "apple_watch_123",
                "device_type": "apple_watch",
                "total_steps": 8500,
                "total_calories_burned": 450.5,
                "total_sleep_minutes": 420,
                "avg_heart_rate": 72.5,
                "avg_spo2": 98.2,
                "heart_rate_data": [
                    {
                        "heart_rate": 75,
                        "hrv_ms": 45,
                        "zone": "rest",
                        "confidence": 0.95,
                        "source": "apple_watch"
                    }
                ],
                "sleep_data": [
                    {
                        "stage": "deep_sleep",
                        "duration_minutes": 120,
                        "start_time": "2024-01-15T02:00:00",
                        "efficiency_percentage": 85.0,
                        "source": "apple_watch"
                    }
                ]
            }
        }

class WearableDataResponse(BaseModel):
    """Response model for wearable data operations"""
    success: bool
    message: str
    log_id: Optional[str] = None
    data_points_processed: Optional[int] = None
    processing_time: Optional[float] = None
    data: Optional[DailyLog] = None

class WearableDataQuery(BaseModel):
    """Query parameters for wearable data retrieval"""
    patient_id: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    data_type: Optional[str] = None  # heart_rate, sleep, activity, etc.
    limit: Optional[int] = Field(100, ge=1, le=1000)
    skip: Optional[int] = Field(0, ge=0)

class WearableDataSummary(BaseModel):
    """Summary statistics for wearable data"""
    patient_id: str
    date_range: Dict[str, datetime]
    total_days: int
    total_steps: int
    total_calories_burned: float
    total_sleep_hours: float
    avg_heart_rate: float
    avg_spo2: float
    sleep_efficiency: float
    activity_minutes: int
    data_completeness: float  # Percentage of days with data
