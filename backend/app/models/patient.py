from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"

class BloodType(str, Enum):
    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"

class PatientBase(BaseModel):
    """Base patient model"""
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    date_of_birth: datetime
    gender: Gender
    email: str = Field(..., pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    phone: Optional[str] = Field(None, pattern=r"^\+?1?\d{9,15}$")
    address: Optional[str] = None
    blood_type: Optional[BloodType] = None
    height_cm: Optional[float] = Field(None, ge=50, le=300)
    weight_kg: Optional[float] = Field(None, ge=1, le=500)
    emergency_contact: Optional[str] = None

class PatientCreate(PatientBase):
    """Model for creating a new patient"""
    pass

class PatientUpdate(BaseModel):
    """Model for updating patient information"""
    first_name: Optional[str] = Field(None, min_length=1, max_length=50)
    last_name: Optional[str] = Field(None, min_length=1, max_length=50)
    date_of_birth: Optional[datetime] = None
    gender: Optional[Gender] = None
    email: Optional[str] = Field(None, pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    phone: Optional[str] = Field(None, pattern=r"^\+?1?\d{9,15}$")
    address: Optional[str] = None
    blood_type: Optional[BloodType] = None
    height_cm: Optional[float] = Field(None, ge=50, le=300)
    weight_kg: Optional[float] = Field(None, ge=1, le=500)
    emergency_contact: Optional[str] = None

class Patient(PatientBase):
    """Complete patient model with ID and timestamps"""
    id: str = Field(alias="_id")
    created_at: datetime
    updated_at: datetime
    medical_history: Optional[List[Dict[str, Any]]] = []
    current_medications: Optional[List[str]] = []
    allergies: Optional[List[str]] = []
    
    class Config:
        validate_by_name = True
        json_schema_extra = {
            "example": {
                "first_name": "John",
                "last_name": "Doe",
                "date_of_birth": "1990-01-01T00:00:00",
                "gender": "male",
                "email": "john.doe@example.com",
                "phone": "+1234567890",
                "blood_type": "A+",
                "height_cm": 175.0,
                "weight_kg": 70.0
            }
        }

class HealthMetrics(BaseModel):
    """Model for health metrics data"""
    patient_id: str
    timestamp: datetime
    heart_rate: Optional[int] = Field(None, ge=30, le=200)
    blood_pressure_systolic: Optional[int] = Field(None, ge=70, le=200)
    blood_pressure_diastolic: Optional[int] = Field(None, ge=40, le=130)
    temperature: Optional[float] = Field(None, ge=35.0, le=42.0)
    oxygen_saturation: Optional[float] = Field(None, ge=70.0, le=100.0)
    respiratory_rate: Optional[int] = Field(None, ge=8, le=40)
    glucose_level: Optional[float] = Field(None, ge=50.0, le=500.0)
    notes: Optional[str] = None

class PredictionRequest(BaseModel):
    """Model for ML prediction requests"""
    patient_id: str
    features: Dict[str, Any]
    model_type: str = Field(..., description="Type of model to use for prediction")

class PredictionResponse(BaseModel):
    """Model for ML prediction responses"""
    patient_id: str
    prediction: Any
    confidence: float
    model_used: str
    timestamp: datetime
    features_used: List[str]
