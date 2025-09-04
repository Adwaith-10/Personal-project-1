from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class ChatMessageRole(str, Enum):
    """Chat message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class HealthMetricTrend(BaseModel):
    """Health metric trend data"""
    metric: str = Field(..., description="Health metric name (e.g., ldl, glucose, heart_rate)")
    current_value: float = Field(..., description="Current value")
    trend_direction: str = Field(..., description="Trend direction (improving, declining, stable)")
    trend_strength: float = Field(..., ge=0, le=1, description="Trend strength (0-1)")
    normal_range: Dict[str, float] = Field(..., description="Normal range for the metric")
    unit: str = Field(..., description="Unit of measurement")
    status: str = Field(..., description="Current status (normal, elevated, low)")

class WearableTrends(BaseModel):
    """Wearable device trends"""
    heart_rate_trend: Optional[HealthMetricTrend] = Field(None, description="Heart rate trend")
    sleep_trend: Optional[HealthMetricTrend] = Field(None, description="Sleep quality trend")
    activity_trend: Optional[HealthMetricTrend] = Field(None, description="Activity level trend")
    steps_trend: Optional[HealthMetricTrend] = Field(None, description="Daily steps trend")
    hrv_trend: Optional[HealthMetricTrend] = Field(None, description="Heart rate variability trend")
    spo2_trend: Optional[HealthMetricTrend] = Field(None, description="Blood oxygen trend")

class HealthPredictions(BaseModel):
    """Health predictions from ML models"""
    ldl_prediction: Optional[HealthMetricTrend] = Field(None, description="LDL cholesterol prediction")
    glucose_prediction: Optional[HealthMetricTrend] = Field(None, description="Glucose prediction")
    hemoglobin_prediction: Optional[HealthMetricTrend] = Field(None, description="Hemoglobin prediction")

class PatientContext(BaseModel):
    """Patient context for chatbot"""
    patient_id: str = Field(..., description="Patient ID")
    age: int = Field(..., description="Patient age")
    gender: str = Field(..., description="Patient gender")
    bmi: Optional[float] = Field(None, description="Body Mass Index")
    medical_conditions: List[str] = Field(default_factory=list, description="Medical conditions")
    medications: List[str] = Field(default_factory=list, description="Current medications")
    lifestyle_factors: Dict[str, Any] = Field(default_factory=dict, description="Lifestyle factors")

class ChatMessage(BaseModel):
    """Individual chat message"""
    role: ChatMessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class ChatSession(BaseModel):
    """Chat session with patient"""
    session_id: str = Field(..., description="Unique session ID")
    patient_id: str = Field(..., description="Patient ID")
    start_time: datetime = Field(default_factory=datetime.now, description="Session start time")
    messages: List[ChatMessage] = Field(default_factory=list, description="Chat messages")
    context: Optional[PatientContext] = Field(None, description="Patient context")
    wearable_trends: Optional[WearableTrends] = Field(None, description="Wearable trends")
    health_predictions: Optional[HealthPredictions] = Field(None, description="Health predictions")
    session_summary: Optional[str] = Field(None, description="Session summary")

class ChatRequest(BaseModel):
    """Chat request from user"""
    patient_id: str = Field(..., description="Patient ID")
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Existing session ID")
    include_health_data: bool = Field(True, description="Include health data in context")
    include_trends: bool = Field(True, description="Include trend analysis in context")

class ChatResponse(BaseModel):
    """Chat response from virtual doctor"""
    session_id: str = Field(..., description="Session ID")
    message: str = Field(..., description="Doctor's response")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    health_insights: Optional[List[str]] = Field(None, description="Health insights")
    recommendations: Optional[List[str]] = Field(None, description="Health recommendations")
    follow_up_questions: Optional[List[str]] = Field(None, description="Follow-up questions")
    urgency_level: str = Field("normal", description="Urgency level (normal, moderate, high)")
    next_steps: Optional[List[str]] = Field(None, description="Recommended next steps")

class HealthAnalysis(BaseModel):
    """Comprehensive health analysis"""
    patient_id: str = Field(..., description="Patient ID")
    analysis_date: datetime = Field(default_factory=datetime.now, description="Analysis date")
    overall_health_score: float = Field(..., ge=0, le=100, description="Overall health score")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    positive_factors: List[str] = Field(default_factory=list, description="Positive health factors")
    lifestyle_assessment: Dict[str, str] = Field(default_factory=dict, description="Lifestyle assessment")
    recommendations: Dict[str, List[str]] = Field(default_factory=dict, description="Recommendations by category")
    priority_actions: List[str] = Field(default_factory=list, description="Priority actions")
    monitoring_plan: Dict[str, str] = Field(default_factory=dict, description="Monitoring plan")

class LifestyleRecommendation(BaseModel):
    """Detailed lifestyle recommendation"""
    category: str = Field(..., description="Recommendation category (diet, exercise, sleep, stress)")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")
    rationale: str = Field(..., description="Scientific rationale")
    implementation_steps: List[str] = Field(..., description="Implementation steps")
    expected_benefits: List[str] = Field(..., description="Expected benefits")
    timeline: str = Field(..., description="Expected timeline for results")
    difficulty_level: str = Field(..., description="Difficulty level (easy, moderate, challenging)")
    priority: int = Field(..., ge=1, le=5, description="Priority level (1-5)")

class VirtualDoctorProfile(BaseModel):
    """Virtual doctor profile and capabilities"""
    doctor_name: str = Field("Dr. Sarah Chen", description="Doctor's name")
    specialty: str = Field("Lifestyle Medicine", description="Medical specialty")
    credentials: List[str] = Field(default_factory=lambda: [
        "MD - Harvard Medical School",
        "Board Certified in Internal Medicine",
        "Fellow of the American College of Lifestyle Medicine",
        "Certified in Functional Medicine"
    ], description="Professional credentials")
    expertise_areas: List[str] = Field(default_factory=lambda: [
        "Preventive Medicine",
        "Nutrition Science",
        "Exercise Physiology",
        "Sleep Medicine",
        "Stress Management",
        "Chronic Disease Prevention"
    ], description="Areas of expertise")
    communication_style: str = Field("Compassionate and evidence-based", description="Communication style")
    consultation_approach: str = Field("Holistic and personalized", description="Consultation approach")

class ChatSessionSummary(BaseModel):
    """Summary of chat session"""
    session_id: str = Field(..., description="Session ID")
    patient_id: str = Field(..., description="Patient ID")
    session_duration: float = Field(..., description="Session duration in minutes")
    topics_discussed: List[str] = Field(..., description="Topics discussed")
    key_insights: List[str] = Field(..., description="Key insights from session")
    action_items: List[str] = Field(..., description="Action items for patient")
    follow_up_required: bool = Field(False, description="Whether follow-up is required")
    follow_up_date: Optional[datetime] = Field(None, description="Recommended follow-up date")
    session_rating: Optional[int] = Field(None, ge=1, le=5, description="Session rating (1-5)")
    patient_feedback: Optional[str] = Field(None, description="Patient feedback")
