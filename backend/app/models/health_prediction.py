from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class HealthMetricType(str, Enum):
    """Health metric types that can be predicted"""
    LDL = "ldl"
    GLUCOSE = "glucose"
    HEMOGLOBIN = "hemoglobin"

class WearableFeatures(BaseModel):
    """Wearable device features for health prediction"""
    avg_heart_rate: Optional[float] = Field(None, ge=40, le=200, description="Average heart rate (bpm)")
    avg_spo2: Optional[float] = Field(None, ge=90, le=100, description="Average blood oxygen saturation (%)")
    total_steps: Optional[int] = Field(None, ge=0, description="Total daily steps")
    total_calories_burned: Optional[float] = Field(None, ge=0, description="Total calories burned")
    total_sleep_minutes: Optional[int] = Field(None, ge=0, description="Total sleep duration (minutes)")
    avg_sleep_efficiency: Optional[float] = Field(None, ge=0, le=100, description="Average sleep efficiency (%)")
    total_activity_minutes: Optional[int] = Field(None, ge=0, description="Total activity minutes")
    hrv_avg: Optional[float] = Field(None, ge=0, description="Average heart rate variability (ms)")
    resting_heart_rate: Optional[float] = Field(None, ge=40, le=100, description="Resting heart rate (bpm)")
    max_heart_rate: Optional[float] = Field(None, ge=100, le=200, description="Maximum heart rate (bpm)")
    min_heart_rate: Optional[float] = Field(None, ge=40, le=80, description="Minimum heart rate (bpm)")
    heart_rate_variability: Optional[float] = Field(None, ge=0, description="Heart rate variability (ms)")
    sleep_deep_minutes: Optional[int] = Field(None, ge=0, description="Deep sleep duration (minutes)")
    sleep_light_minutes: Optional[int] = Field(None, ge=0, description="Light sleep duration (minutes)")
    sleep_rem_minutes: Optional[int] = Field(None, ge=0, description="REM sleep duration (minutes)")
    sleep_awake_minutes: Optional[int] = Field(None, ge=0, description="Awake time during sleep (minutes)")
    activity_intensity_high: Optional[int] = Field(None, ge=0, description="High intensity activity minutes")
    activity_intensity_medium: Optional[int] = Field(None, ge=0, description="Medium intensity activity minutes")
    activity_intensity_low: Optional[int] = Field(None, ge=0, description="Low intensity activity minutes")
    steps_goal_achievement: Optional[float] = Field(None, ge=0, description="Steps goal achievement percentage")

class DietFeatures(BaseModel):
    """Diet and nutrition features for health prediction"""
    total_calories: Optional[float] = Field(None, ge=0, description="Total daily calories")
    total_protein: Optional[float] = Field(None, ge=0, description="Total protein (g)")
    total_carbs: Optional[float] = Field(None, ge=0, description="Total carbohydrates (g)")
    total_fat: Optional[float] = Field(None, ge=0, description="Total fat (g)")
    total_fiber: Optional[float] = Field(None, ge=0, description="Total fiber (g)")
    total_sugar: Optional[float] = Field(None, ge=0, description="Total sugar (g)")
    total_sodium: Optional[float] = Field(None, ge=0, description="Total sodium (mg)")
    total_potassium: Optional[float] = Field(None, ge=0, description="Total potassium (mg)")
    total_vitamin_c: Optional[float] = Field(None, ge=0, description="Total vitamin C (mg)")
    total_calcium: Optional[float] = Field(None, ge=0, description="Total calcium (mg)")
    total_iron: Optional[float] = Field(None, ge=0, description="Total iron (mg)")
    avg_meal_size: Optional[float] = Field(None, ge=0, description="Average meal size (calories)")
    meals_per_day: Optional[int] = Field(None, ge=1, le=10, description="Number of meals per day")
    snacks_per_day: Optional[int] = Field(None, ge=0, le=10, description="Number of snacks per day")
    water_intake: Optional[float] = Field(None, ge=0, description="Water intake (ml)")
    alcohol_intake: Optional[float] = Field(None, ge=0, description="Alcohol intake (ml)")
    caffeine_intake: Optional[float] = Field(None, ge=0, description="Caffeine intake (mg)")
    processed_food_ratio: Optional[float] = Field(None, ge=0, le=100, description="Processed food ratio (%)")
    fruits_servings: Optional[float] = Field(None, ge=0, description="Fruit servings")
    vegetables_servings: Optional[float] = Field(None, ge=0, description="Vegetable servings")
    protein_servings: Optional[float] = Field(None, ge=0, description="Protein servings")
    grains_servings: Optional[float] = Field(None, ge=0, description="Grain servings")
    dairy_servings: Optional[float] = Field(None, ge=0, description="Dairy servings")
    sweets_servings: Optional[float] = Field(None, ge=0, description="Sweet servings")
    beverages_servings: Optional[float] = Field(None, ge=0, description="Beverage servings")

class DemographicFeatures(BaseModel):
    """Demographic features for health prediction"""
    age: int = Field(..., ge=18, le=100, description="Age in years")
    gender: str = Field(..., description="Gender (male/female)")
    bmi: Optional[float] = Field(None, ge=15, le=50, description="Body Mass Index")
    weight: Optional[float] = Field(None, ge=30, le=200, description="Weight (kg)")
    height: Optional[float] = Field(None, ge=120, le=220, description="Height (cm)")
    activity_level: Optional[str] = Field(None, description="Activity level (sedentary/light/moderate/active)")
    smoking_status: Optional[str] = Field(None, description="Smoking status (never/former/current)")
    alcohol_consumption: Optional[str] = Field(None, description="Alcohol consumption (none/light/moderate/heavy)")
    medical_conditions: Optional[str] = Field(None, description="Medical conditions")

class LifestyleFeatures(BaseModel):
    """Lifestyle features for health prediction"""
    stress_level: Optional[int] = Field(None, ge=1, le=5, description="Stress level (1-5)")
    sleep_quality: Optional[int] = Field(None, ge=1, le=5, description="Sleep quality (1-5)")
    exercise_frequency: Optional[int] = Field(None, ge=0, le=7, description="Exercise frequency (days per week)")
    meditation_practice: Optional[bool] = Field(None, description="Meditation practice")
    social_activity: Optional[int] = Field(None, ge=1, le=5, description="Social activity level (1-5)")
    work_hours: Optional[float] = Field(None, ge=0, le=16, description="Work hours per day")
    screen_time: Optional[float] = Field(None, ge=0, le=24, description="Screen time (hours per day)")
    outdoor_time: Optional[float] = Field(None, ge=0, le=24, description="Outdoor time (hours per day)")
    social_support: Optional[int] = Field(None, ge=1, le=5, description="Social support level (1-5)")

class HealthPredictionRequest(BaseModel):
    """Request model for health prediction"""
    patient_id: str = Field(..., description="Patient ID")
    wearable_features: Optional[WearableFeatures] = Field(None, description="Wearable device features")
    diet_features: Optional[DietFeatures] = Field(None, description="Diet and nutrition features")
    demographic_features: DemographicFeatures = Field(..., description="Demographic features")
    lifestyle_features: Optional[LifestyleFeatures] = Field(None, description="Lifestyle features")
    target_metrics: Optional[List[HealthMetricType]] = Field(
        default=[HealthMetricType.LDL, HealthMetricType.GLUCOSE, HealthMetricType.HEMOGLOBIN],
        description="Health metrics to predict"
    )
    confidence_threshold: Optional[float] = Field(0.7, ge=0, le=1, description="Confidence threshold for predictions")

class HealthPrediction(BaseModel):
    """Individual health metric prediction"""
    metric: HealthMetricType = Field(..., description="Health metric type")
    predicted_value: float = Field(..., description="Predicted value")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    unit: str = Field(..., description="Unit of measurement")
    normal_range: Dict[str, float] = Field(..., description="Normal range for the metric")
    status: str = Field(..., description="Health status (normal/elevated/low)")
    risk_level: str = Field(..., description="Risk level (low/medium/high)")
    recommendations: List[str] = Field(..., description="Health recommendations")

class HealthPredictionResponse(BaseModel):
    """Response model for health prediction"""
    success: bool = Field(..., description="Prediction success status")
    patient_id: str = Field(..., description="Patient ID")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    predictions: List[HealthPrediction] = Field(..., description="Health predictions")
    model_version: str = Field(..., description="Model version used")
    processing_time: float = Field(..., description="Processing time in seconds")
    data_quality_score: float = Field(..., ge=0, le=1, description="Data quality score")
    missing_features: List[str] = Field(..., description="Missing features")
    overall_health_score: Optional[float] = Field(None, ge=0, le=100, description="Overall health score")
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    recommendations: List[str] = Field(..., description="General health recommendations")

class HealthPredictionLog(BaseModel):
    """Model for storing health prediction logs in database"""
    id: Optional[str] = Field(alias="_id")
    user_id: str = Field(..., description="User ID")
    patient_id: str = Field(..., description="Patient ID")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")
    model_type: str = Field(..., description="Type of model (xgboost, neural_network, etc.)")
    predictions: List[HealthPrediction] = Field(..., description="Health predictions")
    input_features: Dict[str, Any] = Field(..., description="Input features used for prediction")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores for each prediction")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    processing_time: float = Field(..., description="Processing time in seconds")
    data_quality_score: float = Field(..., ge=0, le=1, description="Data quality score")
    missing_features: List[str] = Field(..., description="Missing features")
    overall_health_score: Optional[float] = Field(None, ge=0, le=100, description="Overall health score")
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    recommendations: List[str] = Field(..., description="General health recommendations")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

class ModelTrainingRequest(BaseModel):
    """Request model for training health prediction models"""
    data_source: str = Field(..., description="Data source for training")
    target_metrics: List[HealthMetricType] = Field(..., description="Metrics to train models for")
    training_parameters: Optional[Dict[str, Any]] = Field(None, description="Training parameters")
    validation_split: Optional[float] = Field(0.2, ge=0.1, le=0.5, description="Validation split ratio")

class ModelTrainingResponse(BaseModel):
    """Response model for model training"""
    success: bool = Field(..., description="Training success status")
    training_id: str = Field(..., description="Training session ID")
    models_trained: List[str] = Field(..., description="List of trained models")
    training_metrics: Dict[str, Dict[str, float]] = Field(..., description="Training metrics for each model")
    training_time: float = Field(..., description="Total training time in seconds")
    model_version: str = Field(..., description="New model version")
    deployment_status: str = Field(..., description="Model deployment status")

class ModelEvaluationRequest(BaseModel):
    """Request model for model evaluation"""
    model_version: str = Field(..., description="Model version to evaluate")
    evaluation_dataset: str = Field(..., description="Dataset for evaluation")
    metrics: List[str] = Field(..., description="Metrics to calculate")

class ModelEvaluationResponse(BaseModel):
    """Response model for model evaluation"""
    success: bool = Field(..., description="Evaluation success status")
    model_version: str = Field(..., description="Evaluated model version")
    evaluation_metrics: Dict[str, Dict[str, float]] = Field(..., description="Evaluation metrics")
    performance_summary: Dict[str, Any] = Field(..., description="Performance summary")
    recommendations: List[str] = Field(..., description="Model improvement recommendations")

class HealthTrendAnalysis(BaseModel):
    """Health trend analysis over time"""
    patient_id: str = Field(..., description="Patient ID")
    metric: HealthMetricType = Field(..., description="Health metric")
    time_period: str = Field(..., description="Time period analyzed")
    trend_direction: str = Field(..., description="Trend direction (improving/declining/stable)")
    trend_strength: float = Field(..., ge=0, le=1, description="Trend strength")
    data_points: int = Field(..., description="Number of data points")
    average_value: float = Field(..., description="Average value over period")
    min_value: float = Field(..., description="Minimum value")
    max_value: float = Field(..., description="Maximum value")
    volatility: float = Field(..., description="Value volatility")
    predictions: List[Dict[str, Any]] = Field(..., description="Future predictions")

class HealthInsights(BaseModel):
    """Health insights and recommendations"""
    patient_id: str = Field(..., description="Patient ID")
    timestamp: datetime = Field(..., description="Insights timestamp")
    overall_health_score: float = Field(..., ge=0, le=100, description="Overall health score")
    risk_assessment: Dict[str, str] = Field(..., description="Risk assessment by category")
    key_insights: List[str] = Field(..., description="Key health insights")
    recommendations: List[str] = Field(..., description="Health recommendations")
    priority_actions: List[str] = Field(..., description="Priority actions")
    follow_up_schedule: Dict[str, str] = Field(..., description="Follow-up schedule")
    progress_tracking: Dict[str, Any] = Field(..., description="Progress tracking metrics")
