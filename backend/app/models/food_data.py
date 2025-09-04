from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class FoodCategory(str, Enum):
    """Food categories"""
    FRUITS = "fruits"
    VEGETABLES = "vegetables"
    PROTEINS = "proteins"
    GRAINS = "grains"
    DAIRY = "dairy"
    SWEETS = "sweets"
    BEVERAGES = "beverages"
    SNACKS = "snacks"
    UNKNOWN = "unknown"

class NutritionInfo(BaseModel):
    """Nutrition information for a food item"""
    calories: float = Field(..., ge=0, description="Calories per 100g")
    protein: float = Field(..., ge=0, description="Protein in grams per 100g")
    carbs: float = Field(..., ge=0, description="Carbohydrates in grams per 100g")
    fat: float = Field(..., ge=0, description="Fat in grams per 100g")
    fiber: float = Field(..., ge=0, description="Fiber in grams per 100g")
    sugar: Optional[float] = Field(None, ge=0, description="Sugar in grams per 100g")
    sodium: Optional[float] = Field(None, ge=0, description="Sodium in mg per 100g")
    potassium: Optional[float] = Field(None, ge=0, description="Potassium in mg per 100g")
    vitamin_c: Optional[float] = Field(None, ge=0, description="Vitamin C in mg per 100g")
    calcium: Optional[float] = Field(None, ge=0, description="Calcium in mg per 100g")
    iron: Optional[float] = Field(None, ge=0, description="Iron in mg per 100g")

class FoodPrediction(BaseModel):
    """Food classification prediction"""
    food_name: str = Field(..., description="Name of the predicted food")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    category: FoodCategory = Field(..., description="Food category")
    nutrition: NutritionInfo = Field(..., description="Nutrition information")
    bbox: Optional[List[float]] = Field(None, description="Bounding box coordinates [x1, y1, x2, y2]")

class FoodImageUpload(BaseModel):
    """Model for food image upload request"""
    patient_id: str = Field(..., description="Patient ID")
    meal_type: Optional[str] = Field(None, description="Type of meal (breakfast, lunch, dinner, snack)")
    portion_size: Optional[float] = Field(1.0, ge=0.1, le=10.0, description="Portion size multiplier")
    notes: Optional[str] = Field(None, description="Additional notes")

class FoodLog(BaseModel):
    """Complete food log model for MongoDB storage"""
    id: Optional[str] = Field(alias="_id")
    user_id: str = Field(..., description="User ID")
    patient_id: str = Field(..., description="Patient ID")
    timestamp: datetime = Field(..., description="When the food was logged")
    meal_type: Optional[str] = Field(None, description="Type of meal")
    
    # Image information
    image_filename: Optional[str] = Field(None, description="Original image filename")
    image_size: Optional[int] = Field(None, description="Image file size in bytes")
    
    # Classification results
    predictions: List[FoodPrediction] = Field(..., description="Food classification predictions")
    top_prediction: FoodPrediction = Field(..., description="Top prediction")
    
    # Nutrition summary
    total_calories: float = Field(..., ge=0, description="Total calories")
    total_protein: float = Field(..., ge=0, description="Total protein in grams")
    total_carbs: float = Field(..., ge=0, description="Total carbohydrates in grams")
    total_fat: float = Field(..., ge=0, description="Total fat in grams")
    total_fiber: float = Field(..., ge=0, description="Total fiber in grams")
    
    # Portion and serving information
    portion_size: float = Field(1.0, ge=0.1, le=10.0, description="Portion size multiplier")
    estimated_weight: Optional[float] = Field(None, ge=0, description="Estimated weight in grams")
    
    # Metadata
    device_info: Optional[Dict[str, Any]] = Field(None, description="Device information")
    processing_time: Optional[float] = Field(None, ge=0, description="Processing time in seconds")
    model_version: Optional[str] = Field(None, description="Model version used")
    confidence_threshold: Optional[float] = Field(0.5, ge=0, le=1, description="Confidence threshold used")
    
    # User feedback
    user_correction: Optional[str] = Field(None, description="User correction if prediction was wrong")
    user_rating: Optional[int] = Field(None, ge=1, le=5, description="User rating of prediction accuracy")
    notes: Optional[str] = Field(None, description="Additional notes")
    
    # Timestamps
    created_at: datetime = Field(..., description="When the log was created")
    updated_at: datetime = Field(..., description="When the log was last updated")
    
    class Config:
        validate_by_name = True
        json_schema_extra = {
            "example": {
                "patient_id": "507f1f77bcf86cd799439011",
                "timestamp": "2024-01-15T12:30:00",
                "meal_type": "lunch",
                "image_filename": "food_image_20240115.jpg",
                "image_size": 1024000,
                "predictions": [
                    {
                        "food_name": "apple",
                        "confidence": 0.95,
                        "category": "fruits",
                        "nutrition": {
                            "calories": 95,
                            "protein": 0.5,
                            "carbs": 25,
                            "fat": 0.3,
                            "fiber": 4.4
                        }
                    }
                ],
                "top_prediction": {
                    "food_name": "apple",
                    "confidence": 0.95,
                    "category": "fruits",
                    "nutrition": {
                        "calories": 95,
                        "protein": 0.5,
                        "carbs": 25,
                        "fat": 0.3,
                        "fiber": 4.4
                    }
                },
                "total_calories": 95,
                "total_protein": 0.5,
                "total_carbs": 25,
                "total_fat": 0.3,
                "total_fiber": 4.4,
                "portion_size": 1.0,
                "estimated_weight": 100,
                "processing_time": 2.5,
                "model_version": "resnet18_v1",
                "confidence_threshold": 0.5,
                "created_at": "2024-01-15T12:30:00",
                "updated_at": "2024-01-15T12:30:00"
            }
        }

class FoodLogResponse(BaseModel):
    """Response model for food log operations"""
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    log_id: Optional[str] = Field(None, description="Food log ID")
    predictions_count: Optional[int] = Field(None, description="Number of predictions made")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    data: Optional[FoodLog] = Field(None, description="Food log data")

class DailyNutritionSummary(BaseModel):
    """Daily nutrition summary"""
    patient_id: str = Field(..., description="Patient ID")
    date: datetime = Field(..., description="Date of summary")
    total_calories: float = Field(..., ge=0, description="Total calories for the day")
    total_protein: float = Field(..., ge=0, description="Total protein for the day")
    total_carbs: float = Field(..., ge=0, description="Total carbohydrates for the day")
    total_fat: float = Field(..., ge=0, description="Total fat for the day")
    total_fiber: float = Field(..., ge=0, description="Total fiber for the day")
    meals_count: int = Field(..., ge=0, description="Number of meals logged")
    foods_count: int = Field(..., ge=0, description="Number of different foods logged")
    calorie_goal: Optional[float] = Field(None, ge=0, description="Daily calorie goal")
    protein_goal: Optional[float] = Field(None, ge=0, description="Daily protein goal")
    carbs_goal: Optional[float] = Field(None, ge=0, description="Daily carbs goal")
    fat_goal: Optional[float] = Field(None, ge=0, description="Daily fat goal")
    
    # Goal progress percentages
    calorie_progress: Optional[float] = Field(None, ge=0, le=200, description="Calorie goal progress percentage")
    protein_progress: Optional[float] = Field(None, ge=0, le=200, description="Protein goal progress percentage")
    carbs_progress: Optional[float] = Field(None, ge=0, le=200, description="Carbs goal progress percentage")
    fat_progress: Optional[float] = Field(None, ge=0, le=200, description="Fat goal progress percentage")

class FoodRecommendation(BaseModel):
    """Food recommendation based on nutrition goals"""
    food_name: str = Field(..., description="Recommended food name")
    category: FoodCategory = Field(..., description="Food category")
    nutrition: NutritionInfo = Field(..., description="Nutrition information")
    reason: str = Field(..., description="Reason for recommendation")
    priority: int = Field(..., ge=1, le=5, description="Recommendation priority (1-5)")
    serving_size: Optional[str] = Field(None, description="Recommended serving size")
