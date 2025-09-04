from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from decimal import Decimal


# User schemas
class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None


class UserResponse(UserBase):
    id: str
    is_active: bool
    is_admin: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Auth schemas
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    email: Optional[str] = None


class RefreshTokenRequest(BaseModel):
    refresh_token: str


# Meal item schemas
class MealItemBase(BaseModel):
    label: str
    confidence: Optional[float] = None
    grams_estimated: Optional[float] = None
    grams_actual: Optional[float] = None
    calories: Optional[float] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    mask_polygon: Optional[List[List[float]]] = None


class MealItemCreate(MealItemBase):
    pass


class MealItemUpdate(BaseModel):
    label: Optional[str] = None
    grams_actual: Optional[float] = None
    calories: Optional[float] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None


class MealItemResponse(MealItemBase):
    id: str
    meal_id: str
    created_at: datetime

    class Config:
        from_attributes = True


# Meal schemas
class MealBase(BaseModel):
    name: Optional[str] = None
    plate_diameter_cm: Optional[float] = None
    notes: Optional[str] = None


class MealCreate(MealBase):
    items: List[MealItemCreate]


class MealUpdate(BaseModel):
    name: Optional[str] = None
    notes: Optional[str] = None
    items: Optional[List[MealItemUpdate]] = None


class MealResponse(MealBase):
    id: str
    user_id: str
    image_path: Optional[str] = None
    total_calories: Optional[float] = None
    total_protein: Optional[float] = None
    total_carbs: Optional[float] = None
    total_fat: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    items: List[MealItemResponse]

    class Config:
        from_attributes = True


# Analysis schemas
class AnalysisRequest(BaseModel):
    plate_diameter_cm: Optional[float] = Field(None, ge=5, le=50)
    barcode: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = None


class AnalysisItem(BaseModel):
    mask_id: str
    label: str
    candidates: List[str]
    grams_est: float
    calories: float
    protein_g: float
    carb_g: float
    fat_g: float
    polygon: List[List[float]]


class AnalysisResponse(BaseModel):
    items: List[AnalysisItem]
    totals: Dict[str, float]
    processing_time_ms: int


# Food search schemas
class FoodSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=100)


class FoodSearchResponse(BaseModel):
    foods: List[Dict[str, Any]]
    total: int


# Feedback schemas
class FeedbackCreate(BaseModel):
    meal_id: str
    item_id: str
    original_label: str
    corrected_label: Optional[str] = None
    original_grams: Optional[float] = None
    corrected_grams: Optional[float] = None
    feedback_type: str
    notes: Optional[str] = None


class FeedbackResponse(FeedbackCreate):
    id: str
    user_id: str
    created_at: datetime

    class Config:
        from_attributes = True


# Billing schemas
class CheckoutSessionRequest(BaseModel):
    price_id: str
    success_url: str
    cancel_url: str


class CheckoutSessionResponse(BaseModel):
    session_id: str
    checkout_url: str


# Health check schema
class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    database: str = "unknown"
    redis: str = "unknown"


# Pagination schemas
class PaginationParams(BaseModel):
    page: int = Field(1, ge=1)
    size: int = Field(20, ge=1, le=100)


class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int
