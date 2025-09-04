from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    meals = relationship("Meal", back_populates="user")
    refresh_tokens = relationship("RefreshToken", back_populates="user")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    token = Column(String, unique=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="refresh_tokens")


class Meal(Base):
    __tablename__ = "meals"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String)
    image_path = Column(String)
    plate_diameter_cm = Column(Float)
    total_calories = Column(Float)
    total_protein = Column(Float)
    total_carbs = Column(Float)
    total_fat = Column(Float)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="meals")
    items = relationship("MealItem", back_populates="meal", cascade="all, delete-orphan")


class MealItem(Base):
    __tablename__ = "meal_items"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    meal_id = Column(String, ForeignKey("meals.id"), nullable=False)
    label = Column(String, nullable=False)
    confidence = Column(Float)
    grams_estimated = Column(Float)
    grams_actual = Column(Float)
    calories = Column(Float)
    protein_g = Column(Float)
    carbs_g = Column(Float)
    fat_g = Column(Float)
    mask_polygon = Column(JSON)  # [[x, y], [x, y], ...]
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    meal = relationship("Meal", back_populates="items")


class FoodCache(Base):
    __tablename__ = "food_cache"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    barcode = Column(String, index=True)
    fdc_id = Column(String, index=True)
    openfoodfacts_id = Column(String, index=True)
    calories_per_100g = Column(Float)
    protein_per_100g = Column(Float)
    carbs_per_100g = Column(Float)
    fat_per_100g = Column(Float)
    fiber_per_100g = Column(Float)
    sugar_per_100g = Column(Float)
    sodium_per_100g = Column(Float)
    source = Column(String)  # "usda", "openfoodfacts", "manual"
    last_updated = Column(DateTime(timezone=True), server_default=func.now())
    
    # Index for search
    __table_args__ = (
        {'postgresql_gin_index': True, 'postgresql_gin_index_trgm_ops': True}
    )


class Feedback(Base):
    __tablename__ = "feedback"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    meal_id = Column(String, ForeignKey("meals.id"), nullable=False)
    item_id = Column(String, ForeignKey("meal_items.id"), nullable=False)
    original_label = Column(String, nullable=False)
    corrected_label = Column(String)
    original_grams = Column(Float)
    corrected_grams = Column(Float)
    feedback_type = Column(String)  # "label_correction", "portion_correction", "both"
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User")
    meal = relationship("Meal")
    item = relationship("MealItem")


class AnalysisLog(Base):
    __tablename__ = "analysis_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    image_path = Column(String)
    processing_time_ms = Column(Integer)
    items_detected = Column(Integer)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User")
