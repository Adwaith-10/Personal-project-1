from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form, Query
from typing import List, Optional
from datetime import datetime, timedelta
from bson import ObjectId
import sys
import os
import time
from PIL import Image
import io

# Add the parent directory to the path to import models and services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.food_data import (
    FoodImageUpload, FoodLog, FoodLogResponse, DailyNutritionSummary
)
from services.database import get_database
from services.food_classifier_service import AdvancedFoodClassifier

router = APIRouter(prefix="/api/v1/food-classification", tags=["food-classification"])

# Initialize the food classifier service
food_classifier = AdvancedFoodClassifier()

@router.post("/classify", response_model=FoodLogResponse)
async def classify_food_image(
    file: UploadFile = File(..., description="Food image file"),
    patient_id: str = Form(..., description="Patient ID"),
    meal_type: Optional[str] = Form(None, description="Type of meal (breakfast, lunch, dinner, snack)"),
    portion_size: Optional[float] = Form(1.0, description="Portion size multiplier"),
    notes: Optional[str] = Form(None, description="Additional notes")
):
    """
    Classify food image and estimate nutrition.
    
    This endpoint accepts an image file and returns:
    - Food classification predictions
    - Nutrition information (calories, protein, carbs, fat, fiber)
    - Portion size estimation
    - Confidence scores
    """
    
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Validate patient ID
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(
                status_code=400,
                detail="Invalid patient ID format"
            )
        
        # Get database connection
        db = get_database()
        
        # Check if patient exists
        patient = await db.patients.find_one({"_id": ObjectId(patient_id)})
        if not patient:
            raise HTTPException(
                status_code=404,
                detail="Patient not found"
            )
        
        # Load model if not already loaded
        if food_classifier.model is None:
            if not food_classifier.load_model():
                raise HTTPException(
                    status_code=500,
                    detail="Failed to load food classification model"
                )
        
        # Read and process image
        image_content = await file.read()
        image = Image.open(io.BytesIO(image_content))
        
        # Classify the food image
        predictions = food_classifier.classify_food_image(image)
        
        if not predictions:
            raise HTTPException(
                status_code=400,
                detail="No food items detected in the image. Please try a clearer image of food."
            )
        
        # Get top prediction
        top_prediction = predictions[0]
        
        # Estimate portion size if not provided
        if portion_size == 1.0:
            portion_size = food_classifier.estimate_portion_size(image)
        
        # Calculate total nutrition
        total_nutrition = food_classifier.calculate_total_nutrition(predictions, portion_size)
        
        # Create food log data
        food_log_data = {
            "patient_id": patient_id,
            "timestamp": datetime.now(),
            "meal_type": meal_type,
            
            # Image information
            "image_filename": file.filename,
            "image_size": len(image_content),
            
            # Classification results
            "predictions": [pred.dict() for pred in predictions],
            "top_prediction": top_prediction.dict(),
            
            # Nutrition summary
            "total_calories": total_nutrition["calories"],
            "total_protein": total_nutrition["protein"],
            "total_carbs": total_nutrition["carbs"],
            "total_fat": total_nutrition["fat"],
            "total_fiber": total_nutrition["fiber"],
            
            # Portion and serving information
            "portion_size": portion_size,
            "estimated_weight": 100.0 * portion_size,  # Assume 100g base serving
            
            # Metadata
            "processing_time": time.time() - start_time,
            "model_version": "resnet18_v1",
            "confidence_threshold": 0.5,
            "notes": notes,
            
            # Timestamps
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        # Save to database
        result = await db.food_logs.insert_one(food_log_data)
        log_id = str(result.inserted_id)
        
        # Get the created log
        food_log = await db.food_logs.find_one({"_id": ObjectId(log_id)})
        food_log["_id"] = str(food_log["_id"])
        
        processing_time = time.time() - start_time
        
        return FoodLogResponse(
            success=True,
            message="Food classification completed successfully",
            log_id=log_id,
            predictions_count=len(predictions),
            processing_time=round(processing_time, 2),
            data=FoodLog(**food_log)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/logs", response_model=List[FoodLog])
async def get_food_logs(
    patient_id: str = Query(..., description="Patient ID"),
    start_date: Optional[datetime] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[datetime] = Query(None, description="End date (ISO format)"),
    meal_type: Optional[str] = Query(None, description="Filter by meal type"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    db=Depends(get_database)
):
    """Get food logs with optional filtering"""
    try:
        # Validate patient ID
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Build query
        query = {"patient_id": patient_id}
        
        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = start_date
            if end_date:
                query["timestamp"]["$lte"] = end_date
        
        if meal_type:
            query["meal_type"] = meal_type
        
        # Get logs
        logs = await db.food_logs.find(query).sort("timestamp", -1).skip(skip).limit(limit).to_list(limit)
        
        # Convert ObjectId to string
        for log in logs:
            log["_id"] = str(log["_id"])
        
        return logs
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch food logs: {str(e)}")

@router.get("/logs/{log_id}", response_model=FoodLog)
async def get_food_log(log_id: str, db=Depends(get_database)):
    """Get a specific food log by ID"""
    try:
        if not ObjectId.is_valid(log_id):
            raise HTTPException(status_code=400, detail="Invalid log ID format")
        
        log = await db.food_logs.find_one({"_id": ObjectId(log_id)})
        if not log:
            raise HTTPException(status_code=404, detail="Food log not found")
        
        log["_id"] = str(log["_id"])
        return log
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch food log: {str(e)}")

@router.get("/patient/{patient_id}/daily-summary")
async def get_daily_nutrition_summary(
    patient_id: str,
    date: Optional[datetime] = Query(None, description="Date for summary (defaults to today)"),
    db=Depends(get_database)
):
    """Get daily nutrition summary for a patient"""
    try:
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Use today's date if not provided
        if date is None:
            date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Calculate date range
        start_date = date
        end_date = date + timedelta(days=1)
        
        # Get logs for the day
        logs = await db.food_logs.find({
            "patient_id": patient_id,
            "timestamp": {"$gte": start_date, "$lt": end_date}
        }).to_list(1000)
        
        if not logs:
            return {
                "patient_id": patient_id,
                "date": date,
                "total_calories": 0,
                "total_protein": 0,
                "total_carbs": 0,
                "total_fat": 0,
                "total_fiber": 0,
                "meals_count": 0,
                "foods_count": 0,
                "message": "No food logs found for this date"
            }
        
        # Calculate totals
        total_calories = sum(log.get("total_calories", 0) for log in logs)
        total_protein = sum(log.get("total_protein", 0) for log in logs)
        total_carbs = sum(log.get("total_carbs", 0) for log in logs)
        total_fat = sum(log.get("total_fat", 0) for log in logs)
        total_fiber = sum(log.get("total_fiber", 0) for log in logs)
        
        # Count unique meals and foods
        meal_types = set(log.get("meal_type") for log in logs if log.get("meal_type"))
        unique_foods = set()
        for log in logs:
            if log.get("top_prediction") and log["top_prediction"].get("food_name"):
                unique_foods.add(log["top_prediction"]["food_name"])
        
        # Get patient's nutrition goals (placeholder - would come from patient profile)
        calorie_goal = 2000  # Default daily calorie goal
        protein_goal = 50    # Default daily protein goal (g)
        carbs_goal = 250     # Default daily carbs goal (g)
        fat_goal = 65        # Default daily fat goal (g)
        
        # Calculate progress percentages
        calorie_progress = (total_calories / calorie_goal) * 100 if calorie_goal > 0 else 0
        protein_progress = (total_protein / protein_goal) * 100 if protein_goal > 0 else 0
        carbs_progress = (total_carbs / carbs_goal) * 100 if carbs_goal > 0 else 0
        fat_progress = (total_fat / fat_goal) * 100 if fat_goal > 0 else 0
        
        return {
            "patient_id": patient_id,
            "date": date,
            "total_calories": round(total_calories, 1),
            "total_protein": round(total_protein, 1),
            "total_carbs": round(total_carbs, 1),
            "total_fat": round(total_fat, 1),
            "total_fiber": round(total_fiber, 1),
            "meals_count": len(meal_types),
            "foods_count": len(unique_foods),
            "calorie_goal": calorie_goal,
            "protein_goal": protein_goal,
            "carbs_goal": carbs_goal,
            "fat_goal": fat_goal,
            "calorie_progress": round(calorie_progress, 1),
            "protein_progress": round(protein_progress, 1),
            "carbs_progress": round(carbs_progress, 1),
            "fat_progress": round(fat_progress, 1)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate daily summary: {str(e)}")

@router.get("/patient/{patient_id}/nutrition-trends")
async def get_nutrition_trends(
    patient_id: str,
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze"),
    db=Depends(get_database)
):
    """Get nutrition trends over time"""
    try:
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get logs in date range
        logs = await db.food_logs.find({
            "patient_id": patient_id,
            "timestamp": {"$gte": start_date, "$lte": end_date}
        }).sort("timestamp", 1).to_list(1000)
        
        if not logs:
            return {
                "patient_id": patient_id,
                "date_range": {"start": start_date, "end": end_date},
                "trends": {},
                "message": "No food logs found for trend analysis"
            }
        
        # Group by date and calculate daily totals
        daily_totals = {}
        for log in logs:
            date_key = log["timestamp"].date()
            if date_key not in daily_totals:
                daily_totals[date_key] = {
                    "calories": 0,
                    "protein": 0,
                    "carbs": 0,
                    "fat": 0,
                    "fiber": 0,
                    "meals": 0
                }
            
            daily_totals[date_key]["calories"] += log.get("total_calories", 0)
            daily_totals[date_key]["protein"] += log.get("total_protein", 0)
            daily_totals[date_key]["carbs"] += log.get("total_carbs", 0)
            daily_totals[date_key]["fat"] += log.get("total_fat", 0)
            daily_totals[date_key]["fiber"] += log.get("total_fiber", 0)
            daily_totals[date_key]["meals"] += 1
        
        # Calculate trends
        dates = sorted(daily_totals.keys())
        calories_trend = [daily_totals[date]["calories"] for date in dates]
        protein_trend = [daily_totals[date]["protein"] for date in dates]
        carbs_trend = [daily_totals[date]["carbs"] for date in dates]
        fat_trend = [daily_totals[date]["fat"] for date in dates]
        
        # Calculate averages
        avg_calories = sum(calories_trend) / len(calories_trend) if calories_trend else 0
        avg_protein = sum(protein_trend) / len(protein_trend) if protein_trend else 0
        avg_carbs = sum(carbs_trend) / len(carbs_trend) if carbs_trend else 0
        avg_fat = sum(fat_trend) / len(fat_trend) if fat_trend else 0
        
        return {
            "patient_id": patient_id,
            "date_range": {"start": start_date, "end": end_date},
            "total_days": len(dates),
            "trends": {
                "calories": {
                    "daily_values": calories_trend,
                    "average": round(avg_calories, 1),
                    "min": min(calories_trend) if calories_trend else 0,
                    "max": max(calories_trend) if calories_trend else 0
                },
                "protein": {
                    "daily_values": protein_trend,
                    "average": round(avg_protein, 1),
                    "min": min(protein_trend) if protein_trend else 0,
                    "max": max(protein_trend) if protein_trend else 0
                },
                "carbs": {
                    "daily_values": carbs_trend,
                    "average": round(avg_carbs, 1),
                    "min": min(carbs_trend) if carbs_trend else 0,
                    "max": max(carbs_trend) if carbs_trend else 0
                },
                "fat": {
                    "daily_values": fat_trend,
                    "average": round(avg_fat, 1),
                    "min": min(fat_trend) if fat_trend else 0,
                    "max": max(fat_trend) if fat_trend else 0
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze nutrition trends: {str(e)}")

@router.delete("/logs/{log_id}")
async def delete_food_log(log_id: str, db=Depends(get_database)):
    """Delete a food log"""
    try:
        if not ObjectId.is_valid(log_id):
            raise HTTPException(status_code=400, detail="Invalid log ID format")
        
        # Check if log exists
        existing_log = await db.food_logs.find_one({"_id": ObjectId(log_id)})
        if not existing_log:
            raise HTTPException(status_code=404, detail="Food log not found")
        
        # Delete the log
        await db.food_logs.delete_one({"_id": ObjectId(log_id)})
        
        return {"message": "Food log deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete food log: {str(e)}")

@router.put("/logs/{log_id}/feedback")
async def update_food_log_feedback(
    log_id: str,
    user_correction: Optional[str] = Form(None, description="User correction if prediction was wrong"),
    user_rating: Optional[int] = Form(None, ge=1, le=5, description="User rating of prediction accuracy (1-5)"),
    db=Depends(get_database)
):
    """Update food log with user feedback"""
    try:
        if not ObjectId.is_valid(log_id):
            raise HTTPException(status_code=400, detail="Invalid log ID format")
        
        # Check if log exists
        existing_log = await db.food_logs.find_one({"_id": ObjectId(log_id)})
        if not existing_log:
            raise HTTPException(status_code=404, detail="Food log not found")
        
        # Prepare update data
        update_data = {"updated_at": datetime.now()}
        
        if user_correction is not None:
            update_data["user_correction"] = user_correction
        
        if user_rating is not None:
            update_data["user_rating"] = user_rating
        
        # Update the log
        await db.food_logs.update_one(
            {"_id": ObjectId(log_id)},
            {"$set": update_data}
        )
        
        return {"message": "Food log feedback updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update food log feedback: {str(e)}")
