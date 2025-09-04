from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from datetime import datetime, timedelta
from bson import ObjectId
import sys
import os
import time

# Add the parent directory to the path to import models and services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.wearable_data import (
    WearableDataUpload, DailyLog, WearableDataResponse, 
    WearableDataQuery, WearableDataSummary
)
from services.database import get_database
from services.wearable_data_processor import WearableDataProcessor

router = APIRouter(prefix="/api/v1/wearable-data", tags=["wearable-data"])

@router.post("/", response_model=WearableDataResponse)
async def upload_wearable_data(data: WearableDataUpload):
    """
    Upload wearable device data.
    
    This endpoint accepts JSON payloads containing:
    - Heart rate and HRV data
    - SpO2 (blood oxygen) data
    - Sleep stage data
    - Activity data
    - Steps and calories data
    - Temperature data
    
    The data is processed, analyzed, and stored in the daily_logs collection.
    """
    
    start_time = time.time()
    
    try:
        # Validate patient ID
        if not ObjectId.is_valid(data.patient_id):
            raise HTTPException(
                status_code=400, 
                detail="Invalid patient ID format"
            )
        
        # Get database connection
        db = get_database()
        
        # Check if patient exists
        patient = await db.patients.find_one({"_id": ObjectId(data.patient_id)})
        if not patient:
            raise HTTPException(
                status_code=404, 
                detail="Patient not found"
            )
        
        # Process the wearable data
        processor = WearableDataProcessor()
        processing_result = await processor.process_wearable_data(data)
        
        if processing_result["processing_status"] == "failed":
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process wearable data: {processing_result.get('error', 'Unknown error')}"
            )
        
        # Check if daily log already exists for this date
        date_start = data.date.replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = date_start + timedelta(days=1)
        
        existing_log = await db.daily_logs.find_one({
            "patient_id": data.patient_id,
            "date": {"$gte": date_start, "$lt": date_end}
        })
        
        if existing_log:
            # Update existing log
            update_data = processing_result["daily_log_data"].copy()
            update_data["updated_at"] = datetime.now()
            
            await db.daily_logs.update_one(
                {"_id": existing_log["_id"]},
                {"$set": update_data}
            )
            
            log_id = str(existing_log["_id"])
            message = "Wearable data updated successfully"
        else:
            # Create new log
            result = await db.daily_logs.insert_one(processing_result["daily_log_data"])
            log_id = str(result.inserted_id)
            message = "Wearable data uploaded successfully"
        
        # Get the created/updated log
        daily_log = await db.daily_logs.find_one({"_id": ObjectId(log_id)})
        daily_log["_id"] = str(daily_log["_id"])
        
        processing_time = time.time() - start_time
        
        return WearableDataResponse(
            success=True,
            message=message,
            log_id=log_id,
            data_points_processed=processing_result["total_data_points"],
            processing_time=round(processing_time, 2),
            data=DailyLog(**daily_log)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/", response_model=List[DailyLog])
async def get_wearable_data(
    patient_id: str = Query(..., description="Patient ID"),
    start_date: Optional[datetime] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[datetime] = Query(None, description="End date (ISO format)"),
    data_type: Optional[str] = Query(None, description="Filter by data type"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    db=Depends(get_database)
):
    """Get wearable data logs with optional filtering"""
    try:
        # Validate patient ID
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Build query
        query = {"patient_id": patient_id}
        
        if start_date or end_date:
            query["date"] = {}
            if start_date:
                query["date"]["$gte"] = start_date
            if end_date:
                query["date"]["$lte"] = end_date
        
        # Get logs
        logs = await db.daily_logs.find(query).sort("date", -1).skip(skip).limit(limit).to_list(limit)
        
        # Convert ObjectId to string
        for log in logs:
            log["_id"] = str(log["_id"])
        
        return logs
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch wearable data: {str(e)}")

@router.get("/{log_id}", response_model=DailyLog)
async def get_wearable_data_log(log_id: str, db=Depends(get_database)):
    """Get a specific wearable data log by ID"""
    try:
        if not ObjectId.is_valid(log_id):
            raise HTTPException(status_code=400, detail="Invalid log ID format")
        
        log = await db.daily_logs.find_one({"_id": ObjectId(log_id)})
        if not log:
            raise HTTPException(status_code=404, detail="Wearable data log not found")
        
        log["_id"] = str(log["_id"])
        return log
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch wearable data log: {str(e)}")

@router.get("/patient/{patient_id}/summary")
async def get_patient_wearable_summary(
    patient_id: str,
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db=Depends(get_database)
):
    """Get summary statistics for a patient's wearable data"""
    try:
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get logs in date range
        logs = await db.daily_logs.find({
            "patient_id": patient_id,
            "date": {"$gte": start_date, "$lte": end_date}
        }).sort("date", -1).to_list(1000)
        
        if not logs:
            return {
                "patient_id": patient_id,
                "date_range": {"start": start_date, "end": end_date},
                "total_days": 0,
                "message": "No wearable data found for the specified period"
            }
        
        # Calculate summary statistics
        total_steps = sum(log.get("total_steps", 0) for log in logs)
        total_calories = sum(log.get("total_calories_burned", 0) for log in logs)
        total_sleep_hours = sum(log.get("total_sleep_minutes", 0) for log in logs) / 60
        
        # Calculate averages
        heart_rates = [log.get("avg_heart_rate") for log in logs if log.get("avg_heart_rate")]
        spo2_values = [log.get("avg_spo2") for log in logs if log.get("avg_spo2")]
        
        avg_heart_rate = sum(heart_rates) / len(heart_rates) if heart_rates else 0
        avg_spo2 = sum(spo2_values) / len(spo2_values) if spo2_values else 0
        
        # Calculate sleep efficiency
        sleep_efficiencies = []
        for log in logs:
            if log.get("sleep_data"):
                for sleep in log["sleep_data"]:
                    if sleep.get("efficiency_percentage"):
                        sleep_efficiencies.append(sleep["efficiency_percentage"])
        
        avg_sleep_efficiency = sum(sleep_efficiencies) / len(sleep_efficiencies) if sleep_efficiencies else 0
        
        # Calculate activity minutes
        total_activity_minutes = 0
        for log in logs:
            if log.get("activity_data"):
                for activity in log["activity_data"]:
                    total_activity_minutes += activity.get("duration_minutes", 0)
        
        # Calculate data completeness
        data_completeness = (len(logs) / days) * 100
        
        return {
            "patient_id": patient_id,
            "date_range": {"start": start_date, "end": end_date},
            "total_days": len(logs),
            "total_steps": total_steps,
            "total_calories_burned": round(total_calories, 1),
            "total_sleep_hours": round(total_sleep_hours, 1),
            "avg_heart_rate": round(avg_heart_rate, 1),
            "avg_spo2": round(avg_spo2, 1),
            "sleep_efficiency": round(avg_sleep_efficiency, 1),
            "activity_minutes": total_activity_minutes,
            "data_completeness": round(data_completeness, 1)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

@router.get("/patient/{patient_id}/trends")
async def get_patient_trends(
    patient_id: str,
    days: int = Query(30, ge=7, le=365, description="Number of days to analyze"),
    db=Depends(get_database)
):
    """Get trend analysis for a patient's wearable data"""
    try:
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get logs in date range
        logs = await db.daily_logs.find({
            "patient_id": patient_id,
            "date": {"$gte": start_date, "$lte": end_date}
        }).sort("date", 1).to_list(1000)
        
        if not logs:
            return {
                "patient_id": patient_id,
                "date_range": {"start": start_date, "end": end_date},
                "trends": {},
                "message": "No wearable data found for trend analysis"
            }
        
        # Convert to DailyLog objects for processing
        from models.wearable_data import DailyLog
        daily_logs = [DailyLog(**log) for log in logs]
        
        # Analyze trends
        processor = WearableDataProcessor()
        trends = await processor.analyze_trends(daily_logs)
        
        return {
            "patient_id": patient_id,
            "date_range": {"start": start_date, "end": end_date},
            "total_days": len(logs),
            "trends": trends
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze trends: {str(e)}")

@router.get("/patient/{patient_id}/insights")
async def get_patient_insights(
    patient_id: str,
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze"),
    db=Depends(get_database)
):
    """Get health insights from wearable data"""
    try:
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get logs in date range
        logs = await db.daily_logs.find({
            "patient_id": patient_id,
            "date": {"$gte": start_date, "$lte": end_date}
        }).sort("date", -1).to_list(1000)
        
        if not logs:
            return {
                "patient_id": patient_id,
                "date_range": {"start": start_date, "end": end_date},
                "insights": [],
                "message": "No wearable data found for insights"
            }
        
        # Convert to DailyLog objects for processing
        from models.wearable_data import DailyLog
        daily_logs = [DailyLog(**log) for log in logs]
        
        # Generate insights
        processor = WearableDataProcessor()
        insights = await processor.generate_health_insights(daily_logs)
        
        return {
            "patient_id": patient_id,
            "date_range": {"start": start_date, "end": end_date},
            "total_days": len(logs),
            "insights": insights
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")

@router.delete("/{log_id}")
async def delete_wearable_data_log(log_id: str, db=Depends(get_database)):
    """Delete a wearable data log"""
    try:
        if not ObjectId.is_valid(log_id):
            raise HTTPException(status_code=400, detail="Invalid log ID format")
        
        # Check if log exists
        existing_log = await db.daily_logs.find_one({"_id": ObjectId(log_id)})
        if not existing_log:
            raise HTTPException(status_code=404, detail="Wearable data log not found")
        
        # Delete the log
        await db.daily_logs.delete_one({"_id": ObjectId(log_id)})
        
        return {"message": "Wearable data log deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete wearable data log: {str(e)}")
