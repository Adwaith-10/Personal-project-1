from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime
from bson import ObjectId
import sys
import os

# Add the parent directory to the path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.patient import Patient, PatientCreate, PatientUpdate
from services.database import get_database

router = APIRouter(prefix="/api/v1/patients", tags=["patients"])

@router.post("/", response_model=Patient)
async def create_patient(patient: PatientCreate, db=Depends(get_database)):
    """Create a new patient"""
    try:
        # Check if patient with same email already exists
        existing_patient = await db.patients.find_one({"email": patient.email})
        if existing_patient:
            raise HTTPException(status_code=400, detail="Patient with this email already exists")
        
        # Prepare patient data
        patient_data = patient.dict()
        patient_data["created_at"] = datetime.utcnow()
        patient_data["updated_at"] = datetime.utcnow()
        patient_data["medical_history"] = []
        patient_data["current_medications"] = []
        patient_data["allergies"] = []
        
        # Insert into database
        result = await db.patients.insert_one(patient_data)
        
        # Get the created patient
        created_patient = await db.patients.find_one({"_id": result.inserted_id})
        created_patient["_id"] = str(created_patient["_id"])
        
        return created_patient
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create patient: {str(e)}")

@router.get("/", response_model=List[Patient])
async def get_patients(
    skip: int = 0, 
    limit: int = 100, 
    search: Optional[str] = None,
    db=Depends(get_database)
):
    """Get all patients with optional search and pagination"""
    try:
        query = {}
        if search:
            query = {
                "$or": [
                    {"first_name": {"$regex": search, "$options": "i"}},
                    {"last_name": {"$regex": search, "$options": "i"}},
                    {"email": {"$regex": search, "$options": "i"}}
                ]
            }
        
        patients = await db.patients.find(query).skip(skip).limit(limit).to_list(limit)
        
        # Convert ObjectId to string
        for patient in patients:
            patient["_id"] = str(patient["_id"])
        
        return patients
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch patients: {str(e)}")

@router.get("/{patient_id}", response_model=Patient)
async def get_patient(patient_id: str, db=Depends(get_database)):
    """Get a specific patient by ID"""
    try:
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        patient = await db.patients.find_one({"_id": ObjectId(patient_id)})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        patient["_id"] = str(patient["_id"])
        return patient
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch patient: {str(e)}")

@router.put("/{patient_id}", response_model=Patient)
async def update_patient(
    patient_id: str, 
    patient_update: PatientUpdate, 
    db=Depends(get_database)
):
    """Update a patient's information"""
    try:
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Check if patient exists
        existing_patient = await db.patients.find_one({"_id": ObjectId(patient_id)})
        if not existing_patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Prepare update data
        update_data = patient_update.dict(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        # Update patient
        await db.patients.update_one(
            {"_id": ObjectId(patient_id)},
            {"$set": update_data}
        )
        
        # Get updated patient
        updated_patient = await db.patients.find_one({"_id": ObjectId(patient_id)})
        updated_patient["_id"] = str(updated_patient["_id"])
        
        return updated_patient
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update patient: {str(e)}")

@router.delete("/{patient_id}")
async def delete_patient(patient_id: str, db=Depends(get_database)):
    """Delete a patient"""
    try:
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Check if patient exists
        existing_patient = await db.patients.find_one({"_id": ObjectId(patient_id)})
        if not existing_patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Delete patient
        await db.patients.delete_one({"_id": ObjectId(patient_id)})
        
        return {"message": "Patient deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete patient: {str(e)}")

@router.get("/{patient_id}/health-metrics")
async def get_patient_health_metrics(
    patient_id: str, 
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db=Depends(get_database)
):
    """Get health metrics for a specific patient"""
    try:
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Check if patient exists
        patient = await db.patients.find_one({"_id": ObjectId(patient_id)})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Build query for health metrics
        query = {"patient_id": patient_id}
        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = start_date
            if end_date:
                query["timestamp"]["$lte"] = end_date
        
        # Get health metrics
        metrics = await db.health_metrics.find(query).sort("timestamp", -1).to_list(1000)
        
        # Convert ObjectId to string
        for metric in metrics:
            metric["_id"] = str(metric["_id"])
        
        return {"metrics": metrics, "count": len(metrics)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch health metrics: {str(e)}")
