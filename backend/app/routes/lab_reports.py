from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from datetime import datetime
from bson import ObjectId
import sys
import os
import time

# Add the parent directory to the path to import models and services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lab_report import LabReport, LabReportResponse, LabReportUpload
from services.database import get_database
from services.lab_report_processor import LabReportProcessor

router = APIRouter(prefix="/api/v1/lab-reports", tags=["lab-reports"])

@router.post("/upload-lab-report", response_model=LabReportResponse)
async def upload_lab_report(
    file: UploadFile = File(..., description="PDF lab report file"),
    patient_id: str = Form(..., description="Patient ID"),
    report_date: Optional[str] = Form(None, description="Report date (YYYY-MM-DD)"),
    lab_name: Optional[str] = Form(None, description="Laboratory name"),
    notes: Optional[str] = Form(None, description="Additional notes")
):
    """
    Upload and process a lab report PDF file.
    
    This endpoint:
    1. Accepts a PDF file upload
    2. Extracts text using pdfplumber
    3. Identifies and extracts biomarkers (LDL, glucose, hemoglobin, etc.)
    4. Stores the processed data in MongoDB
    5. Returns the extracted information
    """
    
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF files are supported"
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
        
        # Read file content
        file_content = await file.read()
        
        # Process the lab report
        processor = LabReportProcessor()
        processing_result = await processor.process_lab_report(file_content, file.filename)
        
        if processing_result["processing_status"] == "failed":
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process lab report: {processing_result.get('error', 'Unknown error')}"
            )
        
        # Parse report date
        parsed_report_date = None
        if report_date:
            try:
                parsed_report_date = datetime.strptime(report_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid report date format. Use YYYY-MM-DD"
                )
        elif processing_result["metadata"].get("report_date"):
            parsed_report_date = processing_result["metadata"]["report_date"]
        else:
            parsed_report_date = datetime.now()
        
        # Use extracted lab name if not provided
        if not lab_name and processing_result["metadata"].get("lab_name"):
            lab_name = processing_result["metadata"]["lab_name"]
        
        # Create lab report document
        lab_report_data = {
            "patient_id": patient_id,
            "report_date": parsed_report_date,
            "lab_name": lab_name,
            "original_filename": file.filename,
            "file_size": len(file_content),
            "biomarkers": [biomarker.dict() for biomarker in processing_result["biomarkers"]],
            "extracted_text": processing_result["extracted_text"],
            "processing_status": processing_result["processing_status"],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "notes": notes
        }
        
        # Store in MongoDB
        result = await db.lab_reports.insert_one(lab_report_data)
        
        # Get the created report
        created_report = await db.lab_reports.find_one({"_id": result.inserted_id})
        created_report["_id"] = str(created_report["_id"])
        
        processing_time = time.time() - start_time
        
        return LabReportResponse(
            success=True,
            message="Lab report uploaded and processed successfully",
            report_id=str(result.inserted_id),
            biomarkers_found=len(processing_result["biomarkers"]),
            processing_time=round(processing_time, 2),
            data=LabReport(**created_report)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/", response_model=List[LabReport])
async def get_lab_reports(
    patient_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db=Depends(get_database)
):
    """Get lab reports with optional filtering by patient"""
    try:
        query = {}
        if patient_id:
            if not ObjectId.is_valid(patient_id):
                raise HTTPException(status_code=400, detail="Invalid patient ID format")
            query["patient_id"] = patient_id
        
        lab_reports = await db.lab_reports.find(query).skip(skip).limit(limit).to_list(limit)
        
        # Convert ObjectId to string
        for report in lab_reports:
            report["_id"] = str(report["_id"])
        
        return lab_reports
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch lab reports: {str(e)}")

@router.get("/{report_id}", response_model=LabReport)
async def get_lab_report(report_id: str, db=Depends(get_database)):
    """Get a specific lab report by ID"""
    try:
        if not ObjectId.is_valid(report_id):
            raise HTTPException(status_code=400, detail="Invalid report ID format")
        
        report = await db.lab_reports.find_one({"_id": ObjectId(report_id)})
        if not report:
            raise HTTPException(status_code=404, detail="Lab report not found")
        
        report["_id"] = str(report["_id"])
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch lab report: {str(e)}")

@router.get("/patient/{patient_id}/biomarkers")
async def get_patient_biomarkers(
    patient_id: str,
    biomarker_name: Optional[str] = None,
    db=Depends(get_database)
):
    """Get biomarker trends for a specific patient"""
    try:
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Get all lab reports for the patient
        reports = await db.lab_reports.find({"patient_id": patient_id}).sort("report_date", -1).to_list(1000)
        
        # Extract biomarkers
        all_biomarkers = []
        for report in reports:
            for biomarker in report.get("biomarkers", []):
                biomarker_data = biomarker.copy()
                biomarker_data["report_date"] = report["report_date"]
                biomarker_data["report_id"] = str(report["_id"])
                all_biomarkers.append(biomarker_data)
        
        # Filter by biomarker name if specified
        if biomarker_name:
            all_biomarkers = [b for b in all_biomarkers if b["name"].lower() == biomarker_name.lower()]
        
        return {
            "patient_id": patient_id,
            "biomarkers": all_biomarkers,
            "total_reports": len(reports),
            "total_biomarkers": len(all_biomarkers)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch patient biomarkers: {str(e)}")

@router.delete("/{report_id}")
async def delete_lab_report(report_id: str, db=Depends(get_database)):
    """Delete a lab report"""
    try:
        if not ObjectId.is_valid(report_id):
            raise HTTPException(status_code=400, detail="Invalid report ID format")
        
        # Check if report exists
        existing_report = await db.lab_reports.find_one({"_id": ObjectId(report_id)})
        if not existing_report:
            raise HTTPException(status_code=404, detail="Lab report not found")
        
        # Delete the report
        await db.lab_reports.delete_one({"_id": ObjectId(report_id)})
        
        return {"message": "Lab report deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete lab report: {str(e)}")
