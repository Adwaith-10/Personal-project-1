from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class BiomarkerType(str, Enum):
    """Types of biomarkers that can be extracted"""
    LDL = "LDL"
    HDL = "HDL"
    GLUCOSE = "glucose"
    HEMOGLOBIN = "hemoglobin"
    CHOLESTEROL = "cholesterol"
    TRIGLYCERIDES = "triglycerides"
    CREATININE = "creatinine"
    BUN = "BUN"
    SODIUM = "sodium"
    POTASSIUM = "potassium"
    CALCIUM = "calcium"
    MAGNESIUM = "magnesium"
    PHOSPHORUS = "phosphorus"
    ALBUMIN = "albumin"
    BILIRUBIN = "bilirubin"
    ALT = "ALT"
    AST = "AST"
    ALKALINE_PHOSPHATASE = "alkaline_phosphatase"
    WBC = "WBC"
    RBC = "RBC"
    PLATELETS = "platelets"
    HEMATOCRIT = "hematocrit"
    MCV = "MCV"
    MCH = "MCH"
    MCHC = "MCHC"
    RDW = "RDW"

class BiomarkerResult(BaseModel):
    """Individual biomarker result"""
    name: str
    value: float
    unit: str
    reference_range: Optional[str] = None
    status: Optional[str] = None  # normal, high, low
    extracted_confidence: float = Field(..., ge=0.0, le=1.0)

class LabReportUpload(BaseModel):
    """Model for lab report upload request"""
    patient_id: str
    report_date: Optional[datetime] = None
    lab_name: Optional[str] = None
    notes: Optional[str] = None

class LabReport(BaseModel):
    """Complete lab report model"""
    id: Optional[str] = Field(alias="_id")
    user_id: str = Field(..., description="User ID")
    patient_id: str
    report_date: datetime
    lab_name: Optional[str] = None
    original_filename: str
    file_size: int
    biomarkers: List[BiomarkerResult]
    extracted_text: str
    processing_status: str = "completed"  # processing, completed, failed
    processing_errors: Optional[List[str]] = None
    created_at: datetime
    updated_at: datetime
    notes: Optional[str] = None
    
    class Config:
        validate_by_name = True
        json_schema_extra = {
            "example": {
                "patient_id": "507f1f77bcf86cd799439011",
                "report_date": "2024-01-15T00:00:00",
                "lab_name": "LabCorp",
                "original_filename": "lab_report_2024.pdf",
                "file_size": 1024000,
                "biomarkers": [
                    {
                        "name": "LDL",
                        "value": 120.0,
                        "unit": "mg/dL",
                        "reference_range": "<100",
                        "status": "high",
                        "extracted_confidence": 0.95
                    },
                    {
                        "name": "glucose",
                        "value": 95.0,
                        "unit": "mg/dL",
                        "reference_range": "70-100",
                        "status": "normal",
                        "extracted_confidence": 0.92
                    }
                ],
                "extracted_text": "Complete extracted text from PDF...",
                "processing_status": "completed",
                "notes": "Annual checkup lab report"
            }
        }

class LabReportResponse(BaseModel):
    """Response model for lab report operations"""
    success: bool
    message: str
    report_id: Optional[str] = None
    biomarkers_found: Optional[int] = None
    processing_time: Optional[float] = None
    data: Optional[LabReport] = None

class BiomarkerExtractionConfig(BaseModel):
    """Configuration for biomarker extraction"""
    biomarker_patterns: Dict[str, List[str]] = {
        "LDL": ["LDL", "LDL-C", "Low Density Lipoprotein"],
        "HDL": ["HDL", "HDL-C", "High Density Lipoprotein"],
        "glucose": ["Glucose", "Blood Sugar", "FBS", "Random Glucose"],
        "hemoglobin": ["Hemoglobin", "Hgb", "Hb"],
        "cholesterol": ["Total Cholesterol", "Cholesterol"],
        "triglycerides": ["Triglycerides", "TG"],
        "creatinine": ["Creatinine", "Cr"],
        "BUN": ["BUN", "Blood Urea Nitrogen"],
        "sodium": ["Sodium", "Na"],
        "potassium": ["Potassium", "K"],
        "calcium": ["Calcium", "Ca"],
        "magnesium": ["Magnesium", "Mg"],
        "phosphorus": ["Phosphorus", "Phos"],
        "albumin": ["Albumin", "Alb"],
        "bilirubin": ["Bilirubin", "Total Bilirubin"],
        "ALT": ["ALT", "Alanine Aminotransferase", "SGPT"],
        "AST": ["AST", "Aspartate Aminotransferase", "SGOT"],
        "alkaline_phosphatase": ["Alkaline Phosphatase", "ALP"],
        "WBC": ["WBC", "White Blood Cell Count", "Leukocytes"],
        "RBC": ["RBC", "Red Blood Cell Count", "Erythrocytes"],
        "platelets": ["Platelets", "PLT", "Thrombocytes"],
        "hematocrit": ["Hematocrit", "Hct"],
        "MCV": ["MCV", "Mean Corpuscular Volume"],
        "MCH": ["MCH", "Mean Corpuscular Hemoglobin"],
        "MCHC": ["MCHC", "Mean Corpuscular Hemoglobin Concentration"],
        "RDW": ["RDW", "Red Cell Distribution Width"]
    }
    
    unit_patterns: Dict[str, List[str]] = {
        "mg/dL": ["mg/dL", "mg/dl", "mg/dL"],
        "mmol/L": ["mmol/L", "mmol/l"],
        "g/dL": ["g/dL", "g/dl"],
        "mEq/L": ["mEq/L", "mEq/l"],
        "U/L": ["U/L", "u/L"],
        "K/uL": ["K/uL", "k/uL", "K/μL"],
        "M/uL": ["M/uL", "m/uL", "M/μL"],
        "%": ["%", "percent"]
    }
    
    reference_range_patterns: List[str] = [
        r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",  # 70-100
        r"<\s*(\d+\.?\d*)",  # <100
        r">\s*(\d+\.?\d*)",  # >12
        r"(\d+\.?\d*)\s*to\s*(\d+\.?\d*)",  # 70 to 100
    ]
