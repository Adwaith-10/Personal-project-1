import re
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone: str) -> bool:
    """Validate phone number format"""
    pattern = r'^\+?1?\d{9,15}$'
    return re.match(pattern, phone) is not None

def calculate_age(date_of_birth: datetime) -> int:
    """Calculate age from date of birth"""
    today = datetime.now()
    age = today.year - date_of_birth.year
    if today.month < date_of_birth.month or (today.month == date_of_birth.month and today.day < date_of_birth.day):
        age -= 1
    return age

def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Calculate BMI from weight and height"""
    if height_cm <= 0 or weight_kg <= 0:
        return 0.0
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 2)

def get_bmi_category(bmi: float) -> str:
    """Get BMI category based on BMI value"""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def format_datetime(dt: datetime) -> str:
    """Format datetime to ISO string"""
    return dt.isoformat()

def parse_datetime(date_string: str) -> datetime:
    """Parse datetime from string"""
    try:
        return datetime.fromisoformat(date_string.replace('Z', '+00:00'))
    except ValueError:
        return datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')

def generate_patient_summary(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary of patient information"""
    summary = {
        "id": patient_data.get("_id"),
        "name": f"{patient_data.get('first_name', '')} {patient_data.get('last_name', '')}",
        "age": calculate_age(parse_datetime(patient_data.get('date_of_birth'))) if patient_data.get('date_of_birth') else None,
        "gender": patient_data.get('gender'),
        "email": patient_data.get('email'),
        "phone": patient_data.get('phone'),
        "blood_type": patient_data.get('blood_type'),
        "bmi": None,
        "bmi_category": None
    }
    
    # Calculate BMI if weight and height are available
    weight = patient_data.get('weight_kg')
    height = patient_data.get('height_cm')
    if weight and height:
        bmi = calculate_bmi(weight, height)
        summary["bmi"] = bmi
        summary["bmi_category"] = get_bmi_category(bmi)
    
    return summary

def validate_health_metrics(metrics: Dict[str, Any]) -> List[str]:
    """Validate health metrics and return list of validation errors"""
    errors = []
    
    # Heart rate validation
    heart_rate = metrics.get('heart_rate')
    if heart_rate is not None and (heart_rate < 30 or heart_rate > 200):
        errors.append("Heart rate must be between 30 and 200 bpm")
    
    # Blood pressure validation
    systolic = metrics.get('blood_pressure_systolic')
    diastolic = metrics.get('blood_pressure_diastolic')
    if systolic is not None and (systolic < 70 or systolic > 200):
        errors.append("Systolic blood pressure must be between 70 and 200 mmHg")
    if diastolic is not None and (diastolic < 40 or diastolic > 130):
        errors.append("Diastolic blood pressure must be between 40 and 130 mmHg")
    if systolic and diastolic and systolic <= diastolic:
        errors.append("Systolic blood pressure must be higher than diastolic")
    
    # Temperature validation
    temperature = metrics.get('temperature')
    if temperature is not None and (temperature < 35.0 or temperature > 42.0):
        errors.append("Temperature must be between 35.0 and 42.0 °C")
    
    # Oxygen saturation validation
    oxygen_sat = metrics.get('oxygen_saturation')
    if oxygen_sat is not None and (oxygen_sat < 70.0 or oxygen_sat > 100.0):
        errors.append("Oxygen saturation must be between 70.0 and 100.0%")
    
    # Respiratory rate validation
    resp_rate = metrics.get('respiratory_rate')
    if resp_rate is not None and (resp_rate < 8 or resp_rate > 40):
        errors.append("Respiratory rate must be between 8 and 40 breaths/min")
    
    # Glucose level validation
    glucose = metrics.get('glucose_level')
    if glucose is not None and (glucose < 50.0 or glucose > 500.0):
        errors.append("Glucose level must be between 50.0 and 500.0 mg/dL")
    
    return errors

def get_date_range(days: int = 30) -> tuple:
    """Get date range for the last N days"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return start_date, end_date

def serialize_object_id(obj):
    """Custom JSON serializer for ObjectId"""
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    elif hasattr(obj, '__str__'):
        return str(obj)
    else:
        return obj

def create_response(data: Any, message: str = "Success", status: str = "success") -> Dict[str, Any]:
    """Create a standardized API response"""
    return {
        "status": status,
        "message": message,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }

def paginate_results(results: List[Any], page: int = 1, page_size: int = 10) -> Dict[str, Any]:
    """Paginate results"""
    total = len(results)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    paginated_results = results[start_idx:end_idx]
    
    return {
        "results": paginated_results,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size,
            "has_next": end_idx < total,
            "has_prev": page > 1
        }
    }
