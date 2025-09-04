from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from datetime import datetime
from bson import ObjectId
import sys
import os

# Add the parent directory to the path to import models and services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.chatbot import (
    ChatRequest, ChatResponse, ChatSession, ChatMessage,
    HealthAnalysis, ChatSessionSummary, VirtualDoctorProfile
)
from services.database import get_database
from services.virtual_doctor_service import VirtualDoctorService

router = APIRouter(prefix="/api/v1/virtual-doctor", tags=["virtual-doctor"])

# Initialize the virtual doctor service
virtual_doctor = VirtualDoctorService()

@router.post("/chat", response_model=ChatResponse)
async def chat_with_doctor(request: ChatRequest):
    """
    Chat with the virtual doctor for lifestyle medicine consultation.
    
    This endpoint allows patients to have a conversation with Dr. Sarah Chen,
    a board-certified lifestyle medicine specialist. The doctor analyzes:
    - Wearable device trends (heart rate, sleep, activity, steps)
    - Health predictions (LDL, glucose, hemoglobin)
    - Patient demographics and medical history
    
    Returns personalized health advice, lifestyle recommendations, and follow-up guidance.
    """
    
    try:
        # Validate patient ID
        if not ObjectId.is_valid(request.patient_id):
            raise HTTPException(
                status_code=400,
                detail="Invalid patient ID format"
            )
        
        # Get database connection
        db = get_database()
        
        # Check if patient exists
        patient = await db.patients.find_one({"_id": ObjectId(request.patient_id)})
        if not patient:
            raise HTTPException(
                status_code=404,
                detail="Patient not found"
            )
        
        # Chat with virtual doctor
        response = await virtual_doctor.chat_with_doctor(request)
        
        # Store chat session in database
        chat_session_data = {
            "session_id": response.session_id,
            "patient_id": request.patient_id,
            "user_message": request.message,
            "doctor_response": response.message,
            "timestamp": response.timestamp,
            "health_insights": response.health_insights,
            "recommendations": response.recommendations,
            "urgency_level": response.urgency_level,
            "include_health_data": request.include_health_data,
            "include_trends": request.include_trends
        }
        
        await db.chat_sessions.insert_one(chat_session_data)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )

@router.get("/sessions/{patient_id}", response_model=List[ChatSession])
async def get_chat_sessions(
    patient_id: str,
    limit: int = Query(50, ge=1, le=100, description="Number of sessions to return"),
    skip: int = Query(0, ge=0, description="Number of sessions to skip"),
    db=Depends(get_database)
):
    """Get chat session history for a patient"""
    try:
        # Validate patient ID
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Get chat sessions
        sessions = await db.chat_sessions.find({
            "patient_id": patient_id
        }).sort("timestamp", -1).skip(skip).limit(limit).to_list(limit)
        
        # Convert to response format
        chat_sessions = []
        for session_data in sessions:
            # Create messages
            messages = []
            
            # Add user message
            if "user_message" in session_data:
                messages.append({
                    "role": "user",
                    "content": session_data["user_message"],
                    "timestamp": session_data["timestamp"]
                })
            
            # Add doctor response
            if "doctor_response" in session_data:
                messages.append({
                    "role": "assistant",
                    "content": session_data["doctor_response"],
                    "timestamp": session_data["timestamp"]
                })
            
            chat_session = ChatSession(
                session_id=session_data["session_id"],
                patient_id=session_data["patient_id"],
                start_time=session_data["timestamp"],
                messages=messages
            )
            chat_sessions.append(chat_session)
        
        return chat_sessions
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chat sessions: {str(e)}")

@router.get("/sessions/{session_id}/summary", response_model=ChatSessionSummary)
async def get_session_summary(session_id: str):
    """Get summary of a specific chat session"""
    try:
        summary = await virtual_doctor.get_session_summary(session_id)
        if not summary:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session summary: {str(e)}")

@router.get("/analysis/{patient_id}", response_model=HealthAnalysis)
async def get_health_analysis(patient_id: str):
    """
    Get comprehensive health analysis from the virtual doctor.
    
    This endpoint provides a detailed health assessment including:
    - Overall health score
    - Risk factors and positive factors
    - Lifestyle assessment by category
    - Personalized recommendations
    - Priority actions
    - Monitoring plan
    """
    
    try:
        # Validate patient ID
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Get database connection
        db = get_database()
        
        # Check if patient exists
        patient = await db.patients.find_one({"_id": ObjectId(patient_id)})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Generate health analysis
        analysis = await virtual_doctor.generate_health_analysis(patient_id)
        
        # Store analysis in database
        analysis_data = {
            "patient_id": patient_id,
            "analysis_date": analysis.analysis_date,
            "overall_health_score": analysis.overall_health_score,
            "risk_factors": analysis.risk_factors,
            "positive_factors": analysis.positive_factors,
            "lifestyle_assessment": analysis.lifestyle_assessment,
            "recommendations": analysis.recommendations,
            "priority_actions": analysis.priority_actions,
            "monitoring_plan": analysis.monitoring_plan
        }
        
        await db.health_analyses.insert_one(analysis_data)
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health analysis failed: {str(e)}")

@router.get("/doctor-profile", response_model=VirtualDoctorProfile)
async def get_doctor_profile():
    """Get the virtual doctor's profile and credentials"""
    return virtual_doctor.doctor_profile

@router.get("/health-trends/{patient_id}")
async def get_health_trends(patient_id: str):
    """Get health trends analysis for a patient"""
    try:
        # Validate patient ID
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Get health trends
        trends = await virtual_doctor.get_health_trends(patient_id)
        
        if not trends:
            return {
                "patient_id": patient_id,
                "message": "No wearable data available for trend analysis",
                "trends": {}
            }
        
        # Convert to response format
        trends_data = {}
        if trends.heart_rate_trend:
            trends_data["heart_rate"] = {
                "current_value": trends.heart_rate_trend.current_value,
                "trend_direction": trends.heart_rate_trend.trend_direction,
                "trend_strength": trends.heart_rate_trend.trend_strength,
                "status": trends.heart_rate_trend.status,
                "unit": trends.heart_rate_trend.unit
            }
        
        if trends.sleep_trend:
            trends_data["sleep"] = {
                "current_value": trends.sleep_trend.current_value,
                "trend_direction": trends.sleep_trend.trend_direction,
                "trend_strength": trends.sleep_trend.trend_strength,
                "status": trends.sleep_trend.status,
                "unit": trends.sleep_trend.unit
            }
        
        if trends.steps_trend:
            trends_data["steps"] = {
                "current_value": trends.steps_trend.current_value,
                "trend_direction": trends.steps_trend.trend_direction,
                "trend_strength": trends.steps_trend.trend_strength,
                "status": trends.steps_trend.status,
                "unit": trends.steps_trend.unit
            }
        
        return {
            "patient_id": patient_id,
            "analysis_date": datetime.now(),
            "trends": trends_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health trends: {str(e)}")

@router.get("/health-predictions/{patient_id}")
async def get_health_predictions(patient_id: str):
    """Get health predictions for a patient"""
    try:
        # Validate patient ID
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Get health predictions
        predictions = await virtual_doctor.get_health_predictions(patient_id)
        
        if not predictions:
            return {
                "patient_id": patient_id,
                "message": "No health predictions available",
                "predictions": {}
            }
        
        # Convert to response format
        predictions_data = {}
        if predictions.ldl_prediction:
            predictions_data["ldl"] = {
                "current_value": predictions.ldl_prediction.current_value,
                "status": predictions.ldl_prediction.status,
                "unit": predictions.ldl_prediction.unit,
                "normal_range": predictions.ldl_prediction.normal_range
            }
        
        if predictions.glucose_prediction:
            predictions_data["glucose"] = {
                "current_value": predictions.glucose_prediction.current_value,
                "status": predictions.glucose_prediction.status,
                "unit": predictions.glucose_prediction.unit,
                "normal_range": predictions.glucose_prediction.normal_range
            }
        
        if predictions.hemoglobin_prediction:
            predictions_data["hemoglobin"] = {
                "current_value": predictions.hemoglobin_prediction.current_value,
                "status": predictions.hemoglobin_prediction.status,
                "unit": predictions.hemoglobin_prediction.unit,
                "normal_range": predictions.hemoglobin_prediction.normal_range
            }
        
        return {
            "patient_id": patient_id,
            "analysis_date": datetime.now(),
            "predictions": predictions_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health predictions: {str(e)}")

@router.get("/patient-context/{patient_id}")
async def get_patient_context(patient_id: str):
    """Get patient context for the virtual doctor"""
    try:
        # Validate patient ID
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Get patient context
        context = await virtual_doctor.get_patient_context(patient_id)
        
        return {
            "patient_id": patient_id,
            "context": {
                "age": context.age,
                "gender": context.gender,
                "bmi": context.bmi,
                "medical_conditions": context.medical_conditions,
                "medications": context.medications,
                "lifestyle_factors": context.lifestyle_factors
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get patient context: {str(e)}")

@router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str, db=Depends(get_database)):
    """Delete a specific chat session"""
    try:
        # Check if session exists
        existing_session = await db.chat_sessions.find_one({"session_id": session_id})
        if not existing_session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Delete the session
        await db.chat_sessions.delete_one({"session_id": session_id})
        
        return {"message": "Chat session deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chat session: {str(e)}")

@router.get("/status")
async def get_service_status():
    """Get the status of the virtual doctor service"""
    try:
        status = {
            "service": "Virtual Doctor Service",
            "status": "active",
            "doctor_name": virtual_doctor.doctor_profile.doctor_name,
            "specialty": virtual_doctor.doctor_profile.specialty,
            "langchain_available": hasattr(virtual_doctor, 'llm') and virtual_doctor.llm is not None,
            "active_sessions": len(virtual_doctor.sessions),
            "timestamp": datetime.now()
        }
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get service status: {str(e)}")

@router.post("/sessions/{session_id}/feedback")
async def submit_session_feedback(
    session_id: str,
    rating: int = Query(..., ge=1, le=5, description="Session rating (1-5)"),
    feedback: str = Query(..., description="Patient feedback"),
    db=Depends(get_database)
):
    """Submit feedback for a chat session"""
    try:
        # Check if session exists
        existing_session = await db.chat_sessions.find_one({"session_id": session_id})
        if not existing_session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Update session with feedback
        await db.chat_sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "session_rating": rating,
                    "patient_feedback": feedback,
                    "feedback_timestamp": datetime.now()
                }
            }
        )
        
        return {"message": "Feedback submitted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

@router.get("/recommendations/{patient_id}")
async def get_personalized_recommendations(patient_id: str):
    """Get personalized lifestyle recommendations for a patient"""
    try:
        # Validate patient ID
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Get health analysis
        analysis = await virtual_doctor.generate_health_analysis(patient_id)
        
        # Format recommendations
        recommendations = {
            "patient_id": patient_id,
            "analysis_date": datetime.now(),
            "overall_health_score": analysis.overall_health_score,
            "risk_factors": analysis.risk_factors,
            "recommendations": analysis.recommendations,
            "priority_actions": analysis.priority_actions,
            "monitoring_plan": analysis.monitoring_plan
        }
        
        return recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")
