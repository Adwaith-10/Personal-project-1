from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from datetime import datetime, timedelta
from bson import ObjectId
import sys
import os
import time

# Add the parent directory to the path to import models and services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.health_prediction import (
    HealthPredictionRequest, HealthPredictionResponse, HealthPrediction,
    ModelTrainingRequest, ModelTrainingResponse, ModelEvaluationRequest, ModelEvaluationResponse,
    HealthTrendAnalysis, HealthInsights, HealthMetricType
)
from services.database import get_database
from services.health_prediction_service import HealthPredictionService

router = APIRouter(prefix="/api/v1/health-prediction", tags=["health-prediction"])

# Initialize the health prediction service
health_predictor = HealthPredictionService()

@router.post("/predict", response_model=HealthPredictionResponse)
async def predict_health_metrics(request: HealthPredictionRequest):
    """
    Predict internal health metrics (LDL, glucose, hemoglobin) using ML models.
    
    This endpoint uses trained XGBoost models to predict health metrics based on:
    - Wearable device data (heart rate, steps, sleep, etc.)
    - Diet and nutrition data
    - Demographic information
    - Lifestyle factors
    
    Returns predictions with confidence scores, risk assessments, and recommendations.
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
        
        # Make predictions
        prediction_response = health_predictor.predict_health_metrics(request)
        
        # Store prediction in database
        prediction_log = {
            "patient_id": request.patient_id,
            "timestamp": prediction_response.timestamp,
            "predictions": [pred.dict() for pred in prediction_response.predictions],
            "overall_health_score": prediction_response.overall_health_score,
            "risk_factors": prediction_response.risk_factors,
            "data_quality_score": prediction_response.data_quality_score,
            "model_version": prediction_response.model_version,
            "processing_time": prediction_response.processing_time,
            "input_features": {
                "wearable_features": request.wearable_features.dict() if request.wearable_features else None,
                "diet_features": request.diet_features.dict() if request.diet_features else None,
                "demographic_features": request.demographic_features.dict(),
                "lifestyle_features": request.lifestyle_features.dict() if request.lifestyle_features else None
            }
        }
        
        await db.health_predictions.insert_one(prediction_log)
        
        return prediction_response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@router.get("/predictions", response_model=List[HealthPredictionResponse])
async def get_health_predictions(
    patient_id: str = Query(..., description="Patient ID"),
    start_date: Optional[datetime] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[datetime] = Query(None, description="End date (ISO format)"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    db=Depends(get_database)
):
    """Get historical health predictions for a patient"""
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
        
        # Get predictions
        predictions = await db.health_predictions.find(query).sort("timestamp", -1).skip(skip).limit(limit).to_list(limit)
        
        # Convert to response format
        responses = []
        for pred in predictions:
            # Convert predictions back to HealthPrediction objects
            health_predictions = []
            for pred_data in pred.get("predictions", []):
                health_predictions.append(HealthPrediction(**pred_data))
            
            response = HealthPredictionResponse(
                success=True,
                patient_id=pred["patient_id"],
                timestamp=pred["timestamp"],
                predictions=health_predictions,
                model_version=pred.get("model_version", "1.0.0"),
                processing_time=pred.get("processing_time", 0.0),
                data_quality_score=pred.get("data_quality_score", 0.0),
                missing_features=pred.get("missing_features", []),
                overall_health_score=pred.get("overall_health_score"),
                risk_factors=pred.get("risk_factors", []),
                recommendations=pred.get("recommendations", [])
            )
            responses.append(response)
        
        return responses
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch predictions: {str(e)}")

@router.post("/train", response_model=ModelTrainingResponse)
async def train_health_prediction_models(request: ModelTrainingRequest):
    """
    Train new health prediction models.
    
    This endpoint triggers the training of XGBoost models for predicting health metrics.
    Training can be done with new data or to update existing models.
    """
    
    try:
        # Generate training ID
        training_id = f"training_{int(time.time())}"
        
        # Import the ML pipeline
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "ml_models"))
        
        try:
            from health_prediction_pipeline import HealthMetricsPredictor
            
            # Initialize predictor
            predictor = HealthMetricsPredictor()
            
            # Generate synthetic data for training (in production, this would use real data)
            print("📊 Generating training data...")
            data = predictor.generate_synthetic_data(n_samples=10000)
            
            # Train models
            print("🚀 Training models...")
            results = predictor.train_all_models(data)
            
            # Save models
            predictor.save_models()
            
            # Prepare response
            models_trained = list(results['models'].keys())
            training_metrics = results['metrics']
            
            return ModelTrainingResponse(
                success=True,
                training_id=training_id,
                models_trained=models_trained,
                training_metrics=training_metrics,
                training_time=time.time(),  # Simplified
                model_version="1.0.0",
                deployment_status="deployed"
            )
            
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="ML pipeline not available. Please ensure the health_prediction_pipeline.py is in the ml_models directory."
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )

@router.post("/evaluate", response_model=ModelEvaluationResponse)
async def evaluate_health_prediction_models(request: ModelEvaluationRequest):
    """
    Evaluate health prediction models.
    
    This endpoint evaluates the performance of trained models using test data
    and provides detailed metrics and recommendations for improvement.
    """
    
    try:
        # Import the ML pipeline
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "ml_models"))
        
        try:
            from health_prediction_pipeline import HealthMetricsPredictor
            
            # Initialize predictor
            predictor = HealthMetricsPredictor()
            
            # Load existing models
            predictor.load_models()
            
            # Generate evaluation data
            eval_data = predictor.generate_synthetic_data(n_samples=2000)
            
            # Evaluate models
            evaluation_results = predictor.evaluate_models()
            
            # Generate recommendations
            recommendations = [
                "Consider collecting more diverse training data",
                "Regular model retraining with new data",
                "Monitor model drift over time",
                "Validate predictions with clinical data"
            ]
            
            return ModelEvaluationResponse(
                success=True,
                model_version=request.model_version,
                evaluation_metrics=evaluation_results,
                performance_summary={
                    "total_models": len(evaluation_results),
                    "average_rmse": sum(metrics['rmse'] for metrics in evaluation_results.values()) / len(evaluation_results),
                    "average_r2": sum(metrics['r2'] for metrics in evaluation_results.values()) / len(evaluation_results)
                },
                recommendations=recommendations
            )
            
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="ML pipeline not available for evaluation."
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )

@router.get("/trends/{patient_id}")
async def analyze_health_trends(
    patient_id: str,
    metric: HealthMetricType = Query(..., description="Health metric to analyze"),
    days: int = Query(30, ge=7, le=365, description="Number of days to analyze"),
    db=Depends(get_database)
):
    """Analyze health trends over time for a specific metric"""
    try:
        # Validate patient ID
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get predictions for the metric
        predictions = await db.health_predictions.find({
            "patient_id": patient_id,
            "timestamp": {"$gte": start_date, "$lte": end_date}
        }).sort("timestamp", 1).to_list(1000)
        
        if not predictions:
            return {
                "patient_id": patient_id,
                "metric": metric,
                "time_period": f"Last {days} days",
                "message": "No prediction data available for trend analysis"
            }
        
        # Extract metric values
        metric_values = []
        for pred in predictions:
            for pred_data in pred.get("predictions", []):
                if pred_data.get("metric") == metric.value:
                    metric_values.append({
                        "timestamp": pred["timestamp"],
                        "value": pred_data.get("predicted_value", 0)
                    })
        
        if not metric_values:
            return {
                "patient_id": patient_id,
                "metric": metric,
                "time_period": f"Last {days} days",
                "message": f"No {metric.value} data available for trend analysis"
            }
        
        # Calculate trend statistics
        values = [mv["value"] for mv in metric_values]
        avg_value = sum(values) / len(values)
        min_value = min(values)
        max_value = max(values)
        
        # Simple trend calculation
        if len(values) > 1:
            trend_direction = "improving" if values[-1] < values[0] else "declining" if values[-1] > values[0] else "stable"
            trend_strength = abs(values[-1] - values[0]) / avg_value if avg_value > 0 else 0
        else:
            trend_direction = "stable"
            trend_strength = 0
        
        # Calculate volatility
        if len(values) > 1:
            volatility = sum(abs(values[i] - values[i-1]) for i in range(1, len(values))) / (len(values) - 1)
        else:
            volatility = 0
        
        return HealthTrendAnalysis(
            patient_id=patient_id,
            metric=metric,
            time_period=f"Last {days} days",
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            data_points=len(metric_values),
            average_value=avg_value,
            min_value=min_value,
            max_value=max_value,
            volatility=volatility,
            predictions=[]  # Could add future predictions here
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")

@router.get("/insights/{patient_id}")
async def generate_health_insights(
    patient_id: str,
    db=Depends(get_database)
):
    """Generate comprehensive health insights and recommendations"""
    try:
        # Validate patient ID
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Get recent predictions
        recent_predictions = await db.health_predictions.find({
            "patient_id": patient_id
        }).sort("timestamp", -1).limit(10).to_list(10)
        
        if not recent_predictions:
            return {
                "patient_id": patient_id,
                "message": "No prediction data available for insights"
            }
        
        # Analyze recent predictions
        all_predictions = []
        for pred in recent_predictions:
            for pred_data in pred.get("predictions", []):
                all_predictions.append(pred_data)
        
        # Calculate overall health score
        if all_predictions:
            health_scores = [pred.get("overall_health_score", 0) for pred in recent_predictions if pred.get("overall_health_score")]
            overall_health_score = sum(health_scores) / len(health_scores) if health_scores else 0
        else:
            overall_health_score = 0
        
        # Identify risk factors
        risk_factors = set()
        for pred in recent_predictions:
            risk_factors.update(pred.get("risk_factors", []))
        
        # Generate insights
        key_insights = []
        if overall_health_score < 70:
            key_insights.append("Overall health score indicates room for improvement")
        if len(risk_factors) > 3:
            key_insights.append("Multiple risk factors identified - consider lifestyle changes")
        
        # Generate recommendations
        recommendations = [
            "Schedule regular health checkups",
            "Monitor key health metrics regularly",
            "Maintain a balanced diet and exercise routine",
            "Get adequate sleep and manage stress"
        ]
        
        # Priority actions
        priority_actions = []
        if "Elevated LDL" in risk_factors:
            priority_actions.append("Focus on heart-healthy diet changes")
        if "Elevated GLUCOSE" in risk_factors:
            priority_actions.append("Monitor blood sugar and increase activity")
        
        return HealthInsights(
            patient_id=patient_id,
            timestamp=datetime.now(),
            overall_health_score=overall_health_score,
            risk_assessment={
                "cardiovascular": "medium" if "Elevated LDL" in risk_factors else "low",
                "metabolic": "medium" if "Elevated GLUCOSE" in risk_factors else "low",
                "nutritional": "medium" if "Elevated HEMOGLOBIN" in risk_factors else "low"
            },
            key_insights=key_insights,
            recommendations=recommendations,
            priority_actions=priority_actions,
            follow_up_schedule={
                "next_checkup": "3 months",
                "blood_work": "6 months",
                "lifestyle_review": "1 month"
            },
            progress_tracking={
                "health_score_trend": "stable",
                "risk_factor_count": len(risk_factors),
                "prediction_accuracy": 0.85
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insights generation failed: {str(e)}")

@router.get("/models/status")
async def get_model_status():
    """Get the status of trained models"""
    try:
        model_status = {
            "models_loaded": len(health_predictor.models),
            "available_models": list(health_predictor.models.keys()),
            "model_version": health_predictor.model_version,
            "last_updated": "2024-01-01T00:00:00Z",  # Could be stored in metadata
            "status": "ready" if health_predictor.models else "not_loaded"
        }
        
        return model_status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

@router.delete("/predictions/{prediction_id}")
async def delete_health_prediction(prediction_id: str, db=Depends(get_database)):
    """Delete a specific health prediction"""
    try:
        if not ObjectId.is_valid(prediction_id):
            raise HTTPException(status_code=400, detail="Invalid prediction ID format")
        
        # Check if prediction exists
        existing_prediction = await db.health_predictions.find_one({"_id": ObjectId(prediction_id)})
        if not existing_prediction:
            raise HTTPException(status_code=404, detail="Health prediction not found")
        
        # Delete the prediction
        await db.health_predictions.delete_one({"_id": ObjectId(prediction_id)})
        
        return {"message": "Health prediction deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete prediction: {str(e)}")
