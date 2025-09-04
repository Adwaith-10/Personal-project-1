import pandas as pd
import numpy as np
import sys
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import joblib
import json

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.health_prediction import (
    HealthPredictionRequest, HealthPredictionResponse, HealthPrediction,
    HealthMetricType, WearableFeatures, DietFeatures, DemographicFeatures, LifestyleFeatures
)

class HealthPredictionService:
    """Service for health prediction using trained ML models"""
    
    def __init__(self, model_dir: str = "ml_models/models"):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        self.model_version = "1.0.0"
        
        # Load models if available
        self.load_models()
        
        # Define normal ranges for health metrics
        self.normal_ranges = {
            "ldl": {
                "unit": "mg/dL",
                "normal": {"min": 0, "max": 100},
                "borderline": {"min": 100, "max": 129},
                "high": {"min": 130, "max": 159},
                "very_high": {"min": 160, "max": 1000}
            },
            "glucose": {
                "unit": "mg/dL",
                "normal": {"min": 70, "max": 100},
                "prediabetes": {"min": 100, "max": 125},
                "diabetes": {"min": 126, "max": 1000}
            },
            "hemoglobin": {
                "unit": "g/dL",
                "male": {"min": 13.5, "max": 17.5},
                "female": {"min": 12.0, "max": 15.5},
                "low": {"min": 0, "max": 12.0}
            }
        }
    
    def load_models(self):
        """Load trained ML models"""
        try:
            # Load models for each target variable
            target_variables = ['ldl', 'glucose', 'hemoglobin']
            
            for target in target_variables:
                model_path = os.path.join(self.model_dir, f"{target}_model.pkl")
                scaler_path = os.path.join(self.model_dir, f"{target}_scaler.pkl")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[target] = joblib.load(model_path)
                    self.scalers[target] = joblib.load(scaler_path)
                    print(f"✅ Loaded {target} model and scaler")
                else:
                    print(f"⚠️ Model files not found for {target}")
            
            # Load metadata
            metadata_path = os.path.join(self.model_dir, "model_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print("✅ Loaded model metadata")
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
    
    def prepare_input_data(self, request: HealthPredictionRequest) -> pd.DataFrame:
        """Prepare input data for prediction"""
        data = {}
        
        # Add demographic features
        if request.demographic_features:
            demo = request.demographic_features
            data.update({
                'age': demo.age,
                'gender': demo.gender,
                'bmi': demo.bmi or 25.0,  # Default BMI
                'weight': demo.weight or 70.0,
                'height': demo.height or 170.0,
                'activity_level': demo.activity_level or 'moderate',
                'smoking_status': demo.smoking_status or 'never',
                'alcohol_consumption': demo.alcohol_consumption or 'none',
                'medical_conditions': demo.medical_conditions or 'none'
            })
        
        # Add wearable features
        if request.wearable_features:
            wearable = request.wearable_features
            data.update({
                'avg_heart_rate': wearable.avg_heart_rate or 72.0,
                'avg_spo2': wearable.avg_spo2 or 98.0,
                'total_steps': wearable.total_steps or 8000,
                'total_calories_burned': wearable.total_calories_burned or 2000.0,
                'total_sleep_minutes': wearable.total_sleep_minutes or 420,
                'avg_sleep_efficiency': wearable.avg_sleep_efficiency or 85.0,
                'total_activity_minutes': wearable.total_activity_minutes or 45,
                'hrv_avg': wearable.hrv_avg or 45.0,
                'resting_heart_rate': wearable.resting_heart_rate or 65.0,
                'max_heart_rate': wearable.max_heart_rate or 140.0,
                'min_heart_rate': wearable.min_heart_rate or 55.0,
                'heart_rate_variability': wearable.heart_rate_variability or 35.0,
                'sleep_deep_minutes': wearable.sleep_deep_minutes or 120,
                'sleep_light_minutes': wearable.sleep_light_minutes or 200,
                'sleep_rem_minutes': wearable.sleep_rem_minutes or 90,
                'sleep_awake_minutes': wearable.sleep_awake_minutes or 30,
                'activity_intensity_high': wearable.activity_intensity_high or 15,
                'activity_intensity_medium': wearable.activity_intensity_medium or 25,
                'activity_intensity_low': wearable.activity_intensity_low or 35,
                'steps_goal_achievement': wearable.steps_goal_achievement or 80.0
            })
        
        # Add diet features
        if request.diet_features:
            diet = request.diet_features
            data.update({
                'total_calories': diet.total_calories or 2000.0,
                'total_protein': diet.total_protein or 80.0,
                'total_carbs': diet.total_carbs or 250.0,
                'total_fat': diet.total_fat or 70.0,
                'total_fiber': diet.total_fiber or 25.0,
                'total_sugar': diet.total_sugar or 50.0,
                'total_sodium': diet.total_sodium or 2300.0,
                'total_potassium': diet.total_potassium or 3500.0,
                'total_vitamin_c': diet.total_vitamin_c or 90.0,
                'total_calcium': diet.total_calcium or 1000.0,
                'total_iron': diet.total_iron or 18.0,
                'avg_meal_size': diet.avg_meal_size or 600.0,
                'meals_per_day': diet.meals_per_day or 3,
                'snacks_per_day': diet.snacks_per_day or 1,
                'water_intake': diet.water_intake or 2000.0,
                'alcohol_intake': diet.alcohol_intake or 50.0,
                'caffeine_intake': diet.caffeine_intake or 200.0,
                'processed_food_ratio': diet.processed_food_ratio or 30.0,
                'fruits_servings': diet.fruits_servings or 2.0,
                'vegetables_servings': diet.vegetables_servings or 3.0,
                'protein_servings': diet.protein_servings or 2.0,
                'grains_servings': diet.grains_servings or 6.0,
                'dairy_servings': diet.dairy_servings or 2.0,
                'sweets_servings': diet.sweets_servings or 1.0,
                'beverages_servings': diet.beverages_servings or 6.0
            })
        
        # Add lifestyle features
        if request.lifestyle_features:
            lifestyle = request.lifestyle_features
            data.update({
                'stress_level': lifestyle.stress_level or 3,
                'sleep_quality': lifestyle.sleep_quality or 3,
                'exercise_frequency': lifestyle.exercise_frequency or 3,
                'meditation_practice': lifestyle.meditation_practice or False,
                'social_activity': lifestyle.social_activity or 3,
                'work_hours': lifestyle.work_hours or 8.0,
                'screen_time': lifestyle.screen_time or 6.0,
                'outdoor_time': lifestyle.outdoor_time or 2.0,
                'social_support': lifestyle.social_support or 3
            })
        
        # Create feature interactions
        data['bmi_age_interaction'] = data.get('bmi', 25.0) * data.get('age', 45)
        data['calories_activity_ratio'] = data.get('total_calories', 2000.0) / (data.get('total_activity_minutes', 45) + 1)
        data['sleep_efficiency_ratio'] = data.get('avg_sleep_efficiency', 85.0) / (data.get('total_sleep_minutes', 420) + 1)
        data['protein_carb_ratio'] = data.get('total_protein', 80.0) / (data.get('total_carbs', 250.0) + 1)
        data['fiber_calorie_ratio'] = data.get('total_fiber', 25.0) / (data.get('total_calories', 2000.0) + 1)
        
        # Add polynomial features
        data['bmi_squared'] = data.get('bmi', 25.0) ** 2
        data['age_squared'] = data.get('age', 45) ** 2
        data['heart_rate_squared'] = data.get('avg_heart_rate', 72.0) ** 2
        
        return pd.DataFrame([data])
    
    def predict_health_metrics(self, request: HealthPredictionRequest) -> HealthPredictionResponse:
        """Predict health metrics using trained models"""
        start_time = time.time()
        
        try:
            # Check if models are loaded
            if not self.models:
                raise ValueError("No trained models available. Please train models first.")
            
            # Prepare input data
            input_data = self.prepare_input_data(request)
            
            # Calculate data quality score
            data_quality_score = self.calculate_data_quality_score(request)
            
            # Get missing features
            missing_features = self.get_missing_features(request)
            
            # Make predictions
            predictions = []
            for target_metric in request.target_metrics:
                metric_name = target_metric.value
                
                if metric_name in self.models and metric_name in self.scalers:
                    # Scale features
                    scaler = self.scalers[metric_name]
                    X_scaled = scaler.transform(input_data)
                    
                    # Make prediction
                    model = self.models[metric_name]
                    predicted_value = model.predict(X_scaled)[0]
                    
                    # Calculate confidence (simplified - could be enhanced with uncertainty estimation)
                    confidence = self.calculate_prediction_confidence(request, metric_name)
                    
                    # Get metric information
                    metric_info = self.get_metric_info(metric_name, predicted_value, request.demographic_features.gender)
                    
                    # Create prediction object
                    prediction = HealthPrediction(
                        metric=target_metric,
                        predicted_value=round(predicted_value, 2),
                        confidence=confidence,
                        unit=metric_info['unit'],
                        normal_range=metric_info['normal_range'],
                        status=metric_info['status'],
                        risk_level=metric_info['risk_level'],
                        recommendations=metric_info['recommendations']
                    )
                    
                    predictions.append(prediction)
            
            # Calculate overall health score
            overall_health_score = self.calculate_overall_health_score(predictions)
            
            # Identify risk factors
            risk_factors = self.identify_risk_factors(predictions, request)
            
            # Generate general recommendations
            general_recommendations = self.generate_general_recommendations(predictions, request)
            
            processing_time = time.time() - start_time
            
            return HealthPredictionResponse(
                success=True,
                patient_id=request.patient_id,
                timestamp=datetime.now(),
                predictions=predictions,
                model_version=self.model_version,
                processing_time=round(processing_time, 2),
                data_quality_score=data_quality_score,
                missing_features=missing_features,
                overall_health_score=overall_health_score,
                risk_factors=risk_factors,
                recommendations=general_recommendations
            )
            
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def calculate_data_quality_score(self, request: HealthPredictionRequest) -> float:
        """Calculate data quality score based on completeness"""
        total_features = 0
        available_features = 0
        
        # Count wearable features
        if request.wearable_features:
            wearable_dict = request.wearable_features.dict()
            for value in wearable_dict.values():
                total_features += 1
                if value is not None:
                    available_features += 1
        
        # Count diet features
        if request.diet_features:
            diet_dict = request.diet_features.dict()
            for value in diet_dict.values():
                total_features += 1
                if value is not None:
                    available_features += 1
        
        # Count lifestyle features
        if request.lifestyle_features:
            lifestyle_dict = request.lifestyle_features.dict()
            for value in lifestyle_dict.values():
                total_features += 1
                if value is not None:
                    available_features += 1
        
        # Demographic features are required, so they're always available
        demo_dict = request.demographic_features.dict()
        for value in demo_dict.values():
            total_features += 1
            if value is not None:
                available_features += 1
        
        return available_features / total_features if total_features > 0 else 0.0
    
    def get_missing_features(self, request: HealthPredictionRequest) -> List[str]:
        """Get list of missing features"""
        missing = []
        
        if not request.wearable_features:
            missing.append("wearable_features")
        if not request.diet_features:
            missing.append("diet_features")
        if not request.lifestyle_features:
            missing.append("lifestyle_features")
        
        return missing
    
    def calculate_prediction_confidence(self, request: HealthPredictionRequest, metric: str) -> float:
        """Calculate prediction confidence (simplified implementation)"""
        # Base confidence on data quality
        base_confidence = self.calculate_data_quality_score(request)
        
        # Adjust based on metric-specific factors
        if metric == "ldl":
            # LDL predictions are more reliable with diet data
            if request.diet_features:
                base_confidence += 0.1
        elif metric == "glucose":
            # Glucose predictions are more reliable with activity data
            if request.wearable_features:
                base_confidence += 0.1
        elif metric == "hemoglobin":
            # Hemoglobin predictions are more reliable with diet data
            if request.diet_features:
                base_confidence += 0.1
        
        return min(base_confidence, 0.95)  # Cap at 95%
    
    def get_metric_info(self, metric: str, value: float, gender: str) -> Dict[str, Any]:
        """Get metric information including normal ranges and status"""
        ranges = self.normal_ranges.get(metric, {})
        unit = ranges.get("unit", "")
        
        if metric == "ldl":
            normal_range = ranges["normal"]
            if value <= normal_range["max"]:
                status = "normal"
                risk_level = "low"
            elif value <= ranges["borderline"]["max"]:
                status = "borderline"
                risk_level = "medium"
            elif value <= ranges["high"]["max"]:
                status = "high"
                risk_level = "high"
            else:
                status = "very_high"
                risk_level = "high"
        
        elif metric == "glucose":
            normal_range = ranges["normal"]
            if value <= normal_range["max"]:
                status = "normal"
                risk_level = "low"
            elif value <= ranges["prediabetes"]["max"]:
                status = "prediabetes"
                risk_level = "medium"
            else:
                status = "diabetes"
                risk_level = "high"
        
        elif metric == "hemoglobin":
            if gender == "male":
                normal_range = ranges["male"]
            else:
                normal_range = ranges["female"]
            
            if value >= normal_range["min"] and value <= normal_range["max"]:
                status = "normal"
                risk_level = "low"
            else:
                status = "low"
                risk_level = "medium"
        
        else:
            normal_range = {"min": 0, "max": 100}
            status = "unknown"
            risk_level = "unknown"
        
        # Generate recommendations
        recommendations = self.generate_metric_recommendations(metric, status, risk_level)
        
        return {
            "unit": unit,
            "normal_range": normal_range,
            "status": status,
            "risk_level": risk_level,
            "recommendations": recommendations
        }
    
    def generate_metric_recommendations(self, metric: str, status: str, risk_level: str) -> List[str]:
        """Generate recommendations for a specific metric"""
        recommendations = []
        
        if metric == "ldl":
            if status in ["high", "very_high"]:
                recommendations.extend([
                    "Reduce saturated fat intake",
                    "Increase fiber consumption",
                    "Exercise regularly",
                    "Consider medication if recommended by doctor"
                ])
            elif status == "borderline":
                recommendations.extend([
                    "Monitor diet and exercise",
                    "Increase physical activity",
                    "Reduce processed foods"
                ])
        
        elif metric == "glucose":
            if status == "diabetes":
                recommendations.extend([
                    "Monitor blood sugar regularly",
                    "Follow diabetes management plan",
                    "Exercise regularly",
                    "Consult with healthcare provider"
                ])
            elif status == "prediabetes":
                recommendations.extend([
                    "Reduce carbohydrate intake",
                    "Increase physical activity",
                    "Lose weight if overweight",
                    "Monitor blood sugar"
                ])
        
        elif metric == "hemoglobin":
            if status == "low":
                recommendations.extend([
                    "Increase iron-rich foods",
                    "Consider iron supplements",
                    "Eat more protein",
                    "Consult with healthcare provider"
                ])
        
        return recommendations
    
    def calculate_overall_health_score(self, predictions: List[HealthPrediction]) -> float:
        """Calculate overall health score based on predictions"""
        if not predictions:
            return 0.0
        
        scores = []
        weights = {"ldl": 0.3, "glucose": 0.4, "hemoglobin": 0.3}
        
        for prediction in predictions:
            metric = prediction.metric.value
            weight = weights.get(metric, 0.33)
            
            # Calculate score based on status
            if prediction.status == "normal":
                score = 100
            elif prediction.status in ["borderline", "prediabetes"]:
                score = 70
            elif prediction.status in ["high", "low"]:
                score = 40
            else:
                score = 20
            
            scores.append(score * weight)
        
        return sum(scores)
    
    def identify_risk_factors(self, predictions: List[HealthPrediction], request: HealthPredictionRequest) -> List[str]:
        """Identify risk factors based on predictions and input data"""
        risk_factors = []
        
        # Check predictions
        for prediction in predictions:
            if prediction.risk_level in ["medium", "high"]:
                risk_factors.append(f"Elevated {prediction.metric.value.upper()}")
        
        # Check demographic factors
        demo = request.demographic_features
        if demo.age > 50:
            risk_factors.append("Age over 50")
        if demo.bmi and demo.bmi > 30:
            risk_factors.append("Obesity (BMI > 30)")
        if demo.smoking_status == "current":
            risk_factors.append("Current smoker")
        
        # Check lifestyle factors
        if request.lifestyle_features:
            lifestyle = request.lifestyle_features
            if lifestyle.exercise_frequency and lifestyle.exercise_frequency < 3:
                risk_factors.append("Low physical activity")
            if lifestyle.stress_level and lifestyle.stress_level > 4:
                risk_factors.append("High stress level")
        
        return risk_factors
    
    def generate_general_recommendations(self, predictions: List[HealthPrediction], request: HealthPredictionRequest) -> List[str]:
        """Generate general health recommendations"""
        recommendations = []
        
        # General recommendations based on risk factors
        risk_factors = self.identify_risk_factors(predictions, request)
        
        if "Elevated LDL" in risk_factors:
            recommendations.append("Focus on heart-healthy diet with low saturated fats")
        
        if "Elevated GLUCOSE" in risk_factors:
            recommendations.append("Monitor carbohydrate intake and increase physical activity")
        
        if "Elevated HEMOGLOBIN" in risk_factors:
            recommendations.append("Ensure adequate iron intake through diet or supplements")
        
        if "Age over 50" in risk_factors:
            recommendations.append("Schedule regular health checkups")
        
        if "Obesity (BMI > 30)" in risk_factors:
            recommendations.append("Work with healthcare provider on weight management plan")
        
        if "Current smoker" in risk_factors:
            recommendations.append("Consider smoking cessation programs")
        
        if "Low physical activity" in risk_factors:
            recommendations.append("Aim for at least 150 minutes of moderate exercise per week")
        
        if "High stress level" in risk_factors:
            recommendations.append("Practice stress management techniques like meditation")
        
        # Add general wellness recommendations
        recommendations.extend([
            "Maintain regular sleep schedule",
            "Stay hydrated throughout the day",
            "Eat a balanced diet with plenty of fruits and vegetables"
        ])
        
        return recommendations[:10]  # Limit to top 10 recommendations
