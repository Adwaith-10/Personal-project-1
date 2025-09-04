#!/usr/bin/env python3
"""
Test script for health prediction ML pipeline functionality
"""

import requests
import json
from datetime import datetime
import time

def test_health_prediction():
    """Test the health prediction functionality"""
    
    # API configuration
    API_BASE_URL = "http://localhost:8000"
    
    print("🏥 Testing Health Prediction ML Pipeline")
    print("=" * 50)
    
    # First, get a patient ID to use for testing
    print("📋 Getting available patients...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/patients")
        if response.status_code == 200:
            patients = response.json()
            if isinstance(patients, dict) and "patients" in patients:
                patients_list = patients["patients"]
            else:
                patients_list = patients
            
            if patients_list:
                patient_id = patients_list[0]["_id"]
                print(f"✅ Using patient ID: {patient_id}")
            else:
                print("❌ No patients found. Please create a patient first.")
                return
        else:
            print(f"❌ Failed to get patients: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        print("💡 Make sure the FastAPI server is running on http://localhost:8000")
        return
    
    # Test model training
    print("\n🚀 Testing model training...")
    try:
        training_request = {
            "data_source": "synthetic",
            "target_metrics": ["ldl", "glucose", "hemoglobin"],
            "training_parameters": {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.1
            },
            "validation_split": 0.2
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/health-prediction/train",
            json=training_request
        )
        
        if response.status_code == 200:
            training_result = response.json()
            print("✅ Model training completed successfully!")
            print(f"📊 Training ID: {training_result.get('training_id')}")
            print(f"🎯 Models trained: {training_result.get('models_trained')}")
            print(f"⏱️ Training time: {training_result.get('training_time')} seconds")
            
            # Display training metrics
            metrics = training_result.get('training_metrics', {})
            for metric, values in metrics.items():
                print(f"   {metric.upper()}: RMSE={values.get('rmse', 'N/A'):.2f}, R²={values.get('r2', 'N/A'):.3f}")
        else:
            print(f"❌ Model training failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during model training: {e}")
    
    # Test health prediction
    print("\n🔮 Testing health prediction...")
    try:
        # Create sample health data
        prediction_request = {
            "patient_id": patient_id,
            "wearable_features": {
                "avg_heart_rate": 75.0,
                "avg_spo2": 98.5,
                "total_steps": 8500,
                "total_calories_burned": 2100.0,
                "total_sleep_minutes": 420,
                "avg_sleep_efficiency": 87.0,
                "total_activity_minutes": 45,
                "hrv_avg": 48.0,
                "resting_heart_rate": 68.0,
                "max_heart_rate": 145.0,
                "min_heart_rate": 58.0,
                "heart_rate_variability": 38.0,
                "sleep_deep_minutes": 125,
                "sleep_light_minutes": 210,
                "sleep_rem_minutes": 85,
                "sleep_awake_minutes": 25,
                "activity_intensity_high": 18,
                "activity_intensity_medium": 22,
                "activity_intensity_low": 30,
                "steps_goal_achievement": 85.0
            },
            "diet_features": {
                "total_calories": 1950.0,
                "total_protein": 85.0,
                "total_carbs": 240.0,
                "total_fat": 65.0,
                "total_fiber": 28.0,
                "total_sugar": 45.0,
                "total_sodium": 2200.0,
                "total_potassium": 3600.0,
                "total_vitamin_c": 95.0,
                "total_calcium": 1050.0,
                "total_iron": 18.5,
                "avg_meal_size": 580.0,
                "meals_per_day": 3,
                "snacks_per_day": 2,
                "water_intake": 2200.0,
                "alcohol_intake": 30.0,
                "caffeine_intake": 180.0,
                "processed_food_ratio": 25.0,
                "fruits_servings": 2.5,
                "vegetables_servings": 3.5,
                "protein_servings": 2.2,
                "grains_servings": 6.5,
                "dairy_servings": 2.1,
                "sweets_servings": 1.2,
                "beverages_servings": 6.2
            },
            "demographic_features": {
                "age": 42,
                "gender": "male",
                "bmi": 26.5,
                "weight": 78.0,
                "height": 172.0,
                "activity_level": "moderate",
                "smoking_status": "never",
                "alcohol_consumption": "light",
                "medical_conditions": "none"
            },
            "lifestyle_features": {
                "stress_level": 3,
                "sleep_quality": 4,
                "exercise_frequency": 4,
                "meditation_practice": True,
                "social_activity": 4,
                "work_hours": 8.5,
                "screen_time": 5.5,
                "outdoor_time": 2.5,
                "social_support": 4
            },
            "target_metrics": ["ldl", "glucose", "hemoglobin"],
            "confidence_threshold": 0.7
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/health-prediction/predict",
            json=prediction_request
        )
        
        if response.status_code == 200:
            prediction_result = response.json()
            print("✅ Health prediction completed successfully!")
            print(f"📊 Overall Health Score: {prediction_result.get('overall_health_score', 'N/A')}")
            print(f"🎯 Data Quality Score: {prediction_result.get('data_quality_score', 'N/A'):.2f}")
            print(f"⏱️ Processing Time: {prediction_result.get('processing_time', 'N/A')} seconds")
            
            # Display predictions
            predictions = prediction_result.get('predictions', [])
            print(f"\n📈 Health Predictions:")
            for pred in predictions:
                metric = pred.get('metric', 'Unknown')
                value = pred.get('predicted_value', 'N/A')
                confidence = pred.get('confidence', 'N/A')
                status = pred.get('status', 'Unknown')
                risk_level = pred.get('risk_level', 'Unknown')
                unit = pred.get('unit', '')
                
                print(f"   {metric.upper()}: {value} {unit} (Confidence: {confidence:.1%}, Status: {status}, Risk: {risk_level})")
            
            # Display risk factors
            risk_factors = prediction_result.get('risk_factors', [])
            if risk_factors:
                print(f"\n⚠️ Risk Factors:")
                for risk in risk_factors:
                    print(f"   - {risk}")
            
            # Display recommendations
            recommendations = prediction_result.get('recommendations', [])
            if recommendations:
                print(f"\n💡 Recommendations:")
                for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                    print(f"   {i}. {rec}")
                    
        else:
            print(f"❌ Health prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during health prediction: {e}")
    
    # Test model evaluation
    print("\n📊 Testing model evaluation...")
    try:
        evaluation_request = {
            "model_version": "1.0.0",
            "evaluation_dataset": "test_data",
            "metrics": ["rmse", "mae", "r2"]
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/health-prediction/evaluate",
            json=evaluation_request
        )
        
        if response.status_code == 200:
            evaluation_result = response.json()
            print("✅ Model evaluation completed successfully!")
            
            # Display evaluation metrics
            metrics = evaluation_result.get('evaluation_metrics', {})
            for metric, values in metrics.items():
                print(f"   {metric.upper()}: RMSE={values.get('rmse', 'N/A'):.2f}, MAE={values.get('mae', 'N/A'):.2f}, R²={values.get('r2', 'N/A'):.3f}")
            
            # Display performance summary
            summary = evaluation_result.get('performance_summary', {})
            print(f"📈 Performance Summary:")
            print(f"   Total Models: {summary.get('total_models', 'N/A')}")
            print(f"   Average RMSE: {summary.get('average_rmse', 'N/A'):.2f}")
            print(f"   Average R²: {summary.get('average_r2', 'N/A'):.3f}")
            
        else:
            print(f"❌ Model evaluation failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during model evaluation: {e}")
    
    # Test health trends analysis
    print("\n📈 Testing health trends analysis...")
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/health-prediction/trends/{patient_id}",
            params={
                "metric": "ldl",
                "days": 30
            }
        )
        
        if response.status_code == 200:
            trends_result = response.json()
            print("✅ Health trends analysis completed successfully!")
            print(f"📊 Metric: {trends_result.get('metric', 'N/A')}")
            print(f"📅 Time Period: {trends_result.get('time_period', 'N/A')}")
            print(f"📈 Trend Direction: {trends_result.get('trend_direction', 'N/A')}")
            print(f"💪 Trend Strength: {trends_result.get('trend_strength', 'N/A'):.2f}")
            print(f"📊 Data Points: {trends_result.get('data_points', 'N/A')}")
            print(f"📊 Average Value: {trends_result.get('average_value', 'N/A'):.2f}")
            print(f"📊 Volatility: {trends_result.get('volatility', 'N/A'):.2f}")
        else:
            print(f"❌ Health trends analysis failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during trends analysis: {e}")
    
    # Test health insights generation
    print("\n🧠 Testing health insights generation...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health-prediction/insights/{patient_id}")
        
        if response.status_code == 200:
            insights_result = response.json()
            print("✅ Health insights generated successfully!")
            print(f"📊 Overall Health Score: {insights_result.get('overall_health_score', 'N/A')}")
            
            # Display risk assessment
            risk_assessment = insights_result.get('risk_assessment', {})
            if risk_assessment:
                print(f"⚠️ Risk Assessment:")
                for category, risk in risk_assessment.items():
                    print(f"   {category.title()}: {risk}")
            
            # Display key insights
            key_insights = insights_result.get('key_insights', [])
            if key_insights:
                print(f"💡 Key Insights:")
                for insight in key_insights:
                    print(f"   - {insight}")
            
            # Display priority actions
            priority_actions = insights_result.get('priority_actions', [])
            if priority_actions:
                print(f"🎯 Priority Actions:")
                for action in priority_actions:
                    print(f"   - {action}")
            
            # Display follow-up schedule
            follow_up = insights_result.get('follow_up_schedule', {})
            if follow_up:
                print(f"📅 Follow-up Schedule:")
                for item, schedule in follow_up.items():
                    print(f"   {item.replace('_', ' ').title()}: {schedule}")
                    
        else:
            print(f"❌ Health insights generation failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during insights generation: {e}")
    
    # Test getting historical predictions
    print("\n📋 Testing historical predictions retrieval...")
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/health-prediction/predictions",
            params={
                "patient_id": patient_id,
                "limit": 5
            }
        )
        
        if response.status_code == 200:
            predictions = response.json()
            print(f"✅ Retrieved {len(predictions)} historical predictions")
            
            for i, pred in enumerate(predictions[:3], 1):  # Show first 3
                timestamp = pred.get('timestamp', 'N/A')
                health_score = pred.get('overall_health_score', 'N/A')
                print(f"   {i}. {timestamp}: Health Score = {health_score}")
        else:
            print(f"❌ Failed to retrieve historical predictions: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error retrieving historical predictions: {e}")
    
    # Test model status
    print("\n🔧 Testing model status...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health-prediction/models/status")
        
        if response.status_code == 200:
            status_result = response.json()
            print("✅ Model status retrieved successfully!")
            print(f"📊 Models Loaded: {status_result.get('models_loaded', 'N/A')}")
            print(f"🎯 Available Models: {status_result.get('available_models', [])}")
            print(f"📋 Model Version: {status_result.get('model_version', 'N/A')}")
            print(f"📊 Status: {status_result.get('status', 'N/A')}")
        else:
            print(f"❌ Failed to get model status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting model status: {e}")
    
    print("\n🎉 Health Prediction ML Pipeline testing completed!")

if __name__ == "__main__":
    test_health_prediction()
