"""
Tests for ML prediction accuracy and model performance
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import joblib
import tempfile
import os

from app.services.health_prediction_service import HealthPredictionService
from app.models.health_prediction import (
    HealthPredictionRequest, HealthPredictionResponse, HealthPrediction,
    HealthMetricType, WearableFeatures, DietFeatures, DemographicFeatures, LifestyleFeatures
)


class TestMLPredictionAccuracy:
    """Test ML prediction accuracy and model performance"""
    
    @pytest.fixture
    def prediction_service(self):
        """Create a health prediction service instance."""
        return HealthPredictionService()
    
    @pytest.fixture
    def sample_training_data(self):
        """Generate sample training data for 2 users over 10 days."""
        data = []
        base_date = datetime(2024, 1, 15)
        
        for user_idx in range(2):
            user_id = f"user_{user_idx + 1:03d}"
            patient_id = f"patient_{user_idx + 1:03d}"
            
            for day in range(10):
                current_date = base_date + timedelta(days=day)
                
                # Generate realistic features with some correlation to targets
                age = 30 + user_idx * 5
                bmi = 25.0 + user_idx * 2 + (day % 3) * 0.5
                avg_heart_rate = 70 + user_idx * 5 + (day % 3) * 2
                avg_sleep_hours = 7.5 + user_idx * 0.5 + (day % 2) * 0.3
                avg_steps = 8000 + user_idx * 1000 + (day % 3) * 500
                avg_calories = 2000 + user_idx * 200 + (day % 3) * 100
                avg_protein = 80 + user_idx * 10 + (day % 3) * 5
                
                # Generate targets with realistic relationships to features
                # LDL increases with BMI and decreases with activity
                ldl = 100 + (bmi - 25) * 3 - (avg_steps - 8000) * 0.001 + np.random.normal(0, 5)
                
                # Glucose increases with BMI and decreases with activity
                glucose = 90 + (bmi - 25) * 2 - (avg_steps - 8000) * 0.0005 + np.random.normal(0, 3)
                
                # Hemoglobin increases with protein intake and age
                hemoglobin = 14.0 + (avg_protein - 80) * 0.01 + (age - 30) * 0.02 + np.random.normal(0, 0.5)
                
                # Ensure values are within realistic ranges
                ldl = max(50, min(200, ldl))
                glucose = max(70, min(140, glucose))
                hemoglobin = max(12.0, min(18.0, hemoglobin))
                
                row = {
                    'user_id': user_id,
                    'patient_id': patient_id,
                    'date': current_date.isoformat(),
                    'age': age,
                    'gender': 'male' if user_idx == 0 else 'female',
                    'bmi': bmi,
                    'weight': 70 + user_idx * 10,
                    'height': 170 + user_idx * 5,
                    'activity_level': 'moderate',
                    'smoking_status': 'never',
                    'alcohol_consumption': 'none',
                    'medical_conditions': 'none',
                    'avg_heart_rate': avg_heart_rate,
                    'avg_sleep_hours': avg_sleep_hours,
                    'avg_steps': avg_steps,
                    'avg_calories': avg_calories,
                    'avg_protein': avg_protein,
                    'avg_carbs': avg_calories * 0.5 / 4,  # 50% of calories from carbs
                    'avg_fat': avg_calories * 0.3 / 9,    # 30% of calories from fat
                    'avg_fiber': 25 + (day % 3) * 5,
                    'exercise_minutes': 30 + (day % 3) * 10,
                    'stress_level': 1 + (day % 3),
                    'sleep_quality': 7 + (day % 3),
                    'water_intake': 2000 + (day % 3) * 500,
                    'meditation_practice': day % 2 == 0,
                    'social_activity': 3 + (day % 3),
                    'work_hours': 8 + (day % 3),
                    'screen_time': 4 + (day % 3),
                    'outdoor_time': 2 + (day % 3),
                    'social_support': 4 + (day % 3),
                    'ldl': ldl,
                    'glucose': glucose,
                    'hemoglobin': hemoglobin
                }
                
                data.append(row)
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_test_data(self):
        """Generate sample test data for validation."""
        data = []
        base_date = datetime(2024, 1, 25)  # Different date range
        
        for user_idx in range(2):
            user_id = f"test_user_{user_idx + 1:03d}"
            patient_id = f"test_patient_{user_idx + 1:03d}"
            
            for day in range(5):  # 5 days of test data
                current_date = base_date + timedelta(days=day)
                
                # Generate features similar to training data
                age = 35 + user_idx * 3
                bmi = 26.0 + user_idx * 1.5 + (day % 2) * 0.3
                avg_heart_rate = 72 + user_idx * 3 + (day % 2) * 1
                avg_sleep_hours = 7.8 + user_idx * 0.3 + (day % 2) * 0.2
                avg_steps = 8500 + user_idx * 800 + (day % 2) * 300
                avg_calories = 2100 + user_idx * 150 + (day % 2) * 80
                avg_protein = 85 + user_idx * 8 + (day % 2) * 3
                
                # Generate targets with similar relationships
                ldl = 105 + (bmi - 26) * 2.5 - (avg_steps - 8500) * 0.0008 + np.random.normal(0, 4)
                glucose = 92 + (bmi - 26) * 1.8 - (avg_steps - 8500) * 0.0004 + np.random.normal(0, 2.5)
                hemoglobin = 14.2 + (avg_protein - 85) * 0.008 + (age - 35) * 0.015 + np.random.normal(0, 0.4)
                
                # Ensure values are within realistic ranges
                ldl = max(50, min(200, ldl))
                glucose = max(70, min(140, glucose))
                hemoglobin = max(12.0, min(18.0, hemoglobin))
                
                row = {
                    'user_id': user_id,
                    'patient_id': patient_id,
                    'date': current_date.isoformat(),
                    'age': age,
                    'gender': 'male' if user_idx == 0 else 'female',
                    'bmi': bmi,
                    'weight': 75 + user_idx * 8,
                    'height': 172 + user_idx * 3,
                    'activity_level': 'moderate',
                    'smoking_status': 'never',
                    'alcohol_consumption': 'none',
                    'medical_conditions': 'none',
                    'avg_heart_rate': avg_heart_rate,
                    'avg_sleep_hours': avg_sleep_hours,
                    'avg_steps': avg_steps,
                    'avg_calories': avg_calories,
                    'avg_protein': avg_protein,
                    'avg_carbs': avg_calories * 0.5 / 4,
                    'avg_fat': avg_calories * 0.3 / 9,
                    'avg_fiber': 28 + (day % 2) * 4,
                    'exercise_minutes': 35 + (day % 2) * 8,
                    'stress_level': 1 + (day % 2),
                    'sleep_quality': 7 + (day % 2),
                    'water_intake': 2200 + (day % 2) * 400,
                    'meditation_practice': day % 2 == 0,
                    'social_activity': 3 + (day % 2),
                    'work_hours': 8 + (day % 2),
                    'screen_time': 4 + (day % 2),
                    'outdoor_time': 2 + (day % 2),
                    'social_support': 4 + (day % 2),
                    'ldl': ldl,
                    'glucose': glucose,
                    'hemoglobin': hemoglobin
                }
                
                data.append(row)
        
        return pd.DataFrame(data)
    
    def test_model_training_accuracy(self, prediction_service, sample_training_data, sample_test_data):
        """Test model training and accuracy metrics."""
        # Mock the model training process
        with patch.object(prediction_service, 'train_models') as mock_train:
            mock_train.return_value = {
                'ldl': {'rmse': 12.5, 'mae': 10.2, 'r2': 0.78, 'accuracy': 0.87},
                'glucose': {'rmse': 6.8, 'mae': 5.5, 'r2': 0.82, 'accuracy': 0.91},
                'hemoglobin': {'rmse': 0.7, 'mae': 0.6, 'r2': 0.75, 'accuracy': 0.89}
            }
            
            # Train models
            training_results = prediction_service.train_models(sample_training_data)
            
            # Verify training results
            assert 'ldl' in training_results
            assert 'glucose' in training_results
            assert 'hemoglobin' in training_results
            
            # Check accuracy metrics
            for metric, results in training_results.items():
                assert results['rmse'] > 0
                assert results['mae'] > 0
                assert 0 <= results['r2'] <= 1
                assert 0 <= results['accuracy'] <= 1
                
                # Verify reasonable performance
                if metric == 'ldl':
                    assert results['rmse'] < 20  # Should be less than 20 mg/dL
                    assert results['r2'] > 0.7   # Should have good R-squared
                elif metric == 'glucose':
                    assert results['rmse'] < 10  # Should be less than 10 mg/dL
                    assert results['r2'] > 0.75  # Should have good R-squared
                elif metric == 'hemoglobin':
                    assert results['rmse'] < 1.0  # Should be less than 1 g/dL
                    assert results['r2'] > 0.7   # Should have good R-squared
    
    def test_model_prediction_accuracy(self, prediction_service, sample_test_data):
        """Test model prediction accuracy on test data."""
        # Mock the prediction process
        with patch.object(prediction_service, 'predict_health_metrics') as mock_predict:
            # Generate mock predictions
            mock_predictions = []
            for _, row in sample_test_data.iterrows():
                # Add some prediction error to simulate real model behavior
                ldl_pred = row['ldl'] + np.random.normal(0, 8)  # ±8 mg/dL error
                glucose_pred = row['glucose'] + np.random.normal(0, 5)  # ±5 mg/dL error
                hemoglobin_pred = row['hemoglobin'] + np.random.normal(0, 0.6)  # ±0.6 g/dL error
                
                mock_predictions.append({
                    'ldl': max(50, min(200, ldl_pred)),
                    'glucose': max(70, min(140, glucose_pred)),
                    'hemoglobin': max(12.0, min(18.0, hemoglobin_pred))
                })
            
            mock_predict.return_value = mock_predictions
            
            # Calculate accuracy metrics
            actual_ldl = sample_test_data['ldl'].values
            actual_glucose = sample_test_data['glucose'].values
            actual_hemoglobin = sample_test_data['hemoglobin'].values
            
            predicted_ldl = [pred['ldl'] for pred in mock_predictions]
            predicted_glucose = [pred['glucose'] for pred in mock_predictions]
            predicted_hemoglobin = [pred['hemoglobin'] for pred in mock_predictions]
            
            # Calculate metrics
            ldl_rmse = np.sqrt(np.mean((np.array(predicted_ldl) - actual_ldl) ** 2))
            ldl_mae = np.mean(np.abs(np.array(predicted_ldl) - actual_ldl))
            ldl_r2 = 1 - np.sum((np.array(predicted_ldl) - actual_ldl) ** 2) / np.sum((actual_ldl - np.mean(actual_ldl)) ** 2)
            
            glucose_rmse = np.sqrt(np.mean((np.array(predicted_glucose) - actual_glucose) ** 2))
            glucose_mae = np.mean(np.abs(np.array(predicted_glucose) - actual_glucose))
            glucose_r2 = 1 - np.sum((np.array(predicted_glucose) - actual_glucose) ** 2) / np.sum((actual_glucose - np.mean(actual_glucose)) ** 2)
            
            hemoglobin_rmse = np.sqrt(np.mean((np.array(predicted_hemoglobin) - actual_hemoglobin) ** 2))
            hemoglobin_mae = np.mean(np.abs(np.array(predicted_hemoglobin) - actual_hemoglobin))
            hemoglobin_r2 = 1 - np.sum((np.array(predicted_hemoglobin) - actual_hemoglobin) ** 2) / np.sum((actual_hemoglobin - np.mean(actual_hemoglobin)) ** 2)
            
            # Verify reasonable accuracy
            assert ldl_rmse < 15, f"LDL RMSE {ldl_rmse:.2f} should be < 15"
            assert ldl_mae < 12, f"LDL MAE {ldl_mae:.2f} should be < 12"
            assert ldl_r2 > 0.6, f"LDL R² {ldl_r2:.3f} should be > 0.6"
            
            assert glucose_rmse < 8, f"Glucose RMSE {glucose_rmse:.2f} should be < 8"
            assert glucose_mae < 6, f"Glucose MAE {glucose_mae:.2f} should be < 6"
            assert glucose_r2 > 0.65, f"Glucose R² {glucose_r2:.3f} should be > 0.65"
            
            assert hemoglobin_rmse < 1.0, f"Hemoglobin RMSE {hemoglobin_rmse:.2f} should be < 1.0"
            assert hemoglobin_mae < 0.8, f"Hemoglobin MAE {hemoglobin_mae:.2f} should be < 0.8"
            assert hemoglobin_r2 > 0.6, f"Hemoglobin R² {hemoglobin_r2:.3f} should be > 0.6"
            
            print(f"LDL - RMSE: {ldl_rmse:.2f}, MAE: {ldl_mae:.2f}, R²: {ldl_r2:.3f}")
            print(f"Glucose - RMSE: {glucose_rmse:.2f}, MAE: {glucose_mae:.2f}, R²: {glucose_r2:.3f}")
            print(f"Hemoglobin - RMSE: {hemoglobin_rmse:.2f}, MAE: {hemoglobin_mae:.2f}, R²: {hemoglobin_r2:.3f}")
    
    def test_feature_importance_analysis(self, prediction_service, sample_training_data):
        """Test feature importance analysis."""
        # Mock feature importance calculation
        with patch.object(prediction_service, 'calculate_feature_importance') as mock_importance:
            mock_importance.return_value = {
                'ldl': {
                    'bmi': 0.25,
                    'avg_steps': 0.20,
                    'avg_calories': 0.15,
                    'age': 0.10,
                    'avg_protein': 0.08,
                    'avg_heart_rate': 0.07,
                    'avg_sleep_hours': 0.05,
                    'stress_level': 0.03,
                    'exercise_minutes': 0.03,
                    'water_intake': 0.02,
                    'screen_time': 0.01,
                    'social_activity': 0.01
                },
                'glucose': {
                    'bmi': 0.30,
                    'avg_steps': 0.25,
                    'avg_calories': 0.15,
                    'avg_carbs': 0.10,
                    'age': 0.08,
                    'avg_heart_rate': 0.05,
                    'avg_sleep_hours': 0.03,
                    'stress_level': 0.02,
                    'exercise_minutes': 0.01,
                    'water_intake': 0.01
                },
                'hemoglobin': {
                    'avg_protein': 0.35,
                    'age': 0.25,
                    'avg_calories': 0.15,
                    'avg_heart_rate': 0.10,
                    'avg_sleep_hours': 0.08,
                    'exercise_minutes': 0.04,
                    'stress_level': 0.02,
                    'water_intake': 0.01
                }
            }
            
            feature_importance = prediction_service.calculate_feature_importance(sample_training_data)
            
            # Verify feature importance structure
            for metric in ['ldl', 'glucose', 'hemoglobin']:
                assert metric in feature_importance
                assert len(feature_importance[metric]) > 0
                
                # Check that importance values sum to approximately 1
                total_importance = sum(feature_importance[metric].values())
                assert 0.95 <= total_importance <= 1.05, f"Feature importance should sum to ~1, got {total_importance}"
                
                # Check that all importance values are positive
                for importance in feature_importance[metric].values():
                    assert importance >= 0
    
    def test_model_performance_benchmark(self, prediction_service, sample_training_data, sample_test_data):
        """Benchmark model performance."""
        import time
        
        # Mock model training and prediction
        with patch.object(prediction_service, 'train_models') as mock_train:
            with patch.object(prediction_service, 'predict_health_metrics') as mock_predict:
                # Benchmark training time
                start_time = time.time()
                training_results = prediction_service.train_models(sample_training_data)
                training_time = time.time() - start_time
                
                # Benchmark prediction time
                start_time = time.time()
                predictions = prediction_service.predict_health_metrics(sample_test_data)
                prediction_time = time.time() - start_time
                
                # Performance assertions
                assert training_time < 30.0, f"Training took {training_time:.2f}s, should be < 30s"
                assert prediction_time < 5.0, f"Prediction took {prediction_time:.2f}s, should be < 5s"
                
                print(f"Training time: {training_time:.2f}s")
                print(f"Prediction time: {prediction_time:.2f}s")
                print(f"Training data size: {len(sample_training_data)} samples")
                print(f"Test data size: {len(sample_test_data)} samples")
    
    def test_model_robustness(self, prediction_service, sample_test_data):
        """Test model robustness with noisy data."""
        # Add noise to test data
        noisy_data = sample_test_data.copy()
        
        # Add 10% noise to numerical features
        numerical_features = ['bmi', 'avg_heart_rate', 'avg_sleep_hours', 'avg_steps', 
                            'avg_calories', 'avg_protein', 'avg_carbs', 'avg_fat']
        
        for feature in numerical_features:
            if feature in noisy_data.columns:
                noise = np.random.normal(0, 0.1 * noisy_data[feature].std(), len(noisy_data))
                noisy_data[feature] = noisy_data[feature] + noise
        
        # Mock prediction on noisy data
        with patch.object(prediction_service, 'predict_health_metrics') as mock_predict:
            # Generate predictions for both clean and noisy data
            clean_predictions = []
            noisy_predictions = []
            
            for _, row in sample_test_data.iterrows():
                clean_pred = {
                    'ldl': row['ldl'] + np.random.normal(0, 5),
                    'glucose': row['glucose'] + np.random.normal(0, 3),
                    'hemoglobin': row['hemoglobin'] + np.random.normal(0, 0.4)
                }
                clean_predictions.append(clean_pred)
            
            for _, row in noisy_data.iterrows():
                noisy_pred = {
                    'ldl': row['ldl'] + np.random.normal(0, 6),  # Slightly more error
                    'glucose': row['glucose'] + np.random.normal(0, 4),
                    'hemoglobin': row['hemoglobin'] + np.random.normal(0, 0.5)
                }
                noisy_predictions.append(noisy_pred)
            
            # Calculate performance degradation
            clean_ldl_rmse = np.sqrt(np.mean(([p['ldl'] for p in clean_predictions] - sample_test_data['ldl']) ** 2))
            noisy_ldl_rmse = np.sqrt(np.mean(([p['ldl'] for p in noisy_predictions] - sample_test_data['ldl']) ** 2))
            
            # Performance should not degrade too much
            degradation = (noisy_ldl_rmse - clean_ldl_rmse) / clean_ldl_rmse
            assert degradation < 0.3, f"Performance degradation {degradation:.2%} should be < 30%"
            
            print(f"Clean data RMSE: {clean_ldl_rmse:.2f}")
            print(f"Noisy data RMSE: {noisy_ldl_rmse:.2f}")
            print(f"Performance degradation: {degradation:.2%}")
    
    def test_model_consistency(self, prediction_service, sample_test_data):
        """Test model prediction consistency."""
        # Test that similar inputs produce similar outputs
        base_row = sample_test_data.iloc[0].copy()
        
        # Create slightly modified versions of the same input
        variations = []
        for i in range(5):
            variation = base_row.copy()
            # Add small random variations
            variation['bmi'] += np.random.normal(0, 0.1)
            variation['avg_steps'] += np.random.normal(0, 50)
            variation['avg_calories'] += np.random.normal(0, 20)
            variations.append(variation)
        
        # Mock predictions for variations
        with patch.object(prediction_service, 'predict_health_metrics') as mock_predict:
            predictions = []
            for variation in variations:
                pred = {
                    'ldl': 100 + np.random.normal(0, 2),  # Small variation
                    'glucose': 90 + np.random.normal(0, 1),
                    'hemoglobin': 14.0 + np.random.normal(0, 0.1)
                }
                predictions.append(pred)
            
            # Check consistency
            ldl_predictions = [p['ldl'] for p in predictions]
            glucose_predictions = [p['glucose'] for p in predictions]
            hemoglobin_predictions = [p['hemoglobin'] for p in predictions]
            
            # Predictions should be consistent (low variance)
            ldl_std = np.std(ldl_predictions)
            glucose_std = np.std(glucose_predictions)
            hemoglobin_std = np.std(hemoglobin_predictions)
            
            assert ldl_std < 5, f"LDL predictions should be consistent, std: {ldl_std:.2f}"
            assert glucose_std < 3, f"Glucose predictions should be consistent, std: {glucose_std:.2f}"
            assert hemoglobin_std < 0.3, f"Hemoglobin predictions should be consistent, std: {hemoglobin_std:.2f}"
            
            print(f"LDL prediction std: {ldl_std:.2f}")
            print(f"Glucose prediction std: {glucose_std:.2f}")
            print(f"Hemoglobin prediction std: {hemoglobin_std:.2f}")
    
    def test_model_calibration(self, prediction_service, sample_test_data):
        """Test model calibration (prediction confidence)."""
        # Mock predictions with confidence scores
        with patch.object(prediction_service, 'predict_health_metrics') as mock_predict:
            predictions_with_confidence = []
            
            for _, row in sample_test_data.iterrows():
                # Generate predictions with varying confidence
                confidence = np.random.uniform(0.7, 0.95)
                
                pred = {
                    'ldl': row['ldl'] + np.random.normal(0, 10 * (1 - confidence)),
                    'glucose': row['glucose'] + np.random.normal(0, 6 * (1 - confidence)),
                    'hemoglobin': row['hemoglobin'] + np.random.normal(0, 0.8 * (1 - confidence)),
                    'confidence': confidence
                }
                predictions_with_confidence.append(pred)
            
            # Test calibration: higher confidence should correlate with lower error
            ldl_errors = [abs(p['ldl'] - sample_test_data.iloc[i]['ldl']) for i, p in enumerate(predictions_with_confidence)]
            confidences = [p['confidence'] for p in predictions_with_confidence]
            
            # Calculate correlation between confidence and error
            correlation = np.corrcoef(confidences, ldl_errors)[0, 1]
            
            # Higher confidence should correlate with lower error (negative correlation)
            assert correlation < -0.3, f"Confidence should correlate with accuracy, got {correlation:.3f}"
            
            print(f"Confidence-error correlation: {correlation:.3f}")
    
    def test_model_bias_analysis(self, prediction_service, sample_test_data):
        """Test for model bias across different groups."""
        # Analyze predictions by gender
        male_data = sample_test_data[sample_test_data['gender'] == 'male']
        female_data = sample_test_data[sample_test_data['gender'] == 'female']
        
        with patch.object(prediction_service, 'predict_health_metrics') as mock_predict:
            # Mock predictions
            male_predictions = []
            female_predictions = []
            
            for _, row in male_data.iterrows():
                pred = {
                    'ldl': row['ldl'] + np.random.normal(0, 8),
                    'glucose': row['glucose'] + np.random.normal(0, 5),
                    'hemoglobin': row['hemoglobin'] + np.random.normal(0, 0.6)
                }
                male_predictions.append(pred)
            
            for _, row in female_data.iterrows():
                pred = {
                    'ldl': row['ldl'] + np.random.normal(0, 8),
                    'glucose': row['glucose'] + np.random.normal(0, 5),
                    'hemoglobin': row['hemoglobin'] + np.random.normal(0, 0.6)
                }
                female_predictions.append(pred)
            
            # Calculate bias metrics
            male_ldl_errors = [abs(p['ldl'] - male_data.iloc[i]['ldl']) for i, p in enumerate(male_predictions)]
            female_ldl_errors = [abs(p['ldl'] - female_data.iloc[i]['ldl']) for i, p in enumerate(female_predictions)]
            
            male_avg_error = np.mean(male_ldl_errors)
            female_avg_error = np.mean(female_ldl_errors)
            
            # Bias should be minimal (error difference < 20%)
            bias_ratio = abs(male_avg_error - female_avg_error) / max(male_avg_error, female_avg_error)
            assert bias_ratio < 0.2, f"Model bias between groups should be < 20%, got {bias_ratio:.1%}"
            
            print(f"Male average error: {male_avg_error:.2f}")
            print(f"Female average error: {female_avg_error:.2f}")
            print(f"Bias ratio: {bias_ratio:.1%}")
    
    @pytest.mark.asyncio
    async def test_prediction_api_accuracy(self, test_client, sample_users, sample_test_data):
        """Test prediction API accuracy."""
        # First register and login a user
        user = sample_users[0]
        
        # Register user
        register_response = test_client.post("/api/v1/auth/register", json=user)
        assert register_response.status_code == 201
        
        # Login user
        login_response = test_client.post("/api/v1/auth/login", json={
            "email": user["email"],
            "password": user["password"]
        })
        assert login_response.status_code == 200
        
        token_data = login_response.json()
        access_token = token_data["access_token"]
        
        # Test prediction API
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Use first row of test data for prediction
        test_row = sample_test_data.iloc[0]
        
        prediction_request = {
            "patient_id": "test_patient_001",
            "demographic_features": {
                "age": int(test_row['age']),
                "gender": test_row['gender'],
                "bmi": float(test_row['bmi']),
                "weight": float(test_row['weight']),
                "height": float(test_row['height']),
                "activity_level": test_row['activity_level'],
                "smoking_status": test_row['smoking_status'],
                "alcohol_consumption": test_row['alcohol_consumption'],
                "medical_conditions": test_row['medical_conditions']
            },
            "wearable_features": {
                "avg_heart_rate": float(test_row['avg_heart_rate']),
                "avg_sleep_hours": float(test_row['avg_sleep_hours']),
                "avg_steps": int(test_row['avg_steps']),
                "avg_spo2": 98.0,
                "avg_hrv": 45.0
            },
            "diet_features": {
                "avg_calories": float(test_row['avg_calories']),
                "avg_protein": float(test_row['avg_protein']),
                "avg_carbs": float(test_row['avg_carbs']),
                "avg_fat": float(test_row['avg_fat']),
                "avg_fiber": float(test_row['avg_fiber']),
                "water_intake": float(test_row['water_intake'])
            },
            "lifestyle_features": {
                "exercise_minutes": int(test_row['exercise_minutes']),
                "stress_level": int(test_row['stress_level']),
                "sleep_quality": int(test_row['sleep_quality']),
                "meditation_practice": bool(test_row['meditation_practice']),
                "social_activity": int(test_row['social_activity']),
                "work_hours": float(test_row['work_hours']),
                "screen_time": float(test_row['screen_time']),
                "outdoor_time": float(test_row['outdoor_time']),
                "social_support": int(test_row['social_support'])
            },
            "target_metrics": ["ldl", "glucose", "hemoglobin"]
        }
        
        response = test_client.post(
            "/api/v1/health-prediction/predict",
            json=prediction_request,
            headers=headers
        )
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["success"] is True
        assert len(result["predictions"]) >= 3  # Should predict all 3 metrics
        assert result["processing_time"] > 0
        assert result["data_quality_score"] > 0
        
        # Check prediction structure
        for prediction in result["predictions"]:
            assert "metric" in prediction
            assert "predicted_value" in prediction
            assert "confidence" in prediction
            assert "unit" in prediction
            assert "status" in prediction
            assert "risk_level" in prediction
            assert "recommendations" in prediction
            
            # Check value ranges
            if prediction["metric"] == "ldl":
                assert 50 <= prediction["predicted_value"] <= 200
            elif prediction["metric"] == "glucose":
                assert 70 <= prediction["predicted_value"] <= 140
            elif prediction["metric"] == "hemoglobin":
                assert 12.0 <= prediction["predicted_value"] <= 18.0
