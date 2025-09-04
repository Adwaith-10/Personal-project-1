"""
Tests for wearable data logging functionality
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from app.services.wearable_data_processor import WearableDataProcessor
from app.models.wearable_data import DailyLog, WearableDataRequest


class TestWearableLogging:
    """Test wearable data logging functionality"""
    
    @pytest.fixture
    def processor(self):
        """Create a wearable data processor instance."""
        return WearableDataProcessor()
    
    @pytest.fixture
    def sample_wearable_data(self):
        """Sample wearable data for testing."""
        return {
            "patient_id": "patient_001",
            "date": "2024-01-15T00:00:00",
            "device_type": "apple_watch",
            "device_id": "apple_watch_123",
            "heart_rate_data": [
                {
                    "heart_rate": 75,
                    "hrv_ms": 45,
                    "zone": "rest",
                    "confidence": 0.95,
                    "source": "apple_watch",
                    "timestamp": "2024-01-15T08:00:00"
                },
                {
                    "heart_rate": 120,
                    "hrv_ms": 35,
                    "zone": "active",
                    "confidence": 0.92,
                    "source": "apple_watch",
                    "timestamp": "2024-01-15T09:00:00"
                }
            ],
            "sleep_data": [
                {
                    "stage": "deep_sleep",
                    "duration_minutes": 120,
                    "start_time": "2024-01-15T02:00:00",
                    "efficiency_percentage": 85.0,
                    "source": "apple_watch"
                },
                {
                    "stage": "rem_sleep",
                    "duration_minutes": 90,
                    "start_time": "2024-01-15T04:00:00",
                    "efficiency_percentage": 80.0,
                    "source": "apple_watch"
                }
            ],
            "activity_data": [
                {
                    "activity_type": "walking",
                    "duration_minutes": 30,
                    "calories_burned": 150,
                    "intensity": "moderate",
                    "source": "apple_watch",
                    "timestamp": "2024-01-15T08:30:00"
                }
            ],
            "steps_data": [
                {
                    "steps": 1000,
                    "timestamp": "2024-01-15T08:00:00",
                    "source": "apple_watch"
                },
                {
                    "steps": 2000,
                    "timestamp": "2024-01-15T09:00:00",
                    "source": "apple_watch"
                }
            ]
        }
    
    def test_process_heart_rate_data(self, processor, sample_wearable_data):
        """Test heart rate data processing."""
        heart_rate_data = sample_wearable_data["heart_rate_data"]
        
        # Test summary statistics
        summary = processor.calculate_heart_rate_summary(heart_rate_data)
        
        assert summary["average"] == 97.5  # (75 + 120) / 2
        assert summary["min"] == 75
        assert summary["max"] == 120
        assert summary["resting"] == 75  # Should be the rest zone value
        assert summary["data_points"] == 2
        assert summary["quality_score"] > 0.9
    
    def test_process_sleep_data(self, processor, sample_wearable_data):
        """Test sleep data processing."""
        sleep_data = sample_wearable_data["sleep_data"]
        
        # Test sleep summary
        summary = processor.calculate_sleep_summary(sleep_data)
        
        assert summary["total_minutes"] == 210  # 120 + 90
        assert summary["total_hours"] == 3.5
        assert summary["deep_sleep_minutes"] == 120
        assert summary["rem_sleep_minutes"] == 90
        assert summary["light_sleep_minutes"] == 0
        assert summary["quality_score"] > 0.8
    
    def test_process_activity_data(self, processor, sample_wearable_data):
        """Test activity data processing."""
        activity_data = sample_wearable_data["activity_data"]
        
        # Test activity summary
        summary = processor.calculate_activity_summary(activity_data)
        
        assert summary["total_minutes"] == 30
        assert summary["total_calories"] == 150
        assert summary["activities_count"] == 1
        assert summary["average_intensity"] == "moderate"
    
    def test_process_steps_data(self, processor, sample_wearable_data):
        """Test steps data processing."""
        steps_data = sample_wearable_data["steps_data"]
        
        # Test steps summary
        summary = processor.calculate_steps_summary(steps_data)
        
        assert summary["total_steps"] == 3000  # 1000 + 2000
        assert summary["average_steps_per_hour"] == 1500  # 3000 / 2 hours
        assert summary["peak_hour"] is not None
    
    def test_calculate_data_quality_score(self, processor, sample_wearable_data):
        """Test data quality scoring."""
        # Test with complete data
        quality_score = processor.calculate_data_quality_score(sample_wearable_data)
        
        assert 0 <= quality_score <= 1
        assert quality_score > 0.8  # Should be high for complete data
        
        # Test with incomplete data
        incomplete_data = {
            "patient_id": "patient_001",
            "date": "2024-01-15T00:00:00",
            "device_type": "apple_watch",
            "device_id": "apple_watch_123",
            "heart_rate_data": [],  # Empty data
            "sleep_data": [],
            "activity_data": [],
            "steps_data": []
        }
        
        low_quality_score = processor.calculate_data_quality_score(incomplete_data)
        assert low_quality_score < quality_score
    
    def test_generate_health_insights(self, processor, sample_wearable_data):
        """Test health insights generation."""
        insights = processor.generate_health_insights(sample_wearable_data)
        
        assert len(insights) > 0
        assert isinstance(insights, list)
        
        # Check for specific insight types
        insight_text = " ".join(insights).lower()
        assert any(keyword in insight_text for keyword in ["heart", "sleep", "activity", "steps"])
    
    def test_process_wearable_data(self, processor, sample_wearable_data):
        """Test complete wearable data processing."""
        result = processor.process_wearable_data(sample_wearable_data)
        
        assert result.success is True
        assert result.log_id is not None
        assert result.data_points_processed > 0
        assert result.processing_time > 0
        assert result.data is not None
        
        # Check processed data
        processed_data = result.data
        assert processed_data.patient_id == "patient_001"
        assert processed_data.total_steps > 0
        assert processed_data.total_calories_burned > 0
        assert processed_data.total_sleep_minutes > 0
        assert processed_data.avg_heart_rate > 0
        assert processed_data.data_quality_score > 0
    
    def test_wearable_data_validation(self, processor):
        """Test wearable data validation."""
        # Test valid data
        valid_data = {
            "patient_id": "patient_001",
            "date": "2024-01-15T00:00:00",
            "device_type": "apple_watch",
            "device_id": "apple_watch_123",
            "heart_rate_data": [
                {
                    "heart_rate": 75,
                    "hrv_ms": 45,
                    "zone": "rest",
                    "confidence": 0.95,
                    "source": "apple_watch",
                    "timestamp": "2024-01-15T08:00:00"
                }
            ]
        }
        
        assert processor.validate_wearable_data(valid_data) is True
        
        # Test invalid data
        invalid_data = {
            "patient_id": "",  # Empty patient ID
            "date": "invalid_date",
            "heart_rate_data": [
                {
                    "heart_rate": -50,  # Negative heart rate
                    "hrv_ms": 45,
                    "zone": "rest",
                    "confidence": 0.95,
                    "source": "apple_watch",
                    "timestamp": "2024-01-15T08:00:00"
                }
            ]
        }
        
        assert processor.validate_wearable_data(invalid_data) is False
    
    def test_heart_rate_trend_analysis(self, processor):
        """Test heart rate trend analysis."""
        # Create 7 days of heart rate data
        heart_rate_data = []
        base_date = datetime(2024, 1, 15)
        
        for day in range(7):
            current_date = base_date + timedelta(days=day)
            # Simulate improving heart rate trend
            heart_rate = 80 - day * 2  # Decreasing heart rate (improving)
            
            heart_rate_data.append({
                "heart_rate": heart_rate,
                "hrv_ms": 45 + day,
                "zone": "rest",
                "confidence": 0.95,
                "source": "apple_watch",
                "timestamp": current_date.isoformat()
            })
        
        trends = processor.analyze_heart_rate_trends(heart_rate_data)
        
        assert trends["trend_direction"] == "improving"
        assert trends["trend_strength"] > 0.5
        assert trends["average_heart_rate"] > 0
        assert trends["variability"] > 0
    
    def test_sleep_trend_analysis(self, processor):
        """Test sleep trend analysis."""
        # Create 7 days of sleep data
        sleep_data = []
        base_date = datetime(2024, 1, 15)
        
        for day in range(7):
            current_date = base_date + timedelta(days=day)
            # Simulate improving sleep duration
            sleep_minutes = 360 + day * 30  # Increasing sleep duration
            
            sleep_data.append({
                "stage": "deep_sleep",
                "duration_minutes": sleep_minutes * 0.2,
                "start_time": current_date.isoformat(),
                "efficiency_percentage": 80 + day * 2,
                "source": "apple_watch"
            })
        
        trends = processor.analyze_sleep_trends(sleep_data)
        
        assert trends["trend_direction"] == "improving"
        assert trends["trend_strength"] > 0.5
        assert trends["average_sleep_hours"] > 0
        assert trends["sleep_efficiency"] > 0
    
    def test_activity_trend_analysis(self, processor):
        """Test activity trend analysis."""
        # Create 7 days of activity data
        activity_data = []
        base_date = datetime(2024, 1, 15)
        
        for day in range(7):
            current_date = base_date + timedelta(days=day)
            # Simulate increasing activity
            duration = 20 + day * 5  # Increasing activity duration
            
            activity_data.append({
                "activity_type": "walking",
                "duration_minutes": duration,
                "calories_burned": duration * 5,
                "intensity": "moderate",
                "source": "apple_watch",
                "timestamp": current_date.isoformat()
            })
        
        trends = processor.analyze_activity_trends(activity_data)
        
        assert trends["trend_direction"] == "improving"
        assert trends["trend_strength"] > 0.5
        assert trends["average_daily_activity_minutes"] > 0
        assert trends["total_calories_burned"] > 0
    
    def test_wearable_data_performance(self, processor):
        """Test wearable data processing performance."""
        import time
        
        # Create large dataset
        large_data = {
            "patient_id": "patient_001",
            "date": "2024-01-15T00:00:00",
            "device_type": "apple_watch",
            "device_id": "apple_watch_123",
            "heart_rate_data": [
                {
                    "heart_rate": 70 + i % 20,
                    "hrv_ms": 40 + i % 10,
                    "zone": "rest" if i % 2 == 0 else "active",
                    "confidence": 0.95,
                    "source": "apple_watch",
                    "timestamp": f"2024-01-15T{i:02d}:00:00"
                }
                for i in range(1000)  # 1000 heart rate readings
            ],
            "sleep_data": [
                {
                    "stage": "deep_sleep",
                    "duration_minutes": 120,
                    "start_time": "2024-01-15T02:00:00",
                    "efficiency_percentage": 85.0,
                    "source": "apple_watch"
                }
            ]
        }
        
        # Benchmark processing
        start_time = time.time()
        result = processor.process_wearable_data(large_data)
        processing_time = time.time() - start_time
        
        # Performance assertions
        assert processing_time < 5.0, f"Processing took {processing_time:.2f}s, should be < 5.0s"
        assert result.success is True
        assert result.data_points_processed == 1000
        
        print(f"Processed {result.data_points_processed} data points in {processing_time:.3f}s")
    
    def test_wearable_data_accuracy(self, processor):
        """Test wearable data processing accuracy."""
        # Test data with known values
        test_data = {
            "patient_id": "patient_001",
            "date": "2024-01-15T00:00:00",
            "device_type": "apple_watch",
            "device_id": "apple_watch_123",
            "heart_rate_data": [
                {"heart_rate": 60, "hrv_ms": 50, "zone": "rest", "confidence": 0.95, "source": "apple_watch", "timestamp": "2024-01-15T08:00:00"},
                {"heart_rate": 80, "hrv_ms": 40, "zone": "active", "confidence": 0.95, "source": "apple_watch", "timestamp": "2024-01-15T09:00:00"},
                {"heart_rate": 100, "hrv_ms": 30, "zone": "active", "confidence": 0.95, "source": "apple_watch", "timestamp": "2024-01-15T10:00:00"}
            ],
            "sleep_data": [
                {"stage": "deep_sleep", "duration_minutes": 120, "start_time": "2024-01-15T02:00:00", "efficiency_percentage": 85.0, "source": "apple_watch"},
                {"stage": "rem_sleep", "duration_minutes": 90, "start_time": "2024-01-15T04:00:00", "efficiency_percentage": 80.0, "source": "apple_watch"},
                {"stage": "light_sleep", "duration_minutes": 210, "start_time": "2024-01-15T00:00:00", "efficiency_percentage": 75.0, "source": "apple_watch"}
            ],
            "steps_data": [
                {"steps": 1000, "timestamp": "2024-01-15T08:00:00", "source": "apple_watch"},
                {"steps": 2000, "timestamp": "2024-01-15T09:00:00", "source": "apple_watch"},
                {"steps": 3000, "timestamp": "2024-01-15T10:00:00", "source": "apple_watch"}
            ]
        }
        
        result = processor.process_wearable_data(test_data)
        
        # Verify calculations
        processed_data = result.data
        
        # Heart rate calculations
        assert processed_data.avg_heart_rate == 80.0  # (60 + 80 + 100) / 3
        assert processed_data.heart_rate_data[0].heart_rate == 60
        assert processed_data.heart_rate_data[1].heart_rate == 80
        assert processed_data.heart_rate_data[2].heart_rate == 100
        
        # Sleep calculations
        assert processed_data.total_sleep_minutes == 420  # 120 + 90 + 210
        assert len(processed_data.sleep_data) == 3
        
        # Steps calculations
        assert processed_data.total_steps == 6000  # 1000 + 2000 + 3000
        assert len(processed_data.steps_data) == 3
    
    def test_wearable_data_edge_cases(self, processor):
        """Test wearable data edge cases."""
        # Test with empty data
        empty_data = {
            "patient_id": "patient_001",
            "date": "2024-01-15T00:00:00",
            "device_type": "apple_watch",
            "device_id": "apple_watch_123",
            "heart_rate_data": [],
            "sleep_data": [],
            "activity_data": [],
            "steps_data": []
        }
        
        result = processor.process_wearable_data(empty_data)
        assert result.success is True
        assert result.data_points_processed == 0
        assert result.data.data_quality_score < 0.5
        
        # Test with single data point
        single_data = {
            "patient_id": "patient_001",
            "date": "2024-01-15T00:00:00",
            "device_type": "apple_watch",
            "device_id": "apple_watch_123",
            "heart_rate_data": [
                {"heart_rate": 75, "hrv_ms": 45, "zone": "rest", "confidence": 0.95, "source": "apple_watch", "timestamp": "2024-01-15T08:00:00"}
            ]
        }
        
        result = processor.process_wearable_data(single_data)
        assert result.success is True
        assert result.data_points_processed == 1
        assert result.data.avg_heart_rate == 75.0
    
    def test_wearable_data_error_handling(self, processor):
        """Test wearable data error handling."""
        # Test with missing required fields
        invalid_data = {
            "patient_id": "patient_001",
            # Missing date
            "heart_rate_data": [
                {"heart_rate": 75, "hrv_ms": 45, "zone": "rest", "confidence": 0.95, "source": "apple_watch", "timestamp": "2024-01-15T08:00:00"}
            ]
        }
        
        result = processor.process_wearable_data(invalid_data)
        assert result.success is False
        assert "error" in result.message.lower()
        
        # Test with invalid heart rate values
        invalid_heart_rate_data = {
            "patient_id": "patient_001",
            "date": "2024-01-15T00:00:00",
            "device_type": "apple_watch",
            "device_id": "apple_watch_123",
            "heart_rate_data": [
                {"heart_rate": -50, "hrv_ms": 45, "zone": "rest", "confidence": 0.95, "source": "apple_watch", "timestamp": "2024-01-15T08:00:00"}
            ]
        }
        
        result = processor.process_wearable_data(invalid_heart_rate_data)
        assert result.success is False or result.data.data_quality_score < 0.5
    
    @pytest.mark.asyncio
    async def test_wearable_data_api(self, test_client, sample_users, sample_wearable_data):
        """Test wearable data API endpoint."""
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
        
        # Add user_id to sample data
        sample_wearable_data["user_id"] = token_data["user"]["user_id"]
        
        # Upload wearable data
        headers = {"Authorization": f"Bearer {access_token}"}
        response = test_client.post(
            "/api/v1/wearable-data",
            json=sample_wearable_data,
            headers=headers
        )
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["success"] is True
        assert result["log_id"] is not None
        assert result["data_points_processed"] > 0
        assert result["processing_time"] > 0
    
    def test_wearable_data_statistical_analysis(self, processor):
        """Test statistical analysis of wearable data."""
        # Create test data with known statistical properties
        heart_rate_data = []
        for i in range(100):
            heart_rate_data.append({
                "heart_rate": 70 + (i % 30),  # Values from 70 to 99
                "hrv_ms": 40 + (i % 20),      # Values from 40 to 59
                "zone": "rest" if i % 2 == 0 else "active",
                "confidence": 0.95,
                "source": "apple_watch",
                "timestamp": f"2024-01-15T{i:02d}:00:00"
            })
        
        # Test statistical calculations
        summary = processor.calculate_heart_rate_summary(heart_rate_data)
        
        assert summary["average"] > 70 and summary["average"] < 100
        assert summary["min"] == 70
        assert summary["max"] == 99
        assert summary["standard_deviation"] > 0
        assert summary["data_points"] == 100
        
        # Test trend analysis
        trends = processor.analyze_heart_rate_trends(heart_rate_data)
        assert trends["trend_direction"] in ["improving", "declining", "stable"]
        assert trends["trend_strength"] >= 0 and trends["trend_strength"] <= 1
        assert trends["variability"] > 0
    
    def test_wearable_data_quality_metrics(self, processor):
        """Test wearable data quality metrics."""
        # Test with high-quality data
        high_quality_data = {
            "patient_id": "patient_001",
            "date": "2024-01-15T00:00:00",
            "device_type": "apple_watch",
            "device_id": "apple_watch_123",
            "heart_rate_data": [
                {
                    "heart_rate": 75,
                    "hrv_ms": 45,
                    "zone": "rest",
                    "confidence": 0.95,
                    "source": "apple_watch",
                    "timestamp": "2024-01-15T08:00:00"
                }
            ],
            "sleep_data": [
                {
                    "stage": "deep_sleep",
                    "duration_minutes": 120,
                    "start_time": "2024-01-15T02:00:00",
                    "efficiency_percentage": 85.0,
                    "source": "apple_watch"
                }
            ]
        }
        
        result = processor.process_wearable_data(high_quality_data)
        assert result.data.data_quality_score > 0.8
        
        # Test with low-quality data
        low_quality_data = {
            "patient_id": "patient_001",
            "date": "2024-01-15T00:00:00",
            "device_type": "apple_watch",
            "device_id": "apple_watch_123",
            "heart_rate_data": [
                {
                    "heart_rate": 200,  # Unrealistic value
                    "hrv_ms": 45,
                    "zone": "rest",
                    "confidence": 0.5,  # Low confidence
                    "source": "apple_watch",
                    "timestamp": "2024-01-15T08:00:00"
                }
            ]
        }
        
        result = processor.process_wearable_data(low_quality_data)
        assert result.data.data_quality_score < 0.6
