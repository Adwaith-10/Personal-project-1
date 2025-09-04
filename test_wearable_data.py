#!/usr/bin/env python3
"""
Test script for wearable data upload functionality
"""

import requests
import json
from datetime import datetime, timedelta
from pathlib import Path

def test_wearable_data_upload():
    """Test the wearable data upload endpoint"""
    
    # API configuration
    API_BASE_URL = "http://localhost:8000"
    
    print("⌚ Testing Wearable Data Upload Functionality")
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
    
    # Create sample wearable data
    print("\n📊 Creating sample wearable data...")
    
    # Sample heart rate data throughout the day
    heart_rate_data = []
    base_time = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
    
    for i in range(24):  # 24 hours of data
        current_time = base_time + timedelta(hours=i)
        heart_rate = 65 + (i % 12) * 5  # Varying heart rate
        hrv = 40 + (i % 8) * 5  # Varying HRV
        
        heart_rate_data.append({
            "heart_rate": heart_rate,
            "hrv_ms": hrv,
            "zone": "rest" if heart_rate < 70 else "fat_burn" if heart_rate < 100 else "cardio",
            "confidence": 0.95,
            "source": "apple_watch"
        })
    
    # Sample SpO2 data
    spo2_data = [
        {
            "spo2_percentage": 98.5,
            "confidence": 0.92,
            "source": "apple_watch"
        },
        {
            "spo2_percentage": 97.8,
            "confidence": 0.89,
            "source": "apple_watch"
        },
        {
            "spo2_percentage": 99.1,
            "confidence": 0.94,
            "source": "apple_watch"
        }
    ]
    
    # Sample sleep data
    sleep_data = [
        {
            "stage": "light_sleep",
            "duration_minutes": 180,
            "start_time": datetime.now().replace(hour=22, minute=0, second=0, microsecond=0),
            "efficiency_percentage": 85.0,
            "source": "apple_watch"
        },
        {
            "stage": "deep_sleep",
            "duration_minutes": 120,
            "start_time": datetime.now().replace(hour=1, minute=0, second=0, microsecond=0),
            "efficiency_percentage": 90.0,
            "source": "apple_watch"
        },
        {
            "stage": "rem_sleep",
            "duration_minutes": 90,
            "start_time": datetime.now().replace(hour=3, minute=0, second=0, microsecond=0),
            "efficiency_percentage": 88.0,
            "source": "apple_watch"
        }
    ]
    
    # Sample activity data
    activity_data = [
        {
            "activity_type": "walking",
            "duration_minutes": 30,
            "calories_burned": 150.0,
            "distance_meters": 2000.0,
            "steps": 2500,
            "start_time": datetime.now().replace(hour=8, minute=0, second=0, microsecond=0),
            "source": "apple_watch"
        },
        {
            "activity_type": "running",
            "duration_minutes": 45,
            "calories_burned": 400.0,
            "distance_meters": 5000.0,
            "steps": 6000,
            "start_time": datetime.now().replace(hour=17, minute=0, second=0, microsecond=0),
            "source": "apple_watch"
        }
    ]
    
    # Sample steps data
    steps_data = [
        {
            "steps_count": 8500,
            "distance_meters": 6800.0,
            "calories_burned": 425.0,
            "source": "apple_watch"
        }
    ]
    
    # Sample calories data
    calories_data = [
        {
            "calories_burned": 1850.0,
            "calories_consumed": 2100.0,
            "net_calories": -250.0,
            "source": "apple_watch"
        }
    ]
    
    # Sample temperature data
    temperature_data = [
        {
            "temperature_celsius": 37.2,
            "confidence": 0.88,
            "source": "apple_watch"
        },
        {
            "temperature_celsius": 36.8,
            "confidence": 0.91,
            "source": "apple_watch"
        }
    ]
    
    # Create the complete wearable data payload
    wearable_data = {
        "patient_id": patient_id,
        "device_id": "apple_watch_123",
        "device_type": "apple_watch",
        "date": datetime.now().isoformat(),
        "heart_rate_data": heart_rate_data,
        "spo2_data": spo2_data,
        "sleep_data": sleep_data,
        "activity_data": activity_data,
        "steps_data": steps_data,
        "calories_data": calories_data,
        "temperature_data": temperature_data,
        "raw_data": {
            "device_battery": 85,
            "sync_timestamp": datetime.now().isoformat(),
            "data_quality": "high"
        }
    }
    
    print(f"📈 Created sample data with:")
    print(f"   - {len(heart_rate_data)} heart rate readings")
    print(f"   - {len(spo2_data)} SpO2 readings")
    print(f"   - {len(sleep_data)} sleep sessions")
    print(f"   - {len(activity_data)} activity sessions")
    print(f"   - {len(steps_data)} steps records")
    print(f"   - {len(calories_data)} calories records")
    print(f"   - {len(temperature_data)} temperature readings")
    
    # Test the upload endpoint
    print("\n📤 Testing wearable data upload...")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/wearable-data/",
            json=wearable_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Wearable data uploaded successfully!")
            print(f"📊 Log ID: {result.get('log_id')}")
            print(f"🔢 Data points processed: {result.get('data_points_processed')}")
            print(f"⏱️ Processing time: {result.get('processing_time')} seconds")
            
            # Display summary statistics
            if result.get('data'):
                data = result['data']
                print(f"\n📈 Summary Statistics:")
                print(f"   - Total Steps: {data.get('total_steps', 'N/A')}")
                print(f"   - Calories Burned: {data.get('total_calories_burned', 'N/A')}")
                print(f"   - Sleep Duration: {data.get('total_sleep_minutes', 'N/A')} minutes")
                print(f"   - Avg Heart Rate: {data.get('avg_heart_rate', 'N/A')} bpm")
                print(f"   - Avg SpO2: {data.get('avg_spo2', 'N/A')}%")
                print(f"   - Data Quality Score: {data.get('data_quality_score', 'N/A')}")
            
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during upload: {e}")
    
    # Test getting wearable data
    print("\n📋 Testing wearable data retrieval...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/wearable-data/?patient_id={patient_id}")
        if response.status_code == 200:
            logs = response.json()
            print(f"✅ Found {len(logs)} wearable data logs for patient")
        else:
            print(f"❌ Failed to get wearable data: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting wearable data: {e}")
    
    # Test summary endpoint
    print("\n📊 Testing summary endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/wearable-data/patient/{patient_id}/summary?days=7")
        if response.status_code == 200:
            summary = response.json()
            print("✅ Summary generated successfully!")
            print(f"📈 Summary for last 7 days:")
            print(f"   - Total Steps: {summary.get('total_steps', 'N/A')}")
            print(f"   - Total Calories: {summary.get('total_calories_burned', 'N/A')}")
            print(f"   - Total Sleep: {summary.get('total_sleep_hours', 'N/A')} hours")
            print(f"   - Avg Heart Rate: {summary.get('avg_heart_rate', 'N/A')} bpm")
            print(f"   - Data Completeness: {summary.get('data_completeness', 'N/A')}%")
        else:
            print(f"❌ Failed to get summary: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting summary: {e}")
    
    # Test trends endpoint
    print("\n📈 Testing trends endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/wearable-data/patient/{patient_id}/trends?days=7")
        if response.status_code == 200:
            trends = response.json()
            print("✅ Trends analysis completed!")
            if trends.get('trends'):
                print(f"📊 Trend Analysis:")
                for metric, trend_data in trends['trends'].items():
                    print(f"   - {metric}: {trend_data.get('trend', 'N/A')} (avg: {trend_data.get('avg', 'N/A')})")
            else:
                print("   - No trend data available")
        else:
            print(f"❌ Failed to get trends: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting trends: {e}")
    
    # Test insights endpoint
    print("\n💡 Testing insights endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/wearable-data/patient/{patient_id}/insights?days=7")
        if response.status_code == 200:
            insights = response.json()
            print("✅ Health insights generated!")
            if insights.get('insights'):
                print(f"🔍 Health Insights:")
                for insight in insights['insights']:
                    severity_emoji = {
                        "warning": "⚠️",
                        "info": "ℹ️",
                        "success": "✅"
                    }.get(insight.get('severity'), "❓")
                    print(f"   {severity_emoji} {insight.get('message', 'N/A')}")
            else:
                print("   - No insights available")
        else:
            print(f"❌ Failed to get insights: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting insights: {e}")

if __name__ == "__main__":
    test_wearable_data_upload()
