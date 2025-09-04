#!/usr/bin/env python3
"""
Health AI Twin - Comprehensive System Testing Script
Tests all components: Food Recognition, AI Doctor, Health Predictions, etc.
"""

import requests
import json
import time
import os
from datetime import datetime
import base64

# Configuration
BASE_URLS = {
    "part1": "http://localhost:8003",
    "part2": "http://localhost:8004", 
    "part3": "http://localhost:8005",
    "part4": "http://localhost:8006",
    "frontend": "http://localhost:8501"
}

# Test credentials
TEST_USER = {
    "email": "test@example.com",
    "password": "test123"
}

class HealthAITwinTester:
    def __init__(self):
        self.session = requests.Session()
        self.auth_token = None
        self.test_results = []
        
    def log_test(self, test_name, status, details=""):
        """Log test results"""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        print(f"{'✅' if status == 'PASS' else '❌'} {test_name}: {details}")
        
    def test_service_health(self):
        """Test if all services are running"""
        print("\n🏥 Testing Service Health...")
        
        for service, url in BASE_URLS.items():
            try:
                response = self.session.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    self.log_test(f"{service} Health Check", "PASS", f"Service running on {url}")
                else:
                    self.log_test(f"{service} Health Check", "FAIL", f"Status: {response.status_code}")
            except Exception as e:
                self.log_test(f"{service} Health Check", "FAIL", f"Connection error: {str(e)}")
    
    def test_authentication(self):
        """Test user authentication"""
        print("\n🔐 Testing Authentication...")
        
        try:
            # Test login
            login_data = {
                "email": TEST_USER["email"],
                "password": TEST_USER["password"]
            }
            
            response = self.session.post(
                f"{BASE_URLS['part4']}/api/v1/auth/login",
                json=login_data,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if "access_token" in data:
                    self.auth_token = data["access_token"]
                    self.session.headers.update({"Authorization": f"Bearer {self.auth_token}"})
                    self.log_test("Authentication", "PASS", "Login successful")
                else:
                    self.log_test("Authentication", "FAIL", "No access token in response")
            else:
                self.log_test("Authentication", "FAIL", f"Login failed: {response.status_code}")
                
        except Exception as e:
            self.log_test("Authentication", "FAIL", f"Authentication error: {str(e)}")
    
    def test_food_recognition(self):
        """Test food image classification"""
        print("\n🍎 Testing Food Recognition...")
        
        # Create a simple test image (base64 encoded)
        test_image_data = """
        iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==
        """
        
        try:
            # Test food classification endpoint
            files = {
                'file': ('test_food.jpg', base64.b64decode(test_image_data), 'image/jpeg')
            }
            data = {
                'patient_id': '507f1f77bcf86cd799439011',  # Test patient ID
                'meal_type': 'lunch',
                'portion_size': '1.0'
            }
            
            response = self.session.post(
                f"{BASE_URLS['part2']}/api/v1/food-classification/classify",
                files=files,
                data=data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                if "classification" in result and "nutrition" in result:
                    food_name = result["classification"]["food_name"]
                    confidence = result["classification"]["confidence"]
                    self.log_test("Food Recognition", "PASS", 
                                f"Classified as {food_name} (confidence: {confidence})")
                else:
                    self.log_test("Food Recognition", "FAIL", "Missing classification data")
            else:
                self.log_test("Food Recognition", "FAIL", f"Status: {response.status_code}")
                
        except Exception as e:
            self.log_test("Food Recognition", "FAIL", f"Food recognition error: {str(e)}")
    
    def test_ai_doctor(self):
        """Test AI doctor consultation"""
        print("\n🤖 Testing AI Doctor...")
        
        try:
            # Test AI doctor chat
            chat_data = {
                "question": "How can I improve my diet?",
                "include_food_data": True,
                "recent_food_log": [
                    {
                        "food_name": "apple",
                        "nutrition": {"calories": 95, "protein": 0.5, "carbs": 25, "fat": 0.3}
                    }
                ]
            }
            
            response = self.session.post(
                f"{BASE_URLS['part4']}/api/v1/virtual-doctor/chat",
                json=chat_data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                if "conversational_response" in result:
                    self.log_test("AI Doctor", "PASS", "Conversational response generated")
                elif "data" in result:
                    self.log_test("AI Doctor", "PASS", "Health advice generated")
                else:
                    self.log_test("AI Doctor", "FAIL", "No response data")
            else:
                self.log_test("AI Doctor", "FAIL", f"Status: {response.status_code}")
                
        except Exception as e:
            self.log_test("AI Doctor", "FAIL", f"AI doctor error: {str(e)}")
    
    def test_health_predictions(self):
        """Test health prediction system"""
        print("\n📊 Testing Health Predictions...")
        
        try:
            # Test health prediction
            prediction_data = {
                "patient_id": "507f1f77bcf86cd799439011",
                "age": 35,
                "bmi": 24.5,
                "heart_rate_avg": 72,
                "steps_avg": 8500,
                "sleep_hours_avg": 7.5,
                "calories_avg": 2100
            }
            
            response = self.session.post(
                f"{BASE_URLS['part3']}/api/v1/health-prediction/predict",
                json=prediction_data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                if "predictions" in result:
                    self.log_test("Health Predictions", "PASS", "Predictions generated successfully")
                else:
                    self.log_test("Health Predictions", "FAIL", "No predictions in response")
            else:
                self.log_test("Health Predictions", "FAIL", f"Status: {response.status_code}")
                
        except Exception as e:
            self.log_test("Health Predictions", "FAIL", f"Prediction error: {str(e)}")
    
    def test_wearable_data(self):
        """Test wearable data processing"""
        print("\n⌚ Testing Wearable Data...")
        
        try:
            # Test wearable data submission
            wearable_data = {
                "patient_id": "507f1f77bcf86cd799439011",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "heart_rate_avg": 72,
                "steps_count": 8500,
                "sleep_hours": 7.5,
                "spo2_avg": 98
            }
            
            response = self.session.post(
                f"{BASE_URLS['part2']}/api/v1/wearable-data",
                json=wearable_data,
                timeout=10
            )
            
            if response.status_code == 200:
                self.log_test("Wearable Data", "PASS", "Data processed successfully")
            else:
                self.log_test("Wearable Data", "FAIL", f"Status: {response.status_code}")
                
        except Exception as e:
            self.log_test("Wearable Data", "FAIL", f"Wearable data error: {str(e)}")
    
    def test_frontend_access(self):
        """Test frontend accessibility"""
        print("\n🌐 Testing Frontend...")
        
        try:
            response = self.session.get(f"{BASE_URLS['frontend']}", timeout=10)
            if response.status_code == 200:
                self.log_test("Frontend Access", "PASS", "Frontend accessible")
            else:
                self.log_test("Frontend Access", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Frontend Access", "FAIL", f"Frontend error: {str(e)}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🧪 Health AI Twin - Comprehensive System Test")
        print("=" * 50)
        
        # Run all test categories
        self.test_service_health()
        self.test_authentication()
        self.test_food_recognition()
        self.test_ai_doctor()
        self.test_health_predictions()
        self.test_wearable_data()
        self.test_frontend_access()
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate test summary"""
        print("\n" + "=" * 50)
        print("📋 TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASS"])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"  - {result['test']}: {result['details']}")
        
        # Save results to file
        with open("test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: test_results.json")
        
        if passed_tests == total_tests:
            print("\n🎉 All tests passed! Your Health AI Twin system is working perfectly!")
        else:
            print(f"\n⚠️ {failed_tests} test(s) failed. Please check the issues above.")

def main():
    """Main function"""
    tester = HealthAITwinTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()



