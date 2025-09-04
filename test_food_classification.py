#!/usr/bin/env python3
"""
Test script for food classification functionality
"""

import requests
import json
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io

def create_sample_food_image(food_name="apple"):
    """Create a sample food image for testing"""
    # Create a simple colored rectangle as a placeholder
    width, height = 400, 300
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw a colored rectangle based on food type
    colors = {
        "apple": (255, 0, 0),      # Red
        "banana": (255, 255, 0),   # Yellow
        "orange": (255, 165, 0),   # Orange
        "pizza": (255, 69, 0),     # Red-orange
        "hamburger": (139, 69, 19), # Brown
        "salad": (34, 139, 34),    # Green
        "bread": (210, 180, 140),  # Tan
        "chicken": (255, 215, 0),  # Gold
        "fish": (0, 191, 255),     # Deep sky blue
        "rice": (255, 255, 240),   # Ivory
        "pasta": (255, 228, 196),  # Bisque
        "milk": (255, 255, 255),   # White
        "cheese": (255, 215, 0),   # Gold
        "yogurt": (255, 255, 224), # Light yellow
        "ice cream": (255, 182, 193), # Light pink
        "cake": (255, 192, 203),   # Pink
        "cookie": (160, 82, 45),   # Saddle brown
        "chocolate": (139, 69, 19), # Saddle brown
        "coffee": (101, 67, 33),   # Dark brown
        "tea": (210, 180, 140),    # Tan
        "water": (173, 216, 230),  # Light blue
        "soda": (255, 20, 147),    # Deep pink
        "beer": (255, 215, 0),     # Gold
        "wine": (128, 0, 128),     # Purple
        "carrot": (255, 165, 0),   # Orange
        "broccoli": (34, 139, 34), # Green
        "tomato": (255, 0, 0),     # Red
        "onion": (255, 255, 255),  # White
        "potato": (210, 180, 140), # Tan
        "corn": (255, 255, 0),     # Yellow
        "peas": (34, 139, 34),     # Green
        "lettuce": (50, 205, 50),  # Lime green
        "spinach": (0, 128, 0),    # Green
        "cucumber": (0, 255, 127), # Spring green
        "bell pepper": (255, 0, 0), # Red
        "mushroom": (210, 180, 140), # Tan
        "garlic": (255, 255, 255), # White
        "ginger": (255, 140, 0),   # Dark orange
        "lemon": (255, 255, 0),    # Yellow
        "lime": (0, 255, 0),       # Lime
        "avocado": (0, 128, 0),    # Green
        "olive": (128, 128, 0),    # Olive
        "almond": (210, 180, 140), # Tan
        "walnut": (139, 69, 19),   # Saddle brown
        "peanut": (210, 180, 140), # Tan
        "cashew": (255, 215, 0),   # Gold
        "sunflower seed": (255, 255, 0), # Yellow
        "pumpkin seed": (255, 165, 0),   # Orange
        "chia seed": (34, 139, 34),      # Green
        "flax seed": (210, 180, 140),    # Tan
        "quinoa": (255, 255, 240),       # Ivory
        "oatmeal": (210, 180, 140),      # Tan
        "cereal": (255, 255, 224),       # Light yellow
        "toast": (210, 180, 140),        # Tan
        "bagel": (210, 180, 140),        # Tan
        "muffin": (255, 228, 196),       # Bisque
        "donut": (255, 192, 203),        # Pink
        "croissant": (255, 228, 196),    # Bisque
        "pancake": (255, 228, 196),      # Bisque
        "waffle": (255, 228, 196),       # Bisque
        "french fries": (255, 215, 0),   # Gold
        "chips": (255, 165, 0),          # Orange
        "popcorn": (255, 255, 224),      # Light yellow
        "pretzel": (210, 180, 140),      # Tan
        "crackers": (255, 228, 196),     # Bisque
        "nuts": (139, 69, 19),           # Saddle brown
        "dried fruit": (255, 69, 0),     # Red-orange
        "jerky": (139, 69, 19),          # Saddle brown
        "granola": (210, 180, 140),      # Tan
        "energy bar": (255, 215, 0),     # Gold
        "protein bar": (255, 215, 0),    # Gold
        "smoothie": (255, 182, 193),     # Light pink
        "juice": (255, 69, 0),           # Red-orange
        "sports drink": (0, 191, 255),   # Deep sky blue
        "protein shake": (255, 255, 224), # Light yellow
        "meal replacement": (255, 228, 196), # Bisque
        "supplement": (255, 255, 255),   # White
        "vitamin": (255, 255, 255),      # White
        "medicine": (255, 255, 255),     # White
        "unknown": (128, 128, 128)       # Gray
    }
    
    color = colors.get(food_name.lower(), colors["unknown"])
    
    # Draw a rectangle representing the food
    margin = 50
    draw.rectangle([margin, margin, width-margin, height-margin], fill=color, outline='black', width=3)
    
    # Add text label
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    text = f"Sample {food_name.title()}"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    text_x = (width - text_width) // 2
    text_y = height - margin - text_height - 10
    
    draw.text((text_x, text_y), text, fill='black', font=font)
    
    return image

def test_food_classification():
    """Test the food classification endpoint"""
    
    # API configuration
    API_BASE_URL = "http://localhost:8000"
    
    print("🍎 Testing Food Classification Functionality")
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
    
    # Test different food types
    test_foods = ["apple", "banana", "pizza", "salad", "chicken", "bread", "coffee"]
    
    for food_name in test_foods:
        print(f"\n🍽️ Testing classification for: {food_name}")
        
        # Create sample food image
        image = create_sample_food_image(food_name)
        
        # Save image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        # Test the classification endpoint
        try:
            files = {
                "file": (f"sample_{food_name}.jpg", img_byte_arr, "image/jpeg")
            }
            data = {
                "patient_id": patient_id,
                "meal_type": "lunch",
                "portion_size": 1.0,
                "notes": f"Sample {food_name} for testing"
            }
            
            response = requests.post(
                f"{API_BASE_URL}/api/v1/food-classification/classify",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Classification successful!")
                print(f"📊 Log ID: {result.get('log_id')}")
                print(f"🔢 Predictions made: {result.get('predictions_count')}")
                print(f"⏱️ Processing time: {result.get('processing_time')} seconds")
                
                # Display top prediction
                if result.get('data') and result['data'].get('top_prediction'):
                    top_pred = result['data']['top_prediction']
                    print(f"🥇 Top Prediction: {top_pred.get('food_name', 'Unknown')}")
                    print(f"🎯 Confidence: {top_pred.get('confidence', 0):.1%}")
                    print(f"📂 Category: {top_pred.get('category', 'Unknown')}")
                
                # Display nutrition info
                if result.get('data'):
                    data = result['data']
                    print(f"📈 Nutrition Summary:")
                    print(f"   - Calories: {data.get('total_calories', 'N/A')}")
                    print(f"   - Protein: {data.get('total_protein', 'N/A')}g")
                    print(f"   - Carbs: {data.get('total_carbs', 'N/A')}g")
                    print(f"   - Fat: {data.get('total_fat', 'N/A')}g")
                    print(f"   - Fiber: {data.get('total_fiber', 'N/A')}g")
                    print(f"   - Portion Size: {data.get('portion_size', 'N/A')}x")
                
            else:
                print(f"❌ Classification failed: {response.status_code}")
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Error during classification: {e}")
    
    # Test getting food logs
    print("\n📋 Testing food logs retrieval...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/food-classification/logs?patient_id={patient_id}")
        if response.status_code == 200:
            logs = response.json()
            print(f"✅ Found {len(logs)} food logs for patient")
        else:
            print(f"❌ Failed to get food logs: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting food logs: {e}")
    
    # Test daily summary
    print("\n📊 Testing daily nutrition summary...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/food-classification/patient/{patient_id}/daily-summary")
        if response.status_code == 200:
            summary = response.json()
            print("✅ Daily summary generated successfully!")
            print(f"📈 Today's Nutrition Summary:")
            print(f"   - Total Calories: {summary.get('total_calories', 'N/A')}")
            print(f"   - Total Protein: {summary.get('total_protein', 'N/A')}g")
            print(f"   - Total Carbs: {summary.get('total_carbs', 'N/A')}g")
            print(f"   - Total Fat: {summary.get('total_fat', 'N/A')}g")
            print(f"   - Meals Count: {summary.get('meals_count', 'N/A')}")
            print(f"   - Foods Count: {summary.get('foods_count', 'N/A')}")
            
            # Show progress towards goals
            if summary.get('calorie_progress'):
                print(f"🎯 Calorie Goal Progress: {summary.get('calorie_progress', 'N/A')}%")
            if summary.get('protein_progress'):
                print(f"🎯 Protein Goal Progress: {summary.get('protein_progress', 'N/A')}%")
        else:
            print(f"❌ Failed to get daily summary: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting daily summary: {e}")
    
    # Test nutrition trends
    print("\n📈 Testing nutrition trends...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/food-classification/patient/{patient_id}/nutrition-trends?days=7")
        if response.status_code == 200:
            trends = response.json()
            print("✅ Nutrition trends analysis completed!")
            if trends.get('trends'):
                print(f"📊 Trend Analysis (last 7 days):")
                for metric, trend_data in trends['trends'].items():
                    print(f"   - {metric.title()}: avg {trend_data.get('average', 'N/A')} (min: {trend_data.get('min', 'N/A')}, max: {trend_data.get('max', 'N/A')})")
            else:
                print("   - No trend data available")
        else:
            print(f"❌ Failed to get nutrition trends: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting nutrition trends: {e}")

if __name__ == "__main__":
    test_food_classification()
