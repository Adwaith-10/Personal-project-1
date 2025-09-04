import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import requests
import json
from datetime import datetime
import os
import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="Food Classifier & Nutrition Estimator",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .nutrition-card {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .upload-area {
        border: 2px dashed #2E8B57;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8fff8;
    }
</style>
""", unsafe_allow_html=True)

class FoodClassifier:
    """Food classification and nutrition estimation system"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = []
        self.nutrition_db = self._load_nutrition_database()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _load_nutrition_database(self):
        """Load nutrition database with common foods"""
        return {
            "apple": {"calories": 95, "protein": 0.5, "carbs": 25, "fat": 0.3, "fiber": 4.4},
            "banana": {"calories": 105, "protein": 1.3, "carbs": 27, "fat": 0.4, "fiber": 3.1},
            "orange": {"calories": 62, "protein": 1.2, "carbs": 15, "fat": 0.2, "fiber": 3.1},
            "strawberry": {"calories": 49, "protein": 1.0, "carbs": 12, "fat": 0.5, "fiber": 3.0},
            "grape": {"calories": 62, "protein": 0.6, "carbs": 16, "fat": 0.2, "fiber": 0.9},
            "pizza": {"calories": 266, "protein": 11, "carbs": 33, "fat": 10, "fiber": 2.5},
            "hamburger": {"calories": 354, "protein": 16, "carbs": 30, "fat": 17, "fiber": 2.0},
            "hot dog": {"calories": 151, "protein": 5, "carbs": 2, "fat": 14, "fiber": 0.0},
            "sandwich": {"calories": 200, "protein": 12, "carbs": 25, "fat": 8, "fiber": 2.0},
            "salad": {"calories": 20, "protein": 2, "carbs": 4, "fat": 0.2, "fiber": 1.5},
            "soup": {"calories": 100, "protein": 5, "carbs": 15, "fat": 3, "fiber": 2.0},
            "rice": {"calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3, "fiber": 0.4},
            "pasta": {"calories": 131, "protein": 5, "carbs": 25, "fat": 1.1, "fiber": 1.8},
            "bread": {"calories": 79, "protein": 3.1, "carbs": 15, "fat": 1.0, "fiber": 1.2},
            "chicken": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6, "fiber": 0.0},
            "fish": {"calories": 100, "protein": 20, "carbs": 0, "fat": 2.5, "fiber": 0.0},
            "beef": {"calories": 250, "protein": 26, "carbs": 0, "fat": 15, "fiber": 0.0},
            "pork": {"calories": 242, "protein": 27, "carbs": 0, "fat": 14, "fiber": 0.0},
            "egg": {"calories": 70, "protein": 6, "carbs": 0.6, "fat": 5, "fiber": 0.0},
            "milk": {"calories": 42, "protein": 3.4, "carbs": 5, "fat": 1, "fiber": 0.0},
            "cheese": {"calories": 113, "protein": 7, "carbs": 0.4, "fat": 9, "fiber": 0.0},
            "yogurt": {"calories": 59, "protein": 10, "carbs": 3.6, "fat": 0.4, "fiber": 0.0},
            "ice cream": {"calories": 137, "protein": 2.3, "carbs": 16, "fat": 7, "fiber": 0.0},
            "cake": {"calories": 257, "protein": 3.2, "carbs": 35, "fat": 12, "fiber": 0.8},
            "cookie": {"calories": 78, "protein": 0.9, "carbs": 10, "fat": 4, "fiber": 0.3},
            "chocolate": {"calories": 546, "protein": 4.9, "carbs": 61, "fat": 31, "fiber": 7.0},
            "coffee": {"calories": 2, "protein": 0.3, "carbs": 0, "fat": 0, "fiber": 0.0},
            "tea": {"calories": 1, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0.0},
            "water": {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0.0},
            "soda": {"calories": 150, "protein": 0, "carbs": 39, "fat": 0, "fiber": 0.0},
            "beer": {"calories": 153, "protein": 1.6, "carbs": 13, "fat": 0, "fiber": 0.0},
            "wine": {"calories": 125, "protein": 0.1, "carbs": 4, "fat": 0, "fiber": 0.0},
            "carrot": {"calories": 41, "protein": 0.9, "carbs": 10, "fat": 0.2, "fiber": 2.8},
            "broccoli": {"calories": 34, "protein": 2.8, "carbs": 7, "fat": 0.4, "fiber": 2.6},
            "tomato": {"calories": 22, "protein": 1.1, "carbs": 5, "fat": 0.2, "fiber": 1.2},
            "onion": {"calories": 40, "protein": 1.1, "carbs": 9, "fat": 0.1, "fiber": 1.7},
            "potato": {"calories": 77, "protein": 2, "carbs": 17, "fat": 0.1, "fiber": 2.2},
            "corn": {"calories": 86, "protein": 3.2, "carbs": 19, "fat": 1.2, "fiber": 2.7},
            "peas": {"calories": 84, "protein": 5.4, "carbs": 14, "fat": 0.4, "fiber": 5.7},
            "lettuce": {"calories": 15, "protein": 1.4, "carbs": 2.9, "fat": 0.1, "fiber": 1.3},
            "spinach": {"calories": 23, "protein": 2.9, "carbs": 3.6, "fat": 0.4, "fiber": 2.2},
            "cucumber": {"calories": 16, "protein": 0.7, "carbs": 3.6, "fat": 0.1, "fiber": 0.5},
            "bell pepper": {"calories": 31, "protein": 1, "carbs": 7, "fat": 0.3, "fiber": 2.1},
            "mushroom": {"calories": 22, "protein": 3.1, "carbs": 3.3, "fat": 0.3, "fiber": 1.0},
            "garlic": {"calories": 149, "protein": 6.4, "carbs": 33, "fat": 0.5, "fiber": 2.1},
            "ginger": {"calories": 80, "protein": 1.8, "carbs": 18, "fat": 0.8, "fiber": 2.0},
            "lemon": {"calories": 29, "protein": 1.1, "carbs": 9, "fat": 0.3, "fiber": 2.8},
            "lime": {"calories": 30, "protein": 0.7, "carbs": 10, "fat": 0.2, "fiber": 2.8},
            "avocado": {"calories": 160, "protein": 2, "carbs": 9, "fat": 15, "fiber": 6.7},
            "olive": {"calories": 115, "protein": 0.8, "carbs": 6, "fat": 11, "fiber": 3.2},
            "almond": {"calories": 164, "protein": 6, "carbs": 6, "fat": 14, "fiber": 3.5},
            "walnut": {"calories": 185, "protein": 4.3, "carbs": 4, "fat": 18, "fiber": 1.9},
            "peanut": {"calories": 166, "protein": 7, "carbs": 6, "fat": 14, "fiber": 2.4},
            "cashew": {"calories": 157, "protein": 5, "carbs": 9, "fat": 12, "fiber": 0.9},
            "sunflower seed": {"calories": 164, "protein": 5.8, "carbs": 6, "fat": 14, "fiber": 3.0},
            "pumpkin seed": {"calories": 151, "protein": 7, "carbs": 5, "fat": 13, "fiber": 1.7},
            "chia seed": {"calories": 137, "protein": 4.4, "carbs": 12, "fat": 9, "fiber": 10.6},
            "flax seed": {"calories": 55, "protein": 1.9, "carbs": 3, "fat": 4.3, "fiber": 2.8},
            "quinoa": {"calories": 120, "protein": 4.4, "carbs": 22, "fat": 1.9, "fiber": 2.8},
            "oatmeal": {"calories": 68, "protein": 2.4, "carbs": 12, "fat": 1.4, "fiber": 1.7},
            "cereal": {"calories": 100, "protein": 2, "carbs": 20, "fat": 1, "fiber": 2.0},
            "toast": {"calories": 75, "protein": 3, "carbs": 14, "fat": 1, "fiber": 1.5},
            "bagel": {"calories": 245, "protein": 9, "carbs": 48, "fat": 1.5, "fiber": 2.0},
            "muffin": {"calories": 265, "protein": 4.5, "carbs": 44, "fat": 8, "fiber": 1.0},
            "donut": {"calories": 253, "protein": 4, "carbs": 31, "fat": 12, "fiber": 1.0},
            "croissant": {"calories": 231, "protein": 5, "carbs": 26, "fat": 12, "fiber": 1.0},
            "pancake": {"calories": 227, "protein": 6, "carbs": 38, "fat": 6, "fiber": 1.0},
            "waffle": {"calories": 218, "protein": 6, "carbs": 35, "fat": 7, "fiber": 1.0},
            "french fries": {"calories": 365, "protein": 4, "carbs": 63, "fat": 17, "fiber": 4.4},
            "chips": {"calories": 536, "protein": 7, "carbs": 53, "fat": 35, "fiber": 4.4},
            "popcorn": {"calories": 31, "protein": 1, "carbs": 6, "fat": 0.4, "fiber": 1.2},
            "pretzel": {"calories": 380, "protein": 10, "carbs": 80, "fat": 3, "fiber": 2.7},
            "crackers": {"calories": 502, "protein": 10, "carbs": 61, "fat": 26, "fiber": 2.0},
            "nuts": {"calories": 607, "protein": 20, "carbs": 23, "fat": 54, "fiber": 7.0},
            "dried fruit": {"calories": 359, "protein": 3.4, "carbs": 93, "fat": 0.5, "fiber": 7.0},
            "jerky": {"calories": 410, "protein": 33, "carbs": 11, "fat": 26, "fiber": 0.0},
            "granola": {"calories": 471, "protein": 10, "carbs": 64, "fat": 20, "fiber": 8.0},
            "energy bar": {"calories": 200, "protein": 8, "carbs": 25, "fat": 8, "fiber": 3.0},
            "protein bar": {"calories": 200, "protein": 20, "carbs": 15, "fat": 8, "fiber": 2.0},
            "smoothie": {"calories": 150, "protein": 2, "carbs": 30, "fat": 2, "fiber": 3.0},
            "juice": {"calories": 111, "protein": 0.5, "carbs": 26, "fat": 0.2, "fiber": 0.5},
            "sports drink": {"calories": 45, "protein": 0, "carbs": 12, "fat": 0, "fiber": 0.0},
            "protein shake": {"calories": 120, "protein": 25, "carbs": 3, "fat": 1, "fiber": 0.0},
            "meal replacement": {"calories": 250, "protein": 15, "carbs": 30, "fat": 8, "fiber": 5.0},
            "supplement": {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0.0},
            "vitamin": {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0.0},
            "medicine": {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0.0},
            "unknown": {"calories": 100, "protein": 5, "carbs": 15, "fat": 3, "fiber": 2.0}
        }
    
    def load_model(self):
        """Load the pre-trained ResNet18 model"""
        try:
            # Load pre-trained ResNet18
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            self.model.eval()
            self.model.to(self.device)
            
            # Load ImageNet class names
            import urllib.request
            url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            urllib.request.urlretrieve(url, "imagenet_classes.txt")
            
            with open("imagenet_classes.txt", "r") as f:
                self.class_names = [s.strip() for s in f.readlines()]
            
            st.success("✅ Model loaded successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return False
    
    def classify_image(self, image):
        """Classify food image and return predictions"""
        try:
            # Preprocess image
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top 5 predictions
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            
            predictions = []
            for i in range(top5_prob.size(0)):
                class_name = self.class_names[top5_catid[i]]
                confidence = top5_prob[i].item()
                
                # Check if it's food-related
                if self._is_food_related(class_name):
                    predictions.append({
                        "class": class_name,
                        "confidence": confidence,
                        "nutrition": self._get_nutrition_info(class_name)
                    })
            
            return predictions
            
        except Exception as e:
            st.error(f"❌ Error classifying image: {str(e)}")
            return []
    
    def _is_food_related(self, class_name):
        """Check if the class name is food-related"""
        food_keywords = [
            'food', 'fruit', 'vegetable', 'meat', 'bread', 'cake', 'cookie', 'pizza',
            'hamburger', 'hot dog', 'sandwich', 'salad', 'soup', 'rice', 'pasta',
            'chicken', 'fish', 'beef', 'pork', 'egg', 'milk', 'cheese', 'yogurt',
            'ice cream', 'chocolate', 'coffee', 'tea', 'water', 'soda', 'beer', 'wine',
            'apple', 'banana', 'orange', 'strawberry', 'grape', 'carrot', 'broccoli',
            'tomato', 'onion', 'potato', 'corn', 'peas', 'lettuce', 'spinach',
            'cucumber', 'bell pepper', 'mushroom', 'garlic', 'ginger', 'lemon', 'lime',
            'avocado', 'olive', 'almond', 'walnut', 'peanut', 'cashew', 'seed',
            'quinoa', 'oatmeal', 'cereal', 'toast', 'bagel', 'muffin', 'donut',
            'croissant', 'pancake', 'waffle', 'french fries', 'chips', 'popcorn',
            'pretzel', 'crackers', 'nuts', 'dried fruit', 'jerky', 'granola',
            'energy bar', 'protein bar', 'smoothie', 'juice', 'sports drink',
            'protein shake', 'meal replacement'
        ]
        
        class_name_lower = class_name.lower()
        return any(keyword in class_name_lower for keyword in food_keywords)
    
    def _get_nutrition_info(self, food_name):
        """Get nutrition information for a food item"""
        food_name_lower = food_name.lower()
        
        # Try exact match first
        if food_name_lower in self.nutrition_db:
            return self.nutrition_db[food_name_lower]
        
        # Try partial matches
        for key, nutrition in self.nutrition_db.items():
            if key in food_name_lower or food_name_lower in key:
                return nutrition
        
        # Return default nutrition for unknown foods
        return self.nutrition_db["unknown"]
    
    def estimate_portion_size(self, image):
        """Estimate portion size based on image analysis (simplified)"""
        # This is a simplified estimation - in a real system, you'd use more sophisticated CV
        width, height = image.size
        
        # Simple heuristic: larger images might indicate larger portions
        area = width * height
        if area > 500000:  # Large image
            return 1.5  # 1.5x normal portion
        elif area > 200000:  # Medium image
            return 1.0  # Normal portion
        else:  # Small image
            return 0.7  # 0.7x normal portion

def main():
    """Main Streamlit application"""
    st.markdown('<h1 class="main-header">🍎 Food Classifier & Nutrition Estimator</h1>', unsafe_allow_html=True)
    
    # Initialize classifier
    if 'classifier' not in st.session_state:
        st.session_state.classifier = FoodClassifier()
        st.session_state.model_loaded = False
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Load model button
    if not st.session_state.model_loaded:
        if st.sidebar.button("🚀 Load AI Model"):
            with st.spinner("Loading ResNet18 model..."):
                st.session_state.model_loaded = st.session_state.classifier.load_model()
    
    if st.session_state.model_loaded:
        st.sidebar.success("✅ Model Ready")
    else:
        st.sidebar.warning("⚠️ Model not loaded")
        st.info("Please load the AI model from the sidebar to start classifying food images.")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Upload Food Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a food image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of food to classify and get nutrition information"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Classify button
            if st.button("🔍 Classify Food", type="primary"):
                with st.spinner("Analyzing image..."):
                    predictions = st.session_state.classifier.classify_image(image)
                    
                    if predictions:
                        st.session_state.predictions = predictions
                        st.session_state.uploaded_image = image
                        st.success("✅ Analysis complete!")
                    else:
                        st.warning("⚠️ No food items detected in the image.")
    
    with col2:
        st.subheader("📊 Results")
        
        if 'predictions' in st.session_state and st.session_state.predictions:
            predictions = st.session_state.predictions
            
            # Display top prediction
            top_prediction = predictions[0]
            st.markdown(f"**Top Prediction:** {top_prediction['class'].title()}")
            
            # Confidence indicator
            confidence = top_prediction['confidence']
            if confidence > 0.7:
                confidence_class = "confidence-high"
            elif confidence > 0.4:
                confidence_class = "confidence-medium"
            else:
                confidence_class = "confidence-low"
            
            st.markdown(f"<span class='{confidence_class}'>Confidence: {confidence:.1%}</span>", unsafe_allow_html=True)
            
            # Nutrition information
            nutrition = top_prediction['nutrition']
            
            st.markdown("### 🥗 Nutrition Information (per 100g)")
            
            col_n1, col_n2, col_n3, col_n4 = st.columns(4)
            
            with col_n1:
                st.metric("Calories", f"{nutrition['calories']}")
            
            with col_n2:
                st.metric("Protein", f"{nutrition['protein']}g")
            
            with col_n3:
                st.metric("Carbs", f"{nutrition['carbs']}g")
            
            with col_n4:
                st.metric("Fat", f"{nutrition['fat']}g")
            
            # Additional nutrition info
            st.markdown(f"**Fiber:** {nutrition['fiber']}g")
            
            # Portion size estimation
            if 'uploaded_image' in st.session_state:
                portion_multiplier = st.session_state.classifier.estimate_portion_size(st.session_state.uploaded_image)
                st.markdown(f"**Estimated Portion Size:** {portion_multiplier:.1f}x normal")
                
                # Adjusted nutrition
                st.markdown("### 📏 Adjusted Nutrition (estimated portion)")
                
                col_a1, col_a2, col_a3, col_a4 = st.columns(4)
                
                with col_a1:
                    st.metric("Calories", f"{nutrition['calories'] * portion_multiplier:.0f}")
                
                with col_a2:
                    st.metric("Protein", f"{nutrition['protein'] * portion_multiplier:.1f}g")
                
                with col_a3:
                    st.metric("Carbs", f"{nutrition['carbs'] * portion_multiplier:.1f}g")
                
                with col_a4:
                    st.metric("Fat", f"{nutrition['fat'] * portion_multiplier:.1f}g")
            
            # Alternative predictions
            if len(predictions) > 1:
                st.markdown("### 🔍 Alternative Predictions")
                for i, pred in enumerate(predictions[1:4], 1):
                    st.markdown(f"{i}. **{pred['class'].title()}** ({pred['confidence']:.1%})")
    
    # Bottom section for additional features
    st.markdown("---")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.subheader("📈 Daily Summary")
        st.info("Track your daily nutrition intake")
        # Placeholder for daily summary feature
        
    with col4:
        st.subheader("🎯 Health Goals")
        st.info("Set and monitor nutrition goals")
        # Placeholder for health goals feature
        
    with col5:
        st.subheader("📱 Save Results")
        st.info("Save classification results to your profile")
        # Placeholder for save feature

if __name__ == "__main__":
    main()
