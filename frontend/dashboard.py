#!/usr/bin/env python3
"""
Health AI Twin - Part 5: Frontend Dashboard
Streamlit Dashboard for Health Monitoring and AI Consultation
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import io
from PIL import Image

# Configuration
API_BASE_URL = "http://localhost:8006"  # Part 4 AI Services
ML_API_URL = "http://localhost:8005"    # Part 3 ML Pipeline
DATA_API_URL = "http://localhost:8004"  # Part 2 Data Processing

# Page configuration
st.set_page_config(
    page_title="Health AI Twin Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .health-score {
        font-size: 2rem;
        font-weight: bold;
        color: #28a745;
    }
    .warning {
        color: #ffc107;
        font-weight: bold;
    }
    .danger {
        color: #dc3545;
        font-weight: bold;
    }
    .food-upload {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = None
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'food_log' not in st.session_state:
    st.session_state.food_log = []

def check_api_health():
    """Check health of all API services"""
    services = {
        "Core Infrastructure": "http://localhost:8003/health",
        "Data Processing": "http://localhost:8004/health", 
        "ML Pipeline": "http://localhost:8005/health",
        "AI Services": "http://localhost:8006/health"
    }
    
    status = {}
    for service_name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            status[service_name] = "🟢 Online" if response.status_code == 200 else "🔴 Offline"
        except:
            status[service_name] = "🔴 Offline"
    
    return status

def login_user(email, password):
    """Login user and get JWT token"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/auth/login",
            json={"email": email, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.auth_token = data["access_token"]
            st.session_state.current_user = data["user"]
            return True
        else:
            st.error("Login failed. Please check your credentials.")
            return False
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return False

def register_user(user_data):
    """Register new user"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/auth/register",
            json=user_data
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.auth_token = data["access_token"]
            st.session_state.current_user = data["user"]
            return True
        else:
            st.error("Registration failed. Please try again.")
            return False
    except Exception as e:
        st.error(f"Registration error: {str(e)}")
        return False

def upload_food_photo(image_file, meal_type):
    """Upload food photo for classification"""
    try:
        # Prepare the file for upload
        files = {"file": ("food_image.jpg", image_file.getvalue(), "image/jpeg")}
        data = {"meal_type": meal_type}
        
        response = requests.post(
            f"{DATA_API_URL}/api/v1/food-classification/classify",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            st.error(f"Failed to classify food: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error uploading food photo: {str(e)}")
        return None

def get_health_analysis():
    """Get comprehensive health analysis"""
    try:
        headers = {"Authorization": f"Bearer {st.session_state.auth_token}"}
        response = requests.get(
            f"{API_BASE_URL}/api/v1/virtual-doctor/analysis",
            headers=headers
        )
        if response.status_code == 200:
            return response.json()["data"]
        else:
            return None
    except Exception as e:
        st.error(f"Error getting health analysis: {str(e)}")
        return None

def chat_with_doctor(question, include_food_data=True):
    """Chat with virtual doctor with enhanced context"""
    try:
        # Check if user is authenticated
        if not st.session_state.auth_token:
            st.error("Please login first to chat with the virtual doctor")
            return None
        
        headers = {"Authorization": f"Bearer {st.session_state.auth_token}"}
        
        # Prepare enhanced context with food data
        context = {
            "question": question,
            "include_food_data": include_food_data,
            "recent_food_log": st.session_state.food_log[-5:] if st.session_state.food_log else []
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/virtual-doctor/chat",
            json=context,
            headers=headers
        )
        if response.status_code == 200:
            return response.json()["data"]
        elif response.status_code == 401:
            st.error("Authentication failed. Please login again.")
            st.session_state.auth_token = None
            return None
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error chatting with doctor: {str(e)}")
        return None

def generate_sample_health_data():
    """Generate sample health data for visualization"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    
    data = {
        'date': dates,
        'heart_rate': [70 + i % 10 for i in range(len(dates))],
        'steps': [8000 + (i * 100) % 3000 for i in range(len(dates))],
        'sleep_hours': [7.5 + (i % 3) * 0.5 for i in range(len(dates))],
        'calories': [2000 + (i % 5) * 100 for i in range(len(dates))],
        'ldl': [120 + (i % 4) * 5 for i in range(len(dates))],
        'glucose': [95 + (i % 3) * 3 for i in range(len(dates))]
    }
    
    return pd.DataFrame(data)

# Food Tracking Page
def food_tracking_page():
    """Food tracking and photo upload page"""
    st.markdown('<h1 class="main-header">🍎 Food Tracking & Photo Analysis</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Upload Food Photo")
        
        # Food photo upload
        uploaded_file = st.file_uploader(
            "Take a photo of your food",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo of your meal for AI analysis"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Your food photo", use_column_width=True)
            
            # Meal type selection
            meal_type = st.selectbox(
                "What meal is this?",
                ["Breakfast", "Lunch", "Dinner", "Snack"]
            )
            
            # Analyze button
            if st.button("🍽️ Analyze Food"):
                with st.spinner("Analyzing your food..."):
                    result = upload_food_photo(uploaded_file, meal_type)
                    
                    if result:
                        st.success("✅ Food analyzed successfully!")
                        
                        # Display results
                        st.subheader("📊 Analysis Results")
                        
                        if "classification" in result:
                            st.write(f"**Detected Food:** {result['classification']}")
                            
                            # Show confidence and category if available
                            if isinstance(result['classification'], dict):
                                st.write(f"**Category:** {result['classification'].get('category', 'Unknown').title()}")
                                st.write(f"**Confidence:** {result['classification'].get('confidence', 0):.1%}")
                        
                        if "nutrition" in result:
                            nutrition = result["nutrition"]
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Calories", f"{nutrition.get('calories', 0)} kcal")
                            with col2:
                                st.metric("Protein", f"{nutrition.get('protein', 0)}g")
                            with col3:
                                st.metric("Carbs", f"{nutrition.get('carbs', 0)}g")
                            with col4:
                                st.metric("Fat", f"{nutrition.get('fat', 0)}g")
                            
                            # Show fiber if available
                            if nutrition.get('fiber', 0) > 0:
                                st.write(f"**Fiber:** {nutrition.get('fiber', 0)}g")
                        
                        # Add to food log
                        food_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "meal_type": meal_type,
                            "food_name": result.get("classification", "Unknown"),
                            "nutrition": result.get("nutrition", {}),
                            "image_uploaded": True
                        }
                        
                        # Add category and confidence if available
                        if isinstance(result.get("classification"), dict):
                            food_entry["category"] = result["classification"].get("category", "unknown")
                            food_entry["confidence"] = result["classification"].get("confidence", 0)
                        st.session_state.food_log.append(food_entry)
                        
                        st.success("✅ Added to your food log!")
    
    with col2:
        st.subheader("📋 Today's Food Log")
        
        # Filter today's entries
        today = datetime.now().date()
        today_entries = [
            entry for entry in st.session_state.food_log 
            if datetime.fromisoformat(entry["timestamp"]).date() == today
        ]
        
        if today_entries:
            for entry in today_entries:
                # Create a nice title with category badge
                title = f"{entry['meal_type']} - {entry['food_name']}"
                if entry.get("category"):
                    category_color = {
                        "fruits": "🍎",
                        "vegetables": "🥬", 
                        "proteins": "🥩",
                        "grains": "🌾",
                        "dairy": "🥛",
                        "snacks": "🍿",
                        "beverages": "🥤"
                    }.get(entry["category"], "🍽️")
                    title = f"{category_color} {title}"
                
                with st.expander(title):
                    st.write(f"**Time:** {datetime.fromisoformat(entry['timestamp']).strftime('%H:%M')}")
                    st.write(f"**Food:** {entry['food_name']}")
                    
                    # Show category and confidence
                    if entry.get("category"):
                        st.write(f"**Category:** {entry['category'].title()}")
                    if entry.get("confidence"):
                        st.write(f"**Confidence:** {entry['confidence']:.1%}")
                    
                    if "nutrition" in entry:
                        nutrition = entry["nutrition"]
                        st.write("**Nutrition:**")
                        st.write(f"- Calories: {nutrition.get('calories', 0)} kcal")
                        st.write(f"- Protein: {nutrition.get('protein', 0)}g")
                        st.write(f"- Carbs: {nutrition.get('carbs', 0)}g")
                        st.write(f"- Fat: {nutrition.get('fat', 0)}g")
                        if nutrition.get('fiber', 0) > 0:
                            st.write(f"- Fiber: {nutrition.get('fiber', 0)}g")
        else:
            st.info("No food logged today. Upload a photo to get started!")
        
        # Daily nutrition summary
        if today_entries:
            st.subheader("📊 Daily Nutrition Summary")
            
            total_calories = sum(entry.get("nutrition", {}).get("calories", 0) for entry in today_entries)
            total_protein = sum(entry.get("nutrition", {}).get("protein", 0) for entry in today_entries)
            total_carbs = sum(entry.get("nutrition", {}).get("carbs", 0) for entry in today_entries)
            total_fat = sum(entry.get("nutrition", {}).get("fat", 0) for entry in today_entries)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Calories", f"{total_calories} kcal")
            with col2:
                st.metric("Total Protein", f"{total_protein}g")
            with col3:
                st.metric("Total Carbs", f"{total_carbs}g")
            with col4:
                st.metric("Total Fat", f"{total_fat}g")

# Enhanced Virtual Doctor Page
def virtual_doctor_page():
    """Enhanced virtual doctor with food context"""
    st.markdown('<h1 class="main-header">🤖 AI Virtual Doctor</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat with Dr. Sarah Chen")
        
        # Enhanced question input
        question = st.text_area(
            "Ask the virtual doctor about your health:",
            height=100,
            placeholder="e.g., How can I improve my diet based on my recent food intake? What should I eat to lower my cholesterol?"
        )
        
        # Include food data option
        include_food_data = st.checkbox(
            "Include my recent food data in analysis",
            value=True,
            help="This will help the doctor give more personalized advice based on your actual food intake"
        )
        
        if st.button("Send Question"):
            if question:
                with st.spinner("Dr. Sarah is analyzing your health data..."):
                    response = chat_with_doctor(question, include_food_data)
                    if response:
                        st.session_state.chat_history.append((question, response))
                        st.success("Response received!")
                        st.rerun()
    
    with col2:
        st.subheader("📝 Recent Conversations")
        
        if st.session_state.chat_history:
            for i, (question, response) in enumerate(st.session_state.chat_history[-3:]):
                with st.expander(f"Q: {question[:50]}..."):
                    st.write(f"**Question:** {question}")
                    st.write("**Answer:**")
                    
                    if isinstance(response, dict):
                        # Debug: Show response keys
                        st.write(f"**DEBUG - Response keys:** {list(response.keys())}")
                        
                        # Show conversational response first (like ChatGPT)
                        if "conversational_response" in response:
                            st.markdown("**Dr. Sarah Chen:**")
                            st.markdown(response["conversational_response"])
                        elif "recommendations" in response:
                            for rec in response["recommendations"]:
                                st.write(f"• **{rec['category']}**: {rec['recommendation']}")
                        else:
                            st.write(str(response))
                    else:
                        st.write(str(response))
        else:
            st.info("No conversations yet. Start chatting with the virtual doctor!")

# Main Dashboard
def main_dashboard():
    """Main dashboard interface"""
    st.markdown('<h1 class="main-header">🏥 Health AI Twin Dashboard</h1>', unsafe_allow_html=True)
    
    # API Health Check
    with st.expander("🔧 System Status", expanded=False):
        status = check_api_health()
        cols = st.columns(len(status))
        for i, (service, health) in enumerate(status.items()):
            with cols[i]:
                st.metric(service, health)
    
    # User Authentication
    if not st.session_state.auth_token:
        st.warning("Please login to access the dashboard")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                
                if submit and email and password:
                    if login_user(email, password):
                        st.success("Login successful!")
                        st.rerun()
        
        with tab2:
            with st.form("register_form"):
                reg_email = st.text_input("Email", key="reg_email")
                reg_password = st.text_input("Password", type="password", key="reg_password")
                reg_first_name = st.text_input("First Name", key="reg_first_name")
                reg_last_name = st.text_input("Last Name", key="reg_last_name")
                reg_submit = st.form_submit_button("Register")
                
                if reg_submit and reg_email and reg_password and reg_first_name and reg_last_name:
                    user_data = {
                        "email": reg_email,
                        "password": reg_password,
                        "first_name": reg_first_name,
                        "last_name": reg_last_name
                    }
                    if register_user(user_data):
                        st.success("Registration successful!")
                        st.rerun()
            
            # Quick test login
            st.markdown("---")
            st.subheader("🧪 Quick Test Login")
            st.write("Use these credentials for testing:")
            st.code("Email: test@example.com\nPassword: test123")
            if st.button("Quick Login"):
                if login_user("test@example.com", "test123"):
                    st.success("Test login successful!")
                    st.rerun()
        
        return
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.auth_token = None
        st.session_state.current_user = None
        st.session_state.chat_history = []
        st.session_state.food_log = []
        st.rerun()
    
    # Welcome message
    if st.session_state.current_user:
        st.sidebar.success(f"Welcome, {st.session_state.current_user['first_name']}!")
    
    # Main dashboard content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Health Overview")
        
        # Get health analysis
        health_data = get_health_analysis()
        
        if health_data:
            # Health score
            health_score = health_data.get("health_score", 75)
            st.markdown(f'<div class="metric-card"><div class="health-score">{health_score}/100</div>Overall Health Score</div>', unsafe_allow_html=True)
            
            # Key metrics
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric("Heart Rate", "72 bpm", "↓ 2 bpm")
            with metrics_col2:
                st.metric("Steps", "8,500", "↑ 500")
            with metrics_col3:
                st.metric("Sleep", "7.5 hrs", "→ 0.2 hrs")
            with metrics_col4:
                st.metric("Calories", "2,100", "↑ 100")
        
        # Health trends chart
        st.subheader("📈 Health Trends")
        df = generate_sample_health_data()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Heart Rate', 'Daily Steps', 'Sleep Hours', 'Calories'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Heart Rate
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['heart_rate'], name='Heart Rate', line=dict(color='red')),
            row=1, col=1
        )
        
        # Steps
        fig.add_trace(
            go.Bar(x=df['date'], y=df['steps'], name='Steps', marker_color='blue'),
            row=1, col=2
        )
        
        # Sleep
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['sleep_hours'], name='Sleep', line=dict(color='purple')),
            row=2, col=1
        )
        
        # Calories
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['calories'], name='Calories', line=dict(color='orange')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🤖 Quick Health Check")
        
        # Quick health questions
        quick_questions = [
            "How can I improve my diet?",
            "What exercises should I do today?",
            "How can I improve my sleep?",
            "What should I eat to lower cholesterol?",
            "How can I increase my energy levels?"
        ]
        
        selected_question = st.selectbox("Choose a quick question:", quick_questions)
        
        if st.button("Ask Doctor"):
            with st.spinner("Getting personalized advice..."):
                response = chat_with_doctor(selected_question, include_food_data=True)
                if response:
                    st.success("✅ Personalized advice received!")
                    
                    if isinstance(response, dict):
                        # Debug: Show response keys
                        st.write(f"**DEBUG - Response keys:** {list(response.keys())}")
                        
                        # Show conversational response first (like ChatGPT)
                        if "conversational_response" in response:
                            st.markdown("**Dr. Sarah Chen:**")
                            st.markdown(response["conversational_response"])
                        elif "recommendations" in response:
                            for rec in response["recommendations"]:
                                st.write(f"**{rec['category']}**: {rec['recommendation']}")
                                st.write(f"*{rec['rationale']}*")
                                st.write("**Action Steps:**")
                                for step in rec.get("actionable_steps", []):
                                    st.write(f"• {step}")
                                st.divider()
    
    # Recommendations section
    if health_data:
        st.subheader("💡 Personalized Health Recommendations")
        
        recommendations = health_data.get("recommendations", [])
        for rec in recommendations:
            with st.expander(f"{rec['category']} - {rec['priority'].title()} Priority"):
                st.write(f"**Recommendation:** {rec['recommendation']}")
                st.write(f"**Rationale:** {rec['rationale']}")
                st.write("**Actionable Steps:**")
                for step in rec.get("actionable_steps", []):
                    st.write(f"• {step}")
    
    # Risk assessment
    if health_data:
        st.subheader("⚠️ Risk Assessment")
        
        risk_assessment = health_data.get("risk_assessment", {})
        risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
        
        with risk_col1:
            risk_level = risk_assessment.get("cardiovascular_risk", "low")
            color = "green" if risk_level == "low" else "orange" if risk_level == "medium" else "red"
            st.markdown(f'<div style="color: {color}; font-weight: bold;">Cardiovascular: {risk_level.title()}</div>', unsafe_allow_html=True)
        
        with risk_col2:
            risk_level = risk_assessment.get("diabetes_risk", "low")
            color = "green" if risk_level == "low" else "orange" if risk_level == "medium" else "red"
            st.markdown(f'<div style="color: {color}; font-weight: bold;">Diabetes: {risk_level.title()}</div>', unsafe_allow_html=True)
        
        with risk_col3:
            risk_level = risk_assessment.get("obesity_risk", "low")
            color = "green" if risk_level == "low" else "orange" if risk_level == "medium" else "red"
            st.markdown(f'<div style="color: {color}; font-weight: bold;">Obesity: {risk_level.title()}</div>', unsafe_allow_html=True)
        
        with risk_col4:
            stress_level = risk_assessment.get("stress_level", "low")
            color = "green" if stress_level == "low" else "orange" if stress_level == "medium" else "red"
            st.markdown(f'<div style="color: {color}; font-weight: bold;">Stress: {stress_level.title()}</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🏥 Health AI Twin")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Food Tracking", "Virtual Doctor", "Health Analysis", "Settings"]
)

if page == "Dashboard":
    main_dashboard()
elif page == "Food Tracking":
    food_tracking_page()
elif page == "Virtual Doctor":
    virtual_doctor_page()
elif page == "Health Analysis":
    st.title("📊 Health Analysis")
    st.write("Detailed health analysis and insights will be displayed here.")
elif page == "Settings":
    st.title("⚙️ Settings")
    st.write("User preferences and system settings.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Health AI Twin Dashboard - Part 5: Frontend with Food Tracking</div>",
    unsafe_allow_html=True
)
