#!/usr/bin/env python3
"""
Food Vision Pro - Streamlit Web Portal

A web interface for viewing meals, analytics, and managing the Food Vision Pro system.
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional
import time

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://192.168.0.104:8000")
API_BASE_URL = f"{BACKEND_URL}/api/v1"

# Page configuration
st.set_page_config(
    page_title="Food Vision Pro",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
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
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = None
if 'user_info' not in st.session_state:
    st.session_state.user_info = None


def api_request(method: str, endpoint: str, data: Dict = None, headers: Dict = None) -> Optional[Dict]:
    """Make API request to backend"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if headers is None:
            headers = {}
        
        if st.session_state.auth_token:
            headers['Authorization'] = f"Bearer {st.session_state.auth_token}"
        
        if method.upper() == "GET":
            response = requests.get(url, headers=headers)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, headers=headers)
        elif method.upper() == "PUT":
            response = requests.put(url, json=data, headers=headers)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return None
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            st.session_state.auth_token = None
            st.session_state.user_info = None
            st.error("Authentication expired. Please log in again.")
            return None
        else:
            st.error(f"API request failed: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return None


def login_page():
    """Login page"""
    st.markdown('<h1 class="main-header">🍽️ Food Vision Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Sign in to your account")
    
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if email and password:
                response = api_request("POST", "/auth/login", {
                    "email": email,
                    "password": password
                })
                
                if response:
                    st.session_state.auth_token = response.get("access_token")
                    st.session_state.user_info = {"email": email}
                    st.success("Login successful!")
                    st.rerun()
            else:
                st.error("Please enter both email and password")
    
    st.markdown("---")
    st.markdown("Don't have an account? Contact your administrator.")


def main_dashboard():
    """Main dashboard after login"""
    st.markdown('<h1 class="main-header">🍽️ Food Vision Pro</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.selectbox(
            "Choose a page",
            ["Dashboard", "Meals", "Analytics", "Food Database", "Settings", "Logout"]
        )
        
        if page == "Logout":
            st.session_state.auth_token = None
            st.session_state.user_info = None
            st.rerun()
    
    # Page routing
    if page == "Dashboard":
        show_dashboard()
    elif page == "Meals":
        show_meals_page()
    elif page == "Analytics":
        show_analytics_page()
    elif page == "Food Database":
        show_food_database_page()
    elif page == "Settings":
        show_settings_page()


def show_dashboard():
    """Show main dashboard"""
    st.markdown("## 📊 Dashboard Overview")
    
    # Get user info
    user_response = api_request("GET", "/auth/me")
    if user_response:
        st.session_state.user_info = user_response
    
    # Welcome message
    if st.session_state.user_info:
        st.markdown(f"**Welcome back, {st.session_state.user_info.get('full_name', st.session_state.user_info.get('email', 'User'))}!**")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Today's Meals</h3>
            <h2>0</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Calories Today</h3>
            <h2>0 kcal</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Protein Today</h3>
            <h2>0g</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Meals</h3>
            <h2>0</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent activity
    st.markdown("## 📝 Recent Activity")
    
    # Placeholder for recent meals
    st.info("No recent meals found. Start by analyzing your first food image!")
    
    # Quick actions
    st.markdown("## ⚡ Quick Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📸 Analyze New Food", use_container_width=True):
            st.info("Use the mobile app to analyze food images!")
    
    with col2:
        if st.button("🔍 Search Foods", use_container_width=True):
            st.info("Use the Food Database page to search for foods!")


def show_meals_page():
    """Show meals management page"""
    st.markdown("## 🍽️ My Meals")
    
    # Date filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        from_date = st.date_input("From Date", value=datetime.now() - timedelta(days=7))
    
    with col2:
        to_date = st.date_input("To Date", value=datetime.now())
    
    with col3:
        if st.button("🔍 Filter Meals"):
            st.rerun()
    
    # Get meals
    meals_response = api_request("GET", f"/meals?from_date={from_date}&to_date={to_date}")
    
    if meals_response and meals_response.get("items"):
        meals = meals_response["items"]
        
        # Display meals
        for meal in meals:
            with st.expander(f"🍽️ {meal.get('name', 'Unnamed Meal')} - {meal.get('created_at', 'Unknown Date')}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Items:**")
                    for item in meal.get("items", []):
                        st.markdown(f"- {item.get('label', 'Unknown')}: {item.get('grams_actual', item.get('grams_estimated', 0))}g")
                
                with col2:
                    st.markdown(f"**Totals:**")
                    st.markdown(f"- Calories: {meal.get('total_calories', 0):.1f} kcal")
                    st.markdown(f"- Protein: {meal.get('total_protein', 0):.1f}g")
                    st.markdown(f"- Carbs: {meal.get('total_carbs', 0):.1f}g")
                    st.markdown(f"- Fat: {meal.get('total_fat', 0):.1f}g")
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✏️ Edit", key=f"edit_{meal.get('id')}"):
                        st.info("Edit functionality coming soon!")
                
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{meal.get('id')}"):
                        st.info("Delete functionality coming soon!")
                
                with col3:
                    if st.button("📊 View Details", key=f"details_{meal.get('id')}"):
                        st.info("Detailed view coming soon!")
    else:
        st.info("No meals found for the selected date range.")
    
    # Add new meal button
    if st.button("➕ Add New Meal", use_container_width=True):
        st.info("Use the mobile app to add new meals!")


def show_analytics_page():
    """Show analytics and insights page"""
    st.markdown("## 📈 Analytics & Insights")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    if st.button("📊 Generate Analytics"):
        # Get daily totals for the date range
        current_date = start_date
        daily_data = []
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            daily_response = api_request("GET", f"/meals/daily-totals/{date_str}")
            
            if daily_response:
                daily_data.append({
                    "date": current_date,
                    "calories": daily_response.get("totals", {}).get("calories", 0),
                    "protein": daily_response.get("totals", {}).get("protein_g", 0),
                    "carbs": daily_response.get("totals", {}).get("carbs_g", 0),
                    "fat": daily_response.get("totals", {}).get("fat_g", 0),
                    "meals_count": daily_response.get("meals_count", 0)
                })
            else:
                daily_data.append({
                    "date": current_date,
                    "calories": 0,
                    "protein": 0,
                    "carbs": 0,
                    "fat": 0,
                    "meals_count": 0
                })
            
            current_date += timedelta(days=1)
        
        if daily_data:
            # Convert to DataFrame
            df = pd.DataFrame(daily_data)
            
            # Calories over time
            st.markdown("### 🔥 Daily Calories")
            fig_calories = px.line(df, x="date", y="calories", title="Daily Calorie Intake")
            st.plotly_chart(fig_calories, use_container_width=True)
            
            # Macronutrients over time
            st.markdown("### 🥩 Macronutrients Over Time")
            fig_macros = px.line(df, x="date", y=["protein", "carbs", "fat"], 
                               title="Daily Macronutrient Intake")
            st.plotly_chart(fig_macros, use_container_width=True)
            
            # Meals per day
            st.markdown("### 🍽️ Meals Per Day")
            fig_meals = px.bar(df, x="date", y="meals_count", title="Daily Meal Count")
            st.plotly_chart(fig_meals, use_container_width=True)
            
            # Summary statistics
            st.markdown("### 📊 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_calories = df["calories"].mean()
                st.metric("Average Daily Calories", f"{avg_calories:.1f} kcal")
            
            with col2:
                avg_protein = df["protein"].mean()
                st.metric("Average Daily Protein", f"{avg_protein:.1f}g")
            
            with col3:
                avg_carbs = df["carbs"].mean()
                st.metric("Average Daily Carbs", f"{avg_carbs:.1f}g")
            
            with col4:
                avg_fat = df["fat"].mean()
                st.metric("Average Daily Fat", f"{avg_fat:.1f}g")
        else:
            st.warning("No data available for the selected date range.")


def show_food_database_page():
    """Show food database search page"""
    st.markdown("## 🔍 Food Database")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Search for foods by name or barcode", placeholder="e.g., chicken, rice, 123456789")
    
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    # Barcode lookup
    st.markdown("### 📱 Barcode Scanner")
    barcode = st.text_input("Or enter barcode directly", placeholder="123456789")
    
    if st.button("🔍 Lookup Barcode") and barcode:
        barcode_response = api_request("GET", f"/foods/barcode/{barcode}")
        if barcode_response:
            display_food_info(barcode_response)
    
    # Search results
    if search_button and search_query:
        search_response = api_request("GET", f"/foods/search?q={search_query}")
        
        if search_response and search_response.get("foods"):
            foods = search_response["foods"]
            st.markdown(f"### 📋 Search Results ({len(foods)} found)")
            
            for food in foods:
                display_food_info(food)
        else:
            st.info("No foods found matching your search.")
    
    # Popular foods
    st.markdown("### 🌟 Popular Foods")
    popular_response = api_request("GET", "/foods/popular?limit=10")
    
    if popular_response and popular_response.get("foods"):
        popular_foods = popular_response["foods"]
        
        # Create DataFrame for display
        df = pd.DataFrame(popular_foods)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Popular foods data not available.")


def display_food_info(food: Dict):
    """Display food information in a formatted way"""
    with st.expander(f"🍽️ {food.get('name', 'Unknown Food')}"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Nutrition (per 100g):**")
            if food.get("calories_per_100g"):
                st.markdown(f"- Calories: {food['calories_per_100g']} kcal")
            if food.get("protein_per_100g"):
                st.markdown(f"- Protein: {food['protein_per_100g']}g")
            if food.get("carbs_per_100g"):
                st.markdown(f"- Carbs: {food['carbs_per_100g']}g")
            if food.get("fat_per_100g"):
                st.markdown(f"- Fat: {food['fat_per_100g']}g")
        
        with col2:
            st.markdown("**Additional Info:**")
            if food.get("source"):
                st.markdown(f"- Source: {food['source']}")
            if food.get("barcode"):
                st.markdown(f"- Barcode: {food['barcode']}")
            if food.get("brand"):
                st.markdown(f"- Brand: {food['brand']}")


def show_settings_page():
    """Show settings page"""
    st.markdown("## ⚙️ Settings")
    
    st.markdown("### 🔐 Account Information")
    if st.session_state.user_info:
        st.markdown(f"**Email:** {st.session_state.user_info.get('email', 'N/A')}")
        st.markdown(f"**Full Name:** {st.session_state.user_info.get('full_name', 'N/A')}")
        st.markdown(f"**Account Type:** {'Admin' if st.session_state.user_info.get('is_admin') else 'User'}")
    
    st.markdown("### 📱 Mobile App")
    st.info("Download the Food Vision Pro mobile app to analyze food images and track your nutrition!")
    
    st.markdown("### 🔒 Privacy & Data")
    st.markdown("Your data is stored securely and used only for nutrition tracking purposes.")
    
    st.markdown("### 📞 Support")
    st.markdown("For support and questions, please contact your system administrator.")


def main():
    """Main application entry point"""
    # Check authentication
    if not st.session_state.auth_token:
        login_page()
    else:
        main_dashboard()


if __name__ == "__main__":
    main()
