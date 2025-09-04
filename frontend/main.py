import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any
import sys
import os

# Add the components directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))

# Page configuration
st.set_page_config(
    page_title="Health AI Twin",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-critical {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def check_api_connection():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_patients():
    """Fetch patients from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/patients")
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

def get_patient_metrics(patient_id: str, days: int = 30):
    """Fetch health metrics for a patient"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        params = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
        
        response = requests.get(
            f"{API_BASE_URL}/api/v1/patients/{patient_id}/health-metrics",
            params=params
        )
        
        if response.status_code == 200:
            return response.json().get("metrics", [])
        return []
    except:
        return []

def create_health_dashboard():
    """Create the main health dashboard"""
    st.markdown('<h1 class="main-header">🏥 Health AI Twin Dashboard</h1>', unsafe_allow_html=True)
    
    # Check API connection
    if not check_api_connection():
        st.error("⚠️ Cannot connect to the backend API. Please make sure the FastAPI server is running on http://localhost:8000")
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Patients", "Health Metrics", "Predictions", "Settings"]
    )
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Patients":
        show_patients_page()
    elif page == "Health Metrics":
        show_health_metrics_page()
    elif page == "Predictions":
        show_predictions_page()
    elif page == "Settings":
        show_settings_page()

def show_dashboard():
    """Show the main dashboard"""
    st.header("📊 Health Overview")
    
    # Get patients data
    patients_data = get_patients()
    patients = patients_data.get("patients", []) if isinstance(patients_data, dict) else patients_data
    
    if not patients:
        st.info("No patients found. Add some patients to see the dashboard.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", len(patients))
    
    with col2:
        # Calculate average age
        ages = []
        for patient in patients:
            if patient.get('date_of_birth'):
                try:
                    dob = datetime.fromisoformat(patient['date_of_birth'].replace('Z', '+00:00'))
                    age = (datetime.now() - dob).days // 365
                    ages.append(age)
                except:
                    pass
        avg_age = sum(ages) / len(ages) if ages else 0
        st.metric("Average Age", f"{avg_age:.1f} years")
    
    with col3:
        # Count by gender
        gender_counts = {}
        for patient in patients:
            gender = patient.get('gender', 'unknown')
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
        st.metric("Most Common Gender", max(gender_counts, key=gender_counts.get) if gender_counts else "N/A")
    
    with col4:
        # Health status summary
        st.metric("Active Monitoring", len(patients))
    
    # Recent activity
    st.subheader("📈 Recent Health Activity")
    
    # Get recent metrics for the first patient (for demo)
    if patients:
        patient_id = patients[0].get('_id')
        recent_metrics = get_patient_metrics(patient_id, days=7)
        
        if recent_metrics:
            # Create a DataFrame for visualization
            df = pd.DataFrame(recent_metrics)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Heart rate chart
            if 'heart_rate' in df.columns and not df['heart_rate'].isna().all():
                fig = px.line(df, x='timestamp', y='heart_rate', 
                            title='Heart Rate Trend (Last 7 Days)',
                            labels={'heart_rate': 'Heart Rate (bpm)', 'timestamp': 'Time'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Blood pressure chart
            if 'blood_pressure_systolic' in df.columns and 'blood_pressure_diastolic' in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['blood_pressure_systolic'],
                                       mode='lines+markers', name='Systolic'))
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['blood_pressure_diastolic'],
                                       mode='lines+markers', name='Diastolic'))
                fig.update_layout(title='Blood Pressure Trend (Last 7 Days)',
                                xaxis_title='Time', yaxis_title='Blood Pressure (mmHg)',
                                height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recent health metrics available.")
    
    # Patient list preview
    st.subheader("👥 Recent Patients")
    if patients:
        # Create a simple table
        patient_data = []
        for patient in patients[:5]:  # Show first 5 patients
            patient_data.append({
                'Name': f"{patient.get('first_name', '')} {patient.get('last_name', '')}",
                'Age': calculate_age(patient.get('date_of_birth')),
                'Gender': patient.get('gender', 'N/A'),
                'Email': patient.get('email', 'N/A')
            })
        
        df = pd.DataFrame(patient_data)
        st.dataframe(df, use_container_width=True)

def show_patients_page():
    """Show the patients management page"""
    st.header("👥 Patient Management")
    
    # Add new patient form
    with st.expander("➕ Add New Patient", expanded=False):
        with st.form("add_patient_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                first_name = st.text_input("First Name")
                date_of_birth = st.date_input("Date of Birth")
                email = st.text_input("Email")
                height_cm = st.number_input("Height (cm)", min_value=50, max_value=300, value=170)
            
            with col2:
                last_name = st.text_input("Last Name")
                gender = st.selectbox("Gender", ["male", "female", "other"])
                phone = st.text_input("Phone")
                weight_kg = st.number_input("Weight (kg)", min_value=1, max_value=500, value=70)
            
            blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
            address = st.text_area("Address")
            emergency_contact = st.text_input("Emergency Contact")
            
            submitted = st.form_submit_button("Add Patient")
            
            if submitted:
                # Create patient data
                patient_data = {
                    "first_name": first_name,
                    "last_name": last_name,
                    "date_of_birth": date_of_birth.isoformat(),
                    "gender": gender,
                    "email": email,
                    "phone": phone,
                    "height_cm": height_cm,
                    "weight_kg": weight_kg,
                    "blood_type": blood_type,
                    "address": address,
                    "emergency_contact": emergency_contact
                }
                
                # Send to API
                try:
                    response = requests.post(f"{API_BASE_URL}/api/v1/patients/", json=patient_data)
                    if response.status_code == 200:
                        st.success("Patient added successfully!")
                        st.rerun()
                    else:
                        st.error(f"Failed to add patient: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Display patients
    st.subheader("📋 Patient List")
    
    patients_data = get_patients()
    patients = patients_data.get("patients", []) if isinstance(patients_data, dict) else patients_data
    
    if patients:
        # Create a search box
        search = st.text_input("🔍 Search patients by name or email")
        
        # Filter patients
        if search:
            filtered_patients = [
                p for p in patients 
                if search.lower() in f"{p.get('first_name', '')} {p.get('last_name', '')}".lower()
                or search.lower() in p.get('email', '').lower()
            ]
        else:
            filtered_patients = patients
        
        # Display patients in a table
        if filtered_patients:
            patient_data = []
            for patient in filtered_patients:
                patient_data.append({
                    'ID': patient.get('_id', 'N/A'),
                    'Name': f"{patient.get('first_name', '')} {patient.get('last_name', '')}",
                    'Age': calculate_age(patient.get('date_of_birth')),
                    'Gender': patient.get('gender', 'N/A'),
                    'Email': patient.get('email', 'N/A'),
                    'Phone': patient.get('phone', 'N/A'),
                    'Blood Type': patient.get('blood_type', 'N/A')
                })
            
            df = pd.DataFrame(patient_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No patients found matching the search criteria.")
    else:
        st.info("No patients found. Add some patients to get started.")

def show_health_metrics_page():
    """Show the health metrics page"""
    st.header("📊 Health Metrics")
    
    # Get patients for selection
    patients_data = get_patients()
    patients = patients_data.get("patients", []) if isinstance(patients_data, dict) else patients_data
    
    if not patients:
        st.info("No patients found. Add some patients to view health metrics.")
        return
    
    # Patient selection
    patient_options = {f"{p.get('first_name', '')} {p.get('last_name', '')}": p.get('_id') for p in patients}
    selected_patient_name = st.selectbox("Select Patient", list(patient_options.keys()))
    selected_patient_id = patient_options[selected_patient_name]
    
    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        days = st.selectbox("Time Range", [7, 14, 30, 60, 90], index=2)
    with col2:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Get metrics
    metrics = get_patient_metrics(selected_patient_id, days)
    
    if metrics:
        df = pd.DataFrame(metrics)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Display metrics
        st.subheader("📈 Health Trends")
        
        # Create tabs for different metrics
        tab1, tab2, tab3, tab4 = st.tabs(["Heart Rate", "Blood Pressure", "Temperature", "All Metrics"])
        
        with tab1:
            if 'heart_rate' in df.columns and not df['heart_rate'].isna().all():
                fig = px.line(df, x='timestamp', y='heart_rate', 
                            title='Heart Rate Trend',
                            labels={'heart_rate': 'Heart Rate (bpm)', 'timestamp': 'Time'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No heart rate data available.")
        
        with tab2:
            if 'blood_pressure_systolic' in df.columns and 'blood_pressure_diastolic' in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['blood_pressure_systolic'],
                                       mode='lines+markers', name='Systolic'))
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['blood_pressure_diastolic'],
                                       mode='lines+markers', name='Diastolic'))
                fig.update_layout(title='Blood Pressure Trend',
                                xaxis_title='Time', yaxis_title='Blood Pressure (mmHg)',
                                height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No blood pressure data available.")
        
        with tab3:
            if 'temperature' in df.columns and not df['temperature'].isna().all():
                fig = px.line(df, x='timestamp', y='temperature', 
                            title='Body Temperature Trend',
                            labels={'temperature': 'Temperature (°C)', 'timestamp': 'Time'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No temperature data available.")
        
        with tab4:
            st.dataframe(df, use_container_width=True)
    else:
        st.info("No health metrics available for the selected patient and time range.")

def show_predictions_page():
    """Show the ML predictions page"""
    st.header("🤖 AI Predictions")
    
    st.info("This feature is under development. ML models will be integrated here for health predictions.")
    
    # Placeholder for future ML integration
    st.subheader("🔮 Health Risk Assessment")
    
    # Mock prediction interface
    with st.form("prediction_form"):
        st.write("Enter patient health data for AI prediction:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=75)
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=200, value=120)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=130, value=80)
        
        with col2:
            temperature = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)
            oxygen_sat = st.number_input("Oxygen Saturation (%)", min_value=70.0, max_value=100.0, value=98.0, step=0.1)
            glucose = st.number_input("Glucose (mg/dL)", min_value=50.0, max_value=500.0, value=100.0, step=1.0)
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        
        submitted = st.form_submit_button("Get Prediction")
        
        if submitted:
            # Mock prediction result
            st.success("✅ Prediction completed!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Health Risk Score", "Low", delta="15%")
            
            with col2:
                st.metric("Predicted Outcome", "Healthy")
            
            with col3:
                st.metric("Confidence", "85%")

def show_settings_page():
    """Show the settings page"""
    st.header("⚙️ Settings")
    
    st.subheader("API Configuration")
    
    # API URL setting
    api_url = st.text_input("API Base URL", value=API_BASE_URL)
    
    # Test connection
    if st.button("Test Connection"):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API connection successful!")
            else:
                st.error("❌ API connection failed!")
        except Exception as e:
            st.error(f"❌ Connection error: {str(e)}")
    
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh All Data"):
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Cache"):
            st.success("Cache cleared!")

def calculate_age(date_of_birth_str):
    """Calculate age from date of birth string"""
    try:
        if date_of_birth_str:
            dob = datetime.fromisoformat(date_of_birth_str.replace('Z', '+00:00'))
            age = (datetime.now() - dob).days // 365
            return age
        return "N/A"
    except:
        return "N/A"

if __name__ == "__main__":
    create_health_dashboard()
