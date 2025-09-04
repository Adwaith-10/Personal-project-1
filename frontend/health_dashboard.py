#!/usr/bin/env python3
"""
Health AI Twin Dashboard
A comprehensive Streamlit dashboard for health monitoring and virtual doctor consultations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Health AI Twin Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
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
        margin: 0.5rem 0;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .danger-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 0.5rem 0;
    }
    .success-card {
        background-color: #d1edff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .doctor-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

class HealthDashboard:
    """Health AI Twin Dashboard class"""
    
    def __init__(self):
        self.patients = []
        self.selected_patient = None
        self.wearable_data = []
        self.lab_reports = []
        self.health_predictions = []
        self.food_logs = []
        self.chat_history = []
        
    def get_patients(self) -> List[Dict]:
        """Get list of patients from API"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/v1/patients")
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "patients" in data:
                    return data["patients"]
                return data
            return []
        except Exception as e:
            st.error(f"Error fetching patients: {e}")
            return []
    
    def get_wearable_data(self, patient_id: str, days: int = 30) -> List[Dict]:
        """Get wearable data for a patient"""
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/v1/wearable-data/{patient_id}",
                params={"days": days}
            )
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            st.error(f"Error fetching wearable data: {e}")
            return []
    
    def get_lab_reports(self, patient_id: str) -> List[Dict]:
        """Get lab reports for a patient"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/v1/lab-reports/{patient_id}")
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            st.error(f"Error fetching lab reports: {e}")
            return []
    
    def get_health_predictions(self, patient_id: str) -> List[Dict]:
        """Get health predictions for a patient"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/v1/health-prediction/predictions", 
                                 params={"patient_id": patient_id})
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            st.error(f"Error fetching health predictions: {e}")
            return []
    
    def get_food_logs(self, patient_id: str) -> List[Dict]:
        """Get food logs for a patient"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/v1/food-classification/logs/{patient_id}")
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            st.error(f"Error fetching food logs: {e}")
            return []
    
    def get_health_monitoring_report(self, patient_id: str) -> Optional[Dict]:
        """Get latest health monitoring report"""
        try:
            response = requests.post(f"{API_BASE_URL}/api/v1/health-monitoring/monitor/{patient_id}")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Error fetching health monitoring report: {e}")
            return None
    
    def chat_with_doctor(self, patient_id: str, message: str, session_id: str = None) -> Optional[Dict]:
        """Chat with virtual doctor"""
        try:
            chat_request = {
                "patient_id": patient_id,
                "message": message,
                "session_id": session_id,
                "include_health_data": True,
                "include_trends": True
            }
            
            response = requests.post(
                f"{API_BASE_URL}/api/v1/virtual-doctor/chat",
                json=chat_request
            )
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Error chatting with doctor: {e}")
            return None
    
    def create_heart_rate_chart(self, data: List[Dict]) -> go.Figure:
        """Create heart rate trend chart"""
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Extract heart rate data
        heart_rate_data = []
        for _, row in df.iterrows():
            if 'heart_rate' in row and row['heart_rate']:
                hr_data = row['heart_rate']
                heart_rate_data.append({
                    'date': row['date'],
                    'average': hr_data.get('average', 0),
                    'min': hr_data.get('min', 0),
                    'max': hr_data.get('max', 0),
                    'resting': hr_data.get('resting', 0)
                })
        
        if not heart_rate_data:
            return go.Figure()
        
        hr_df = pd.DataFrame(heart_rate_data)
        
        fig = go.Figure()
        
        # Add heart rate lines
        fig.add_trace(go.Scatter(
            x=hr_df['date'],
            y=hr_df['average'],
            mode='lines+markers',
            name='Average HR',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=hr_df['date'],
            y=hr_df['resting'],
            mode='lines+markers',
            name='Resting HR',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=6)
        ))
        
        # Add normal range bands
        fig.add_hline(y=60, line_dash="dash", line_color="green", 
                     annotation_text="Normal Range (60-100)")
        fig.add_hline(y=100, line_dash="dash", line_color="green")
        
        fig.update_layout(
            title="Heart Rate Trends",
            xaxis_title="Date",
            yaxis_title="Heart Rate (bpm)",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_sleep_chart(self, data: List[Dict]) -> go.Figure:
        """Create sleep quality chart"""
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Extract sleep data
        sleep_data = []
        for _, row in df.iterrows():
            if 'sleep' in row and row['sleep']:
                sleep_info = row['sleep']
                sleep_data.append({
                    'date': row['date'],
                    'total_minutes': sleep_info.get('total_minutes', 0),
                    'deep_sleep': sleep_info.get('deep_sleep_minutes', 0),
                    'rem_sleep': sleep_info.get('rem_sleep_minutes', 0),
                    'light_sleep': sleep_info.get('light_sleep_minutes', 0),
                    'quality_score': sleep_info.get('quality_score', 0)
                })
        
        if not sleep_data:
            return go.Figure()
        
        sleep_df = pd.DataFrame(sleep_data)
        
        # Create subplot for sleep duration and quality
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Sleep Duration', 'Sleep Quality Score'),
            vertical_spacing=0.1
        )
        
        # Sleep duration
        fig.add_trace(
            go.Bar(
                x=sleep_df['date'],
                y=sleep_df['total_minutes'] / 60,  # Convert to hours
                name='Total Sleep',
                marker_color='#1f77b4'
            ),
            row=1, col=1
        )
        
        # Add recommended sleep range
        fig.add_hline(y=7, line_dash="dash", line_color="green", 
                     annotation_text="Recommended (7-9h)", row=1, col=1)
        fig.add_hline(y=9, line_dash="dash", line_color="green", row=1, col=1)
        
        # Sleep quality
        fig.add_trace(
            go.Scatter(
                x=sleep_df['date'],
                y=sleep_df['quality_score'],
                mode='lines+markers',
                name='Quality Score',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Sleep Analysis",
            height=500,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Hours", row=1, col=1)
        fig.update_yaxes(title_text="Quality Score (0-100)", row=2, col=1)
        
        return fig
    
    def create_nutrition_chart(self, data: List[Dict]) -> go.Figure:
        """Create nutrition intake chart"""
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Extract nutrition data
        nutrition_data = []
        for _, row in df.iterrows():
            if 'nutrition_info' in row and row['nutrition_info']:
                nutrition = row['nutrition_info']
                nutrition_data.append({
                    'date': row['timestamp'].date(),
                    'calories': nutrition.get('calories', 0),
                    'protein': nutrition.get('protein', 0),
                    'carbs': nutrition.get('carbs', 0),
                    'fat': nutrition.get('fat', 0),
                    'fiber': nutrition.get('fiber', 0)
                })
        
        if not nutrition_data:
            return go.Figure()
        
        nutrition_df = pd.DataFrame(nutrition_data)
        
        # Group by date and sum daily intake
        daily_nutrition = nutrition_df.groupby('date').agg({
            'calories': 'sum',
            'protein': 'sum',
            'carbs': 'sum',
            'fat': 'sum',
            'fiber': 'sum'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Calories', 'Protein Intake', 'Carbohydrates', 'Fat Intake'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Calories
        fig.add_trace(
            go.Bar(
                x=daily_nutrition['date'],
                y=daily_nutrition['calories'],
                name='Calories',
                marker_color='#1f77b4'
            ),
            row=1, col=1
        )
        fig.add_hline(y=2000, line_dash="dash", line_color="green", 
                     annotation_text="Recommended", row=1, col=1)
        
        # Protein
        fig.add_trace(
            go.Bar(
                x=daily_nutrition['date'],
                y=daily_nutrition['protein'],
                name='Protein (g)',
                marker_color='#ff7f0e'
            ),
            row=1, col=2
        )
        fig.add_hline(y=50, line_dash="dash", line_color="green", 
                     annotation_text="Recommended", row=1, col=2)
        
        # Carbs
        fig.add_trace(
            go.Bar(
                x=daily_nutrition['date'],
                y=daily_nutrition['carbs'],
                name='Carbs (g)',
                marker_color='#2ca02c'
            ),
            row=2, col=1
        )
        fig.add_hline(y=250, line_dash="dash", line_color="green", 
                     annotation_text="Recommended", row=2, col=1)
        
        # Fat
        fig.add_trace(
            go.Bar(
                x=daily_nutrition['date'],
                y=daily_nutrition['fat'],
                name='Fat (g)',
                marker_color='#d62728'
            ),
            row=2, col=2
        )
        fig.add_hline(y=65, line_dash="dash", line_color="green", 
                     annotation_text="Recommended", row=2, col=2)
        
        fig.update_layout(
            title="Daily Nutrition Intake",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_health_predictions_chart(self, data: List[Dict]) -> go.Figure:
        """Create health predictions chart"""
        if not data:
            return go.Figure()
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Extract prediction data
        prediction_data = []
        for _, row in df.iterrows():
            predictions = row.get('predictions', [])
            for pred in predictions:
                prediction_data.append({
                    'date': row['timestamp'],
                    'metric': pred.get('metric', ''),
                    'predicted_value': pred.get('predicted_value', 0),
                    'confidence': pred.get('confidence', 0),
                    'status': pred.get('status', ''),
                    'unit': pred.get('unit', '')
                })
        
        if not prediction_data:
            return go.Figure()
        
        pred_df = pd.DataFrame(prediction_data)
        
        # Create subplot for each metric
        metrics = pred_df['metric'].unique()
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            return go.Figure()
        
        fig = make_subplots(
            rows=n_metrics, cols=1,
            subplot_titles=[f"{metric.upper()} Predictions" for metric in metrics],
            vertical_spacing=0.1
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, metric in enumerate(metrics):
            metric_data = pred_df[pred_df['metric'] == metric]
            
            fig.add_trace(
                go.Scatter(
                    x=metric_data['date'],
                    y=metric_data['predicted_value'],
                    mode='lines+markers',
                    name=f'{metric.upper()}',
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6)
                ),
                row=i+1, col=1
            )
            
            # Add normal range if available
            if metric == 'ldl':
                fig.add_hline(y=100, line_dash="dash", line_color="red", 
                             annotation_text="High Risk (>100)", row=i+1, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="green", 
                             annotation_text="Optimal (<70)", row=i+1, col=1)
            elif metric == 'glucose':
                fig.add_hline(y=100, line_dash="dash", line_color="red", 
                             annotation_text="High (>100)", row=i+1, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="green", 
                             annotation_text="Normal (70-100)", row=i+1, col=1)
            elif metric == 'hemoglobin':
                fig.add_hline(y=17.5, line_dash="dash", line_color="red", 
                             annotation_text="High (>17.5)", row=i+1, col=1)
                fig.add_hline(y=13.5, line_dash="dash", line_color="green", 
                             annotation_text="Normal (13.5-17.5)", row=i+1, col=1)
        
        fig.update_layout(
            title="Health Predictions Over Time",
            height=300 * n_metrics,
            showlegend=False
        )
        
        return fig
    
    def analyze_trends(self, data: List[Dict]) -> List[Dict]:
        """Analyze health trends and generate warnings"""
        warnings = []
        
        if not data:
            return warnings
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Analyze heart rate trends
        if 'heart_rate' in df.columns:
            hr_data = []
            for _, row in df.iterrows():
                if 'heart_rate' in row and row['heart_rate']:
                    hr_data.append(row['heart_rate'].get('average', 0))
            
            if len(hr_data) >= 7:
                recent_avg = np.mean(hr_data[-7:])
                if recent_avg > 100:
                    warnings.append({
                        'type': 'warning',
                        'metric': 'Heart Rate',
                        'message': f'Average heart rate is elevated ({recent_avg:.1f} bpm). Consider stress management and exercise.',
                        'severity': 'moderate'
                    })
                elif recent_avg < 50:
                    warnings.append({
                        'type': 'warning',
                        'metric': 'Heart Rate',
                        'message': f'Heart rate is unusually low ({recent_avg:.1f} bpm). Consult a healthcare provider.',
                        'severity': 'high'
                    })
        
        # Analyze sleep trends
        if 'sleep' in df.columns:
            sleep_data = []
            for _, row in df.iterrows():
                if 'sleep' in row and row['sleep']:
                    sleep_data.append(row['sleep'].get('total_minutes', 0))
            
            if len(sleep_data) >= 7:
                recent_avg = np.mean(sleep_data[-7:]) / 60  # Convert to hours
                if recent_avg < 6:
                    warnings.append({
                        'type': 'warning',
                        'metric': 'Sleep',
                        'message': f'Average sleep duration is insufficient ({recent_avg:.1f} hours). Aim for 7-9 hours per night.',
                        'severity': 'moderate'
                    })
                elif recent_avg > 10:
                    warnings.append({
                        'type': 'info',
                        'metric': 'Sleep',
                        'message': f'Excessive sleep duration ({recent_avg:.1f} hours). Consider underlying health issues.',
                        'severity': 'low'
                    })
        
        return warnings
    
    def display_warnings(self, warnings: List[Dict]):
        """Display health warnings"""
        if not warnings:
            st.success("✅ No health warnings detected. Your health metrics are within normal ranges.")
            return
        
        st.subheader("⚠️ Health Warnings")
        
        for warning in warnings:
            if warning['severity'] == 'high':
                st.markdown(f"""
                <div class="danger-card">
                    <strong>🚨 {warning['metric']}:</strong> {warning['message']}
                </div>
                """, unsafe_allow_html=True)
            elif warning['severity'] == 'moderate':
                st.markdown(f"""
                <div class="warning-card">
                    <strong>⚠️ {warning['metric']}:</strong> {warning['message']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <strong>ℹ️ {warning['metric']}:</strong> {warning['message']}
                </div>
                """, unsafe_allow_html=True)

def main():
    """Main dashboard function"""
    
    # Initialize dashboard
    dashboard = HealthDashboard()
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Health AI Twin Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for patient selection
    st.sidebar.header("👤 Patient Selection")
    
    # Get patients
    patients = dashboard.get_patients()
    
    if not patients:
        st.error("No patients found. Please ensure the API server is running and patients exist.")
        return
    
    # Patient selection
    patient_names = [f"{p.get('name', 'Unknown')} ({p.get('_id', 'No ID')})" for p in patients]
    selected_patient_name = st.sidebar.selectbox("Select Patient:", patient_names)
    
    # Get selected patient ID
    selected_patient_id = None
    for patient in patients:
        if f"{patient.get('name', 'Unknown')} ({patient.get('_id', 'No ID')})" == selected_patient_name:
            selected_patient_id = patient['_id']
            break
    
    if not selected_patient_id:
        st.error("Invalid patient selection.")
        return
    
    # Date range selection
    st.sidebar.header("📅 Date Range")
    days = st.sidebar.slider("Days to display:", min_value=7, max_value=90, value=30)
    
    # Load data
    with st.spinner("Loading health data..."):
        wearable_data = dashboard.get_wearable_data(selected_patient_id, days)
        lab_reports = dashboard.get_lab_reports(selected_patient_id)
        health_predictions = dashboard.get_health_predictions(selected_patient_id)
        food_logs = dashboard.get_food_logs(selected_patient_id)
        monitoring_report = dashboard.get_health_monitoring_report(selected_patient_id)
    
    # Main dashboard layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Health Overview", 
        "💓 Heart Rate", 
        "😴 Sleep", 
        "🍎 Nutrition", 
        "👩‍⚕️ Virtual Doctor"
    ])
    
    # Tab 1: Health Overview
    with tab1:
        st.header("📊 Health Overview")
        
        # Display monitoring report if available
        if monitoring_report:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Overall Health Status",
                    monitoring_report.get('overall_health_status', 'Unknown')
                )
            
            with col2:
                st.metric(
                    "Risk Level",
                    monitoring_report.get('risk_level', 'Unknown').title()
                )
            
            with col3:
                st.metric(
                    "Deviations Found",
                    monitoring_report.get('deviations_found', 0)
                )
            
            with col4:
                st.metric(
                    "Critical Deviations",
                    monitoring_report.get('critical_deviations', 0)
                )
            
            # Display warnings
            warnings = dashboard.analyze_trends(wearable_data)
            dashboard.display_warnings(warnings)
            
            # Display recommendations
            recommendations = monitoring_report.get('recommendations', [])
            if recommendations:
                st.subheader("💡 Health Recommendations")
                for i, rec in enumerate(recommendations[:5], 1):
                    st.markdown(f"{i}. {rec}")
        
        # Health predictions chart
        if health_predictions:
            st.subheader("🔮 Health Predictions")
            pred_fig = dashboard.create_health_predictions_chart(health_predictions)
            st.plotly_chart(pred_fig, use_container_width=True)
    
    # Tab 2: Heart Rate
    with tab2:
        st.header("💓 Heart Rate Analysis")
        
        if wearable_data:
            hr_fig = dashboard.create_heart_rate_chart(wearable_data)
            st.plotly_chart(hr_fig, use_container_width=True)
            
            # Heart rate statistics
            hr_stats = []
            for data in wearable_data:
                if 'heart_rate' in data and data['heart_rate']:
                    hr_stats.append(data['heart_rate'].get('average', 0))
            
            if hr_stats:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average HR", f"{np.mean(hr_stats):.1f} bpm")
                with col2:
                    st.metric("Min HR", f"{min(hr_stats):.1f} bpm")
                with col3:
                    st.metric("Max HR", f"{max(hr_stats):.1f} bpm")
        else:
            st.info("No heart rate data available for the selected period.")
    
    # Tab 3: Sleep
    with tab3:
        st.header("😴 Sleep Analysis")
        
        if wearable_data:
            sleep_fig = dashboard.create_sleep_chart(wearable_data)
            st.plotly_chart(sleep_fig, use_container_width=True)
            
            # Sleep statistics
            sleep_stats = []
            for data in wearable_data:
                if 'sleep' in data and data['sleep']:
                    sleep_stats.append(data['sleep'].get('total_minutes', 0))
            
            if sleep_stats:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Sleep", f"{np.mean(sleep_stats)/60:.1f} hours")
                with col2:
                    st.metric("Min Sleep", f"{min(sleep_stats)/60:.1f} hours")
                with col3:
                    st.metric("Max Sleep", f"{max(sleep_stats)/60:.1f} hours")
        else:
            st.info("No sleep data available for the selected period.")
    
    # Tab 4: Nutrition
    with tab4:
        st.header("🍎 Nutrition Analysis")
        
        if food_logs:
            nutrition_fig = dashboard.create_nutrition_chart(food_logs)
            st.plotly_chart(nutrition_fig, use_container_width=True)
            
            # Nutrition statistics
            if food_logs:
                total_calories = sum(log.get('nutrition_info', {}).get('calories', 0) for log in food_logs)
                total_protein = sum(log.get('nutrition_info', {}).get('protein', 0) for log in food_logs)
                total_carbs = sum(log.get('nutrition_info', {}).get('carbs', 0) for log in food_logs)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Calories", f"{total_calories:.0f}")
                with col2:
                    st.metric("Total Protein", f"{total_protein:.1f}g")
                with col3:
                    st.metric("Total Carbs", f"{total_carbs:.1f}g")
        else:
            st.info("No nutrition data available for the selected period.")
    
    # Tab 5: Virtual Doctor
    with tab5:
        st.header("👩‍⚕️ Virtual Doctor Consultation")
        
        # Initialize session state for chat
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'session_id' not in st.session_state:
            st.session_state.session_id = None
        
        # Doctor profile
        st.subheader("👩‍⚕️ Dr. Sarah Chen - Lifestyle Medicine Specialist")
        st.markdown("""
        **Credentials:**
        - MD - Harvard Medical School
        - Board Certified in Internal Medicine
        - Fellow of the American College of Lifestyle Medicine
        - Certified in Functional Medicine
        
        **Expertise:** Preventive Medicine, Nutrition Science, Exercise Physiology, Sleep Medicine
        """)
        
        # Chat interface
        st.subheader("💬 Ask Your Health Questions")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message doctor-message">
                    <strong>Dr. Chen:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        user_message = st.text_area("Type your health question:", height=100)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Send", type="primary"):
                if user_message.strip():
                    # Add user message to history
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_message,
                        'timestamp': datetime.now()
                    })
                    
                    # Get doctor response
                    with st.spinner("Dr. Chen is analyzing your health data..."):
                        response = dashboard.chat_with_doctor(
                            selected_patient_id, 
                            user_message, 
                            st.session_state.session_id
                        )
                    
                    if response:
                        # Update session ID
                        st.session_state.session_id = response.get('session_id')
                        
                        # Add doctor response to history
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response.get('message', 'Sorry, I could not process your request.'),
                            'timestamp': datetime.now()
                        })
                        
                        # Display health insights if available
                        insights = response.get('health_insights', [])
                        if insights:
                            st.subheader("💡 Health Insights")
                            for insight in insights:
                                st.info(insight)
                        
                        # Display recommendations if available
                        recommendations = response.get('recommendations', [])
                        if recommendations:
                            st.subheader("💡 Recommendations")
                            for rec in recommendations:
                                st.success(rec)
                        
                        # Display urgency level
                        urgency = response.get('urgency_level', 'normal')
                        if urgency == 'high':
                            st.error("🚨 High urgency - Please consult a healthcare provider immediately.")
                        elif urgency == 'moderate':
                            st.warning("⚠️ Moderate urgency - Consider scheduling a consultation.")
                        
                        st.rerun()
                    else:
                        st.error("Failed to get response from virtual doctor. Please try again.")
        
        with col2:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.session_id = None
                st.rerun()
        
        # Quick question suggestions
        st.subheader("💡 Quick Questions")
        quick_questions = [
            "How can I improve my heart rate variability?",
            "What should I do to get better sleep?",
            "How can I optimize my nutrition for better health?",
            "What lifestyle changes would you recommend based on my data?",
            "Should I be concerned about my current health trends?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(quick_questions):
            with cols[i % 2]:
                if st.button(question, key=f"quick_{i}"):
                    st.session_state.quick_question = question
                    st.rerun()

if __name__ == "__main__":
    main()
