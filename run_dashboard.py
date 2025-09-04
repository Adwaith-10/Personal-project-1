#!/usr/bin/env python3
"""
Launcher script for the Health AI Twin Dashboard
"""

import subprocess
import sys
import os
import time

def main():
    """Launch the Streamlit dashboard"""
    
    print("🏥 Health AI Twin Dashboard Launcher")
    print("=" * 50)
    
    # Check if required packages are installed
    try:
        import streamlit
        import plotly
        import pandas
        import requests
        print("✅ All required packages are installed")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install required packages:")
        print("pip install streamlit plotly pandas requests")
        return
    
    # Check if the dashboard file exists
    dashboard_path = "frontend/health_dashboard.py"
    if not os.path.exists(dashboard_path):
        print(f"❌ Dashboard file not found: {dashboard_path}")
        print("Please ensure the dashboard file exists in the frontend directory.")
        return
    
    print("🚀 Starting Health AI Twin Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8501")
    print("🔗 API should be running at: http://localhost:8000")
    print("\nPress Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        # Launch Streamlit dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            dashboard_path,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    main()
