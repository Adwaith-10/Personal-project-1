#!/usr/bin/env python3
"""
Startup script for Health AI Twin - Part 5: Frontend Dashboard
"""

import subprocess
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Start the Part 5 Streamlit frontend dashboard"""
    print("🏥 Starting Health AI Twin - Part 5: Frontend Dashboard...")
    
    # Configuration
    host = os.getenv("STREAMLIT_HOST", "localhost")
    port = int(os.getenv("STREAMLIT_PORT", 8501))
    
    print(f"📍 Dashboard will run on http://{host}:{port}")
    print("🎯 Part 5: Frontend Dashboard")
    print("   - Streamlit Web Interface")
    print("   - Health Data Visualization")
    print("   - Virtual Doctor Chat")
    print("   - User Authentication")
    print("   - Real-time Health Monitoring")
    
    # Check if required packages are installed
    try:
        import streamlit
        import plotly
        import pandas
        import requests
        print("✅ All required packages are installed")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly", "pandas", "requests"])
        print("✅ Packages installed successfully")
    
    # Start Streamlit
    dashboard_path = os.path.join("frontend", "dashboard.py")
    if not os.path.exists(dashboard_path):
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return
    
    print("🚀 Starting Streamlit dashboard...")
    print("📝 Note: The dashboard will open in your default web browser")
    print("🔄 Press Ctrl+C to stop the dashboard")
    
    # Start Streamlit with the dashboard
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", dashboard_path,
        "--server.port", str(port),
        "--server.address", host,
        "--server.headless", "true"
    ])

if __name__ == "__main__":
    main()
