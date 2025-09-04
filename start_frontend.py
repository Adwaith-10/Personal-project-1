#!/usr/bin/env python3
"""
Startup script for the Health AI Twin Frontend
"""

import subprocess
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Start the Streamlit frontend"""
    print("🏥 Starting Health AI Twin Frontend...")
    
    # Configuration
    host = os.getenv("STREAMLIT_HOST", "localhost")
    port = int(os.getenv("STREAMLIT_PORT", 8501))
    
    print(f"📍 Frontend will run on http://{host}:{port}")
    print("🌐 Make sure the backend API is running on http://localhost:8000")
    
    # Start Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "frontend/main.py",
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "true"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Frontend stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start frontend: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
