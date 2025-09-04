#!/usr/bin/env python3
"""
Startup script for Health AI Twin - Part 3: ML Pipeline
"""

import uvicorn
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Start the Part 3 FastAPI backend server"""
    print("🏥 Starting Health AI Twin - Part 3: ML Pipeline...")
    
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8005))
    reload = os.getenv("API_DEBUG", "True").lower() == "true"
    
    print(f"📍 Server will run on http://{host}:{port}")
    print(f"📚 API Documentation will be available at http://{host}:{port}/docs")
    print(f"🔄 Auto-reload: {'Enabled' if reload else 'Disabled'}")
    print("🎯 Part 3: ML Pipeline")
    print("   - XGBoost Health Prediction Models")
    print("   - Model Training and Evaluation")
    print("   - Feature Engineering")
    print("   - Model Persistence")
    
    # Start the server
    uvicorn.run(
        "backend.app.main_part3:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
