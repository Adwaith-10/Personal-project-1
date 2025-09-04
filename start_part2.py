#!/usr/bin/env python3
"""
Startup script for Health AI Twin - Part 2: Data Processing Services
"""

import uvicorn
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Start the Part 2 FastAPI backend server"""
    print("🏥 Starting Health AI Twin - Part 2: Data Processing Services...")
    
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8004))
    reload = os.getenv("API_DEBUG", "True").lower() == "true"
    
    print(f"📍 Server will run on http://{host}:{port}")
    print(f"📚 API Documentation will be available at http://{host}:{port}/docs")
    print(f"🔄 Auto-reload: {'Enabled' if reload else 'Disabled'}")
    print("🎯 Part 2: Data Processing Services")
    print("   - Lab Report PDF Processing")
    print("   - Wearable Data Processing")
    print("   - Food Image Classification")
    
    # Start the server
    uvicorn.run(
        "backend.app.main_part2:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
