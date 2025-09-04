#!/usr/bin/env python3
"""
Startup script for Health AI Twin - Part 4: AI Services
"""

import uvicorn
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Start the Part 4 FastAPI backend server"""
    print("🏥 Starting Health AI Twin - Part 4: AI Services...")
    
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8006))
    reload = os.getenv("API_DEBUG", "True").lower() == "true"
    
    print(f"📍 Server will run on http://{host}:{port}")
    print(f"📚 API Documentation will be available at http://{host}:{port}/docs")
    print(f"🔄 Auto-reload: {'Enabled' if reload else 'Disabled'}")
    print("🎯 Part 4: AI Services")
    print("   - LangChain Virtual Doctor")
    print("   - JWT Authentication")
    print("   - Health Analysis")
    print("   - User Management")
    
    # Start the server
    uvicorn.run(
        "backend.app.main_part4:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
