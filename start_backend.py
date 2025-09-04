#!/usr/bin/env python3
"""
Startup script for the Health AI Twin Backend
"""

import uvicorn
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Start the FastAPI backend server"""
    print("🏥 Starting Health AI Twin Backend...")
    
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8001))  # Changed to 8001 to avoid conflicts
    reload = os.getenv("API_DEBUG", "True").lower() == "true"
    
    print(f"📍 Server will run on http://{host}:{port}")
    print(f"📚 API Documentation will be available at http://{host}:{port}/docs")
    print(f"🔄 Auto-reload: {'Enabled' if reload else 'Disabled'}")
    
    # Start the server
    uvicorn.run(
        "backend.app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
