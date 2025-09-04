#!/usr/bin/env python3
"""
Simplified Health AI Twin Backend - Part 1: Core Infrastructure
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Health AI Twin API",
    description="Health AI Twin Backend API - Part 1: Core Infrastructure",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for database
client = None
db = None

@app.on_event("startup")
async def startup_db_client():
    """Initialize database connection on startup"""
    global client, db
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        client = AsyncIOMotorClient(mongodb_url)
        db = client.health_ai_twin
        # Test the connection
        await client.admin.command('ping')
        print("✅ Connected to MongoDB")
        print("✅ Part 1: Core Infrastructure Ready")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        # Don't raise the error, just log it for now
        print("⚠️ Running without database connection")

@app.on_event("shutdown")
async def shutdown_db_client():
    """Close database connection on shutdown"""
    global client
    if client:
        client.close()
        print("🔌 Disconnected from MongoDB")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Health AI Twin API - Part 1: Core Infrastructure",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "parts": {
            "part1": "Core Infrastructure ✅",
            "part2": "Data Processing Services (Coming Soon)",
            "part3": "ML Pipeline (Coming Soon)",
            "part4": "AI Services (Coming Soon)",
            "part5": "Frontend Dashboard (Coming Soon)"
        },
        "endpoints": {
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if client:
            # Test database connection
            await client.admin.command('ping')
            return {
                "status": "healthy",
                "database": "connected",
                "part": "Part 1: Core Infrastructure"
            }
        else:
            return {
                "status": "healthy",
                "database": "not connected",
                "part": "Part 1: Core Infrastructure"
            }
    except Exception as e:
        return {
            "status": "degraded",
            "database": f"error: {str(e)}",
            "part": "Part 1: Core Infrastructure"
        }

@app.get("/api/v1/status")
async def api_status():
    """API status endpoint"""
    return {
        "api": "Health AI Twin",
        "part": "Part 1: Core Infrastructure",
        "status": "operational",
        "features": [
            "FastAPI Framework",
            "MongoDB Connection",
            "CORS Support",
            "Health Monitoring"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
