#!/usr/bin/env python3
"""
Simple demo backend for Food Vision Pro
Run this to test the web portal
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Food Vision Pro Demo")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple models
class UserCreate(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

# Demo data
demo_users = {
    "demo@example.com": {"password": "demo123", "id": "1"}
}

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Demo backend running!"}

@app.get("/")
async def root():
    return {"message": "Food Vision Pro Demo API", "docs": "/docs"}

@app.post("/api/v1/auth/signup")
async def signup(user: UserCreate):
    """Demo signup - always succeeds"""
    return {
        "id": "demo_user_123",
        "email": user.email,
        "message": "Demo user created successfully!"
    }

@app.post("/api/v1/auth/login")
async def login(login_data: LoginRequest):
    """Demo login - accepts any email/password"""
    return {
        "access_token": "demo_token_123",
        "refresh_token": "demo_refresh_123",
        "message": "Demo login successful!"
    }

@app.get("/api/v1/meals")
async def get_meals():
    """Demo meals data"""
    return {
        "meals": [
            {
                "id": "1",
                "name": "Demo Breakfast",
                "items": [
                    {
                        "name": "Oatmeal",
                        "calories": 150,
                        "protein": 5,
                        "carbs": 27,
                        "fat": 3,
                        "grams": 100
                    }
                ],
                "total_calories": 150,
                "total_protein": 5,
                "total_carbs": 27,
                "total_fat": 3,
                "created_at": "2024-01-01T08:00:00Z"
            }
        ],
        "total": 1
    }

@app.get("/api/v1/meals/daily-totals")
async def daily_totals():
    """Demo daily totals"""
    return {
        "date": "2024-01-01",
        "totals": {
            "calories": 150,
            "protein": 5,
            "carbs": 27,
            "fat": 3
        }
    }

@app.post("/api/v1/analyze")
async def analyze():
    """Demo analysis"""
    return {
        "items": [
            {
                "label": "grilled_chicken",
                "confidence": 0.85,
                "estimated_grams": 150,
                "calories": 250,
                "protein": 46,
                "carbs": 0,
                "fat": 5
            }
        ],
        "totals": {
            "calories": 250,
            "protein": 46,
            "carbs": 0,
            "fat": 5
        },
        "processing_time_ms": 1500
    }

if __name__ == "__main__":
    print("🚀 Starting Food Vision Pro Demo Backend...")
    print("📱 Web Portal: http://localhost:8501")
    print("🔗 API Docs: http://localhost:8000/docs")
    print("💡 Use any email/password to login!")
    uvicorn.run(app, host="0.0.0.0", port=8000)
