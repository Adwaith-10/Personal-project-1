#!/usr/bin/env python3
"""
Simple working backend for Food Vision Pro
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Food Vision Pro", version="1.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class UserCreate(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Food Vision Pro is running!"}

@app.get("/")
async def root():
    return {
        "message": "🍽️ Food Vision Pro API", 
        "docs": "/docs",
        "status": "running"
    }

@app.post("/api/v1/auth/signup")
async def signup(user: UserCreate):
    """Signup endpoint - accepts any email/password"""
    return {
        "id": "user_123",
        "email": user.email,
        "message": "User created successfully!"
    }

@app.post("/api/v1/auth/login")
async def login(login_data: LoginRequest):
    """Login endpoint - accepts any email/password"""
    return {
        "access_token": "demo_token_123",
        "refresh_token": "demo_refresh_123",
        "message": "Login successful!"
    }

@app.get("/api/v1/meals")
async def get_meals():
    """Demo meals data"""
    return {
        "meals": [
            {
                "id": "1",
                "name": "Breakfast",
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

@app.get("/api/v1/meals/daily-totals/{date}")
async def daily_totals_by_date(date: str):
    """Demo daily totals for specific date"""
    return {
        "date": date,
        "totals": {
            "calories": 150,
            "protein": 5,
            "carbs": 27,
            "fat": 3
        }
    }

@app.get("/api/v1/auth/me")
async def get_current_user():
    """Get current user info"""
    return {
        "id": "user_123",
        "email": "demo@example.com",
        "message": "Current user info"
    }

@app.get("/api/v1/foods/popular")
async def get_popular_foods(limit: int = 10):
    """Get popular foods"""
    return {
        "foods": [
            {"name": "Chicken Breast", "calories": 165, "protein": 31, "carbs": 0, "fat": 3.6},
            {"name": "Brown Rice", "calories": 111, "protein": 2.6, "carbs": 23, "fat": 0.9},
            {"name": "Broccoli", "calories": 34, "protein": 2.8, "carbs": 7, "fat": 0.4},
            {"name": "Salmon", "calories": 208, "protein": 25, "carbs": 0, "fat": 12},
            {"name": "Sweet Potato", "calories": 86, "protein": 1.6, "carbs": 20, "fat": 0.1}
        ],
        "total": 5
    }

@app.post("/api/v1/analyze")
async def analyze():
    """Demo food analysis"""
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
    print("🚀 Starting Food Vision Pro...")
    print("📱 Web Portal: http://localhost:8501")
    print("🔗 API: http://localhost:8000")
    print("🌐 Network API: http://192.168.0.104:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🌐 Network API Docs: http://192.168.0.104:8000/docs")
    print("💡 Use any email/password to login!")
    print("🔧 Access from other devices using: http://192.168.0.104:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
