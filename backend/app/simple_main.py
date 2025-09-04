from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(
    title="Food Vision Pro API",
    description="AI-powered food analysis and nutrition tracking",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple models for demo
class UserCreate(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    message: str

class LoginRequest(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    message: str

# In-memory storage for demo (replace with database later)
users_db = {}
current_user_id = 1

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Food Vision Pro API is running!"}

@app.get("/")
async def root():
    return {"message": "Welcome to Food Vision Pro API", "docs": "/docs"}

@app.post("/api/v1/auth/signup", response_model=UserResponse)
async def signup(user_data: UserCreate):
    """Simple signup endpoint for demo"""
    global current_user_id
    
    if user_data.email in users_db:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_id = str(current_user_id)
    current_user_id += 1
    
    users_db[user_data.email] = {
        "id": user_id,
        "email": user_data.email,
        "password": user_data.password  # In real app, hash this
    }
    
    return UserResponse(
        id=user_id,
        email=user_data.email,
        message="User created successfully"
    )

@app.post("/api/v1/auth/login", response_model=Token)
async def login(login_data: LoginRequest):
    """Simple login endpoint for demo"""
    if login_data.email not in users_db:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user = users_db[login_data.email]
    if user["password"] != login_data.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Simple token (in real app, use JWT)
    access_token = f"demo_token_{user['id']}"
    
    return Token(
        access_token=access_token,
        message="Login successful"
    )

@app.get("/api/v1/meals")
async def get_meals():
    """Demo meals endpoint"""
    return {
        "meals": [
            {
                "id": "1",
                "name": "Breakfast",
                "items": [
                    {"name": "Oatmeal", "calories": 150, "protein": 5, "carbs": 27, "fat": 3}
                ],
                "total_calories": 150,
                "created_at": "2024-01-01T08:00:00Z"
            }
        ],
        "message": "Demo meals data"
    }

@app.post("/api/v1/analyze")
async def analyze_food():
    """Demo analysis endpoint"""
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
        "message": "Demo analysis - upload an image to get real results"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
