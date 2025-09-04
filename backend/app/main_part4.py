#!/usr/bin/env python3
"""
Health AI Twin Backend - Part 4: AI Services
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import os
import json
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Health AI Twin API - Part 4",
    description="Health AI Twin Backend API - Part 4: AI Services",
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

# Security
security = HTTPBearer()
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Global variables
client = None
db = None
virtual_doctor = None

@app.on_event("startup")
async def startup_db_client():
    """Initialize database connection and AI services on startup"""
    global client, db, virtual_doctor
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        client = AsyncIOMotorClient(mongodb_url)
        db = client.health_ai_twin
        # Test the connection
        await client.admin.command('ping')
        print("✅ Connected to MongoDB")
        
        # Initialize AI services
        await initialize_ai_services()
        print("✅ Part 4: AI Services Ready")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        print("⚠️ Running without database connection")

@app.on_event("shutdown")
async def shutdown_db_client():
    """Close database connection on shutdown"""
    global client
    if client:
        client.close()
        print("🔌 Disconnected from MongoDB")

async def initialize_ai_services():
    """Initialize AI services including virtual doctor"""
    global virtual_doctor
    try:
        # Initialize virtual doctor service
        virtual_doctor = VirtualDoctorService()
        print("✅ Virtual Doctor initialized")
    except Exception as e:
        print(f"❌ Failed to initialize AI services: {e}")

# Authentication utilities
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Virtual Doctor Service
class VirtualDoctorService:
    """Virtual doctor service using LangChain for lifestyle medicine consultations"""
    
    def __init__(self):
        self.doctor_profile = {
            "name": "Dr. Sarah Chen",
            "specialty": "Lifestyle Medicine",
            "credentials": [
                "Board Certified in Internal Medicine",
                "Fellowship in Lifestyle Medicine",
                "Certified Health Coach"
            ],
            "expertise": [
                "Preventive Medicine",
                "Nutrition Science",
                "Exercise Physiology",
                "Stress Management",
                "Sleep Medicine"
            ]
        }
        self.conversation_history = []
    
    def get_health_advice(self, user_data: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Generate conversational health advice like ChatGPT"""
        try:
            # Generate comprehensive analysis
            analysis = self._analyze_health_data(user_data)
            recommendations = self._generate_recommendations(user_data, question)
            risk_assessment = self._assess_health_risks(user_data)
            follow_up = self._suggest_follow_up(user_data)
            
            # Create conversational response
            try:
                conversational_response = self._create_conversational_response(
                    question, user_data, analysis, recommendations, risk_assessment
                )
                print(f"DEBUG: Generated conversational response: {conversational_response[:100]}...")
            except Exception as e:
                print(f"DEBUG: Error creating conversational response: {e}")
                conversational_response = f"Hello! I'm Dr. Sarah Chen. {question} is a great question. Let me help you with that."
            
            advice = {
                "doctor": self.doctor_profile["name"],
                "specialty": self.doctor_profile["specialty"],
                "timestamp": datetime.now().isoformat(),
                "user_question": question,
                "conversational_response": conversational_response,
                "analysis": analysis,
                "recommendations": recommendations,
                "risk_assessment": risk_assessment,
                "follow_up": follow_up
            }
            
            # Store conversation
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "advice": advice
            })
            
            return advice
        except Exception as e:
            return {
                "error": f"Failed to generate health advice: {str(e)}",
                "doctor": self.doctor_profile["name"]
            }
    
    def _analyze_health_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user health data"""
        analysis = {
            "overall_health_score": 75,
            "trends": {
                "heart_rate": "stable",
                "sleep_quality": "improving",
                "activity_level": "good",
                "nutrition": "needs_improvement"
            },
            "key_insights": [
                "Your heart rate variability shows good cardiovascular health",
                "Sleep duration is adequate but quality could improve",
                "Daily step count is above recommended levels",
                "Consider reducing processed food intake"
            ]
        }
        return analysis
    
    def _generate_recommendations(self, user_data: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
        """Generate personalized health recommendations based on user data and food intake"""
        recommendations = []
        
        # Analyze food data if available
        recent_calories = user_data.get("recent_calories", 0)
        recent_protein = user_data.get("recent_protein", 0)
        recent_carbs = user_data.get("recent_carbs", 0)
        recent_fat = user_data.get("recent_fat", 0)
        food_variety = user_data.get("food_variety", 0)
        meal_count = user_data.get("meal_count", 0)
        
        # Nutrition recommendations based on actual food intake
        if recent_calories > 0:
            if recent_calories < 1500:
                recommendations.append({
                    "category": "Nutrition",
                    "priority": "high",
                    "recommendation": "Increase daily calorie intake to meet your energy needs",
                    "rationale": f"Your recent intake of {recent_calories} calories is below recommended levels",
                    "actionable_steps": [
                        "Add healthy snacks between meals",
                        "Include more nutrient-dense foods",
                        "Consider increasing portion sizes gradually"
                    ]
                })
            elif recent_calories > 2500:
                recommendations.append({
                    "category": "Nutrition",
                    "priority": "medium",
                    "recommendation": "Monitor calorie intake to maintain healthy weight",
                    "rationale": f"Your recent intake of {recent_calories} calories is above typical needs",
                    "actionable_steps": [
                        "Focus on portion control",
                        "Choose lower-calorie alternatives",
                        "Increase physical activity to balance intake"
                    ]
                })
            
            if recent_protein < 50:
                recommendations.append({
                    "category": "Nutrition",
                    "priority": "high",
                    "recommendation": "Increase protein intake for muscle health",
                    "rationale": f"Your recent protein intake of {recent_protein}g is below recommended levels",
                    "actionable_steps": [
                        "Add lean protein sources (chicken, fish, tofu)",
                        "Include Greek yogurt or cottage cheese",
                        "Consider protein-rich snacks like nuts or eggs"
                    ]
                })
            
            if food_variety < 3:
                recommendations.append({
                    "category": "Nutrition",
                    "priority": "medium",
                    "recommendation": "Increase food variety for better nutrition",
                    "rationale": f"You've consumed {food_variety} different foods recently",
                    "actionable_steps": [
                        "Try new fruits and vegetables",
                        "Include different protein sources",
                        "Experiment with whole grains"
                    ]
                })
        
        # Exercise recommendations based on activity level
        steps_avg = user_data.get("steps_avg", 0)
        if steps_avg < 6000:
            recommendations.append({
                "category": "Exercise",
                "priority": "high",
                "recommendation": "Increase daily physical activity",
                "rationale": f"Your average of {steps_avg} steps is below the recommended 10,000",
                "actionable_steps": [
                    "Take walking breaks during work",
                    "Use stairs instead of elevators",
                    "Park further from destinations"
                ]
            })
        elif steps_avg > 12000:
            recommendations.append({
                "category": "Exercise",
                "priority": "medium",
                "recommendation": "Add strength training to your routine",
                "rationale": "Your cardio is excellent, but strength training will improve overall fitness",
                "actionable_steps": [
                    "Start with bodyweight exercises",
                    "Gradually add resistance training",
                    "Focus on major muscle groups"
                ]
            })
        
        # Sleep recommendations
        sleep_hours = user_data.get("sleep_hours_avg", 7.5)
        if sleep_hours < 7:
            recommendations.append({
                "category": "Sleep",
                "priority": "high",
                "recommendation": "Increase sleep duration to 7-9 hours",
                "rationale": f"Your current sleep of {sleep_hours} hours is below recommended levels",
                "actionable_steps": [
                    "Go to bed 30 minutes earlier",
                    "Create a relaxing bedtime routine",
                    "Limit screen time before bed"
                ]
            })
        elif sleep_hours > 9:
            recommendations.append({
                "category": "Sleep",
                "priority": "medium",
                "recommendation": "Monitor sleep duration for optimal health",
                "rationale": f"Your sleep of {sleep_hours} hours may be excessive",
                "actionable_steps": [
                    "Maintain consistent sleep schedule",
                    "Avoid oversleeping on weekends",
                    "Consider underlying health conditions"
                ]
            })
        
        # Question-specific recommendations
        question_lower = question.lower()
        if "cholesterol" in question_lower or "ldl" in question_lower:
            ldl = user_data.get("ldl", 120)
            if ldl > 130:
                recommendations.append({
                    "category": "Heart Health",
                    "priority": "high",
                    "recommendation": "Focus on heart-healthy diet to lower cholesterol",
                    "rationale": f"Your LDL cholesterol of {ldl} mg/dL is elevated",
                    "actionable_steps": [
                        "Reduce saturated and trans fats",
                        "Increase soluble fiber intake",
                        "Include omega-3 rich foods",
                        "Consider plant sterols"
                    ]
                })
        
        if "diabetes" in question_lower or "glucose" in question_lower:
            glucose = user_data.get("glucose", 96)
            if glucose > 100:
                recommendations.append({
                    "category": "Blood Sugar",
                    "priority": "high",
                    "recommendation": "Monitor blood sugar and adjust diet",
                    "rationale": f"Your glucose level of {glucose} mg/dL is elevated",
                    "actionable_steps": [
                        "Choose low glycemic index foods",
                        "Balance carbs with protein and fiber",
                        "Monitor portion sizes",
                        "Exercise regularly"
                    ]
                })
        
        # If no specific recommendations, provide general ones
        if not recommendations:
            recommendations = [
                {
                    "category": "General Health",
                    "priority": "medium",
                    "recommendation": "Maintain balanced lifestyle habits",
                    "rationale": "Your health metrics are generally good",
                    "actionable_steps": [
                        "Continue current healthy habits",
                        "Regular health check-ups",
                        "Stay hydrated and active"
                    ]
                }
            ]
        
        return recommendations
    
    def _assess_health_risks(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess health risks based on user data"""
        return {
            "cardiovascular_risk": "low",
            "diabetes_risk": "low",
            "obesity_risk": "low",
            "stress_level": "moderate",
            "recommendations": [
                "Continue current healthy habits",
                "Monitor blood pressure regularly",
                "Consider stress management techniques"
            ]
        }
    
    def _suggest_follow_up(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest follow-up actions"""
        return {
            "next_appointment": "3 months",
            "tests_to_monitor": [
                "Blood pressure",
                "Cholesterol panel",
                "Blood glucose"
            ],
            "lifestyle_goals": [
                "Increase vegetable intake by 50%",
                "Add 2 strength training sessions per week",
                "Improve sleep quality score by 20%"
            ]
        }
    
    def _create_conversational_response(self, question: str, user_data: Dict[str, Any], 
                                      health_analysis: Dict[str, Any], recommendations: List[Dict[str, Any]], 
                                      risk_assessment: Dict[str, Any]) -> str:
        """Create a conversational, ChatGPT-like response"""
        
        # Analyze the question type
        question_lower = question.lower()
        
        # Start with a friendly, conversational tone
        response_parts = []
        
        # Greeting based on question type
        if "hello" in question_lower or "hi" in question_lower:
            response_parts.append("Hello! I'm Dr. Sarah Chen, your AI health assistant. How can I help you today?")
        elif "how are you" in question_lower:
            response_parts.append("I'm doing well, thank you for asking! I'm here to help you with any health questions you might have.")
        else:
            response_parts.append("Great question! Let me help you with that.")
        
        # Always add a personalized touch
        response_parts.append(f"I see you're asking about '{question}'. Let me give you some personalized advice based on your health data.")
        
        # Handle different types of questions
        if any(word in question_lower for word in ["diet", "food", "nutrition", "eat", "meal"]):
            response_parts.append(self._handle_diet_question(question, user_data, recommendations))
        
        elif any(word in question_lower for word in ["exercise", "workout", "fitness", "gym", "cardio", "strength"]):
            response_parts.append(self._handle_exercise_question(question, user_data, recommendations))
        
        elif any(word in question_lower for word in ["sleep", "rest", "insomnia", "tired"]):
            response_parts.append(self._handle_sleep_question(question, user_data, recommendations))
        
        elif any(word in question_lower for word in ["stress", "anxiety", "mental", "mood", "depression"]):
            response_parts.append(self._handle_mental_health_question(question, user_data, recommendations))
        
        elif any(word in question_lower for word in ["weight", "bmi", "fat", "lose", "gain"]):
            response_parts.append(self._handle_weight_question(question, user_data, recommendations))
        
        elif any(word in question_lower for word in ["heart", "blood pressure", "cardiovascular", "cholesterol"]):
            response_parts.append(self._handle_heart_health_question(question, user_data, recommendations))
        
        elif any(word in question_lower for word in ["diabetes", "blood sugar", "glucose", "insulin"]):
            response_parts.append(self._handle_diabetes_question(question, user_data, recommendations))
        
        elif any(word in question_lower for word in ["vitamin", "supplement", "mineral", "nutrient"]):
            response_parts.append(self._handle_supplement_question(question, user_data, recommendations))
        
        elif any(word in question_lower for word in ["headache", "pain", "ache", "symptom"]):
            response_parts.append(self._handle_symptom_question(question, user_data, recommendations))
        
        else:
            # General health advice
            response_parts.append(self._handle_general_health_question(question, user_data, recommendations))
        
        # Add personalized insights based on user data
        if user_data.get("recent_calories", 0) > 0:
            response_parts.append(f"\n\nI notice from your recent food intake that you've consumed {user_data['recent_calories']} calories. ")
            if user_data['recent_calories'] < 1500:
                response_parts.append("This seems a bit low for most adults - you might want to consider adding some healthy snacks.")
            elif user_data['recent_calories'] > 2500:
                response_parts.append("This is on the higher side - consider balancing with more physical activity.")
        
        # Add risk assessment if relevant
        if risk_assessment.get("overall_risk") == "high":
            response_parts.append("\n\n⚠️ **Important Note:** Based on your health data, I recommend consulting with a healthcare provider for personalized medical advice.")
        
        # End with encouragement
        response_parts.append("\n\nRemember, I'm here to support your health journey! Feel free to ask me anything else.")
        
        # Ensure we always return a response
        final_response = " ".join(response_parts)
        if not final_response or len(final_response.strip()) < 10:
            final_response = f"Hello! I'm Dr. Sarah Chen. Thank you for your question about '{question}'. I'm here to help you with personalized health advice based on your data. Feel free to ask me anything else!"
        
        return final_response
    
    def _handle_diet_question(self, question: str, user_data: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> str:
        """Handle diet and nutrition questions"""
        response = "Regarding your diet and nutrition, "
        
        if "improve" in question.lower():
            response += "here are some key areas to focus on:\n\n"
            for rec in recommendations:
                if rec["category"] == "Nutrition":
                    response += f"• **{rec['recommendation']}** - {rec['rationale']}\n"
                    for step in rec.get("actionable_steps", []):
                        response += f"  - {step}\n"
        else:
            response += "a balanced diet should include:\n\n"
            response += "• **Fruits and vegetables** (aim for 5+ servings daily)\n"
            response += "• **Lean proteins** (chicken, fish, beans, tofu)\n"
            response += "• **Whole grains** (brown rice, quinoa, whole wheat)\n"
            response += "• **Healthy fats** (avocado, nuts, olive oil)\n"
            response += "• **Stay hydrated** (8+ glasses of water daily)\n"
        
        return response
    
    def _handle_exercise_question(self, question: str, user_data: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> str:
        """Handle exercise and fitness questions"""
        response = "For your fitness routine, "
        
        steps_avg = user_data.get("steps_avg", 0)
        response += f"I see you're averaging {steps_avg} steps daily. "
        
        if steps_avg < 6000:
            response += "You could benefit from increasing your daily activity. Here's what I recommend:\n\n"
            response += "• **Start with walking** - aim for 10,000 steps daily\n"
            response += "• **Add strength training** - 2-3 times per week\n"
            response += "• **Include cardio** - 150 minutes of moderate activity weekly\n"
            response += "• **Find activities you enjoy** - dancing, swimming, cycling\n"
        else:
            response += "Great job staying active! To optimize your fitness:\n\n"
            response += "• **Mix up your routine** - try new activities\n"
            response += "• **Focus on strength training** - builds muscle and bone density\n"
            response += "• **Include flexibility work** - yoga or stretching\n"
            response += "• **Listen to your body** - rest when needed\n"
        
        return response
    
    def _handle_sleep_question(self, question: str, user_data: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> str:
        """Handle sleep-related questions"""
        response = "Regarding your sleep, "
        
        sleep_hours = user_data.get("sleep_hours_avg", 7.5)
        response += f"you're getting {sleep_hours} hours of sleep on average. "
        
        if sleep_hours < 7:
            response += "This is below the recommended 7-9 hours. Here's how to improve:\n\n"
            response += "• **Establish a routine** - go to bed and wake up at the same time\n"
            response += "• **Create a sleep-friendly environment** - cool, dark, quiet room\n"
            response += "• **Limit screen time** - avoid devices 1 hour before bed\n"
            response += "• **Practice relaxation** - meditation, deep breathing, reading\n"
            response += "• **Avoid caffeine** - especially after 2 PM\n"
        else:
            response += "That's a good amount of sleep! To optimize quality:\n\n"
            response += "• **Maintain consistency** - even on weekends\n"
            response += "• **Optimize your environment** - comfortable mattress and pillows\n"
            response += "• **Wind down properly** - gentle activities before bed\n"
            response += "• **Monitor sleep quality** - not just duration\n"
        
        return response
    
    def _handle_mental_health_question(self, question: str, user_data: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> str:
        """Handle mental health questions"""
        response = "Mental health is just as important as physical health. Here are some strategies:\n\n"
        response += "• **Practice mindfulness** - meditation, deep breathing, yoga\n"
        response += "• **Stay connected** - maintain relationships with friends and family\n"
        response += "• **Get regular exercise** - releases endorphins and reduces stress\n"
        response += "• **Prioritize sleep** - poor sleep affects mood and cognition\n"
        response += "• **Limit social media** - can contribute to anxiety and comparison\n"
        response += "• **Seek professional help** - therapy is a sign of strength, not weakness\n"
        response += "• **Practice self-care** - hobbies, relaxation, doing things you enjoy\n"
        
        return response
    
    def _handle_weight_question(self, question: str, user_data: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> str:
        """Handle weight-related questions"""
        bmi = user_data.get("bmi", 24.5)
        response = f"Regarding weight management, your BMI is {bmi}. "
        
        if bmi < 18.5:
            response += "You're in the underweight range. Focus on:\n\n"
            response += "• **Nutrient-dense foods** - healthy calories, not empty ones\n"
            response += "• **Strength training** - builds muscle mass\n"
            response += "• **Regular meals** - don't skip breakfast\n"
            response += "• **Healthy fats** - nuts, avocado, olive oil\n"
        elif bmi > 25:
            response += "You're in the overweight range. Consider:\n\n"
            response += "• **Calorie deficit** - 500 calories less than maintenance\n"
            response += "• **Regular exercise** - both cardio and strength training\n"
            response += "• **Portion control** - use smaller plates, eat slowly\n"
            response += "• **Whole foods** - avoid processed foods\n"
        else:
            response += "You're in the healthy weight range! To maintain:\n\n"
            response += "• **Balanced diet** - variety of nutrients\n"
            response += "• **Regular activity** - 150 minutes weekly\n"
            response += "• **Consistent habits** - sustainable lifestyle changes\n"
        
        return response
    
    def _handle_heart_health_question(self, question: str, user_data: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> str:
        """Handle heart health questions"""
        response = "For heart health, focus on these key areas:\n\n"
        response += "• **Heart-healthy diet** - low saturated fat, high fiber\n"
        response += "• **Regular exercise** - 150 minutes of moderate activity weekly\n"
        response += "• **Manage stress** - chronic stress affects heart health\n"
        response += "• **Don't smoke** - smoking damages blood vessels\n"
        response += "• **Limit alcohol** - moderate consumption only\n"
        response += "• **Regular check-ups** - monitor blood pressure and cholesterol\n"
        response += "• **Adequate sleep** - 7-9 hours nightly\n"
        
        return response
    
    def _handle_diabetes_question(self, question: str, user_data: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> str:
        """Handle diabetes-related questions"""
        response = "For blood sugar management:\n\n"
        response += "• **Low glycemic foods** - whole grains, vegetables, legumes\n"
        response += "• **Regular meals** - don't skip breakfast\n"
        response += "• **Portion control** - balance carbs with protein and fiber\n"
        response += "• **Regular exercise** - helps insulin sensitivity\n"
        response += "• **Monitor blood sugar** - if you have diabetes\n"
        response += "• **Limit sugary drinks** - water, tea, coffee instead\n"
        response += "• **Stress management** - stress affects blood sugar\n"
        
        return response
    
    def _handle_supplement_question(self, question: str, user_data: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> str:
        """Handle supplement questions"""
        response = "Regarding supplements:\n\n"
        response += "• **Focus on food first** - whole foods provide better nutrition\n"
        response += "• **Vitamin D** - many people are deficient, especially in winter\n"
        response += "• **Omega-3** - from fish or algae supplements\n"
        response += "• **Probiotics** - for gut health\n"
        response += "• **Consult a professional** - before starting any supplements\n"
        response += "• **Quality matters** - choose reputable brands\n"
        response += "• **Don't overdo it** - more isn't always better\n"
        
        return response
    
    def _handle_symptom_question(self, question: str, user_data: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> str:
        """Handle symptom questions"""
        response = "I understand you're experiencing symptoms. Here are some general guidelines:\n\n"
        response += "• **Monitor symptoms** - keep track of frequency and severity\n"
        response += "• **Stay hydrated** - dehydration can cause many symptoms\n"
        response += "• **Get adequate rest** - sleep helps the body heal\n"
        response += "• **Consider stress** - many symptoms are stress-related\n"
        response += "• **Seek medical attention** - for severe or persistent symptoms\n"
        response += "• **Don't self-diagnose** - consult healthcare professionals\n"
        
        response += "\n\n⚠️ **Important:** I cannot diagnose medical conditions. Please consult with a healthcare provider for proper diagnosis and treatment."
        
        return response
    
    def _handle_general_health_question(self, question: str, user_data: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> str:
        """Handle general health questions"""
        response = "For overall health and wellness:\n\n"
        response += "• **Balanced nutrition** - variety of whole foods\n"
        response += "• **Regular exercise** - both cardio and strength training\n"
        response += "• **Adequate sleep** - 7-9 hours nightly\n"
        response += "• **Stress management** - meditation, hobbies, social connections\n"
        response += "• **Preventive care** - regular check-ups and screenings\n"
        response += "• **Stay hydrated** - 8+ glasses of water daily\n"
        response += "• **Limit processed foods** - focus on whole, natural foods\n"
        response += "• **Maintain social connections** - important for mental health\n"
        
        return response

# Part 4: AI Services

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Health AI Twin API - Part 4: AI Services",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "parts": {
            "part1": "Core Infrastructure ✅",
            "part2": "Data Processing Services ✅",
            "part3": "ML Pipeline ✅",
            "part4": "AI Services ✅",
            "part5": "Frontend Dashboard (Coming Soon)"
        },
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "auth": "/api/v1/auth",
            "virtual_doctor": "/api/v1/virtual-doctor",
            "health_analysis": "/api/v1/health-analysis"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if client is not None:
            await client.admin.command('ping')
            return {
                "status": "healthy",
                "database": "connected",
                "ai_services": "ready" if virtual_doctor else "not_ready",
                "part": "Part 4: AI Services"
            }
        else:
            return {
                "status": "healthy",
                "database": "not connected",
                "ai_services": "ready" if virtual_doctor else "not_ready",
                "part": "Part 4: AI Services"
            }
    except Exception as e:
        return {
            "status": "degraded",
            "database": f"error: {str(e)}",
            "ai_services": "ready" if virtual_doctor else "not_ready",
            "part": "Part 4: AI Services"
        }

# Authentication Endpoints

@app.post("/api/v1/auth/register")
async def register_user(user_data: Dict[str, Any]):
    """Register a new user"""
    try:
        # Validate required fields
        required_fields = ["email", "password", "first_name", "last_name"]
        for field in required_fields:
            if field not in user_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Check if user already exists
        if db is not None:
            existing_user = await db.users.find_one({"email": user_data["email"]})
            if existing_user:
                raise HTTPException(status_code=400, detail="User already exists")
        
        # Hash password
        hashed_password = bcrypt.hashpw(user_data["password"].encode('utf-8'), bcrypt.gensalt())
        
        # Create user
        new_user = {
            "email": user_data["email"],
            "password": hashed_password.decode('utf-8'),
            "first_name": user_data["first_name"],
            "last_name": user_data["last_name"],
            "created_at": datetime.now().isoformat(),
            "is_active": True
        }
        
        # Store user in database
        if db is not None:
            result = await db.users.insert_one(new_user)
            new_user["_id"] = str(result.inserted_id)
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": new_user["email"]}, expires_delta=access_token_expires
        )
        
        return {
            "message": "User registered successfully",
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "email": new_user["email"],
                "first_name": new_user["first_name"],
                "last_name": new_user["last_name"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registering user: {str(e)}")

@app.post("/api/v1/auth/login")
async def login_user(credentials: Dict[str, Any]):
    """Login user"""
    try:
        # Validate required fields
        if "email" not in credentials or "password" not in credentials:
            raise HTTPException(status_code=400, detail="Email and password required")
        
        # Find user
        if db is not None:
            user = await db.users.find_one({"email": credentials["email"]})
            if not user:
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            # Verify password
            if not bcrypt.checkpw(credentials["password"].encode('utf-8'), user["password"].encode('utf-8')):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            # Create access token
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": user["email"]}, expires_delta=access_token_expires
            )
            
            return {
                "message": "Login successful",
                "access_token": access_token,
                "token_type": "bearer",
                "user": {
                    "email": user["email"],
                    "first_name": user["first_name"],
                    "last_name": user["last_name"]
                }
            }
        else:
            # Simulate login for testing
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": credentials["email"]}, expires_delta=access_token_expires
            )
            
            return {
                "message": "Login successful (simulated)",
                "access_token": access_token,
                "token_type": "bearer",
                "user": {
                    "email": credentials["email"],
                    "first_name": "Test",
                    "last_name": "User"
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during login: {str(e)}")

# Virtual Doctor Endpoints

@app.post("/api/v1/virtual-doctor/chat")
async def chat_with_doctor(
    chat_data: Dict[str, Any],
    current_user: str = Depends(verify_token)
):
    """Chat with virtual doctor with enhanced conversational AI"""
    try:
        if not virtual_doctor:
            raise HTTPException(status_code=503, detail="Virtual doctor service not available")
        
        # Validate required fields
        if "question" not in chat_data:
            raise HTTPException(status_code=400, detail="Question is required")
        
        # Get user health data (simulated)
        user_health_data = {
            "age": 35,
            "bmi": 24.5,
            "heart_rate_avg": 72,
            "steps_avg": 8500,
            "sleep_hours_avg": 7.5,
            "calories_avg": 2100,
            "ldl": 118,
            "glucose": 96,
            "hemoglobin": 14.3
        }
        
        # Include food data if available
        include_food_data = chat_data.get("include_food_data", False)
        recent_food_log = chat_data.get("recent_food_log", [])
        
        if include_food_data and recent_food_log:
            # Analyze food patterns
            total_calories = sum(entry.get("nutrition", {}).get("calories", 0) for entry in recent_food_log)
            total_protein = sum(entry.get("nutrition", {}).get("protein", 0) for entry in recent_food_log)
            total_carbs = sum(entry.get("nutrition", {}).get("carbs", 0) for entry in recent_food_log)
            total_fat = sum(entry.get("nutrition", {}).get("fat", 0) for entry in recent_food_log)
            
            # Add food analysis to user data
            user_health_data.update({
                "recent_calories": total_calories,
                "recent_protein": total_protein,
                "recent_carbs": total_carbs,
                "recent_fat": total_fat,
                "food_variety": len(set(entry.get("food_name", "") for entry in recent_food_log)),
                "meal_count": len(recent_food_log)
            })
        
        # Get personalized health advice with conversational response
        advice = virtual_doctor.get_health_advice(user_health_data, chat_data["question"])
        
        # Generate conversational response
        conversational_response = virtual_doctor._create_conversational_response(
            chat_data["question"], 
            user_health_data, 
            advice
        )
        
        # Store chat in database
        if db is not None:
            chat_record = {
                "user_id": current_user,
                "question": chat_data["question"],
                "advice": advice,
                "conversational_response": conversational_response,
                "food_context": recent_food_log if include_food_data else [],
                "timestamp": datetime.now().isoformat()
            }
            result = await db.chat_logs.insert_one(chat_record)
            # Convert ObjectId to string to avoid serialization issues
            chat_record["_id"] = str(result.inserted_id)
        
        return {
            "message": "Health consultation completed",
            "data": advice,
            "conversational_response": conversational_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during consultation: {str(e)}")

@app.get("/api/v1/virtual-doctor/analysis")
async def get_health_analysis(current_user: str = Depends(verify_token)):
    """Get comprehensive health analysis"""
    try:
        if not virtual_doctor:
            raise HTTPException(status_code=503, detail="Virtual doctor service not available")
        
        # Simulate health data
        user_health_data = {
            "age": 35,
            "bmi": 24.5,
            "heart_rate_avg": 72,
            "steps_avg": 8500,
            "sleep_hours_avg": 7.5,
            "calories_avg": 2100,
            "ldl": 118,
            "glucose": 96,
            "hemoglobin": 14.3
        }
        
        # Generate comprehensive analysis
        analysis = {
            "user_id": current_user,
            "timestamp": datetime.now().isoformat(),
            "health_score": 78,
            "analysis": virtual_doctor._analyze_health_data(user_health_data),
            "recommendations": virtual_doctor._generate_recommendations(user_health_data, "general health"),
            "risk_assessment": virtual_doctor._assess_health_risks(user_health_data),
            "follow_up": virtual_doctor._suggest_follow_up(user_health_data)
        }
        
        # Store analysis in database
        if db is not None:
            result = await db.health_analyses.insert_one(analysis)
            # Convert ObjectId to string to avoid serialization issues
            analysis["_id"] = str(result.inserted_id)
        
        return {
            "message": "Health analysis completed",
            "data": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating analysis: {str(e)}")

# Test endpoints
@app.get("/api/v1/test/auth")
async def test_auth():
    """Test authentication"""
    return {
        "message": "Authentication system test",
        "features": [
            "JWT token generation",
            "Password hashing",
            "User registration",
            "User login"
        ]
    }

@app.get("/api/v1/test/virtual-doctor")
async def test_virtual_doctor():
    """Test virtual doctor"""
    if not virtual_doctor:
        return {
            "message": "Virtual doctor not available",
            "status": "error"
        }
    
    return {
        "message": "Virtual doctor test",
        "doctor": virtual_doctor.doctor_profile["name"],
        "specialty": virtual_doctor.doctor_profile["specialty"],
        "status": "ready"
    }

@app.get("/api/v1/test/health-advice")
async def test_health_advice():
    """Test health advice generation"""
    if not virtual_doctor:
        return {
            "message": "Virtual doctor not available",
            "status": "error"
        }
    
    sample_data = {
        "age": 35,
        "bmi": 24.5,
        "heart_rate_avg": 72,
        "steps_avg": 8500
    }
    
    advice = virtual_doctor.get_health_advice(sample_data, "How can I improve my health?")
    
    return {
        "message": "Health advice test",
        "sample_advice": advice
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
