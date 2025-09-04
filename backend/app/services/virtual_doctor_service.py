import os
import sys
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.chatbot import (
    ChatRequest, ChatResponse, ChatSession, ChatMessage, ChatMessageRole,
    PatientContext, WearableTrends, HealthPredictions, HealthMetricTrend,
    HealthAnalysis, LifestyleRecommendation, VirtualDoctorProfile,
    ChatSessionSummary
)
from services.database import get_database
from services.health_prediction_service import HealthPredictionService

try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage, AIMessage
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.memory import ConversationBufferMemory
    from langchain.output_parsers import PydanticOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain not available. Install with: pip install langchain openai")

class VirtualDoctorService:
    """Virtual doctor service using LangChain for lifestyle medicine consultations"""
    
    def __init__(self):
        self.llm = None
        self.memory = None
        self.doctor_profile = VirtualDoctorProfile()
        self.health_predictor = HealthPredictionService()
        self.sessions = {}
        
        # Initialize LangChain if available
        if LANGCHAIN_AVAILABLE:
            self._initialize_langchain()
            self.memory = ConversationBufferMemory(return_messages=True)
    
    def _initialize_langchain(self):
        """Initialize LangChain components"""
        try:
            # Initialize the language model
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm = ChatOpenAI(
                    model_name="gpt-4",
                    temperature=0.7,
                    max_tokens=2000,
                    openai_api_key=api_key
                )
                print("✅ LangChain initialized with OpenAI GPT-4")
            else:
                print("⚠️ OPENAI_API_KEY not found. Using fallback responses.")
        except Exception as e:
            print(f"❌ Error initializing LangChain: {e}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the virtual doctor"""
        return f"""You are {self.doctor_profile.doctor_name}, a board-certified physician specializing in {self.doctor_profile.specialty}.

CREDENTIALS:
{chr(10).join(f"• {cred}" for cred in self.doctor_profile.credentials)}

EXPERTISE AREAS:
{chr(10).join(f"• {area}" for area in self.doctor_profile.expertise_areas)}

COMMUNICATION STYLE: {self.doctor_profile.communication_style}
CONSULTATION APPROACH: {self.doctor_profile.consultation_approach}

CORE PRINCIPLES:
1. **Evidence-Based Medicine**: Base all recommendations on scientific evidence and clinical guidelines
2. **Preventive Focus**: Emphasize prevention and early intervention
3. **Lifestyle Medicine**: Prioritize lifestyle interventions over medications when appropriate
4. **Personalized Care**: Tailor recommendations to individual patient circumstances
5. **Holistic Approach**: Consider physical, mental, and social health factors
6. **Patient Empowerment**: Educate and empower patients to take control of their health

CONSULTATION GUIDELINES:
- Always start by reviewing the patient's health data and trends
- Provide clear, actionable recommendations
- Explain the scientific rationale behind recommendations
- Consider the patient's current lifestyle and constraints
- Prioritize recommendations by impact and feasibility
- Monitor progress and adjust recommendations as needed
- Maintain a compassionate, supportive tone
- Encourage questions and engagement

HEALTH METRICS INTERPRETATION:
- **LDL Cholesterol**: Target <100 mg/dL for optimal heart health
- **Glucose**: Fasting <100 mg/dL, post-meal <140 mg/dL
- **Hemoglobin**: 13.5-17.5 g/dL (men), 12.0-15.5 g/dL (women)
- **Heart Rate**: Resting 60-100 bpm, lower is generally better
- **Sleep**: 7-9 hours per night with good quality
- **Activity**: 150+ minutes moderate activity per week
- **Steps**: 7,000-10,000 steps per day

RESPONSE FORMAT:
1. **Health Assessment**: Brief overview of current health status
2. **Trend Analysis**: Interpretation of health trends
3. **Risk Assessment**: Identification of risk factors
4. **Recommendations**: Specific, actionable lifestyle recommendations
5. **Implementation Plan**: Step-by-step guidance
6. **Monitoring Plan**: How to track progress
7. **Follow-up**: When to reassess

Remember: You are a trusted healthcare provider. Always prioritize patient safety and recommend professional medical consultation when appropriate."""

    def _get_health_context_prompt(self, patient_context: PatientContext, 
                                 wearable_trends: Optional[WearableTrends] = None,
                                 health_predictions: Optional[HealthPredictions] = None) -> str:
        """Generate health context for the chatbot"""
        context_parts = []
        
        # Patient demographics
        context_parts.append(f"PATIENT PROFILE:")
        context_parts.append(f"• Age: {patient_context.age} years")
        context_parts.append(f"• Gender: {patient_context.gender}")
        if patient_context.bmi:
            context_parts.append(f"• BMI: {patient_context.bmi}")
        if patient_context.medical_conditions:
            context_parts.append(f"• Medical Conditions: {', '.join(patient_context.medical_conditions)}")
        if patient_context.medications:
            context_parts.append(f"• Medications: {', '.join(patient_context.medications)}")
        
        # Wearable trends
        if wearable_trends:
            context_parts.append(f"\nWEARABLE TRENDS:")
            if wearable_trends.heart_rate_trend:
                hr = wearable_trends.heart_rate_trend
                context_parts.append(f"• Heart Rate: {hr.current_value} {hr.unit} ({hr.status}, {hr.trend_direction})")
            if wearable_trends.sleep_trend:
                sleep = wearable_trends.sleep_trend
                context_parts.append(f"• Sleep Quality: {sleep.current_value} {sleep.unit} ({sleep.status}, {sleep.trend_direction})")
            if wearable_trends.activity_trend:
                activity = wearable_trends.activity_trend
                context_parts.append(f"• Activity Level: {activity.current_value} {activity.unit} ({activity.status}, {activity.trend_direction})")
            if wearable_trends.steps_trend:
                steps = wearable_trends.steps_trend
                context_parts.append(f"• Daily Steps: {steps.current_value} {steps.unit} ({steps.status}, {steps.trend_direction})")
            if wearable_trends.hrv_trend:
                hrv = wearable_trends.hrv_trend
                context_parts.append(f"• Heart Rate Variability: {hrv.current_value} {hrv.unit} ({hrv.status}, {hrv.trend_direction})")
        
        # Health predictions
        if health_predictions:
            context_parts.append(f"\nHEALTH PREDICTIONS:")
            if health_predictions.ldl_prediction:
                ldl = health_predictions.ldl_prediction
                context_parts.append(f"• LDL Cholesterol: {ldl.current_value} {ldl.unit} ({ldl.status}, {ldl.trend_direction})")
            if health_predictions.glucose_prediction:
                glucose = health_predictions.glucose_prediction
                context_parts.append(f"• Glucose: {glucose.current_value} {glucose.unit} ({glucose.status}, {glucose.trend_direction})")
            if health_predictions.hemoglobin_prediction:
                hgb = health_predictions.hemoglobin_prediction
                context_parts.append(f"• Hemoglobin: {hgb.current_value} {hgb.unit} ({hgb.status}, {hgb.trend_direction})")
        
        return "\n".join(context_parts)

    def _get_lifestyle_recommendations_prompt(self) -> str:
        """Get prompt for generating lifestyle recommendations"""
        return """Based on the patient's health data and trends, provide specific lifestyle medicine recommendations in the following format:

DIET RECOMMENDATIONS:
- Specific foods to include/exclude
- Meal timing and frequency
- Portion control strategies
- Hydration recommendations

EXERCISE RECOMMENDATIONS:
- Type and intensity of exercise
- Frequency and duration
- Progression strategies
- Safety considerations

SLEEP RECOMMENDATIONS:
- Sleep hygiene practices
- Bedtime routine
- Sleep environment optimization
- Stress management for better sleep

STRESS MANAGEMENT:
- Relaxation techniques
- Mindfulness practices
- Work-life balance strategies
- Social support recommendations

MONITORING PLAN:
- Key metrics to track
- Frequency of monitoring
- Progress indicators
- When to seek professional help

Make recommendations specific, actionable, and evidence-based. Consider the patient's current lifestyle and constraints."""

    async def get_patient_context(self, patient_id: str) -> PatientContext:
        """Get patient context from database"""
        try:
            db = get_database()
            
            # Get patient data
            patient = await db.patients.find_one({"_id": patient_id})
            if not patient:
                raise ValueError(f"Patient {patient_id} not found")
            
            # Get recent health predictions
            recent_predictions = await db.health_predictions.find({
                "patient_id": patient_id
            }).sort("timestamp", -1).limit(1).to_list(1)
            
            # Get recent wearable data
            recent_wearable = await db.daily_logs.find({
                "patient_id": patient_id
            }).sort("date", -1).limit(1).to_list(1)
            
            # Build patient context
            context = PatientContext(
                patient_id=patient_id,
                age=patient.get("age", 30),
                gender=patient.get("gender", "unknown"),
                bmi=patient.get("bmi"),
                medical_conditions=patient.get("medical_conditions", []),
                medications=patient.get("medications", []),
                lifestyle_factors={
                    "activity_level": patient.get("activity_level"),
                    "smoking_status": patient.get("smoking_status"),
                    "alcohol_consumption": patient.get("alcohol_consumption")
                }
            )
            
            return context
            
        except Exception as e:
            print(f"Error getting patient context: {e}")
            # Return default context
            return PatientContext(
                patient_id=patient_id,
                age=35,
                gender="unknown",
                bmi=25.0,
                medical_conditions=[],
                medications=[],
                lifestyle_factors={}
            )

    async def get_health_trends(self, patient_id: str) -> Optional[WearableTrends]:
        """Get health trends from database"""
        try:
            db = get_database()
            
            # Get recent wearable data
            wearable_data = await db.daily_logs.find({
                "patient_id": patient_id
            }).sort("date", -1).limit(7).to_list(7)
            
            if not wearable_data:
                return None
            
            # Calculate trends (simplified)
            latest = wearable_data[0]
            week_ago = wearable_data[-1] if len(wearable_data) > 1 else latest
            
            trends = WearableTrends()
            
            # Heart rate trend
            if "heart_rate" in latest and "heart_rate" in week_ago:
                current_hr = latest["heart_rate"].get("average", 70)
                previous_hr = week_ago["heart_rate"].get("average", 70)
                trend_direction = "improving" if current_hr < previous_hr else "declining" if current_hr > previous_hr else "stable"
                
                trends.heart_rate_trend = HealthMetricTrend(
                    metric="heart_rate",
                    current_value=current_hr,
                    trend_direction=trend_direction,
                    trend_strength=abs(current_hr - previous_hr) / 10,
                    normal_range={"min": 60, "max": 100},
                    unit="bpm",
                    status="normal" if 60 <= current_hr <= 100 else "elevated"
                )
            
            # Sleep trend
            if "sleep" in latest and "sleep" in week_ago:
                current_sleep = latest["sleep"].get("total_minutes", 420)
                previous_sleep = week_ago["sleep"].get("total_minutes", 420)
                trend_direction = "improving" if current_sleep > previous_sleep else "declining" if current_sleep < previous_sleep else "stable"
                
                trends.sleep_trend = HealthMetricTrend(
                    metric="sleep",
                    current_value=current_sleep,
                    trend_direction=trend_direction,
                    trend_strength=abs(current_sleep - previous_sleep) / 60,
                    normal_range={"min": 420, "max": 540},
                    unit="minutes",
                    status="normal" if 420 <= current_sleep <= 540 else "insufficient"
                )
            
            # Steps trend
            if "activity" in latest and "activity" in week_ago:
                current_steps = latest["activity"].get("total_steps", 8000)
                previous_steps = week_ago["activity"].get("total_steps", 8000)
                trend_direction = "improving" if current_steps > previous_steps else "declining" if current_steps < previous_steps else "stable"
                
                trends.steps_trend = HealthMetricTrend(
                    metric="steps",
                    current_value=current_steps,
                    trend_direction=trend_direction,
                    trend_strength=abs(current_steps - previous_steps) / 1000,
                    normal_range={"min": 7000, "max": 10000},
                    unit="steps",
                    status="normal" if 7000 <= current_steps <= 10000 else "low"
                )
            
            return trends
            
        except Exception as e:
            print(f"Error getting health trends: {e}")
            return None

    async def get_health_predictions(self, patient_id: str) -> Optional[HealthPredictions]:
        """Get health predictions from ML models"""
        try:
            db = get_database()
            
            # Get recent predictions
            predictions = await db.health_predictions.find({
                "patient_id": patient_id
            }).sort("timestamp", -1).limit(1).to_list(1)
            
            if not predictions:
                return None
            
            latest_pred = predictions[0]
            pred_data = latest_pred.get("predictions", [])
            
            health_pred = HealthPredictions()
            
            for pred in pred_data:
                metric = pred.get("metric")
                if metric == "ldl":
                    health_pred.ldl_prediction = HealthMetricTrend(
                        metric="ldl",
                        current_value=pred.get("predicted_value", 100),
                        trend_direction=pred.get("trend_direction", "stable"),
                        trend_strength=0.5,
                        normal_range=pred.get("normal_range", {"min": 0, "max": 100}),
                        unit=pred.get("unit", "mg/dL"),
                        status=pred.get("status", "normal")
                    )
                elif metric == "glucose":
                    health_pred.glucose_prediction = HealthMetricTrend(
                        metric="glucose",
                        current_value=pred.get("predicted_value", 95),
                        trend_direction=pred.get("trend_direction", "stable"),
                        trend_strength=0.5,
                        normal_range=pred.get("normal_range", {"min": 70, "max": 100}),
                        unit=pred.get("unit", "mg/dL"),
                        status=pred.get("status", "normal")
                    )
                elif metric == "hemoglobin":
                    health_pred.hemoglobin_prediction = HealthMetricTrend(
                        metric="hemoglobin",
                        current_value=pred.get("predicted_value", 14),
                        trend_direction=pred.get("trend_direction", "stable"),
                        trend_strength=0.5,
                        normal_range=pred.get("normal_range", {"min": 12, "max": 17}),
                        unit=pred.get("unit", "g/dL"),
                        status=pred.get("status", "normal")
                    )
            
            return health_pred
            
        except Exception as e:
            print(f"Error getting health predictions: {e}")
            return None

    async def chat_with_doctor(self, request: ChatRequest) -> ChatResponse:
        """Main chat function with the virtual doctor"""
        try:
            # Get or create session
            session_id = request.session_id or str(uuid.uuid4())
            if session_id not in self.sessions:
                self.sessions[session_id] = ChatSession(
                    session_id=session_id,
                    patient_id=request.patient_id,
                    start_time=datetime.now()
                )
            
            session = self.sessions[session_id]
            
            # Get patient context and health data
            patient_context = await self.get_patient_context(request.patient_id)
            wearable_trends = await self.get_health_trends(request.patient_id) if request.include_trends else None
            health_predictions = await self.get_health_predictions(request.patient_id) if request.include_health_data else None
            
            # Update session context
            session.context = patient_context
            session.wearable_trends = wearable_trends
            session.health_predictions = health_predictions
            
            # Add user message to session
            user_message = ChatMessage(
                role=ChatMessageRole.USER,
                content=request.message,
                timestamp=datetime.now()
            )
            session.messages.append(user_message)
            
            # Generate doctor's response
            if self.llm and LANGCHAIN_AVAILABLE:
                response = await self._generate_llm_response(request.message, patient_context, wearable_trends, health_predictions)
            else:
                response = self._generate_fallback_response(request.message, patient_context, wearable_trends, health_predictions)
            
            # Add doctor's response to session
            doctor_message = ChatMessage(
                role=ChatMessageRole.ASSISTANT,
                content=response.message,
                timestamp=datetime.now()
            )
            session.messages.append(doctor_message)
            
            return response
            
        except Exception as e:
            print(f"Error in chat: {e}")
            return ChatResponse(
                session_id=request.session_id or "error",
                message="I apologize, but I'm experiencing technical difficulties. Please try again or contact support.",
                urgency_level="normal"
            )

    async def _generate_llm_response(self, user_message: str, patient_context: PatientContext,
                                   wearable_trends: Optional[WearableTrends] = None,
                                   health_predictions: Optional[HealthPredictions] = None) -> ChatResponse:
        """Generate response using LangChain LLM"""
        try:
            # Build context
            health_context = self._get_health_context_prompt(patient_context, wearable_trends, health_predictions)
            
            # Create messages
            messages = [
                SystemMessage(content=self._get_system_prompt()),
                HumanMessage(content=f"""
PATIENT HEALTH CONTEXT:
{health_context}

PATIENT MESSAGE:
{user_message}

Please provide a comprehensive response as Dr. Sarah Chen, including:
1. Health assessment based on the data
2. Specific lifestyle recommendations
3. Implementation guidance
4. Monitoring suggestions
5. Follow-up recommendations

Respond in a compassionate, professional manner with clear, actionable advice.
""")
            ]
            
            # Generate response
            response = await self.llm.agenerate([messages])
            response_text = response.generations[0][0].text
            
            # Parse response for structured data
            health_insights = self._extract_health_insights(response_text)
            recommendations = self._extract_recommendations(response_text)
            urgency_level = self._assess_urgency(response_text, health_predictions)
            
            return ChatResponse(
                session_id=str(uuid.uuid4()),
                message=response_text,
                health_insights=health_insights,
                recommendations=recommendations,
                urgency_level=urgency_level,
                follow_up_questions=self._generate_follow_up_questions(response_text)
            )
            
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return self._generate_fallback_response(user_message, patient_context, wearable_trends, health_predictions)

    def _generate_fallback_response(self, user_message: str, patient_context: PatientContext,
                                  wearable_trends: Optional[WearableTrends] = None,
                                  health_predictions: Optional[HealthPredictions] = None) -> ChatResponse:
        """Generate fallback response when LLM is not available"""
        
        # Analyze the user message
        message_lower = user_message.lower()
        
        # Health assessment
        health_insights = []
        recommendations = []
        urgency_level = "normal"
        
        if "ldl" in message_lower or "cholesterol" in message_lower:
            if health_predictions and health_predictions.ldl_prediction:
                ldl = health_predictions.ldl_prediction
                if ldl.status in ["high", "very_high"]:
                    health_insights.append("Your LDL cholesterol is elevated, which increases cardiovascular risk.")
                    recommendations.extend([
                        "Reduce saturated fat intake to less than 7% of daily calories",
                        "Increase fiber consumption to 25-30g daily",
                        "Include omega-3 rich foods like fatty fish",
                        "Exercise regularly for at least 150 minutes per week"
                    ])
                    urgency_level = "moderate"
        
        if "glucose" in message_lower or "blood sugar" in message_lower or "diabetes" in message_lower:
            if health_predictions and health_predictions.glucose_prediction:
                glucose = health_predictions.glucose_prediction
                if glucose.status in ["prediabetes", "diabetes"]:
                    health_insights.append("Your blood glucose levels indicate metabolic concerns.")
                    recommendations.extend([
                        "Monitor carbohydrate intake and timing",
                        "Exercise regularly to improve insulin sensitivity",
                        "Maintain a healthy weight",
                        "Consider working with a registered dietitian"
                    ])
                    urgency_level = "moderate"
        
        if "sleep" in message_lower:
            if wearable_trends and wearable_trends.sleep_trend:
                sleep = wearable_trends.sleep_trend
                if sleep.status == "insufficient":
                    health_insights.append("Your sleep duration is below recommended levels.")
                    recommendations.extend([
                        "Aim for 7-9 hours of sleep per night",
                        "Establish a consistent bedtime routine",
                        "Create a sleep-conducive environment",
                        "Limit screen time before bed"
                    ])
        
        if "exercise" in message_lower or "activity" in message_lower:
            if wearable_trends and wearable_trends.steps_trend:
                steps = wearable_trends.steps_trend
                if steps.status == "low":
                    health_insights.append("Your daily step count is below recommended levels.")
                    recommendations.extend([
                        "Aim for 7,000-10,000 steps daily",
                        "Start with short walks and gradually increase",
                        "Find activities you enjoy",
                        "Use a pedometer or fitness tracker"
                    ])
        
        # Generate response
        response_parts = []
        response_parts.append("Hello! I'm Dr. Sarah Chen, your lifestyle medicine specialist.")
        
        if health_insights:
            response_parts.append("\nBased on your health data, I've identified some important insights:")
            for insight in health_insights:
                response_parts.append(f"• {insight}")
        
        if recommendations:
            response_parts.append("\nHere are my recommendations for improving your health:")
            for i, rec in enumerate(recommendations[:5], 1):
                response_parts.append(f"{i}. {rec}")
        
        if not health_insights and not recommendations:
            response_parts.append("\nI'd be happy to help you with any health concerns or lifestyle questions you have. Could you tell me more about what specific aspects of your health you'd like to discuss?")
        
        response_parts.append("\nRemember, these recommendations are general guidance. For personalized medical advice, please consult with your healthcare provider.")
        
        return ChatResponse(
            session_id=str(uuid.uuid4()),
            message="\n".join(response_parts),
            health_insights=health_insights,
            recommendations=recommendations,
            urgency_level=urgency_level
        )

    def _extract_health_insights(self, response_text: str) -> List[str]:
        """Extract health insights from response text"""
        insights = []
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('•') or line.startswith('-'):
                insights.append(line[1:].strip())
        
        return insights[:5]  # Limit to 5 insights

    def _extract_recommendations(self, response_text: str) -> List[str]:
        """Extract recommendations from response text"""
        recommendations = []
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'try', 'aim', 'focus']):
                recommendations.append(line)
        
        return recommendations[:5]  # Limit to 5 recommendations

    def _assess_urgency(self, response_text: str, health_predictions: Optional[HealthPredictions]) -> str:
        """Assess urgency level based on response and health data"""
        text_lower = response_text.lower()
        
        # High urgency keywords
        high_urgency = ['emergency', 'urgent', 'immediately', 'severe', 'critical', 'dangerous']
        if any(word in text_lower for word in high_urgency):
            return "high"
        
        # Check health predictions for concerning values
        if health_predictions:
            if (health_predictions.ldl_prediction and 
                health_predictions.ldl_prediction.status in ["high", "very_high"]):
                return "moderate"
            if (health_predictions.glucose_prediction and 
                health_predictions.glucose_prediction.status in ["prediabetes", "diabetes"]):
                return "moderate"
        
        return "normal"

    def _generate_follow_up_questions(self, response_text: str) -> List[str]:
        """Generate follow-up questions based on response"""
        questions = [
            "How do you feel about implementing these recommendations?",
            "Do you have any questions about the lifestyle changes I've suggested?",
            "What barriers do you anticipate in making these changes?",
            "Would you like me to help you create a specific action plan?",
            "How can I support you in achieving your health goals?"
        ]
        return questions[:3]

    async def get_session_summary(self, session_id: str) -> Optional[ChatSessionSummary]:
        """Get summary of a chat session"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Calculate session duration
        duration = (datetime.now() - session.start_time).total_seconds() / 60
        
        # Extract topics and insights
        topics = []
        insights = []
        action_items = []
        
        for message in session.messages:
            if message.role == ChatMessageRole.ASSISTANT:
                content = message.content.lower()
                if any(word in content for word in ['diet', 'nutrition', 'food']):
                    topics.append("Diet and Nutrition")
                if any(word in content for word in ['exercise', 'activity', 'workout']):
                    topics.append("Exercise and Physical Activity")
                if any(word in content for word in ['sleep', 'rest']):
                    topics.append("Sleep and Recovery")
                if any(word in content for word in ['stress', 'anxiety', 'mental']):
                    topics.append("Stress Management")
        
        # Remove duplicates
        topics = list(set(topics))
        
        return ChatSessionSummary(
            session_id=session_id,
            patient_id=session.patient_id,
            session_duration=duration,
            topics_discussed=topics,
            key_insights=insights,
            action_items=action_items,
            follow_up_required=len(action_items) > 0
        )

    async def generate_health_analysis(self, patient_id: str) -> HealthAnalysis:
        """Generate comprehensive health analysis"""
        try:
            # Get patient data
            patient_context = await self.get_patient_context(patient_id)
            wearable_trends = await self.get_health_trends(patient_id)
            health_predictions = await self.get_health_predictions(patient_id)
            
            # Calculate health score
            health_score = 75.0  # Base score
            risk_factors = []
            positive_factors = []
            
            # Analyze predictions
            if health_predictions:
                if health_predictions.ldl_prediction:
                    ldl = health_predictions.ldl_prediction
                    if ldl.status in ["high", "very_high"]:
                        risk_factors.append("Elevated LDL cholesterol")
                        health_score -= 10
                    elif ldl.status == "normal":
                        positive_factors.append("Healthy LDL cholesterol levels")
                        health_score += 5
                
                if health_predictions.glucose_prediction:
                    glucose = health_predictions.glucose_prediction
                    if glucose.status in ["prediabetes", "diabetes"]:
                        risk_factors.append("Blood glucose concerns")
                        health_score -= 15
                    elif glucose.status == "normal":
                        positive_factors.append("Healthy blood glucose levels")
                        health_score += 5
            
            # Analyze wearable trends
            if wearable_trends:
                if wearable_trends.sleep_trend and wearable_trends.sleep_trend.status == "insufficient":
                    risk_factors.append("Insufficient sleep")
                    health_score -= 8
                if wearable_trends.steps_trend and wearable_trends.steps_trend.status == "low":
                    risk_factors.append("Low physical activity")
                    health_score -= 7
            
            # Generate recommendations
            recommendations = {
                "diet": [
                    "Focus on whole, unprocessed foods",
                    "Increase fiber intake",
                    "Reduce saturated fat consumption",
                    "Stay hydrated throughout the day"
                ],
                "exercise": [
                    "Aim for 150 minutes of moderate activity weekly",
                    "Include strength training 2-3 times per week",
                    "Find activities you enjoy",
                    "Start gradually and build up"
                ],
                "sleep": [
                    "Maintain consistent sleep schedule",
                    "Create a relaxing bedtime routine",
                    "Optimize sleep environment",
                    "Limit caffeine and screen time before bed"
                ],
                "stress": [
                    "Practice mindfulness or meditation",
                    "Engage in regular physical activity",
                    "Maintain social connections",
                    "Consider stress management techniques"
                ]
            }
            
            return HealthAnalysis(
                patient_id=patient_id,
                overall_health_score=max(0, min(100, health_score)),
                risk_factors=risk_factors,
                positive_factors=positive_factors,
                lifestyle_assessment={
                    "diet": "needs_improvement" if "Elevated LDL" in risk_factors else "good",
                    "exercise": "needs_improvement" if "Low physical activity" in risk_factors else "good",
                    "sleep": "needs_improvement" if "Insufficient sleep" in risk_factors else "good",
                    "stress": "good"
                },
                recommendations=recommendations,
                priority_actions=[
                    "Schedule regular health checkups",
                    "Monitor key health metrics",
                    "Implement lifestyle changes gradually",
                    "Seek professional guidance when needed"
                ],
                monitoring_plan={
                    "frequency": "weekly",
                    "key_metrics": "weight, blood pressure, activity levels",
                    "follow_up": "3 months"
                }
            )
            
        except Exception as e:
            print(f"Error generating health analysis: {e}")
            return HealthAnalysis(
                patient_id=patient_id,
                overall_health_score=75.0,
                risk_factors=["Unable to analyze health data"],
                positive_factors=[],
                lifestyle_assessment={},
                recommendations={},
                priority_actions=["Contact healthcare provider for comprehensive assessment"],
                monitoring_plan={}
            )
