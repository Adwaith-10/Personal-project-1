#!/usr/bin/env python3
"""
Test script for Virtual Doctor Chatbot functionality
"""

import requests
import json
from datetime import datetime
import time

def test_virtual_doctor():
    """Test the virtual doctor chatbot functionality"""
    
    # API configuration
    API_BASE_URL = "http://localhost:8000"
    
    print("👩‍⚕️ Testing Virtual Doctor Chatbot")
    print("=" * 50)
    
    # First, get a patient ID to use for testing
    print("📋 Getting available patients...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/patients")
        if response.status_code == 200:
            patients = response.json()
            if isinstance(patients, dict) and "patients" in patients:
                patients_list = patients["patients"]
            else:
                patients_list = patients
            
            if patients_list:
                patient_id = patients_list[0]["_id"]
                print(f"✅ Using patient ID: {patient_id}")
            else:
                print("❌ No patients found. Please create a patient first.")
                return
        else:
            print(f"❌ Failed to get patients: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        print("💡 Make sure the FastAPI server is running on http://localhost:8000")
        return
    
    # Test doctor profile
    print("\n👩‍⚕️ Testing doctor profile...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/virtual-doctor/doctor-profile")
        
        if response.status_code == 200:
            profile = response.json()
            print("✅ Doctor profile retrieved successfully!")
            print(f"👩‍⚕️ Doctor: {profile.get('doctor_name')}")
            print(f"🏥 Specialty: {profile.get('specialty')}")
            print(f"📚 Credentials: {len(profile.get('credentials', []))} credentials")
            print(f"🎯 Expertise Areas: {len(profile.get('expertise_areas', []))} areas")
        else:
            print(f"❌ Failed to get doctor profile: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting doctor profile: {e}")
    
    # Test service status
    print("\n🔧 Testing service status...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/virtual-doctor/status")
        
        if response.status_code == 200:
            status = response.json()
            print("✅ Service status retrieved successfully!")
            print(f"📊 Service: {status.get('service')}")
            print(f"📈 Status: {status.get('status')}")
            print(f"👩‍⚕️ Doctor: {status.get('doctor_name')}")
            print(f"🤖 LangChain Available: {status.get('langchain_available')}")
            print(f"💬 Active Sessions: {status.get('active_sessions')}")
        else:
            print(f"❌ Failed to get service status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting service status: {e}")
    
    # Test health trends
    print("\n📈 Testing health trends analysis...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/virtual-doctor/health-trends/{patient_id}")
        
        if response.status_code == 200:
            trends = response.json()
            print("✅ Health trends retrieved successfully!")
            trends_data = trends.get('trends', {})
            if trends_data:
                for metric, data in trends_data.items():
                    print(f"   📊 {metric.replace('_', ' ').title()}: {data.get('current_value')} {data.get('unit')} ({data.get('status')}, {data.get('trend_direction')})")
            else:
                print("   📊 No trend data available")
        else:
            print(f"❌ Failed to get health trends: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting health trends: {e}")
    
    # Test health predictions
    print("\n🔮 Testing health predictions...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/virtual-doctor/health-predictions/{patient_id}")
        
        if response.status_code == 200:
            predictions = response.json()
            print("✅ Health predictions retrieved successfully!")
            pred_data = predictions.get('predictions', {})
            if pred_data:
                for metric, data in pred_data.items():
                    print(f"   🔮 {metric.upper()}: {data.get('current_value')} {data.get('unit')} ({data.get('status')})")
            else:
                print("   🔮 No prediction data available")
        else:
            print(f"❌ Failed to get health predictions: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting health predictions: {e}")
    
    # Test patient context
    print("\n👤 Testing patient context...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/virtual-doctor/patient-context/{patient_id}")
        
        if response.status_code == 200:
            context = response.json()
            print("✅ Patient context retrieved successfully!")
            context_data = context.get('context', {})
            print(f"   👤 Age: {context_data.get('age')}")
            print(f"   👤 Gender: {context_data.get('gender')}")
            print(f"   👤 BMI: {context_data.get('bmi')}")
            print(f"   👤 Medical Conditions: {len(context_data.get('medical_conditions', []))}")
            print(f"   👤 Medications: {len(context_data.get('medications', []))}")
        else:
            print(f"❌ Failed to get patient context: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting patient context: {e}")
    
    # Test chat with doctor
    print("\n💬 Testing chat with virtual doctor...")
    try:
        # Test different types of health questions
        test_messages = [
            "Hello Dr. Chen, I'm concerned about my cholesterol levels. Can you help me understand what my LDL numbers mean?",
            "I've been having trouble sleeping lately. My wearable shows I'm only getting about 6 hours per night. What can I do to improve my sleep?",
            "I want to improve my overall health. What lifestyle changes would you recommend based on my data?",
            "My blood sugar has been a bit high lately. Should I be worried about diabetes?",
            "I'm trying to lose weight and improve my fitness. What exercise routine would you suggest?"
        ]
        
        session_id = None
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n💬 Test {i}: {message[:50]}...")
            
            chat_request = {
                "patient_id": patient_id,
                "message": message,
                "session_id": session_id,
                "include_health_data": True,
                "include_trends": True
            }
            
            response = requests.post(
                f"{API_BASE_URL}/api/v1/virtual-doctor/chat",
                json=chat_request
            )
            
            if response.status_code == 200:
                chat_response = response.json()
                session_id = chat_response.get('session_id')
                
                print(f"✅ Chat response received!")
                print(f"   📝 Response length: {len(chat_response.get('message', ''))} characters")
                print(f"   🚨 Urgency level: {chat_response.get('urgency_level')}")
                
                # Display health insights
                insights = chat_response.get('health_insights', [])
                if insights:
                    print(f"   💡 Health insights: {len(insights)} insights")
                    for insight in insights[:2]:  # Show first 2
                        print(f"      • {insight}")
                
                # Display recommendations
                recommendations = chat_response.get('recommendations', [])
                if recommendations:
                    print(f"   💡 Recommendations: {len(recommendations)} recommendations")
                    for rec in recommendations[:2]:  # Show first 2
                        print(f"      • {rec}")
                
                # Display follow-up questions
                follow_up = chat_response.get('follow_up_questions', [])
                if follow_up:
                    print(f"   ❓ Follow-up questions: {len(follow_up)} questions")
                
            else:
                print(f"❌ Chat failed: {response.status_code}")
                print(f"Error: {response.text}")
            
            # Small delay between messages
            time.sleep(1)
            
    except Exception as e:
        print(f"❌ Error during chat: {e}")
    
    # Test health analysis
    print("\n📊 Testing comprehensive health analysis...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/virtual-doctor/analysis/{patient_id}")
        
        if response.status_code == 200:
            analysis = response.json()
            print("✅ Health analysis completed successfully!")
            print(f"📊 Overall Health Score: {analysis.get('overall_health_score')}/100")
            
            # Display risk factors
            risk_factors = analysis.get('risk_factors', [])
            if risk_factors:
                print(f"⚠️ Risk Factors: {len(risk_factors)} identified")
                for risk in risk_factors:
                    print(f"   • {risk}")
            
            # Display positive factors
            positive_factors = analysis.get('positive_factors', [])
            if positive_factors:
                print(f"✅ Positive Factors: {len(positive_factors)} identified")
                for factor in positive_factors:
                    print(f"   • {factor}")
            
            # Display lifestyle assessment
            lifestyle = analysis.get('lifestyle_assessment', {})
            if lifestyle:
                print(f"📋 Lifestyle Assessment:")
                for category, status in lifestyle.items():
                    print(f"   • {category.title()}: {status}")
            
            # Display priority actions
            priority_actions = analysis.get('priority_actions', [])
            if priority_actions:
                print(f"🎯 Priority Actions: {len(priority_actions)} actions")
                for action in priority_actions[:3]:  # Show first 3
                    print(f"   • {action}")
            
            # Display monitoring plan
            monitoring = analysis.get('monitoring_plan', {})
            if monitoring:
                print(f"📈 Monitoring Plan:")
                for item, plan in monitoring.items():
                    print(f"   • {item.replace('_', ' ').title()}: {plan}")
                    
        else:
            print(f"❌ Health analysis failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error during health analysis: {e}")
    
    # Test personalized recommendations
    print("\n💡 Testing personalized recommendations...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/virtual-doctor/recommendations/{patient_id}")
        
        if response.status_code == 200:
            recommendations = response.json()
            print("✅ Personalized recommendations generated successfully!")
            print(f"📊 Health Score: {recommendations.get('overall_health_score')}/100")
            
            # Display recommendations by category
            rec_data = recommendations.get('recommendations', {})
            if rec_data:
                print(f"💡 Recommendations by category:")
                for category, recs in rec_data.items():
                    print(f"   📋 {category.title()}: {len(recs)} recommendations")
                    for rec in recs[:2]:  # Show first 2 per category
                        print(f"      • {rec}")
            
            # Display priority actions
            priority_actions = recommendations.get('priority_actions', [])
            if priority_actions:
                print(f"🎯 Priority Actions: {len(priority_actions)} actions")
                for action in priority_actions:
                    print(f"   • {action}")
                    
        else:
            print(f"❌ Recommendations failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting recommendations: {e}")
    
    # Test chat session history
    print("\n📋 Testing chat session history...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/virtual-doctor/sessions/{patient_id}")
        
        if response.status_code == 200:
            sessions = response.json()
            print(f"✅ Retrieved {len(sessions)} chat sessions")
            
            if sessions:
                latest_session = sessions[0]
                print(f"📅 Latest session: {latest_session.get('start_time')}")
                print(f"💬 Messages: {len(latest_session.get('messages', []))}")
                
                # Show sample messages
                messages = latest_session.get('messages', [])
                if messages:
                    print(f"💬 Sample messages:")
                    for msg in messages[:2]:  # Show first 2 messages
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')[:50] + "..." if len(msg.get('content', '')) > 50 else msg.get('content', '')
                        print(f"   {role.title()}: {content}")
            else:
                print("   📋 No chat sessions found")
        else:
            print(f"❌ Failed to get chat sessions: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting chat sessions: {e}")
    
    # Test session summary (if we have a session)
    if session_id:
        print(f"\n📄 Testing session summary for session: {session_id}")
        try:
            response = requests.get(f"{API_BASE_URL}/api/v1/virtual-doctor/sessions/{session_id}/summary")
            
            if response.status_code == 200:
                summary = response.json()
                print("✅ Session summary retrieved successfully!")
                print(f"⏱️ Session duration: {summary.get('session_duration')} minutes")
                print(f"📋 Topics discussed: {len(summary.get('topics_discussed', []))}")
                print(f"💡 Key insights: {len(summary.get('key_insights', []))}")
                print(f"✅ Action items: {len(summary.get('action_items', []))}")
                print(f"🔄 Follow-up required: {summary.get('follow_up_required')}")
            else:
                print(f"❌ Failed to get session summary: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error getting session summary: {e}")
    
    print("\n🎉 Virtual Doctor Chatbot testing completed!")

if __name__ == "__main__":
    test_virtual_doctor()
