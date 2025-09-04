#!/usr/bin/env python3
"""
Test script for Health Monitoring and Deviation Detection functionality
"""

import requests
import json
from datetime import datetime
import time

def test_health_monitoring():
    """Test the health monitoring and deviation detection functionality"""
    
    # API configuration
    API_BASE_URL = "http://localhost:8000"
    
    print("🔍 Testing Health Monitoring and Deviation Detection")
    print("=" * 60)
    
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
    
    # Test service status
    print("\n🔧 Testing health monitoring service status...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health-monitoring/status")
        
        if response.status_code == 200:
            status = response.json()
            print("✅ Health monitoring service status retrieved successfully!")
            print(f"📊 Service: {status.get('service')}")
            print(f"📈 Status: {status.get('status')}")
            print(f"📋 Monitored Metrics: {', '.join(status.get('monitored_metrics', []))}")
            print(f"🎯 Total Thresholds: {status.get('total_thresholds')}")
        else:
            print(f"❌ Failed to get service status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting service status: {e}")
    
    # Test deviation thresholds
    print("\n📊 Testing deviation thresholds...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health-monitoring/thresholds")
        
        if response.status_code == 200:
            thresholds = response.json()
            print("✅ Deviation thresholds retrieved successfully!")
            for metric, threshold in thresholds.items():
                print(f"   📋 {metric.upper()}:")
                print(f"      • Mild: {threshold.get('mild_threshold')}%")
                print(f"      • Moderate: {threshold.get('moderate_threshold')}%")
                print(f"      • Severe: {threshold.get('severe_threshold')}%")
                print(f"      • Critical: {threshold.get('critical_threshold')}%")
                print(f"      • Unit: {threshold.get('unit')}")
                print(f"      • Requires Consultation: {threshold.get('requires_consultation')}")
        else:
            print(f"❌ Failed to get thresholds: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting thresholds: {e}")
    
    # Test getting latest lab report
    print("\n📄 Testing latest lab report retrieval...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health-monitoring/lab-report/{patient_id}")
        
        if response.status_code == 200:
            lab_report = response.json()
            print("✅ Latest lab report retrieved successfully!")
            print(f"📅 Report Date: {lab_report.get('report_date')}")
            print(f"📋 Report ID: {lab_report.get('report_id')}")
            
            # Show available metrics
            available_metrics = []
            for key, value in lab_report.items():
                if key not in ['report_id', 'patient_id', 'report_date'] and value is not None:
                    available_metrics.append(key)
            
            print(f"📊 Available Metrics: {', '.join(available_metrics[:5])}...")
            
            # Show specific values
            if lab_report.get('ldl'):
                print(f"   • LDL: {lab_report['ldl']} mg/dL")
            if lab_report.get('glucose'):
                print(f"   • Glucose: {lab_report['glucose']} mg/dL")
            if lab_report.get('hemoglobin'):
                print(f"   • Hemoglobin: {lab_report['hemoglobin']} g/dL")
                
        elif response.status_code == 404:
            print("⚠️ No lab report found for this patient")
        else:
            print(f"❌ Failed to get lab report: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting lab report: {e}")
    
    # Test getting latest predictions
    print("\n🔮 Testing latest predictions retrieval...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health-monitoring/predictions/{patient_id}")
        
        if response.status_code == 200:
            predictions = response.json()
            print("✅ Latest predictions retrieved successfully!")
            print(f"📅 Prediction Date: {predictions.get('prediction_date')}")
            print(f"📋 Prediction ID: {predictions.get('prediction_id')}")
            print(f"🤖 Model Version: {predictions.get('model_version')}")
            
            # Show predicted values
            if predictions.get('ldl'):
                print(f"   • Predicted LDL: {predictions['ldl']} mg/dL")
            if predictions.get('glucose'):
                print(f"   • Predicted Glucose: {predictions['glucose']} mg/dL")
            if predictions.get('hemoglobin'):
                print(f"   • Predicted Hemoglobin: {predictions['hemoglobin']} g/dL")
            
            # Show confidence scores
            confidence_scores = predictions.get('confidence_scores', {})
            if confidence_scores:
                print(f"   📊 Confidence Scores: {confidence_scores}")
                
        elif response.status_code == 404:
            print("⚠️ No predictions found for this patient")
        else:
            print(f"❌ Failed to get predictions: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting predictions: {e}")
    
    # Test health comparison
    print("\n⚖️ Testing health comparison...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health-monitoring/comparison/{patient_id}")
        
        if response.status_code == 200:
            comparisons = response.json()
            print(f"✅ Health comparison completed! Found {len(comparisons)} comparisons")
            
            for comparison in comparisons:
                metric = comparison.get('metric', 'unknown')
                predicted = comparison.get('predicted_value')
                actual = comparison.get('actual_value')
                deviation = comparison.get('deviation_percentage')
                severity = comparison.get('severity')
                unit = comparison.get('unit')
                
                print(f"   📊 {metric.upper()}:")
                print(f"      • Predicted: {predicted} {unit}")
                print(f"      • Actual: {actual} {unit}")
                print(f"      • Deviation: {deviation:.1f}%")
                print(f"      • Severity: {severity}")
                print(f"      • Threshold Exceeded: {comparison.get('threshold_exceeded')}")
                print(f"      • Recommendation: {comparison.get('recommendation', 'N/A')[:50]}...")
                
        else:
            print(f"❌ Failed to get health comparison: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting health comparison: {e}")
    
    # Test health monitoring (main function)
    print("\n🔍 Testing health monitoring and deviation detection...")
    try:
        response = requests.post(f"{API_BASE_URL}/api/v1/health-monitoring/monitor/{patient_id}")
        
        if response.status_code == 200:
            monitoring_report = response.json()
            print("✅ Health monitoring completed successfully!")
            print(f"📋 Report ID: {monitoring_report.get('report_id')}")
            print(f"📅 Report Date: {monitoring_report.get('report_date')}")
            print(f"📊 Overall Health Status: {monitoring_report.get('overall_health_status')}")
            print(f"⚠️ Risk Level: {monitoring_report.get('risk_level')}")
            print(f"📈 Metrics Compared: {monitoring_report.get('metrics_compared')}")
            print(f"🚨 Deviations Found: {monitoring_report.get('deviations_found')}")
            print(f"🚨 Critical Deviations: {monitoring_report.get('critical_deviations')}")
            print(f"⚡ Requires Immediate Action: {monitoring_report.get('requires_immediate_action')}")
            print(f"📅 Next Monitoring Date: {monitoring_report.get('next_monitoring_date')}")
            
            # Show alerts generated
            alerts = monitoring_report.get('alerts_generated', [])
            if alerts:
                print(f"🚨 Alerts Generated: {len(alerts)}")
                for alert in alerts:
                    print(f"   • Alert ID: {alert.get('alert_id')}")
                    print(f"   • Severity: {alert.get('severity')}")
                    print(f"   • Triggered Metrics: {', '.join(alert.get('triggered_metrics', []))}")
                    print(f"   • Requires Consultation: {alert.get('requires_medical_consultation')}")
                    print(f"   • Consultation Urgency: {alert.get('consultation_urgency')}")
            else:
                print("✅ No alerts generated - health within normal ranges")
            
            # Show recommendations
            recommendations = monitoring_report.get('recommendations', [])
            if recommendations:
                print(f"💡 Recommendations: {len(recommendations)}")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"   {i}. {rec}")
            
            # Show summary
            summary = monitoring_report.get('summary', '')
            if summary:
                print(f"📝 Summary: {summary}")
                
        else:
            print(f"❌ Health monitoring failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during health monitoring: {e}")
    
    # Test health alerts
    print("\n🚨 Testing health alerts retrieval...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health-monitoring/alerts/{patient_id}")
        
        if response.status_code == 200:
            alerts = response.json()
            print(f"✅ Retrieved {len(alerts)} health alerts")
            
            for alert in alerts[:3]:  # Show first 3 alerts
                print(f"   🚨 Alert ID: {alert.get('alert_id')}")
                print(f"   📅 Timestamp: {alert.get('timestamp')}")
                print(f"   ⚠️ Severity: {alert.get('severity')}")
                print(f"   📊 Triggered Metrics: {', '.join(alert.get('triggered_metrics', []))}")
                print(f"   🏥 Requires Consultation: {alert.get('requires_medical_consultation')}")
                print(f"   ⏰ Consultation Urgency: {alert.get('consultation_urgency')}")
                print(f"   📝 Assessment: {alert.get('overall_assessment', 'N/A')[:50]}...")
                
                # Show urgent actions
                urgent_actions = alert.get('urgent_actions', [])
                if urgent_actions:
                    print(f"   ⚡ Urgent Actions: {len(urgent_actions)}")
                    for action in urgent_actions[:2]:
                        print(f"      • {action}")
        else:
            print(f"❌ Failed to get health alerts: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting health alerts: {e}")
    
    # Test notifications
    print("\n📢 Testing notifications...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health-monitoring/notifications/{patient_id}")
        
        if response.status_code == 200:
            notifications = response.json()
            print(f"✅ Retrieved {len(notifications)} notifications")
            
            for notification in notifications[:3]:  # Show first 3 notifications
                print(f"   📢 Notification ID: {notification.get('notification_id')}")
                print(f"   📅 Timestamp: {notification.get('timestamp')}")
                print(f"   ⚠️ Severity: {notification.get('severity')}")
                print(f"   📝 Title: {notification.get('title')}")
                print(f"   💬 Message: {notification.get('message', 'N/A')[:50]}...")
                print(f"   ⚡ Action Required: {notification.get('action_required')}")
                print(f"   📖 Read: {notification.get('read')}")
                print(f"   ✅ Acknowledged: {notification.get('acknowledged')}")
        else:
            print(f"❌ Failed to get notifications: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting notifications: {e}")
    
    # Test monitoring history
    print("\n📋 Testing monitoring history...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health-monitoring/history/{patient_id}")
        
        if response.status_code == 200:
            history = response.json()
            print(f"✅ Retrieved {len(history)} monitoring reports")
            
            if history:
                latest_report = history[0]
                print(f"   📅 Latest Report Date: {latest_report.get('report_date')}")
                print(f"   📊 Health Status: {latest_report.get('overall_health_status')}")
                print(f"   ⚠️ Risk Level: {latest_report.get('risk_level')}")
                print(f"   📈 Metrics Compared: {latest_report.get('metrics_compared')}")
                print(f"   🚨 Deviations Found: {latest_report.get('deviations_found')}")
                print(f"   🚨 Critical Deviations: {latest_report.get('critical_deviations')}")
            else:
                print("   📋 No monitoring history found")
        else:
            print(f"❌ Failed to get monitoring history: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting monitoring history: {e}")
    
    # Test monitoring summary
    print("\n📊 Testing monitoring summary...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health-monitoring/summary/{patient_id}")
        
        if response.status_code == 200:
            summary = response.json()
            print("✅ Monitoring summary retrieved successfully!")
            print(f"📅 Summary Period: {summary.get('summary_period_days')} days")
            print(f"📋 Total Reports: {summary.get('total_monitoring_reports')}")
            print(f"🚨 Total Deviations: {summary.get('total_deviations_detected')}")
            print(f"🚨 Critical Deviations: {summary.get('total_critical_deviations')}")
            print(f"🚨 Recent Alerts: {summary.get('recent_alerts_count')}")
            print(f"📢 Unread Notifications: {summary.get('unread_notifications')}")
            print(f"📊 Overall Status: {summary.get('overall_health_status')}")
            print(f"⚠️ Risk Level: {summary.get('risk_level')}")
            print(f"🏥 Requires Consultation: {summary.get('requires_medical_consultation')}")
            
            if summary.get('last_monitoring_date'):
                print(f"📅 Last Monitoring: {summary.get('last_monitoring_date')}")
            if summary.get('next_monitoring_date'):
                print(f"📅 Next Monitoring: {summary.get('next_monitoring_date')}")
        else:
            print(f"❌ Failed to get monitoring summary: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error getting monitoring summary: {e}")
    
    print("\n🎉 Health Monitoring and Deviation Detection testing completed!")

if __name__ == "__main__":
    test_health_monitoring()
