from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from datetime import datetime, timedelta
from bson import ObjectId
import sys
import os

# Add the parent directory to the path to import models and services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.health_monitoring import (
    HealthMonitoringReport, HealthDeviationAlert, HealthMetricComparison,
    LabReportMetrics, PredictedHealthMetrics, NotificationMessage,
    DeviationSeverity
)
from services.database import get_database
from services.health_monitoring_service import HealthMonitoringService

router = APIRouter(prefix="/api/v1/health-monitoring", tags=["health-monitoring"])

# Initialize the health monitoring service
health_monitor = HealthMonitoringService()

@router.post("/monitor/{patient_id}", response_model=HealthMonitoringReport)
async def monitor_health_deviations(patient_id: str):
    """
    Monitor health deviations between predicted and actual lab report values.
    
    This endpoint compares today's predicted health values with the most recent
    lab report and flags deviations beyond established thresholds. If significant
    deviations are detected, it generates alerts and notifications recommending
    consultation with a real doctor.
    
    The system monitors:
    - LDL cholesterol deviations
    - Glucose level changes
    - Hemoglobin variations
    - Other relevant health metrics
    
    Returns a comprehensive monitoring report with severity assessment and recommendations.
    """
    
    try:
        # Validate patient ID
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(
                status_code=400,
                detail="Invalid patient ID format"
            )
        
        # Get database connection
        db = get_database()
        
        # Check if patient exists
        patient = await db.patients.find_one({"_id": ObjectId(patient_id)})
        if not patient:
            raise HTTPException(
                status_code=404,
                detail="Patient not found"
            )
        
        # Perform health monitoring
        monitoring_report = await health_monitor.monitor_health_deviations(patient_id)
        
        # Store monitoring report in database
        report_data = {
            "report_id": monitoring_report.report_id,
            "patient_id": patient_id,
            "report_date": monitoring_report.report_date,
            "comparison_date": monitoring_report.comparison_date,
            "overall_health_status": monitoring_report.overall_health_status,
            "risk_level": monitoring_report.risk_level,
            "metrics_compared": monitoring_report.metrics_compared,
            "deviations_found": monitoring_report.deviations_found,
            "critical_deviations": monitoring_report.critical_deviations,
            "alerts_generated": [alert.dict() for alert in monitoring_report.alerts_generated],
            "recommendations": monitoring_report.recommendations,
            "next_monitoring_date": monitoring_report.next_monitoring_date,
            "requires_immediate_action": monitoring_report.requires_immediate_action,
            "summary": monitoring_report.summary
        }
        
        await db.health_monitoring_reports.insert_one(report_data)
        
        return monitoring_report
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health monitoring failed: {str(e)}"
        )

@router.get("/comparison/{patient_id}", response_model=List[HealthMetricComparison])
async def get_health_comparison(patient_id: str):
    """Get detailed comparison between predicted and actual health metrics"""
    try:
        # Validate patient ID
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Get database connection
        db = get_database()
        
        # Check if patient exists
        patient = await db.patients.find_one({"_id": ObjectId(patient_id)})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Get health comparison
        comparisons = await health_monitor.compare_health_metrics(patient_id)
        
        return comparisons
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health comparison: {str(e)}")

@router.get("/alerts/{patient_id}", response_model=List[HealthDeviationAlert])
async def get_health_alerts(
    patient_id: str,
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    severity: Optional[DeviationSeverity] = Query(None, description="Filter by severity level"),
    db=Depends(get_database)
):
    """Get health alerts for a patient"""
    try:
        # Validate patient ID
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Build query
        query = {"patient_id": patient_id}
        
        # Add date filter
        start_date = datetime.now() - timedelta(days=days)
        query["timestamp"] = {"$gte": start_date}
        
        # Add severity filter
        if severity:
            query["severity"] = severity.value
        
        # Get alerts from database
        alerts = await db.health_alerts.find(query).sort("timestamp", -1).to_list(100)
        
        # Convert to response format
        health_alerts = []
        for alert_data in alerts:
            # Convert deviations back to objects
            deviations = []
            for dev_data in alert_data.get("deviations", []):
                deviations.append(HealthMetricComparison(**dev_data))
            
            alert = HealthDeviationAlert(
                alert_id=alert_data["alert_id"],
                patient_id=alert_data["patient_id"],
                timestamp=alert_data["timestamp"],
                severity=DeviationSeverity(alert_data["severity"]),
                triggered_metrics=alert_data["triggered_metrics"],
                deviations=deviations,
                overall_assessment=alert_data["overall_assessment"],
                urgent_actions=alert_data["urgent_actions"],
                recommendations=alert_data["recommendations"],
                requires_medical_consultation=alert_data["requires_medical_consultation"],
                consultation_urgency=alert_data["consultation_urgency"],
                follow_up_timeline=alert_data["follow_up_timeline"]
            )
            health_alerts.append(alert)
        
        return health_alerts
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health alerts: {str(e)}")

@router.get("/lab-report/{patient_id}", response_model=LabReportMetrics)
async def get_latest_lab_report(patient_id: str):
    """Get the latest lab report for a patient"""
    try:
        # Validate patient ID
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Get database connection
        db = get_database()
        
        # Check if patient exists
        patient = await db.patients.find_one({"_id": ObjectId(patient_id)})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Get latest lab report
        lab_report = await health_monitor.get_latest_lab_report(patient_id)
        
        if not lab_report:
            raise HTTPException(
                status_code=404,
                detail="No lab report found for this patient"
            )
        
        return lab_report
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get lab report: {str(e)}")

@router.get("/predictions/{patient_id}", response_model=PredictedHealthMetrics)
async def get_latest_predictions(patient_id: str):
    """Get the latest health predictions for a patient"""
    try:
        # Validate patient ID
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Get database connection
        db = get_database()
        
        # Check if patient exists
        patient = await db.patients.find_one({"_id": ObjectId(patient_id)})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Get latest predictions
        predictions = await health_monitor.get_latest_predictions(patient_id)
        
        if not predictions:
            raise HTTPException(
                status_code=404,
                detail="No health predictions found for this patient"
            )
        
        return predictions
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get predictions: {str(e)}")

@router.get("/notifications/{patient_id}", response_model=List[NotificationMessage])
async def get_notifications(
    patient_id: str,
    unread_only: bool = Query(False, description="Show only unread notifications"),
    severity: Optional[DeviationSeverity] = Query(None, description="Filter by severity level"),
    db=Depends(get_database)
):
    """Get notifications for a patient"""
    try:
        # Validate patient ID
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Build query
        query = {"patient_id": patient_id}
        
        if unread_only:
            query["read"] = False
        
        if severity:
            query["severity"] = severity.value
        
        # Get notifications from database
        notifications = await db.notifications.find(query).sort("timestamp", -1).to_list(50)
        
        # Convert to response format
        notification_messages = []
        for notif_data in notifications:
            notification = NotificationMessage(
                notification_id=notif_data["notification_id"],
                patient_id=notif_data["patient_id"],
                alert_id=notif_data["alert_id"],
                timestamp=notif_data["timestamp"],
                severity=DeviationSeverity(notif_data["severity"]),
                title=notif_data["title"],
                message=notif_data["message"],
                action_required=notif_data["action_required"],
                action_deadline=notif_data.get("action_deadline"),
                notification_type=notif_data["notification_type"],
                sent=notif_data["sent"],
                read=notif_data["read"],
                acknowledged=notif_data["acknowledged"]
            )
            notification_messages.append(notification)
        
        return notification_messages
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get notifications: {str(e)}")

@router.put("/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str, db=Depends(get_database)):
    """Mark a notification as read"""
    try:
        # Update notification
        result = await db.notifications.update_one(
            {"notification_id": notification_id},
            {"$set": {"read": True, "read_timestamp": datetime.now()}}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        return {"message": "Notification marked as read"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to mark notification as read: {str(e)}")

@router.put("/notifications/{notification_id}/acknowledge")
async def acknowledge_notification(notification_id: str, db=Depends(get_database)):
    """Acknowledge a notification"""
    try:
        # Update notification
        result = await db.notifications.update_one(
            {"notification_id": notification_id},
            {"$set": {"acknowledged": True, "acknowledged_timestamp": datetime.now()}}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        return {"message": "Notification acknowledged"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge notification: {str(e)}")

@router.get("/history/{patient_id}", response_model=List[HealthMonitoringReport])
async def get_monitoring_history(
    patient_id: str,
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of reports to return")
):
    """Get monitoring history for a patient"""
    try:
        # Validate patient ID
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Get database connection
        db = get_database()
        
        # Check if patient exists
        patient = await db.patients.find_one({"_id": ObjectId(patient_id)})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Get monitoring history
        history = await health_monitor.get_monitoring_history(patient_id, days)
        
        # Limit results
        return history[:limit]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring history: {str(e)}")

@router.get("/summary/{patient_id}")
async def get_monitoring_summary(patient_id: str, days: int = Query(30, ge=1, le=365)):
    """Get monitoring summary for a patient"""
    try:
        # Validate patient ID
        if not ObjectId.is_valid(patient_id):
            raise HTTPException(status_code=400, detail="Invalid patient ID format")
        
        # Get database connection
        db = get_database()
        
        # Check if patient exists
        patient = await db.patients.find_one({"_id": ObjectId(patient_id)})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Get monitoring history
        history = await health_monitor.get_monitoring_history(patient_id, days)
        
        # Calculate summary statistics
        total_reports = len(history)
        total_deviations = sum(report.deviations_found for report in history)
        total_critical = sum(report.critical_deviations for report in history)
        
        # Get recent alerts
        start_date = datetime.now() - timedelta(days=days)
        recent_alerts = await db.health_alerts.find({
            "patient_id": patient_id,
            "timestamp": {"$gte": start_date}
        }).sort("timestamp", -1).to_list(10)
        
        # Get unread notifications
        unread_notifications = await db.notifications.find({
            "patient_id": patient_id,
            "read": False
        }).count()
        
        # Determine overall status
        if total_critical > 0:
            overall_status = "Requires immediate attention"
            risk_level = "high"
        elif total_deviations > 0:
            overall_status = "Requires monitoring"
            risk_level = "moderate"
        else:
            overall_status = "Within normal ranges"
            risk_level = "low"
        
        return {
            "patient_id": patient_id,
            "summary_period_days": days,
            "total_monitoring_reports": total_reports,
            "total_deviations_detected": total_deviations,
            "total_critical_deviations": total_critical,
            "recent_alerts_count": len(recent_alerts),
            "unread_notifications": unread_notifications,
            "overall_health_status": overall_status,
            "risk_level": risk_level,
            "last_monitoring_date": history[0].report_date if history else None,
            "next_monitoring_date": history[0].next_monitoring_date if history else None,
            "requires_medical_consultation": any(report.requires_immediate_action for report in history)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring summary: {str(e)}")

@router.delete("/alerts/{alert_id}")
async def delete_health_alert(alert_id: str, db=Depends(get_database)):
    """Delete a health alert"""
    try:
        # Check if alert exists
        existing_alert = await db.health_alerts.find_one({"alert_id": alert_id})
        if not existing_alert:
            raise HTTPException(status_code=404, detail="Health alert not found")
        
        # Delete the alert
        await db.health_alerts.delete_one({"alert_id": alert_id})
        
        return {"message": "Health alert deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete health alert: {str(e)}")

@router.get("/thresholds")
async def get_deviation_thresholds():
    """Get deviation thresholds for different health metrics"""
    try:
        thresholds = health_monitor.deviation_thresholds
        
        # Convert to response format
        threshold_data = {}
        for metric, threshold in thresholds.items():
            threshold_data[metric] = {
                "metric": threshold.metric,
                "mild_threshold": threshold.mild_threshold,
                "moderate_threshold": threshold.moderate_threshold,
                "severe_threshold": threshold.severe_threshold,
                "critical_threshold": threshold.critical_threshold,
                "clinical_threshold": threshold.clinical_threshold,
                "unit": threshold.unit,
                "normal_range": threshold.normal_range,
                "requires_consultation": threshold.requires_consultation
            }
        
        return threshold_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get thresholds: {str(e)}")

@router.get("/status")
async def get_monitoring_status():
    """Get the status of the health monitoring service"""
    try:
        status = {
            "service": "Health Monitoring Service",
            "status": "active",
            "monitored_metrics": list(health_monitor.deviation_thresholds.keys()),
            "total_thresholds": len(health_monitor.deviation_thresholds),
            "timestamp": datetime.now()
        }
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get service status: {str(e)}")
