import os
import sys
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import math

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.health_monitoring import (
    HealthMetricComparison, HealthDeviationAlert, LabReportMetrics, 
    PredictedHealthMetrics, DeviationThresholds, HealthMonitoringConfig,
    HealthMonitoringReport, NotificationMessage, DeviationSeverity
)
from services.database import get_database
from services.health_prediction_service import HealthPredictionService

class HealthMonitoringService:
    """Service for monitoring health deviations between predictions and lab reports"""
    
    def __init__(self):
        self.health_predictor = HealthPredictionService()
        self.deviation_thresholds = self._initialize_thresholds()
        
    def _initialize_thresholds(self) -> Dict[str, DeviationThresholds]:
        """Initialize deviation thresholds for different health metrics"""
        return {
            "ldl": DeviationThresholds(
                metric="ldl",
                mild_threshold=10.0,      # 10% deviation
                moderate_threshold=20.0,  # 20% deviation
                severe_threshold=35.0,    # 35% deviation
                critical_threshold=50.0,  # 50% deviation
                clinical_threshold=25.0,  # 25% deviation
                unit="mg/dL",
                normal_range={"min": 0, "max": 100},
                requires_consultation=True
            ),
            "glucose": DeviationThresholds(
                metric="glucose",
                mild_threshold=8.0,       # 8% deviation
                moderate_threshold=15.0,  # 15% deviation
                severe_threshold=25.0,    # 25% deviation
                critical_threshold=40.0,  # 40% deviation
                clinical_threshold=20.0,  # 20% deviation
                unit="mg/dL",
                normal_range={"min": 70, "max": 100},
                requires_consultation=True
            ),
            "hemoglobin": DeviationThresholds(
                metric="hemoglobin",
                mild_threshold=5.0,       # 5% deviation
                moderate_threshold=12.0,  # 12% deviation
                severe_threshold=20.0,    # 20% deviation
                critical_threshold=30.0,  # 30% deviation
                clinical_threshold=15.0,  # 15% deviation
                unit="g/dL",
                normal_range={"min": 12.0, "max": 17.5},
                requires_consultation=True
            ),
            "hdl": DeviationThresholds(
                metric="hdl",
                mild_threshold=12.0,      # 12% deviation
                moderate_threshold=25.0,  # 25% deviation
                severe_threshold=40.0,    # 40% deviation
                critical_threshold=60.0,  # 60% deviation
                clinical_threshold=30.0,  # 30% deviation
                unit="mg/dL",
                normal_range={"min": 40, "max": 60},
                requires_consultation=False
            ),
            "triglycerides": DeviationThresholds(
                metric="triglycerides",
                mild_threshold=15.0,      # 15% deviation
                moderate_threshold=30.0,  # 30% deviation
                severe_threshold=50.0,    # 50% deviation
                critical_threshold=75.0,  # 75% deviation
                clinical_threshold=35.0,  # 35% deviation
                unit="mg/dL",
                normal_range={"min": 0, "max": 150},
                requires_consultation=True
            ),
            "hba1c": DeviationThresholds(
                metric="hba1c",
                mild_threshold=5.0,       # 5% deviation
                moderate_threshold=12.0,  # 12% deviation
                severe_threshold=20.0,    # 20% deviation
                critical_threshold=30.0,  # 30% deviation
                clinical_threshold=15.0,  # 15% deviation
                unit="%",
                normal_range={"min": 4.0, "max": 5.7},
                requires_consultation=True
            )
        }
    
    async def get_latest_lab_report(self, patient_id: str) -> Optional[LabReportMetrics]:
        """Get the latest lab report for a patient"""
        try:
            db = get_database()
            
            # Get the most recent lab report
            lab_report = await db.lab_reports.find_one(
                {"patient_id": patient_id},
                sort=[("upload_date", -1)]
            )
            
            if not lab_report:
                return None
            
            # Extract metrics from lab report
            biomarkers = lab_report.get("biomarkers", {})
            
            return LabReportMetrics(
                report_id=str(lab_report["_id"]),
                patient_id=patient_id,
                report_date=lab_report.get("upload_date", datetime.now()),
                ldl=biomarkers.get("ldl"),
                glucose=biomarkers.get("glucose"),
                hemoglobin=biomarkers.get("hemoglobin"),
                hdl=biomarkers.get("hdl"),
                triglycerides=biomarkers.get("triglycerides"),
                total_cholesterol=biomarkers.get("total_cholesterol"),
                hba1c=biomarkers.get("hba1c"),
                creatinine=biomarkers.get("creatinine"),
                bun=biomarkers.get("bun"),
                sodium=biomarkers.get("sodium"),
                potassium=biomarkers.get("potassium"),
                chloride=biomarkers.get("chloride"),
                co2=biomarkers.get("co2"),
                calcium=biomarkers.get("calcium"),
                phosphorus=biomarkers.get("phosphorus"),
                magnesium=biomarkers.get("magnesium"),
                iron=biomarkers.get("iron"),
                ferritin=biomarkers.get("ferritin"),
                vitamin_d=biomarkers.get("vitamin_d"),
                vitamin_b12=biomarkers.get("vitamin_b12"),
                folate=biomarkers.get("folate"),
                tsh=biomarkers.get("tsh"),
                t4=biomarkers.get("t4"),
                t3=biomarkers.get("t3"),
                crp=biomarkers.get("crp"),
                esr=biomarkers.get("esr"),
                wbc=biomarkers.get("wbc"),
                rbc=biomarkers.get("rbc"),
                platelets=biomarkers.get("platelets"),
                hematocrit=biomarkers.get("hematocrit"),
                mcv=biomarkers.get("mcv"),
                mch=biomarkers.get("mch"),
                mchc=biomarkers.get("mchc"),
                rdw=biomarkers.get("rdw"),
                mpv=biomarkers.get("mpv"),
                pct=biomarkers.get("pct"),
                pdw=biomarkers.get("pdw")
            )
            
        except Exception as e:
            print(f"Error getting latest lab report: {e}")
            return None
    
    async def get_latest_predictions(self, patient_id: str) -> Optional[PredictedHealthMetrics]:
        """Get the latest health predictions for a patient"""
        try:
            db = get_database()
            
            # Get the most recent health prediction
            prediction = await db.health_predictions.find_one(
                {"patient_id": patient_id},
                sort=[("timestamp", -1)]
            )
            
            if not prediction:
                return None
            
            # Extract predictions
            predictions_data = prediction.get("predictions", [])
            predicted_values = {}
            confidence_scores = {}
            
            for pred in predictions_data:
                metric = pred.get("metric")
                predicted_values[metric] = pred.get("predicted_value")
                confidence_scores[metric] = pred.get("confidence", 0.8)
            
            return PredictedHealthMetrics(
                prediction_id=str(prediction["_id"]),
                patient_id=patient_id,
                prediction_date=prediction.get("timestamp", datetime.now()),
                ldl=predicted_values.get("ldl"),
                glucose=predicted_values.get("glucose"),
                hemoglobin=predicted_values.get("hemoglobin"),
                confidence_scores=confidence_scores,
                model_version=prediction.get("model_version", "1.0.0"),
                input_features=prediction.get("input_features", {})
            )
            
        except Exception as e:
            print(f"Error getting latest predictions: {e}")
            return None
    
    def calculate_deviation(self, predicted: float, actual: float) -> Tuple[float, float]:
        """Calculate deviation between predicted and actual values"""
        if actual == 0:
            return 0.0, 0.0
        
        deviation_absolute = predicted - actual
        deviation_percentage = (deviation_absolute / actual) * 100
        
        return deviation_percentage, deviation_absolute
    
    def determine_severity(self, deviation_percentage: float, thresholds: DeviationThresholds) -> DeviationSeverity:
        """Determine severity level based on deviation percentage"""
        abs_deviation = abs(deviation_percentage)
        
        if abs_deviation >= thresholds.critical_threshold:
            return DeviationSeverity.CRITICAL
        elif abs_deviation >= thresholds.severe_threshold:
            return DeviationSeverity.SEVERE
        elif abs_deviation >= thresholds.moderate_threshold:
            return DeviationSeverity.MODERATE
        elif abs_deviation >= thresholds.mild_threshold:
            return DeviationSeverity.MILD
        else:
            return DeviationSeverity.NONE
    
    def assess_clinical_significance(self, metric: str, deviation_percentage: float, 
                                   predicted: float, actual: float, thresholds: DeviationThresholds) -> str:
        """Assess clinical significance of deviation"""
        abs_deviation = abs(deviation_percentage)
        
        if abs_deviation >= thresholds.critical_threshold:
            return f"Critical deviation requiring immediate medical attention"
        elif abs_deviation >= thresholds.severe_threshold:
            return f"Severe deviation suggesting significant health changes"
        elif abs_deviation >= thresholds.moderate_threshold:
            return f"Moderate deviation indicating potential health concerns"
        elif abs_deviation >= thresholds.mild_threshold:
            return f"Mild deviation within normal variation range"
        else:
            return f"Minimal deviation, likely within normal measurement variation"
    
    def generate_recommendation(self, metric: str, severity: DeviationSeverity, 
                              deviation_percentage: float, thresholds: DeviationThresholds) -> str:
        """Generate recommendation based on deviation severity"""
        if severity == DeviationSeverity.CRITICAL:
            return f"URGENT: Consult healthcare provider immediately. {metric.upper()} deviation of {abs(deviation_percentage):.1f}% requires immediate medical evaluation."
        elif severity == DeviationSeverity.SEVERE:
            return f"Schedule medical consultation within 1-2 weeks. {metric.upper()} deviation of {abs(deviation_percentage):.1f}% suggests significant changes requiring professional assessment."
        elif severity == DeviationSeverity.MODERATE:
            return f"Monitor closely and consider consultation if trend continues. {metric.upper()} deviation of {abs(deviation_percentage):.1f}% may indicate health changes."
        elif severity == DeviationSeverity.MILD:
            return f"Continue monitoring. {metric.upper()} deviation of {abs(deviation_percentage):.1f}% is within acceptable range but worth noting."
        else:
            return f"No significant deviation detected in {metric.upper()}. Continue regular monitoring."
    
    async def compare_health_metrics(self, patient_id: str) -> List[HealthMetricComparison]:
        """Compare predicted health metrics with actual lab report values"""
        try:
            # Get latest lab report and predictions
            lab_report = await self.get_latest_lab_report(patient_id)
            predictions = await self.get_latest_predictions(patient_id)
            
            if not lab_report or not predictions:
                return []
            
            comparisons = []
            
            # Compare each metric
            for metric in ["ldl", "glucose", "hemoglobin"]:
                predicted_value = getattr(predictions, metric, None)
                actual_value = getattr(lab_report, metric, None)
                
                if predicted_value is not None and actual_value is not None:
                    # Calculate deviation
                    deviation_percentage, deviation_absolute = self.calculate_deviation(
                        predicted_value, actual_value
                    )
                    
                    # Get thresholds for this metric
                    thresholds = self.deviation_thresholds.get(metric)
                    if not thresholds:
                        continue
                    
                    # Determine severity
                    severity = self.determine_severity(deviation_percentage, thresholds)
                    
                    # Assess clinical significance
                    clinical_significance = self.assess_clinical_significance(
                        metric, deviation_percentage, predicted_value, actual_value, thresholds
                    )
                    
                    # Generate recommendation
                    recommendation = self.generate_recommendation(
                        metric, severity, deviation_percentage, thresholds
                    )
                    
                    # Create comparison
                    comparison = HealthMetricComparison(
                        metric=metric,
                        predicted_value=predicted_value,
                        actual_value=actual_value,
                        unit=thresholds.unit,
                        deviation_percentage=deviation_percentage,
                        deviation_absolute=deviation_absolute,
                        severity=severity,
                        threshold_exceeded=abs(deviation_percentage) >= thresholds.mild_threshold,
                        normal_range=thresholds.normal_range,
                        clinical_significance=clinical_significance,
                        recommendation=recommendation
                    )
                    
                    comparisons.append(comparison)
            
            return comparisons
            
        except Exception as e:
            print(f"Error comparing health metrics: {e}")
            return []
    
    async def generate_health_alert(self, patient_id: str, 
                                  comparisons: List[HealthMetricComparison]) -> Optional[HealthDeviationAlert]:
        """Generate health alert based on deviations"""
        try:
            if not comparisons:
                return None
            
            # Find the highest severity deviation
            severity_levels = [DeviationSeverity.CRITICAL, DeviationSeverity.SEVERE, 
                             DeviationSeverity.MODERATE, DeviationSeverity.MILD, DeviationSeverity.NONE]
            
            max_severity = DeviationSeverity.NONE
            triggered_metrics = []
            critical_deviations = []
            
            for comparison in comparisons:
                if comparison.threshold_exceeded:
                    triggered_metrics.append(comparison.metric)
                    if comparison.severity in [DeviationSeverity.CRITICAL, DeviationSeverity.SEVERE]:
                        critical_deviations.append(comparison)
                    
                    # Update max severity
                    for i, level in enumerate(severity_levels):
                        if comparison.severity == level:
                            if i < severity_levels.index(max_severity):
                                max_severity = comparison.severity
                            break
            
            if not triggered_metrics:
                return None
            
            # Generate alert
            alert_id = str(uuid.uuid4())
            
            # Determine overall assessment
            if max_severity == DeviationSeverity.CRITICAL:
                overall_assessment = "CRITICAL: Multiple significant health deviations detected requiring immediate medical attention."
                urgent_actions = [
                    "Contact healthcare provider immediately",
                    "Schedule emergency consultation if symptoms present",
                    "Monitor for any new symptoms",
                    "Review recent lifestyle changes"
                ]
                consultation_urgency = "immediate"
                follow_up_timeline = "within 24-48 hours"
                requires_consultation = True
            elif max_severity == DeviationSeverity.SEVERE:
                overall_assessment = "SEVERE: Significant health deviations detected requiring prompt medical evaluation."
                urgent_actions = [
                    "Schedule medical consultation within 1-2 weeks",
                    "Monitor health metrics closely",
                    "Review and adjust lifestyle factors",
                    "Prepare for detailed health discussion"
                ]
                consultation_urgency = "within 1-2 weeks"
                follow_up_timeline = "within 1-2 weeks"
                requires_consultation = True
            elif max_severity == DeviationSeverity.MODERATE:
                overall_assessment = "MODERATE: Health deviations detected requiring monitoring and potential consultation."
                urgent_actions = [
                    "Monitor health metrics regularly",
                    "Consider lifestyle adjustments",
                    "Schedule consultation if trend continues",
                    "Track any new symptoms"
                ]
                consultation_urgency = "within 1 month"
                follow_up_timeline = "within 1 month"
                requires_consultation = len(critical_deviations) > 0
            else:
                overall_assessment = "MILD: Minor health deviations detected within normal variation range."
                urgent_actions = [
                    "Continue regular monitoring",
                    "Maintain healthy lifestyle",
                    "Note any persistent changes",
                    "Follow up with routine care"
                ]
                consultation_urgency = "routine"
                follow_up_timeline = "next routine visit"
                requires_consultation = False
            
            # Generate recommendations
            recommendations = []
            for comparison in comparisons:
                if comparison.threshold_exceeded:
                    recommendations.append(comparison.recommendation)
            
            # Add general recommendations
            if max_severity in [DeviationSeverity.CRITICAL, DeviationSeverity.SEVERE]:
                recommendations.extend([
                    "Bring this alert to your healthcare provider's attention",
                    "Prepare a list of recent lifestyle changes",
                    "Note any new symptoms or concerns",
                    "Consider bringing recent lab reports for comparison"
                ])
            
            return HealthDeviationAlert(
                alert_id=alert_id,
                patient_id=patient_id,
                timestamp=datetime.now(),
                severity=max_severity,
                triggered_metrics=triggered_metrics,
                deviations=comparisons,
                overall_assessment=overall_assessment,
                urgent_actions=urgent_actions,
                recommendations=recommendations,
                requires_medical_consultation=requires_consultation,
                consultation_urgency=consultation_urgency,
                follow_up_timeline=follow_up_timeline
            )
            
        except Exception as e:
            print(f"Error generating health alert: {e}")
            return None
    
    async def create_notification(self, patient_id: str, alert: HealthDeviationAlert) -> NotificationMessage:
        """Create notification message for health deviation alert"""
        try:
            notification_id = str(uuid.uuid4())
            
            # Determine notification title and message based on severity
            if alert.severity == DeviationSeverity.CRITICAL:
                title = "🚨 CRITICAL Health Alert - Immediate Action Required"
                message = f"Critical health deviations detected. {alert.overall_assessment} Please consult your healthcare provider immediately."
                action_required = True
                action_deadline = datetime.now() + timedelta(hours=24)
            elif alert.severity == DeviationSeverity.SEVERE:
                title = "⚠️ Health Alert - Medical Consultation Recommended"
                message = f"Significant health deviations detected. {alert.overall_assessment} Schedule a consultation within 1-2 weeks."
                action_required = True
                action_deadline = datetime.now() + timedelta(days=14)
            elif alert.severity == DeviationSeverity.MODERATE:
                title = "📊 Health Monitoring Alert"
                message = f"Health deviations detected requiring attention. {alert.overall_assessment} Monitor closely and consider consultation if trend continues."
                action_required = False
                action_deadline = None
            else:
                title = "📈 Health Update"
                message = f"Minor health variations detected. {alert.overall_assessment} Continue regular monitoring."
                action_required = False
                action_deadline = None
            
            return NotificationMessage(
                notification_id=notification_id,
                patient_id=patient_id,
                alert_id=alert.alert_id,
                timestamp=datetime.now(),
                severity=alert.severity,
                title=title,
                message=message,
                action_required=action_required,
                action_deadline=action_deadline,
                notification_type="system_alert",
                sent=False,
                read=False,
                acknowledged=False
            )
            
        except Exception as e:
            print(f"Error creating notification: {e}")
            return None
    
    async def monitor_health_deviations(self, patient_id: str) -> HealthMonitoringReport:
        """Main function to monitor health deviations and generate comprehensive report"""
        try:
            # Compare health metrics
            comparisons = await self.compare_health_metrics(patient_id)
            
            if not comparisons:
                return HealthMonitoringReport(
                    report_id=str(uuid.uuid4()),
                    patient_id=patient_id,
                    report_date=datetime.now(),
                    comparison_date=datetime.now(),
                    overall_health_status="No data available for comparison",
                    risk_level="unknown",
                    metrics_compared=0,
                    deviations_found=0,
                    critical_deviations=0,
                    alerts_generated=[],
                    recommendations=["Ensure lab reports and predictions are available for monitoring"],
                    next_monitoring_date=datetime.now() + timedelta(days=7),
                    requires_immediate_action=False,
                    summary="No health data available for comparison"
                )
            
            # Generate health alert
            alert = await self.generate_health_alert(patient_id, comparisons)
            
            # Create notification if alert exists
            notification = None
            if alert:
                notification = await self.create_notification(patient_id, alert)
            
            # Calculate statistics
            metrics_compared = len(comparisons)
            deviations_found = len([c for c in comparisons if c.threshold_exceeded])
            critical_deviations = len([c for c in comparisons if c.severity in [DeviationSeverity.CRITICAL, DeviationSeverity.SEVERE]])
            
            # Determine overall health status
            if critical_deviations > 0:
                overall_health_status = "Requires immediate medical attention"
                risk_level = "high"
                requires_immediate_action = True
            elif deviations_found > 0:
                overall_health_status = "Requires monitoring and potential consultation"
                risk_level = "moderate"
                requires_immediate_action = False
            else:
                overall_health_status = "Within normal ranges"
                risk_level = "low"
                requires_immediate_action = False
            
            # Generate recommendations
            recommendations = []
            if alert:
                recommendations.extend(alert.recommendations)
            else:
                recommendations.append("Continue regular health monitoring")
                recommendations.append("Maintain healthy lifestyle practices")
                recommendations.append("Schedule routine checkups as recommended")
            
            # Determine next monitoring date
            if requires_immediate_action:
                next_monitoring_date = datetime.now() + timedelta(days=1)
            elif deviations_found > 0:
                next_monitoring_date = datetime.now() + timedelta(days=3)
            else:
                next_monitoring_date = datetime.now() + timedelta(days=7)
            
            # Generate summary
            summary = f"Health monitoring completed for {patient_id}. "
            summary += f"Compared {metrics_compared} metrics, found {deviations_found} deviations "
            summary += f"({critical_deviations} critical). Overall status: {overall_health_status}."
            
            # Store alert and notification in database
            db = get_database()
            if alert:
                await db.health_alerts.insert_one(alert.dict())
            if notification:
                await db.notifications.insert_one(notification.dict())
            
            return HealthMonitoringReport(
                report_id=str(uuid.uuid4()),
                patient_id=patient_id,
                report_date=datetime.now(),
                comparison_date=datetime.now(),
                overall_health_status=overall_health_status,
                risk_level=risk_level,
                metrics_compared=metrics_compared,
                deviations_found=deviations_found,
                critical_deviations=critical_deviations,
                alerts_generated=[alert] if alert else [],
                recommendations=recommendations,
                next_monitoring_date=next_monitoring_date,
                requires_immediate_action=requires_immediate_action,
                summary=summary
            )
            
        except Exception as e:
            print(f"Error in health monitoring: {e}")
            return HealthMonitoringReport(
                report_id=str(uuid.uuid4()),
                patient_id=patient_id,
                report_date=datetime.now(),
                comparison_date=datetime.now(),
                overall_health_status="Error occurred during monitoring",
                risk_level="unknown",
                metrics_compared=0,
                deviations_found=0,
                critical_deviations=0,
                alerts_generated=[],
                recommendations=["Contact support if issue persists"],
                next_monitoring_date=datetime.now() + timedelta(days=1),
                requires_immediate_action=False,
                summary=f"Error occurred during health monitoring: {str(e)}"
            )
    
    async def get_monitoring_history(self, patient_id: str, days: int = 30) -> List[HealthMonitoringReport]:
        """Get monitoring history for a patient"""
        try:
            db = get_database()
            
            # Get monitoring reports from database
            start_date = datetime.now() - timedelta(days=days)
            
            reports = await db.health_monitoring_reports.find({
                "patient_id": patient_id,
                "report_date": {"$gte": start_date}
            }).sort("report_date", -1).to_list(100)
            
            # Convert to HealthMonitoringReport objects
            monitoring_reports = []
            for report_data in reports:
                # Convert alerts back to objects
                alerts = []
                for alert_data in report_data.get("alerts_generated", []):
                    alerts.append(HealthDeviationAlert(**alert_data))
                
                report = HealthMonitoringReport(
                    report_id=report_data["report_id"],
                    patient_id=report_data["patient_id"],
                    report_date=report_data["report_date"],
                    comparison_date=report_data["comparison_date"],
                    overall_health_status=report_data["overall_health_status"],
                    risk_level=report_data["risk_level"],
                    metrics_compared=report_data["metrics_compared"],
                    deviations_found=report_data["deviations_found"],
                    critical_deviations=report_data["critical_deviations"],
                    alerts_generated=alerts,
                    recommendations=report_data["recommendations"],
                    next_monitoring_date=report_data["next_monitoring_date"],
                    requires_immediate_action=report_data["requires_immediate_action"],
                    summary=report_data["summary"]
                )
                monitoring_reports.append(report)
            
            return monitoring_reports
            
        except Exception as e:
            print(f"Error getting monitoring history: {e}")
            return []
