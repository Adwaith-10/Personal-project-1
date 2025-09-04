from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class DeviationSeverity(str, Enum):
    """Deviation severity levels"""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

class HealthMetricComparison(BaseModel):
    """Comparison between predicted and actual health metrics"""
    metric: str = Field(..., description="Health metric name (e.g., ldl, glucose, hemoglobin)")
    predicted_value: float = Field(..., description="Predicted value from ML model")
    actual_value: float = Field(..., description="Actual value from lab report")
    unit: str = Field(..., description="Unit of measurement")
    deviation_percentage: float = Field(..., description="Percentage deviation from actual")
    deviation_absolute: float = Field(..., description="Absolute deviation from actual")
    severity: DeviationSeverity = Field(..., description="Deviation severity level")
    threshold_exceeded: bool = Field(..., description="Whether deviation exceeds threshold")
    normal_range: Dict[str, float] = Field(..., description="Normal range for the metric")
    clinical_significance: str = Field(..., description="Clinical significance of deviation")
    recommendation: str = Field(..., description="Recommended action")

class HealthDeviationAlert(BaseModel):
    """Alert for health metric deviations"""
    alert_id: str = Field(..., description="Unique alert ID")
    patient_id: str = Field(..., description="Patient ID")
    timestamp: datetime = Field(..., description="Alert timestamp")
    severity: DeviationSeverity = Field(..., description="Overall alert severity")
    triggered_metrics: List[str] = Field(..., description="Metrics that triggered the alert")
    deviations: List[HealthMetricComparison] = Field(..., description="Detailed deviations")
    overall_assessment: str = Field(..., description="Overall health assessment")
    urgent_actions: List[str] = Field(..., description="Urgent actions required")
    recommendations: List[str] = Field(..., description="General recommendations")
    requires_medical_consultation: bool = Field(..., description="Whether medical consultation is required")
    consultation_urgency: str = Field(..., description="Consultation urgency level")
    follow_up_timeline: str = Field(..., description="Recommended follow-up timeline")

class LabReportMetrics(BaseModel):
    """Lab report metrics for comparison"""
    report_id: str = Field(..., description="Lab report ID")
    patient_id: str = Field(..., description="Patient ID")
    report_date: datetime = Field(..., description="Lab report date")
    ldl: Optional[float] = Field(None, description="LDL cholesterol (mg/dL)")
    glucose: Optional[float] = Field(None, description="Glucose (mg/dL)")
    hemoglobin: Optional[float] = Field(None, description="Hemoglobin (g/dL)")
    hdl: Optional[float] = Field(None, description="HDL cholesterol (mg/dL)")
    triglycerides: Optional[float] = Field(None, description="Triglycerides (mg/dL)")
    total_cholesterol: Optional[float] = Field(None, description="Total cholesterol (mg/dL)")
    hba1c: Optional[float] = Field(None, description="HbA1c (%)")
    creatinine: Optional[float] = Field(None, description="Creatinine (mg/dL)")
    bun: Optional[float] = Field(None, description="Blood urea nitrogen (mg/dL)")
    sodium: Optional[float] = Field(None, description="Sodium (mEq/L)")
    potassium: Optional[float] = Field(None, description="Potassium (mEq/L)")
    chloride: Optional[float] = Field(None, description="Chloride (mEq/L)")
    co2: Optional[float] = Field(None, description="CO2 (mEq/L)")
    calcium: Optional[float] = Field(None, description="Calcium (mg/dL)")
    phosphorus: Optional[float] = Field(None, description="Phosphorus (mg/dL)")
    magnesium: Optional[float] = Field(None, description="Magnesium (mg/dL)")
    iron: Optional[float] = Field(None, description="Iron (mcg/dL)")
    ferritin: Optional[float] = Field(None, description="Ferritin (ng/mL)")
    vitamin_d: Optional[float] = Field(None, description="Vitamin D (ng/mL)")
    vitamin_b12: Optional[float] = Field(None, description="Vitamin B12 (pg/mL)")
    folate: Optional[float] = Field(None, description="Folate (ng/mL)")
    tsh: Optional[float] = Field(None, description="TSH (mIU/L)")
    t4: Optional[float] = Field(None, description="T4 (mcg/dL)")
    t3: Optional[float] = Field(None, description="T3 (ng/dL)")
    crp: Optional[float] = Field(None, description="C-reactive protein (mg/L)")
    esr: Optional[float] = Field(None, description="Erythrocyte sedimentation rate (mm/hr)")
    wbc: Optional[float] = Field(None, description="White blood cell count (K/uL)")
    rbc: Optional[float] = Field(None, description="Red blood cell count (M/uL)")
    platelets: Optional[float] = Field(None, description="Platelet count (K/uL)")
    hematocrit: Optional[float] = Field(None, description="Hematocrit (%)")
    mcv: Optional[float] = Field(None, description="Mean corpuscular volume (fL)")
    mch: Optional[float] = Field(None, description="Mean corpuscular hemoglobin (pg)")
    mchc: Optional[float] = Field(None, description="Mean corpuscular hemoglobin concentration (g/dL)")
    rdw: Optional[float] = Field(None, description="Red cell distribution width (%)")
    mpv: Optional[float] = Field(None, description="Mean platelet volume (fL)")
    pct: Optional[float] = Field(None, description="Plateletcrit (%)")
    pdw: Optional[float] = Field(None, description="Platelet distribution width (%)")

class PredictedHealthMetrics(BaseModel):
    """Predicted health metrics from ML models"""
    prediction_id: str = Field(..., description="Prediction ID")
    patient_id: str = Field(..., description="Patient ID")
    prediction_date: datetime = Field(..., description="Prediction date")
    ldl: Optional[float] = Field(None, description="Predicted LDL cholesterol (mg/dL)")
    glucose: Optional[float] = Field(None, description="Predicted glucose (mg/dL)")
    hemoglobin: Optional[float] = Field(None, description="Predicted hemoglobin (g/dL)")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores for predictions")
    model_version: str = Field(..., description="ML model version used")
    input_features: Dict[str, Any] = Field(..., description="Input features used for prediction")

class DeviationThresholds(BaseModel):
    """Thresholds for deviation detection"""
    metric: str = Field(..., description="Health metric name")
    mild_threshold: float = Field(..., description="Mild deviation threshold (%)")
    moderate_threshold: float = Field(..., description="Moderate deviation threshold (%)")
    severe_threshold: float = Field(..., description="Severe deviation threshold (%)")
    critical_threshold: float = Field(..., description="Critical deviation threshold (%)")
    clinical_threshold: float = Field(..., description="Clinical significance threshold")
    unit: str = Field(..., description="Unit of measurement")
    normal_range: Dict[str, float] = Field(..., description="Normal range")
    requires_consultation: bool = Field(..., description="Whether deviation requires medical consultation")

class HealthMonitoringConfig(BaseModel):
    """Configuration for health monitoring"""
    patient_id: str = Field(..., description="Patient ID")
    monitoring_enabled: bool = Field(True, description="Whether monitoring is enabled")
    alert_threshold: float = Field(15.0, description="Default alert threshold (%)")
    critical_threshold: float = Field(30.0, description="Critical alert threshold (%)")
    monitoring_frequency: str = Field("daily", description="Monitoring frequency")
    notification_preferences: Dict[str, bool] = Field(..., description="Notification preferences")
    auto_consultation_trigger: bool = Field(True, description="Auto-trigger consultation for severe deviations")
    consultation_threshold: float = Field(25.0, description="Consultation trigger threshold (%)")

class HealthMonitoringReport(BaseModel):
    """Comprehensive health monitoring report"""
    report_id: str = Field(..., description="Report ID")
    patient_id: str = Field(..., description="Patient ID")
    report_date: datetime = Field(..., description="Report date")
    comparison_date: datetime = Field(..., description="Date of lab report being compared")
    overall_health_status: str = Field(..., description="Overall health status")
    risk_level: str = Field(..., description="Overall risk level")
    metrics_compared: int = Field(..., description="Number of metrics compared")
    deviations_found: int = Field(..., description="Number of deviations found")
    critical_deviations: int = Field(..., description="Number of critical deviations")
    alerts_generated: List[HealthDeviationAlert] = Field(..., description="Generated alerts")
    recommendations: List[str] = Field(..., description="General recommendations")
    next_monitoring_date: datetime = Field(..., description="Next monitoring date")
    requires_immediate_action: bool = Field(..., description="Whether immediate action is required")
    summary: str = Field(..., description="Executive summary")

class NotificationMessage(BaseModel):
    """Notification message for health deviations"""
    notification_id: str = Field(..., description="Notification ID")
    patient_id: str = Field(..., description="Patient ID")
    alert_id: str = Field(..., description="Associated alert ID")
    timestamp: datetime = Field(..., description="Notification timestamp")
    severity: DeviationSeverity = Field(..., description="Notification severity")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    action_required: bool = Field(..., description="Whether action is required")
    action_deadline: Optional[datetime] = Field(None, description="Action deadline")
    notification_type: str = Field(..., description="Notification type (email, sms, push, etc.)")
    sent: bool = Field(False, description="Whether notification was sent")
    read: bool = Field(False, description="Whether notification was read")
    acknowledged: bool = Field(False, description="Whether notification was acknowledged")
