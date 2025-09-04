import statistics
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from models.wearable_data import (
    DailyLog, HeartRateData, SpO2Data, SleepData, ActivityData, 
    StepsData, CaloriesData, TemperatureData
)

class WearableDataProcessor:
    """Service for processing and analyzing wearable data"""
    
    def __init__(self):
        self.max_data_points_per_day = 10000  # Prevent excessive data storage
        
    async def process_wearable_data(self, data_upload) -> Dict[str, Any]:
        """Process wearable data upload and create daily log"""
        try:
            # Validate data volume
            total_data_points = self._count_data_points(data_upload)
            if total_data_points > self.max_data_points_per_day:
                raise ValueError(f"Too many data points: {total_data_points}. Maximum allowed: {self.max_data_points_per_day}")
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(data_upload)
            
            # Calculate data quality score
            data_quality_score = self._calculate_data_quality_score(data_upload)
            
            # Create daily log
            daily_log_data = {
                "patient_id": data_upload.patient_id,
                "date": data_upload.date,
                "device_id": data_upload.device_id,
                "device_type": data_upload.device_type,
                
                # Summary statistics
                "total_steps": summary_stats.get("total_steps"),
                "total_calories_burned": summary_stats.get("total_calories_burned"),
                "total_sleep_minutes": summary_stats.get("total_sleep_minutes"),
                "avg_heart_rate": summary_stats.get("avg_heart_rate"),
                "avg_spo2": summary_stats.get("avg_spo2"),
                
                # Detailed data
                "heart_rate_data": [hr.dict() for hr in data_upload.heart_rate_data],
                "spo2_data": [spo2.dict() for spo2 in data_upload.spo2_data],
                "sleep_data": [sleep.dict() for sleep in data_upload.sleep_data],
                "activity_data": [activity.dict() for activity in data_upload.activity_data],
                "steps_data": [steps.dict() for steps in data_upload.steps_data],
                "calories_data": [calories.dict() for calories in data_upload.calories_data],
                "temperature_data": [temp.dict() for temp in data_upload.temperature_data],
                
                # Raw data
                "raw_data": data_upload.raw_data,
                
                # Metadata
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "data_quality_score": data_quality_score
            }
            
            return {
                "daily_log_data": daily_log_data,
                "summary_stats": summary_stats,
                "data_quality_score": data_quality_score,
                "total_data_points": total_data_points,
                "processing_status": "completed"
            }
            
        except Exception as e:
            return {
                "daily_log_data": None,
                "summary_stats": {},
                "data_quality_score": 0.0,
                "total_data_points": 0,
                "processing_status": "failed",
                "error": str(e)
            }
    
    def _count_data_points(self, data_upload) -> int:
        """Count total number of data points"""
        count = 0
        count += len(data_upload.heart_rate_data or [])
        count += len(data_upload.spo2_data or [])
        count += len(data_upload.sleep_data or [])
        count += len(data_upload.activity_data or [])
        count += len(data_upload.steps_data or [])
        count += len(data_upload.calories_data or [])
        count += len(data_upload.temperature_data or [])
        return count
    
    def _calculate_summary_statistics(self, data_upload) -> Dict[str, Any]:
        """Calculate summary statistics from wearable data"""
        stats = {}
        
        # Heart rate statistics
        if data_upload.heart_rate_data:
            heart_rates = [hr.heart_rate for hr in data_upload.heart_rate_data if hr.heart_rate]
            if heart_rates:
                stats["avg_heart_rate"] = round(statistics.mean(heart_rates), 1)
                stats["min_heart_rate"] = min(heart_rates)
                stats["max_heart_rate"] = max(heart_rates)
                stats["heart_rate_count"] = len(heart_rates)
        
        # SpO2 statistics
        if data_upload.spo2_data:
            spo2_values = [spo2.spo2_percentage for spo2 in data_upload.spo2_data if spo2.spo2_percentage]
            if spo2_values:
                stats["avg_spo2"] = round(statistics.mean(spo2_values), 1)
                stats["min_spo2"] = min(spo2_values)
                stats["max_spo2"] = max(spo2_values)
                stats["spo2_count"] = len(spo2_values)
        
        # Sleep statistics
        if data_upload.sleep_data:
            total_sleep_minutes = sum(sleep.duration_minutes for sleep in data_upload.sleep_data)
            stats["total_sleep_minutes"] = total_sleep_minutes
            stats["total_sleep_hours"] = round(total_sleep_minutes / 60, 1)
            
            # Calculate sleep efficiency
            sleep_efficiencies = [sleep.efficiency_percentage for sleep in data_upload.sleep_data if sleep.efficiency_percentage]
            if sleep_efficiencies:
                stats["avg_sleep_efficiency"] = round(statistics.mean(sleep_efficiencies), 1)
        
        # Activity statistics
        if data_upload.activity_data:
            total_activity_minutes = sum(activity.duration_minutes for activity in data_upload.activity_data)
            stats["total_activity_minutes"] = total_activity_minutes
            stats["total_activity_hours"] = round(total_activity_minutes / 60, 1)
            
            # Activity type breakdown
            activity_types = {}
            for activity in data_upload.activity_data:
                activity_type = activity.activity_type
                if activity_type not in activity_types:
                    activity_types[activity_type] = 0
                activity_types[activity_type] += activity.duration_minutes
            stats["activity_breakdown"] = activity_types
        
        # Steps statistics
        if data_upload.steps_data:
            total_steps = sum(steps.steps_count for steps in data_upload.steps_data)
            stats["total_steps"] = total_steps
            
            total_distance = sum(steps.distance_meters for steps in data_upload.steps_data if steps.distance_meters)
            if total_distance:
                stats["total_distance_km"] = round(total_distance / 1000, 2)
        
        # Calories statistics
        if data_upload.calories_data:
            total_calories_burned = sum(calories.calories_burned for calories in data_upload.calories_data)
            stats["total_calories_burned"] = round(total_calories_burned, 1)
            
            total_calories_consumed = sum(calories.calories_consumed for calories in data_upload.calories_data if calories.calories_consumed)
            if total_calories_consumed:
                stats["total_calories_consumed"] = round(total_calories_consumed, 1)
                stats["net_calories"] = round(total_calories_burned - total_calories_consumed, 1)
        
        # Temperature statistics
        if data_upload.temperature_data:
            temperatures = [temp.temperature_celsius for temp in data_upload.temperature_data if temp.temperature_celsius]
            if temperatures:
                stats["avg_temperature"] = round(statistics.mean(temperatures), 1)
                stats["min_temperature"] = min(temperatures)
                stats["max_temperature"] = max(temperatures)
        
        return stats
    
    def _calculate_data_quality_score(self, data_upload) -> float:
        """Calculate data quality score based on completeness and consistency"""
        score = 0.0
        max_score = 100.0
        
        # Check data completeness
        data_types = [
            ("heart_rate_data", 20),
            ("spo2_data", 15),
            ("sleep_data", 20),
            ("activity_data", 15),
            ("steps_data", 15),
            ("calories_data", 10),
            ("temperature_data", 5)
        ]
        
        for data_type, weight in data_types:
            data_list = getattr(data_upload, data_type, [])
            if data_list:
                score += weight
                
                # Check data quality within each type
                if data_type == "heart_rate_data":
                    valid_hr = [hr for hr in data_list if 30 <= hr.heart_rate <= 220]
                    if valid_hr:
                        score += weight * 0.5 * (len(valid_hr) / len(data_list))
                
                elif data_type == "spo2_data":
                    valid_spo2 = [spo2 for spo2 in data_list if 70 <= spo2.spo2_percentage <= 100]
                    if valid_spo2:
                        score += weight * 0.5 * (len(valid_spo2) / len(data_list))
        
        # Check for device information
        if data_upload.device_id:
            score += 5
        if data_upload.device_type:
            score += 5
        
        # Check for raw data (bonus for flexibility)
        if data_upload.raw_data:
            score += 5
        
        return min(score, max_score) / max_score  # Normalize to 0-1
    
    async def analyze_trends(self, daily_logs: List[DailyLog]) -> Dict[str, Any]:
        """Analyze trends across multiple daily logs"""
        if not daily_logs:
            return {}
        
        trends = {}
        
        # Heart rate trends
        heart_rates = []
        dates = []
        for log in daily_logs:
            if log.avg_heart_rate:
                heart_rates.append(log.avg_heart_rate)
                dates.append(log.date)
        
        if heart_rates:
            trends["heart_rate"] = {
                "trend": self._calculate_trend(heart_rates),
                "avg": round(statistics.mean(heart_rates), 1),
                "min": min(heart_rates),
                "max": max(heart_rates),
                "volatility": round(statistics.stdev(heart_rates), 1) if len(heart_rates) > 1 else 0
            }
        
        # Sleep trends
        sleep_minutes = []
        for log in daily_logs:
            if log.total_sleep_minutes:
                sleep_minutes.append(log.total_sleep_minutes)
        
        if sleep_minutes:
            trends["sleep"] = {
                "trend": self._calculate_trend(sleep_minutes),
                "avg_hours": round(statistics.mean(sleep_minutes) / 60, 1),
                "min_hours": round(min(sleep_minutes) / 60, 1),
                "max_hours": round(max(sleep_minutes) / 60, 1)
            }
        
        # Activity trends
        activity_minutes = []
        for log in daily_logs:
            if log.total_calories_burned:
                # Estimate activity minutes from calories (rough approximation)
                estimated_minutes = log.total_calories_burned / 5  # Assume 5 calories per minute
                activity_minutes.append(estimated_minutes)
        
        if activity_minutes:
            trends["activity"] = {
                "trend": self._calculate_trend(activity_minutes),
                "avg_minutes": round(statistics.mean(activity_minutes), 1),
                "total_hours": round(sum(activity_minutes) / 60, 1)
            }
        
        # Steps trends
        steps = []
        for log in daily_logs:
            if log.total_steps:
                steps.append(log.total_steps)
        
        if steps:
            trends["steps"] = {
                "trend": self._calculate_trend(steps),
                "avg_daily": round(statistics.mean(steps)),
                "total": sum(steps),
                "goal_achievement": round(statistics.mean(steps) / 10000 * 100, 1)  # 10k steps goal
            }
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        if not first_half or not second_half:
            return "insufficient_data"
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        change_percent = ((second_avg - first_avg) / first_avg) * 100
        
        if change_percent > 5:
            return "increasing"
        elif change_percent < -5:
            return "decreasing"
        else:
            return "stable"
    
    async def generate_health_insights(self, daily_logs: List[DailyLog]) -> List[Dict[str, Any]]:
        """Generate health insights from wearable data"""
        insights = []
        
        if not daily_logs:
            return insights
        
        # Analyze recent data (last 7 days)
        recent_logs = [log for log in daily_logs if log.date >= datetime.now() - timedelta(days=7)]
        
        if recent_logs:
            # Sleep insights
            avg_sleep_hours = statistics.mean([log.total_sleep_minutes / 60 for log in recent_logs if log.total_sleep_minutes])
            if avg_sleep_hours < 7:
                insights.append({
                    "type": "sleep",
                    "severity": "warning",
                    "message": f"Average sleep duration is {avg_sleep_hours:.1f} hours, below recommended 7-9 hours",
                    "recommendation": "Try to maintain a consistent sleep schedule and create a relaxing bedtime routine"
                })
            
            # Activity insights
            avg_steps = statistics.mean([log.total_steps for log in recent_logs if log.total_steps])
            if avg_steps < 5000:
                insights.append({
                    "type": "activity",
                    "severity": "warning",
                    "message": f"Average daily steps is {avg_steps:.0f}, below recommended 10,000 steps",
                    "recommendation": "Try to increase daily activity by taking short walks or using stairs"
                })
            
            # Heart rate insights
            avg_hr = statistics.mean([log.avg_heart_rate for log in recent_logs if log.avg_heart_rate])
            if avg_hr > 100:
                insights.append({
                    "type": "heart_rate",
                    "severity": "warning",
                    "message": f"Average heart rate is {avg_hr:.0f} bpm, which is elevated",
                    "recommendation": "Consider stress management techniques and consult with healthcare provider if persistent"
                })
        
        return insights
