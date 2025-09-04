#!/usr/bin/env python3
"""
Test script for lab report upload functionality
"""

import requests
import json
from pathlib import Path

def test_lab_report_upload():
    """Test the lab report upload endpoint"""
    
    # API configuration
    API_BASE_URL = "http://localhost:8000"
    
    print("🧪 Testing Lab Report Upload Functionality")
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
    
    # Test file path (you would need to create a sample PDF)
    test_pdf_path = Path("sample_lab_report.pdf")
    
    if not test_pdf_path.exists():
        print(f"📄 Creating sample lab report content...")
        create_sample_lab_report()
        print(f"✅ Sample lab report created at: {test_pdf_path}")
    
    # Test the upload endpoint
    print("\n📤 Testing lab report upload...")
    
    try:
        with open(test_pdf_path, "rb") as f:
            files = {"file": ("sample_lab_report.pdf", f, "application/pdf")}
            data = {
                "patient_id": patient_id,
                "report_date": "2024-01-15",
                "lab_name": "Test Laboratory",
                "notes": "Sample lab report for testing"
            }
            
            response = requests.post(
                f"{API_BASE_URL}/api/v1/lab-reports/upload-lab-report",
                files=files,
                data=data
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Lab report uploaded successfully!")
            print(f"📊 Report ID: {result.get('report_id')}")
            print(f"🔬 Biomarkers found: {result.get('biomarkers_found')}")
            print(f"⏱️ Processing time: {result.get('processing_time')} seconds")
            
            # Display extracted biomarkers
            if result.get('data', {}).get('biomarkers'):
                print("\n📈 Extracted Biomarkers:")
                for biomarker in result['data']['biomarkers']:
                    status_emoji = {
                        "normal": "✅",
                        "high": "⚠️",
                        "low": "⚠️",
                        "unknown": "❓"
                    }.get(biomarker['status'], "❓")
                    
                    print(f"  {status_emoji} {biomarker['name']}: {biomarker['value']} {biomarker['unit']} ({biomarker['status']}) - Confidence: {biomarker['extracted_confidence']:.2f}")
            
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during upload: {e}")
    
    # Test getting lab reports
    print("\n📋 Testing lab reports retrieval...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/lab-reports?patient_id={patient_id}")
        if response.status_code == 200:
            reports = response.json()
            print(f"✅ Found {len(reports)} lab reports for patient")
        else:
            print(f"❌ Failed to get lab reports: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting lab reports: {e}")

def create_sample_lab_report():
    """Create a sample lab report PDF for testing"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create a simple PDF with lab report data
        c = canvas.Canvas("sample_lab_report.pdf", pagesize=letter)
        width, height = letter
        
        # Add header
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "LABORATORY REPORT")
        
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, "Patient: John Doe")
        c.drawString(50, height - 100, "Date: 01/15/2024")
        c.drawString(50, height - 120, "Lab: Test Laboratory")
        
        # Add biomarker results
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 160, "LABORATORY RESULTS")
        
        c.setFont("Helvetica", 10)
        y_position = height - 190
        
        # Sample biomarker data
        biomarkers = [
            ("LDL", "120", "mg/dL", "<100", "High"),
            ("HDL", "45", "mg/dL", ">40", "Normal"),
            ("Glucose", "95", "mg/dL", "70-100", "Normal"),
            ("Hemoglobin", "14.2", "g/dL", "12-16", "Normal"),
            ("Total Cholesterol", "200", "mg/dL", "<200", "Normal"),
            ("Triglycerides", "150", "mg/dL", "<150", "Normal"),
            ("Creatinine", "1.1", "mg/dL", "0.7-1.3", "Normal"),
            ("Sodium", "140", "mEq/L", "135-145", "Normal"),
            ("Potassium", "4.0", "mEq/L", "3.5-5.0", "Normal"),
            ("WBC", "7.5", "K/uL", "4.5-11.0", "Normal"),
            ("RBC", "4.8", "M/uL", "4.2-5.8", "Normal"),
            ("Platelets", "250", "K/uL", "150-450", "Normal")
        ]
        
        for name, value, unit, ref_range, status in biomarkers:
            c.drawString(50, y_position, f"{name}: {value} {unit} ({ref_range}) - {status}")
            y_position -= 20
        
        c.save()
        print("✅ Sample lab report PDF created successfully")
        
    except ImportError:
        print("⚠️ reportlab not available, creating text file instead")
        # Create a simple text file as fallback
        with open("sample_lab_report.pdf", "w") as f:
            f.write("""LABORATORY REPORT
Patient: John Doe
Date: 01/15/2024
Lab: Test Laboratory

LABORATORY RESULTS:
LDL: 120 mg/dL (<100) - High
HDL: 45 mg/dL (>40) - Normal
Glucose: 95 mg/dL (70-100) - Normal
Hemoglobin: 14.2 g/dL (12-16) - Normal
Total Cholesterol: 200 mg/dL (<200) - Normal
Triglycerides: 150 mg/dL (<150) - Normal
Creatinine: 1.1 mg/dL (0.7-1.3) - Normal
Sodium: 140 mEq/L (135-145) - Normal
Potassium: 4.0 mEq/L (3.5-5.0) - Normal
WBC: 7.5 K/uL (4.5-11.0) - Normal
RBC: 4.8 M/uL (4.2-5.8) - Normal
Platelets: 250 K/uL (150-450) - Normal
""")
        print("✅ Sample lab report text file created")

if __name__ == "__main__":
    test_lab_report_upload()
