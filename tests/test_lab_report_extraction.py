"""
Tests for lab report extraction functionality
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import io

from app.services.lab_report_processor import LabReportProcessor
from app.models.lab_report import LabReport, BiomarkerResult, LabReportUpload


class TestLabReportExtraction:
    """Test lab report extraction functionality"""
    
    @pytest.fixture
    def processor(self):
        """Create a lab report processor instance."""
        return LabReportProcessor()
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF content for testing."""
        return """
        LABORATORY REPORT
        Patient: John Doe
        Date: 2024-01-15
        Lab: LabCorp
        
        RESULTS:
        LDL Cholesterol: 120 mg/dL (Reference: <100)
        Glucose: 95 mg/dL (Reference: 70-100)
        Hemoglobin: 14.5 g/dL (Reference: 13.5-17.5)
        HDL Cholesterol: 55 mg/dL (Reference: >40)
        Triglycerides: 150 mg/dL (Reference: <150)
        
        INTERPRETATION:
        LDL is elevated. Consider lifestyle modifications.
        Other values are within normal range.
        """
    
    @pytest.fixture
    def complex_pdf_content(self):
        """Complex PDF content with various formats."""
        return """
        COMPREHENSIVE METABOLIC PANEL
        Patient: Sarah Smith
        Date: 2024-01-20
        Lab: Quest Diagnostics
        
        CHEMISTRY PANEL:
        Glucose, Fasting: 88 mg/dL (70-100)
        BUN: 15 mg/dL (7-20)
        Creatinine: 0.9 mg/dL (0.6-1.2)
        Sodium: 140 mEq/L (135-145)
        Potassium: 4.0 mEq/L (3.5-5.0)
        Chloride: 102 mEq/L (96-106)
        CO2: 24 mEq/L (22-28)
        Calcium: 9.5 mg/dL (8.5-10.5)
        
        LIPID PANEL:
        Total Cholesterol: 180 mg/dL (<200)
        HDL Cholesterol: 65 mg/dL (>40)
        LDL Cholesterol: 95 mg/dL (<100)
        Triglycerides: 120 mg/dL (<150)
        
        HEMATOLOGY:
        Hemoglobin: 13.8 g/dL (12.0-15.5)
        Hematocrit: 41% (36-46)
        WBC: 7.5 K/uL (4.5-11.0)
        RBC: 4.8 M/uL (4.0-5.2)
        Platelets: 250 K/uL (150-450)
        
        COMMENTS:
        All values within normal range.
        """
    
    def test_extract_text_from_pdf(self, processor, sample_pdf_content):
        """Test PDF text extraction."""
        # Create a mock PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(sample_pdf_content.encode())
            pdf_path = f.name
        
        try:
            # Mock pdfplumber to return our content
            with patch('pdfplumber.open') as mock_pdfplumber:
                mock_pdf = Mock()
                mock_page = Mock()
                mock_page.extract_text.return_value = sample_pdf_content
                mock_pdf.pages = [mock_page]
                mock_pdfplumber.return_value.__enter__.return_value = mock_pdf
                
                extracted_text = processor.extract_text_from_pdf(pdf_path)
                
                assert extracted_text is not None
                assert "LDL Cholesterol" in extracted_text
                assert "Glucose" in extracted_text
                assert "Hemoglobin" in extracted_text
                
        finally:
            os.unlink(pdf_path)
    
    def test_extract_biomarkers(self, processor, sample_pdf_content):
        """Test biomarker extraction from text."""
        biomarkers = processor.extract_biomarkers(sample_pdf_content)
        
        assert len(biomarkers) >= 3
        
        # Check for specific biomarkers
        ldl_found = any(b.name.lower() == 'ldl' for b in biomarkers)
        glucose_found = any(b.name.lower() == 'glucose' for b in biomarkers)
        hemoglobin_found = any(b.name.lower() == 'hemoglobin' for b in biomarkers)
        
        assert ldl_found, "LDL should be extracted"
        assert glucose_found, "Glucose should be extracted"
        assert hemoglobin_found, "Hemoglobin should be extracted"
        
        # Check biomarker values
        for biomarker in biomarkers:
            assert biomarker.value > 0
            assert biomarker.unit in ['mg/dL', 'g/dL', 'mEq/L', 'K/uL', 'M/uL', '%']
            assert 0 <= biomarker.extracted_confidence <= 1
    
    def test_extract_biomarkers_complex(self, processor, complex_pdf_content):
        """Test biomarker extraction from complex PDF content."""
        biomarkers = processor.extract_biomarkers(complex_pdf_content)
        
        assert len(biomarkers) >= 10  # Should extract many biomarkers
        
        # Check for various biomarker types
        expected_biomarkers = ['glucose', 'ldl', 'hdl', 'hemoglobin', 'creatinine', 'sodium']
        found_biomarkers = [b.name.lower() for b in biomarkers]
        
        for expected in expected_biomarkers:
            assert expected in found_biomarkers, f"{expected} should be extracted"
    
    def test_extract_biomarkers_with_various_formats(self, processor):
        """Test biomarker extraction with various text formats."""
        test_cases = [
            "LDL: 120 mg/dL",
            "LDL Cholesterol = 120 mg/dL",
            "LDL Cholesterol 120 mg/dL",
            "LDL Cholesterol: 120 mg/dL (Reference: <100)",
            "LDL Cholesterol: 120 mg/dL (<100)",
            "LDL Cholesterol: 120 mg/dL Reference: <100",
        ]
        
        for test_case in test_cases:
            biomarkers = processor.extract_biomarkers(test_case)
            assert len(biomarkers) >= 1, f"Should extract biomarker from: {test_case}"
            
            if biomarkers:
                biomarker = biomarkers[0]
                assert biomarker.name.lower() == 'ldl'
                assert biomarker.value == 120.0
                assert biomarker.unit == 'mg/dL'
    
    def test_extract_biomarkers_edge_cases(self, processor):
        """Test biomarker extraction edge cases."""
        edge_cases = [
            "LDL: 0 mg/dL",  # Zero value
            "LDL: 999 mg/dL",  # High value
            "LDL: 12.5 mg/dL",  # Decimal value
            "LDL: 12,500 mg/dL",  # Comma in number
            "LDL: 12.5e2 mg/dL",  # Scientific notation
        ]
        
        for test_case in edge_cases:
            biomarkers = processor.extract_biomarkers(test_case)
            assert len(biomarkers) >= 1, f"Should extract biomarker from: {test_case}"
    
    def test_extract_biomarkers_invalid_data(self, processor):
        """Test biomarker extraction with invalid data."""
        invalid_cases = [
            "LDL: mg/dL",  # No value
            "LDL: abc mg/dL",  # Non-numeric value
            "LDL: -50 mg/dL",  # Negative value
            "LDL: 120",  # No unit
        ]
        
        for test_case in invalid_cases:
            biomarkers = processor.extract_biomarkers(test_case)
            # Should either not extract or extract with low confidence
            if biomarkers:
                for biomarker in biomarkers:
                    assert biomarker.extracted_confidence < 0.5
    
    def test_determine_health_status(self, processor):
        """Test health status determination."""
        test_cases = [
            ({"ldl": 85, "glucose": 95, "hemoglobin": 14.5}, "excellent"),
            ({"ldl": 120, "glucose": 95, "hemoglobin": 14.5}, "good"),
            ({"ldl": 150, "glucose": 110, "hemoglobin": 14.5}, "fair"),
            ({"ldl": 180, "glucose": 130, "hemoglobin": 12.0}, "poor"),
        ]
        
        for biomarkers, expected_status in test_cases:
            status = processor.determine_health_status(biomarkers)
            assert status == expected_status, f"Expected {expected_status} for biomarkers {biomarkers}"
    
    def test_generate_recommendations(self, processor):
        """Test recommendation generation."""
        # Test with elevated LDL
        biomarkers = [BiomarkerResult(
            name="LDL",
            value=120.0,
            unit="mg/dL",
            reference_range="<100",
            status="high",
            extracted_confidence=0.95
        )]
        
        recommendations = processor.generate_recommendations(biomarkers)
        
        assert len(recommendations) > 0
        assert any("LDL" in rec or "cholesterol" in rec.lower() for rec in recommendations)
    
    def test_process_lab_report(self, processor, sample_pdf_content):
        """Test complete lab report processing."""
        # Mock PDF extraction
        with patch.object(processor, 'extract_text_from_pdf', return_value=sample_pdf_content):
            with patch.object(processor, 'extract_biomarkers') as mock_extract:
                # Mock biomarker extraction
                mock_biomarkers = [
                    BiomarkerResult(
                        name="LDL",
                        value=120.0,
                        unit="mg/dL",
                        reference_range="<100",
                        status="high",
                        extracted_confidence=0.95
                    ),
                    BiomarkerResult(
                        name="Glucose",
                        value=95.0,
                        unit="mg/dL",
                        reference_range="70-100",
                        status="normal",
                        extracted_confidence=0.92
                    )
                ]
                mock_extract.return_value = mock_biomarkers
                
                # Mock file operations
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                    pdf_path = f.name
                
                try:
                    result = processor.process_lab_report(
                        pdf_path,
                        "patient_001",
                        "2024-01-15T00:00:00",
                        "LabCorp"
                    )
                    
                    assert result.success is True
                    assert result.report_id is not None
                    assert result.biomarkers_found == 2
                    assert result.processing_time > 0
                    
                finally:
                    os.unlink(pdf_path)
    
    def test_process_lab_report_with_errors(self, processor):
        """Test lab report processing with errors."""
        # Test with non-existent file
        result = processor.process_lab_report(
            "non_existent_file.pdf",
            "patient_001",
            "2024-01-15T00:00:00",
            "LabCorp"
        )
        
        assert result.success is False
        assert "error" in result.message.lower()
    
    def test_extract_biomarkers_performance(self, processor):
        """Test biomarker extraction performance with large text."""
        # Create large text with many biomarkers
        large_text = ""
        for i in range(100):
            large_text += f"LDL: {100 + i} mg/dL\n"
            large_text += f"Glucose: {90 + i} mg/dL\n"
            large_text += f"Hemoglobin: {14 + i * 0.1} g/dL\n"
        
        import time
        start_time = time.time()
        biomarkers = processor.extract_biomarkers(large_text)
        end_time = time.time()
        
        # Should process within reasonable time (less than 1 second)
        assert end_time - start_time < 1.0
        assert len(biomarkers) > 0
    
    def test_biomarker_confidence_scoring(self, processor):
        """Test biomarker confidence scoring."""
        test_cases = [
            ("LDL: 120 mg/dL", 0.9),  # Clear format
            ("LDL Cholesterol: 120 mg/dL (Reference: <100)", 0.95),  # With reference
            ("LDL: 120", 0.5),  # No unit
            ("LDL: mg/dL", 0.1),  # No value
            ("LDL: abc mg/dL", 0.1),  # Invalid value
        ]
        
        for test_case, expected_min_confidence in test_cases:
            biomarkers = processor.extract_biomarkers(test_case)
            if biomarkers:
                confidence = biomarkers[0].extracted_confidence
                assert confidence >= expected_min_confidence, \
                    f"Confidence {confidence} should be >= {expected_min_confidence} for: {test_case}"
    
    def test_extract_biomarkers_with_units(self, processor):
        """Test biomarker extraction with various units."""
        unit_test_cases = [
            "LDL: 120 mg/dL",
            "Glucose: 95 mg/dL",
            "Hemoglobin: 14.5 g/dL",
            "Sodium: 140 mEq/L",
            "WBC: 7.5 K/uL",
            "RBC: 4.8 M/uL",
            "Hematocrit: 41%",
        ]
        
        for test_case in unit_test_cases:
            biomarkers = processor.extract_biomarkers(test_case)
            assert len(biomarkers) >= 1, f"Should extract biomarker from: {test_case}"
            
            if biomarkers:
                biomarker = biomarkers[0]
                assert biomarker.unit in ['mg/dL', 'g/dL', 'mEq/L', 'K/uL', 'M/uL', '%']
    
    def test_extract_biomarkers_with_reference_ranges(self, processor):
        """Test biomarker extraction with reference ranges."""
        reference_test_cases = [
            "LDL: 120 mg/dL (Reference: <100)",
            "Glucose: 95 mg/dL (70-100)",
            "Hemoglobin: 14.5 g/dL (13.5-17.5)",
            "Sodium: 140 mEq/L (135-145)",
        ]
        
        for test_case in reference_test_cases:
            biomarkers = processor.extract_biomarkers(test_case)
            assert len(biomarkers) >= 1, f"Should extract biomarker from: {test_case}"
            
            if biomarkers:
                biomarker = biomarkers[0]
                assert biomarker.reference_range is not None
                assert len(biomarker.reference_range) > 0
    
    @pytest.mark.asyncio
    async def test_upload_lab_report_api(self, test_client, sample_users):
        """Test lab report upload API endpoint."""
        # First register and login a user
        user = sample_users[0]
        
        # Register user
        register_response = test_client.post("/api/v1/auth/register", json=user)
        assert register_response.status_code == 201
        
        # Login user
        login_response = test_client.post("/api/v1/auth/login", json={
            "email": user["email"],
            "password": user["password"]
        })
        assert login_response.status_code == 200
        
        token_data = login_response.json()
        access_token = token_data["access_token"]
        
        # Create mock PDF content
        pdf_content = """
        LABORATORY REPORT
        Patient: John Doe
        Date: 2024-01-15
        Lab: LabCorp
        
        RESULTS:
        LDL Cholesterol: 120 mg/dL (Reference: <100)
        Glucose: 95 mg/dL (Reference: 70-100)
        Hemoglobin: 14.5 g/dL (Reference: 13.5-17.5)
        """
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(pdf_content.encode())
            pdf_path = f.name
        
        try:
            # Upload lab report
            with open(pdf_path, 'rb') as pdf_file:
                files = {"file": ("test_report.pdf", pdf_file, "application/pdf")}
                data = {
                    "patient_id": "patient_001",
                    "report_date": "2024-01-15T00:00:00",
                    "lab_name": "LabCorp",
                    "notes": "Test lab report"
                }
                
                headers = {"Authorization": f"Bearer {access_token}"}
                response = test_client.post(
                    "/api/v1/lab-reports/upload",
                    files=files,
                    data=data,
                    headers=headers
                )
                
                assert response.status_code == 200
                result = response.json()
                
                assert result["success"] is True
                assert result["report_id"] is not None
                assert result["biomarkers_found"] >= 3
                assert result["processing_time"] > 0
                
        finally:
            os.unlink(pdf_path)
    
    def test_lab_report_data_validation(self, processor):
        """Test lab report data validation."""
        # Test valid data
        valid_biomarkers = [
            BiomarkerResult(
                name="LDL",
                value=120.0,
                unit="mg/dL",
                reference_range="<100",
                status="high",
                extracted_confidence=0.95
            )
        ]
        
        # Test invalid data
        invalid_biomarkers = [
            BiomarkerResult(
                name="LDL",
                value=-50.0,  # Negative value
                unit="mg/dL",
                reference_range="<100",
                status="high",
                extracted_confidence=0.95
            )
        ]
        
        # Valid data should pass validation
        assert processor.validate_biomarkers(valid_biomarkers) is True
        
        # Invalid data should fail validation
        assert processor.validate_biomarkers(invalid_biomarkers) is False
    
    def test_lab_report_error_handling(self, processor):
        """Test lab report error handling."""
        # Test with empty text
        biomarkers = processor.extract_biomarkers("")
        assert len(biomarkers) == 0
        
        # Test with None text
        biomarkers = processor.extract_biomarkers(None)
        assert len(biomarkers) == 0
        
        # Test with very long text
        long_text = "LDL: 120 mg/dL\n" * 10000
        biomarkers = processor.extract_biomarkers(long_text)
        assert len(biomarkers) > 0  # Should still extract some biomarkers
    
    def test_lab_report_performance_benchmark(self, processor):
        """Benchmark lab report processing performance."""
        import time
        
        # Create test data
        test_text = ""
        for i in range(1000):
            test_text += f"LDL: {100 + i % 50} mg/dL\n"
            test_text += f"Glucose: {90 + i % 20} mg/dL\n"
            test_text += f"Hemoglobin: {14 + i % 3} g/dL\n"
        
        # Benchmark extraction
        start_time = time.time()
        biomarkers = processor.extract_biomarkers(test_text)
        extraction_time = time.time() - start_time
        
        # Performance assertions
        assert extraction_time < 2.0, f"Extraction took {extraction_time:.2f}s, should be < 2.0s"
        assert len(biomarkers) > 0, "Should extract at least some biomarkers"
        
        print(f"Extracted {len(biomarkers)} biomarkers in {extraction_time:.3f}s")
    
    def test_lab_report_accuracy_validation(self, processor):
        """Test lab report extraction accuracy."""
        # Known test cases with expected results
        test_cases = [
            {
                "input": "LDL: 120 mg/dL",
                "expected": {
                    "name": "LDL",
                    "value": 120.0,
                    "unit": "mg/dL"
                }
            },
            {
                "input": "Glucose: 95 mg/dL (70-100)",
                "expected": {
                    "name": "Glucose",
                    "value": 95.0,
                    "unit": "mg/dL"
                }
            },
            {
                "input": "Hemoglobin: 14.5 g/dL",
                "expected": {
                    "name": "Hemoglobin",
                    "value": 14.5,
                    "unit": "g/dL"
                }
            }
        ]
        
        for test_case in test_cases:
            biomarkers = processor.extract_biomarkers(test_case["input"])
            assert len(biomarkers) >= 1, f"Should extract biomarker from: {test_case['input']}"
            
            if biomarkers:
                biomarker = biomarkers[0]
                expected = test_case["expected"]
                
                assert biomarker.name.lower() == expected["name"].lower()
                assert abs(biomarker.value - expected["value"]) < 0.1
                assert biomarker.unit == expected["unit"]
