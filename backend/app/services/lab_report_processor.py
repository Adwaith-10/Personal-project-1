import pdfplumber
import re
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from models.lab_report import BiomarkerResult, BiomarkerExtractionConfig

logger = logging.getLogger(__name__)

class LabReportProcessor:
    """Service for processing lab reports and extracting biomarkers"""
    
    def __init__(self):
        self.config = BiomarkerExtractionConfig()
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        
    async def process_lab_report(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process a lab report PDF and extract biomarkers"""
        try:
            # Validate file size
            if len(file_content) > self.max_file_size:
                raise ValueError(f"File size exceeds maximum allowed size of {self.max_file_size} bytes")
            
            # Extract text from PDF
            extracted_text = await self._extract_text_from_pdf(file_content)
            
            # Extract biomarkers
            biomarkers = await self._extract_biomarkers(extracted_text)
            
            # Extract report metadata
            metadata = await self._extract_metadata(extracted_text, filename)
            
            return {
                "extracted_text": extracted_text,
                "biomarkers": biomarkers,
                "metadata": metadata,
                "processing_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error processing lab report: {str(e)}")
            return {
                "extracted_text": "",
                "biomarkers": [],
                "metadata": {},
                "processing_status": "failed",
                "error": str(e)
            }
    
    async def _extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF using pdfplumber"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            extracted_text = ""
            
            with pdfplumber.open(temp_file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\n"
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return extracted_text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise e
    
    async def _extract_biomarkers(self, text: str) -> List[BiomarkerResult]:
        """Extract biomarkers from extracted text"""
        biomarkers = []
        
        # Normalize text for better matching
        normalized_text = text.upper()
        lines = text.split('\n')
        
        for biomarker_name, patterns in self.config.biomarker_patterns.items():
            for pattern in patterns:
                pattern_upper = pattern.upper()
                
                # Find all occurrences of the biomarker pattern
                matches = self._find_biomarker_matches(lines, pattern_upper, biomarker_name)
                
                for match in matches:
                    biomarkers.append(match)
        
        # Remove duplicates based on biomarker name and value
        unique_biomarkers = self._remove_duplicate_biomarkers(biomarkers)
        
        return unique_biomarkers
    
    def _find_biomarker_matches(self, lines: List[str], pattern: str, biomarker_name: str) -> List[BiomarkerResult]:
        """Find biomarker matches in text lines"""
        matches = []
        
        for line in lines:
            if pattern in line.upper():
                # Extract value and unit from the line
                result = self._parse_biomarker_line(line, biomarker_name)
                if result:
                    matches.append(result)
        
        return matches
    
    def _parse_biomarker_line(self, line: str, biomarker_name: str) -> Optional[BiomarkerResult]:
        """Parse a single line to extract biomarker information"""
        try:
            # Common patterns for lab results
            patterns = [
                # Pattern: Test Name: Value Unit (Reference Range)
                r'([A-Za-z\s]+)[:\s]+(\d+\.?\d*)\s*([A-Za-z/%]+)\s*(?:\(([^)]+)\))?',
                # Pattern: Test Name Value Unit
                r'([A-Za-z\s]+)\s+(\d+\.?\d*)\s*([A-Za-z/%]+)',
                # Pattern: Value Unit Test Name
                r'(\d+\.?\d*)\s*([A-Za-z/%]+)\s+([A-Za-z\s]+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    
                    if len(groups) >= 3:
                        # Determine which group is the test name, value, and unit
                        if groups[0].replace(' ', '').isalpha():
                            # First group is test name
                            test_name = groups[0].strip()
                            value_str = groups[1]
                            unit = groups[2]
                            reference_range = groups[3] if len(groups) > 3 else None
                        else:
                            # First group is value
                            value_str = groups[0]
                            unit = groups[1]
                            test_name = groups[2].strip()
                            reference_range = groups[3] if len(groups) > 3 else None
                        
                        # Validate that this line contains our target biomarker
                        if biomarker_name.upper() in test_name.upper():
                            try:
                                value = float(value_str)
                                unit = self._normalize_unit(unit)
                                status = self._determine_status(value, reference_range, biomarker_name)
                                confidence = self._calculate_confidence(line, biomarker_name)
                                
                                return BiomarkerResult(
                                    name=biomarker_name,
                                    value=value,
                                    unit=unit,
                                    reference_range=reference_range,
                                    status=status,
                                    extracted_confidence=confidence
                                )
                            except ValueError:
                                continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Error parsing biomarker line '{line}': {str(e)}")
            return None
    
    def _normalize_unit(self, unit: str) -> str:
        """Normalize unit strings"""
        unit = unit.strip().upper()
        
        # Map common unit variations
        unit_mapping = {
            'MG/DL': 'mg/dL',
            'MMOL/L': 'mmol/L',
            'G/DL': 'g/dL',
            'MEQ/L': 'mEq/L',
            'U/L': 'U/L',
            'K/UL': 'K/uL',
            'M/UL': 'M/uL',
            'PERCENT': '%',
            '%': '%'
        }
        
        return unit_mapping.get(unit, unit)
    
    def _determine_status(self, value: float, reference_range: Optional[str], biomarker_name: str) -> str:
        """Determine if biomarker value is normal, high, or low"""
        if not reference_range:
            return "unknown"
        
        try:
            # Parse reference range
            if '-' in reference_range:
                # Range format: 70-100
                parts = reference_range.split('-')
                min_val = float(parts[0].strip())
                max_val = float(parts[1].strip())
                
                if value < min_val:
                    return "low"
                elif value > max_val:
                    return "high"
                else:
                    return "normal"
            
            elif reference_range.startswith('<'):
                # Less than format: <100
                max_val = float(reference_range[1:].strip())
                if value >= max_val:
                    return "high"
                else:
                    return "normal"
            
            elif reference_range.startswith('>'):
                # Greater than format: >12
                min_val = float(reference_range[1:].strip())
                if value <= min_val:
                    return "low"
                else:
                    return "normal"
            
            else:
                return "unknown"
                
        except (ValueError, IndexError):
            return "unknown"
    
    def _calculate_confidence(self, line: str, biomarker_name: str) -> float:
        """Calculate confidence score for extracted biomarker"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on pattern matching
        if biomarker_name.upper() in line.upper():
            confidence += 0.2
        
        # Check for numeric value
        if re.search(r'\d+\.?\d*', line):
            confidence += 0.1
        
        # Check for unit
        if re.search(r'(mg/dL|mmol/L|g/dL|mEq/L|U/L|K/uL|M/uL|%)', line, re.IGNORECASE):
            confidence += 0.1
        
        # Check for reference range
        if re.search(r'\([^)]+\)', line):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _remove_duplicate_biomarkers(self, biomarkers: List[BiomarkerResult]) -> List[BiomarkerResult]:
        """Remove duplicate biomarkers, keeping the one with highest confidence"""
        unique_biomarkers = {}
        
        for biomarker in biomarkers:
            key = biomarker.name
            if key not in unique_biomarkers or biomarker.extracted_confidence > unique_biomarkers[key].extracted_confidence:
                unique_biomarkers[key] = biomarker
        
        return list(unique_biomarkers.values())
    
    async def _extract_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        """Extract metadata from lab report"""
        metadata = {
            "lab_name": None,
            "report_date": None,
            "patient_name": None
        }
        
        # Extract lab name
        lab_patterns = [
            r'Lab:\s*([^\n]+)',
            r'Laboratory:\s*([^\n]+)',
            r'Lab Name:\s*([^\n]+)',
            r'([A-Za-z\s]+)\s+Laboratory',
        ]
        
        for pattern in lab_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["lab_name"] = match.group(1).strip()
                break
        
        # Extract report date
        date_patterns = [
            r'Date:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'Report Date:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    date_str = match.group(1)
                    # Try to parse the date
                    for fmt in ['%m/%d/%Y', '%m/%d/%y', '%m-%d-%Y', '%m-%d-%y']:
                        try:
                            metadata["report_date"] = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
                    if metadata["report_date"]:
                        break
                except:
                    continue
        
        # Extract patient name
        name_patterns = [
            r'Patient:\s*([^\n]+)',
            r'Name:\s*([^\n]+)',
            r'Patient Name:\s*([^\n]+)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["patient_name"] = match.group(1).strip()
                break
        
        return metadata
