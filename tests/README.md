# Health AI Twin - Test Suite

This directory contains comprehensive tests for the Health AI Twin system, covering report extraction, wearable logging, and ML prediction accuracy.

## 🧪 Test Overview

The test suite is designed to validate the core functionality of the Health AI Twin system with realistic data scenarios for 2 users over 10 days.

### Test Categories

1. **Lab Report Extraction Tests** (`test_lab_report_extraction.py`)
   - PDF text extraction accuracy
   - Biomarker identification and parsing
   - Confidence scoring validation
   - Error handling and edge cases
   - Performance benchmarking

2. **Wearable Data Logging Tests** (`test_wearable_logging.py`)
   - Heart rate data processing
   - Sleep stage classification
   - Activity tracking accuracy
   - Data quality assessment
   - Trend analysis validation

3. **ML Prediction Accuracy Tests** (`test_ml_prediction_accuracy.py`)
   - Model training performance
   - Prediction accuracy metrics (RMSE, MAE, R²)
   - Feature importance analysis
   - Model robustness testing
   - Bias detection and validation

## 📊 Test Data

### Sample Users
- **User 1**: John Doe (male, 35 years old)
- **User 2**: Sarah Smith (female, 40 years old)

### Data Coverage (10 Days)
- **Lab Reports**: 2 comprehensive metabolic panels
- **Wearable Data**: 20 daily logs (2 users × 10 days)
- **Food Logs**: 60 meal entries (2 users × 10 days × 3 meals)
- **Health Predictions**: 20 prediction records (2 users × 10 days)

### Realistic Data Generation
- Heart rate: 70-120 BPM with daily variations
- Sleep: 7-8 hours with different sleep stages
- Activity: 8,000-10,000 steps with exercise sessions
- Nutrition: Balanced meals with realistic calorie counts
- Health metrics: Correlated with lifestyle factors

## 🚀 Running Tests

### Prerequisites
```bash
# Install test dependencies
pip install -r requirements.txt

# Ensure MongoDB is running (for integration tests)
mongod --dbpath /path/to/data/db
```

### Quick Start
```bash
# Run all tests
python run_tests.py

# Run specific test category
python run_tests.py --test tests/test_lab_report_extraction.py

# Run performance tests only
python run_tests.py --performance
```

### Individual Test Files
```bash
# Lab report extraction tests
pytest tests/test_lab_report_extraction.py -v

# Wearable data tests
pytest tests/test_wearable_logging.py -v

# ML prediction tests
pytest tests/test_ml_prediction_accuracy.py -v
```

### Test Configuration
```bash
# Run with detailed output
pytest -v --tb=long

# Run with coverage report
pytest --cov=app --cov-report=html

# Run specific test method
pytest tests/test_lab_report_extraction.py::TestLabReportExtraction::test_extract_biomarkers -v
```

## 📈 Performance Benchmarks

### Expected Performance Metrics

#### Lab Report Extraction
- **Processing Time**: < 2 seconds per report
- **Accuracy**: > 90% biomarker identification
- **Confidence**: > 0.8 for valid biomarkers
- **Error Handling**: Graceful handling of malformed PDFs

#### Wearable Data Processing
- **Processing Time**: < 5 seconds per daily log
- **Data Quality Score**: > 0.8 for complete data
- **Trend Analysis**: Accurate trend detection
- **Statistical Accuracy**: Valid summary statistics

#### ML Prediction Accuracy
- **Training Time**: < 30 seconds for full dataset
- **Prediction Time**: < 5 seconds per prediction
- **LDL RMSE**: < 15 mg/dL
- **Glucose RMSE**: < 8 mg/dL
- **Hemoglobin RMSE**: < 1.0 g/dL
- **R² Score**: > 0.7 for all metrics

## 🔍 Test Validation Criteria

### Data Quality Thresholds
```python
# Wearable Data
min_heart_rate = 40
max_heart_rate = 200
min_sleep_hours = 4
max_sleep_hours = 12
min_steps = 100
max_steps = 50000

# Lab Reports
min_ldl = 50
max_ldl = 300
min_glucose = 40
max_glucose = 400
min_hemoglobin = 8
max_hemoglobin = 20

# Food Logs
min_calories = 10
max_calories = 2000
min_protein = 0
max_protein = 100
min_carbs = 0
max_carbs = 300
```

### Accuracy Benchmarks
```python
# ML Model Performance
expected_metrics = {
    "ldl": {
        "rmse": 15.0,  # mg/dL
        "mae": 12.0,   # mg/dL
        "r2": 0.75,    # R-squared
        "accuracy": 0.85  # Classification accuracy
    },
    "glucose": {
        "rmse": 8.0,   # mg/dL
        "mae": 6.5,    # mg/dL
        "r2": 0.80,    # R-squared
        "accuracy": 0.90  # Classification accuracy
    },
    "hemoglobin": {
        "rmse": 0.8,   # g/dL
        "mae": 0.6,    # g/dL
        "r2": 0.70,    # R-squared
        "accuracy": 0.88  # Classification accuracy
    }
}
```

## 🛠️ Test Fixtures

### Database Setup
- Test database: `health_aitwin_test`
- Automatic cleanup after tests
- Sample data generation for 2 users over 10 days

### Mock Services
- PDF processing with mock content
- Image classification with mock predictions
- ML model training with synthetic data
- API endpoints with authentication

### Sample Data
- Realistic biomarker values with correlations
- Wearable data with daily variations
- Food logs with nutritional information
- Health predictions with confidence scores

## 📋 Test Reports

### Generated Reports
- **Comprehensive Report**: `test_report_YYYYMMDD_HHMMSS.txt`
- **Performance Metrics**: Processing times and accuracy scores
- **Error Analysis**: Detailed error messages and stack traces
- **Recommendations**: Actionable insights for improvement

### Report Contents
```
Health AI Twin - Comprehensive Test Report
============================================================
Generated: 2024-01-15 14:30:25

SUMMARY:
Total Categories: 3
Total Tests: 45
Passed: 3
Failed: 0
Success Rate: 100.0%

DETAILED RESULTS:
Lab Report Extraction:
  Status: ✅ PASSED
  Duration: 12.34s
  Tests: 15

Wearable Data Logging:
  Status: ✅ PASSED
  Duration: 8.76s
  Tests: 18

ML Prediction Accuracy:
  Status: ✅ PASSED
  Duration: 15.23s
  Tests: 12
```

## 🔧 Troubleshooting

### Common Issues

#### MongoDB Connection
```bash
# Start MongoDB service
sudo systemctl start mongod

# Check MongoDB status
sudo systemctl status mongod

# Connect to test database
mongo health_aitwin_test
```

#### Test Dependencies
```bash
# Install missing dependencies
pip install pytest pytest-asyncio pytest-mock

# Update requirements
pip install -r requirements.txt --upgrade
```

#### Test Timeouts
```bash
# Increase timeout for slow tests
pytest --timeout=600

# Run tests in parallel
pytest -n auto
```

### Debug Mode
```bash
# Run with debug output
pytest -v -s --tb=long

# Run specific failing test
pytest tests/test_lab_report_extraction.py::TestLabReportExtraction::test_extract_biomarkers -v -s
```

## 📚 Test Documentation

### Test Structure
```
tests/
├── __init__.py
├── conftest.py                    # Pytest configuration and fixtures
├── test_lab_report_extraction.py  # Lab report processing tests
├── test_wearable_logging.py       # Wearable data tests
├── test_ml_prediction_accuracy.py # ML model tests
└── README.md                      # This file
```

### Fixture Usage
```python
# Use sample data fixtures
def test_with_sample_data(sample_users, sample_wearable_data):
    # Test with realistic data
    pass

# Use test client for API tests
@pytest.mark.asyncio
async def test_api_endpoint(test_client):
    # Test API functionality
    pass
```

## 🎯 Continuous Integration

### GitHub Actions
```yaml
name: Health AI Twin Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python run_tests.py
```

### Local CI
```bash
# Pre-commit hook
#!/bin/bash
python run_tests.py
if [ $? -ne 0 ]; then
    echo "Tests failed. Please fix issues before committing."
    exit 1
fi
```

## 📞 Support

For test-related issues:
1. Check the troubleshooting section
2. Review test logs and error messages
3. Verify data quality thresholds
4. Ensure all dependencies are installed
5. Contact the development team

---

**Note**: These tests are designed to validate the Health AI Twin system's core functionality. Regular test runs help ensure system reliability and performance.
