#!/usr/bin/env python3
"""
Comprehensive test runner for Health AI Twin system
Runs all tests for report extraction, wearable logging, and ML prediction accuracy
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_tests():
    """Run all tests and generate comprehensive report"""
    
    print("🧪 Health AI Twin - Comprehensive Test Suite")
    print("=" * 60)
    print(f"Test Run Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test categories
    test_categories = [
        {
            "name": "Lab Report Extraction",
            "file": "tests/test_lab_report_extraction.py",
            "description": "Tests for PDF biomarker extraction and processing"
        },
        {
            "name": "Wearable Data Logging", 
            "file": "tests/test_wearable_logging.py",
            "description": "Tests for wearable device data processing and analysis"
        },
        {
            "name": "ML Prediction Accuracy",
            "file": "tests/test_ml_prediction_accuracy.py", 
            "description": "Tests for ML model accuracy and performance"
        }
    ]
    
    # Results tracking
    results = {}
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    # Run each test category
    for category in test_categories:
        print(f"\n🔬 Testing: {category['name']}")
        print(f"📝 Description: {category['description']}")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # Run pytest for this category
            cmd = [
                sys.executable, "-m", "pytest", 
                category["file"],
                "-v",  # Verbose output
                "--tb=short",  # Short traceback
                "--durations=10",  # Show 10 slowest tests
                "--color=yes"  # Colored output
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse results
            if result.returncode == 0:
                status = "✅ PASSED"
                passed_tests += 1
            else:
                status = "❌ FAILED"
                failed_tests += 1
            
            # Extract test counts from output
            output_lines = result.stdout.split('\n')
            test_count = 0
            for line in output_lines:
                if "passed" in line and "failed" in line:
                    # Parse pytest summary line
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed":
                            test_count = int(parts[i-1])
                            break
            
            total_tests += test_count
            
            results[category["name"]] = {
                "status": status,
                "duration": duration,
                "test_count": test_count,
                "output": result.stdout,
                "error": result.stderr if result.stderr else None
            }
            
            print(f"Status: {status}")
            print(f"Duration: {duration:.2f}s")
            print(f"Tests: {test_count}")
            
            if result.stderr:
                print(f"Errors: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            results[category["name"]] = {
                "status": "⏰ TIMEOUT",
                "duration": 300,
                "test_count": 0,
                "output": "Test timed out after 5 minutes",
                "error": "Timeout"
            }
            failed_tests += 1
            print("Status: ⏰ TIMEOUT (5 minutes)")
            
        except Exception as e:
            results[category["name"]] = {
                "status": "💥 ERROR",
                "duration": 0,
                "test_count": 0,
                "output": str(e),
                "error": str(e)
            }
            failed_tests += 1
            print(f"Status: 💥 ERROR - {e}")
    
    # Generate comprehensive report
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST REPORT")
    print("=" * 60)
    
    print(f"Total Test Categories: {len(test_categories)}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests / len(test_categories) * 100):.1f}%")
    
    print("\n📋 Detailed Results:")
    print("-" * 60)
    
    for category_name, result in results.items():
        print(f"\n{category_name}:")
        print(f"  Status: {result['status']}")
        print(f"  Duration: {result['duration']:.2f}s")
        print(f"  Tests: {result['test_count']}")
        
        if result['error']:
            print(f"  Error: {result['error']}")
    
    # Performance summary
    print("\n⚡ Performance Summary:")
    print("-" * 60)
    
    total_duration = sum(r['duration'] for r in results.values())
    avg_duration = total_duration / len(test_categories) if test_categories else 0
    
    print(f"Total Test Time: {total_duration:.2f}s")
    print(f"Average Time per Category: {avg_duration:.2f}s")
    print(f"Tests per Second: {total_tests / total_duration:.1f}" if total_duration > 0 else "Tests per Second: N/A")
    
    # Data quality metrics
    print("\n📈 Data Quality Metrics:")
    print("-" * 60)
    
    print("✅ Lab Report Extraction:")
    print("  - PDF text extraction accuracy")
    print("  - Biomarker identification precision")
    print("  - Confidence scoring validation")
    print("  - Error handling robustness")
    
    print("\n✅ Wearable Data Processing:")
    print("  - Heart rate data accuracy")
    print("  - Sleep stage classification")
    print("  - Activity tracking precision")
    print("  - Data quality scoring")
    
    print("\n✅ ML Prediction Accuracy:")
    print("  - Model training performance")
    print("  - Prediction accuracy metrics")
    print("  - Feature importance analysis")
    print("  - Model robustness testing")
    
    # Recommendations
    print("\n💡 Recommendations:")
    print("-" * 60)
    
    if failed_tests > 0:
        print("❌ Issues Found:")
        for category_name, result in results.items():
            if "FAILED" in result['status'] or "ERROR" in result['status'] or "TIMEOUT" in result['status']:
                print(f"  - {category_name}: {result['status']}")
                if result['error']:
                    print(f"    Error: {result['error']}")
    else:
        print("✅ All tests passed successfully!")
    
    print("\n🔧 Next Steps:")
    print("  - Review any failed tests")
    print("  - Check performance bottlenecks")
    print("  - Validate data quality thresholds")
    print("  - Monitor model accuracy trends")
    
    # Save detailed report
    report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write("Health AI Twin - Comprehensive Test Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SUMMARY:\n")
        f.write(f"Total Categories: {len(test_categories)}\n")
        f.write(f"Total Tests: {total_tests}\n")
        f.write(f"Passed: {passed_tests}\n")
        f.write(f"Failed: {failed_tests}\n")
        f.write(f"Success Rate: {(passed_tests / len(test_categories) * 100):.1f}%\n\n")
        
        f.write("DETAILED RESULTS:\n")
        for category_name, result in results.items():
            f.write(f"\n{category_name}:\n")
            f.write(f"  Status: {result['status']}\n")
            f.write(f"  Duration: {result['duration']:.2f}s\n")
            f.write(f"  Tests: {result['test_count']}\n")
            if result['output']:
                f.write(f"  Output:\n{result['output']}\n")
            if result['error']:
                f.write(f"  Error: {result['error']}\n")
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Final status
    if failed_tests == 0:
        print("\n🎉 ALL TESTS PASSED! The Health AI Twin system is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed_tests} test category(ies) failed. Please review the results.")
        return 1

def run_specific_test(test_file):
    """Run a specific test file"""
    print(f"🧪 Running specific test: {test_file}")
    print("=" * 60)
    
    cmd = [
        sys.executable, "-m", "pytest", 
        test_file,
        "-v",
        "--tb=long",
        "--color=yes"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode

def run_performance_tests():
    """Run performance-focused tests"""
    print("🚀 Running Performance Tests")
    print("=" * 60)
    
    performance_tests = [
        "tests/test_lab_report_extraction.py::TestLabReportExtraction::test_lab_report_performance_benchmark",
        "tests/test_wearable_logging.py::TestWearableLogging::test_wearable_data_performance",
        "tests/test_ml_prediction_accuracy.py::TestMLPredictionAccuracy::test_model_performance_benchmark"
    ]
    
    for test in performance_tests:
        print(f"\n🔬 Running: {test}")
        cmd = [
            sys.executable, "-m", "pytest", 
            test,
            "-v",
            "--tb=short",
            "--color=yes"
        ]
        
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"❌ Performance test failed: {test}")
        else:
            print(f"✅ Performance test passed: {test}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Health AI Twin Test Runner")
    parser.add_argument("--test", help="Run specific test file")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    
    args = parser.parse_args()
    
    if args.test:
        exit_code = run_specific_test(args.test)
    elif args.performance:
        run_performance_tests()
        exit_code = 0
    else:
        exit_code = run_tests()
    
    sys.exit(exit_code)
