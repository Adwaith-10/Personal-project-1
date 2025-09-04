#!/usr/bin/env python3
"""
Simple test runner for Health AI Twin system
Runs basic functionality tests without complex dependencies
"""

import subprocess
import sys
import time
from datetime import datetime

def run_simple_tests():
    """Run simple tests that don't require complex dependencies"""
    
    print("🧪 Health AI Twin - Simple Test Suite")
    print("=" * 60)
    print(f"Test Run Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test categories
    test_categories = [
        {
            "name": "Basic Functionality",
            "file": "tests/test_basic.py",
            "description": "Basic mathematical and data processing tests"
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
                "--color=yes"  # Colored output
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
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
                "duration": 60,
                "test_count": 0,
                "output": "Test timed out after 1 minute",
                "error": "Timeout"
            }
            failed_tests += 1
            print("Status: ⏰ TIMEOUT (1 minute)")
            
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
    
    # Generate report
    print("\n" + "=" * 60)
    print("📊 SIMPLE TEST REPORT")
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
    
    # Test coverage summary
    print("\n📈 Test Coverage:")
    print("-" * 60)
    
    print("✅ Basic Functionality:")
    print("  - Mathematical operations")
    print("  - Data structure operations")
    print("  - Health metrics calculations")
    print("  - Data validation logic")
    print("  - Statistical calculations")
    print("  - Trend analysis")
    print("  - Performance benchmarking")
    print("  - Error handling")
    print("  - Health score calculations")
    
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
        print("✅ All basic tests passed successfully!")
        print("🎉 The test infrastructure is working correctly!")
    
    print("\n🔧 Next Steps:")
    print("  - Install additional dependencies for full test suite")
    print("  - Set up MongoDB for integration tests")
    print("  - Configure environment variables")
    print("  - Run comprehensive test suite")
    
    # Save detailed report
    report_file = f"simple_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write("Health AI Twin - Simple Test Report\n")
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
        print("\n🎉 ALL BASIC TESTS PASSED! The test infrastructure is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed_tests} test category(ies) failed. Please review the results.")
        return 1

if __name__ == "__main__":
    exit_code = run_simple_tests()
    sys.exit(exit_code)
