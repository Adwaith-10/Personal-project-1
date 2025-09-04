#!/usr/bin/env python3
"""
Health AI Twin - Start All Services Script
Launches all services in the correct order
"""

import subprocess
import time
import sys
import os
from pathlib import Path

# Service configurations
SERVICES = [
    {
        "name": "Part 1: Core Infrastructure",
        "script": "start_part1.py",
        "port": 8003,
        "description": "Basic FastAPI + MongoDB setup"
    },
    {
        "name": "Part 2: Data Processing",
        "script": "start_part2.py", 
        "port": 8004,
        "description": "Lab reports, wearable data, food classification"
    },
    {
        "name": "Part 3: ML Pipeline",
        "script": "start_part3.py",
        "port": 8005,
        "description": "Health predictions, model training"
    },
    {
        "name": "Part 4: AI Services",
        "script": "start_part4.py",
        "port": 8006,
        "description": "Virtual doctor, authentication"
    },
    {
        "name": "Frontend Dashboard",
        "script": "start_part5.py",
        "port": 8501,
        "description": "Streamlit dashboard"
    }
]

def check_port_available(port):
    """Check if a port is available"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def kill_process_on_port(port):
    """Kill process using a specific port"""
    try:
        result = subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                subprocess.run(['kill', '-9', pid])
            print(f"🔌 Killed process on port {port}")
            time.sleep(2)
    except Exception as e:
        print(f"⚠️ Could not kill process on port {port}: {e}")

def start_service(service_config):
    """Start a single service"""
    name = service_config["name"]
    script = service_config["script"]
    port = service_config["port"]
    description = service_config["description"]
    
    print(f"\n🚀 Starting {name}...")
    print(f"   📝 {description}")
    print(f"   🌐 Port: {port}")
    
    # Check if port is available
    if not check_port_available(port):
        print(f"   ⚠️ Port {port} is in use, killing existing process...")
        kill_process_on_port(port)
    
    # Check if script exists
    if not os.path.exists(script):
        print(f"   ❌ Script {script} not found!")
        return False
    
    try:
        # Start service in background
        process = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit for service to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print(f"   ✅ {name} started successfully (PID: {process.pid})")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"   ❌ {name} failed to start")
            print(f"   📄 Error: {stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error starting {name}: {e}")
        return False

def main():
    """Main function"""
    print("🏥 Health AI Twin - Starting All Services")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("start_part1.py"):
        print("❌ Error: Please run this script from the Health AI Twin project root directory")
        sys.exit(1)
    
    # Kill any existing processes
    print("🔌 Cleaning up existing processes...")
    for service in SERVICES:
        kill_process_on_port(service["port"])
    
    # Start services in order
    started_services = []
    failed_services = []
    
    for service in SERVICES:
        if start_service(service):
            started_services.append(service)
        else:
            failed_services.append(service)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 STARTUP SUMMARY")
    print("=" * 50)
    
    print(f"✅ Successfully started: {len(started_services)} services")
    for service in started_services:
        print(f"   - {service['name']} (Port: {service['port']})")
    
    if failed_services:
        print(f"\n❌ Failed to start: {len(failed_services)} services")
        for service in failed_services:
            print(f"   - {service['name']} (Port: {service['port']})")
    
    if started_services:
        print(f"\n🌐 Access URLs:")
        print(f"   Frontend Dashboard: http://localhost:8501")
        print(f"   API Documentation: http://localhost:8003/docs")
        print(f"   Health Checks:")
        for service in started_services:
            print(f"     - {service['name']}: http://localhost:{service['port']}/health")
    
    if len(started_services) == len(SERVICES):
        print(f"\n🎉 All services started successfully!")
        print(f"🧪 Run 'python3 test_system.py' to test the system")
    else:
        print(f"\n⚠️ Some services failed to start. Check the errors above.")
    
    print(f"\n💡 To stop all services, run: pkill -f 'uvicorn'")

if __name__ == "__main__":
    main()



