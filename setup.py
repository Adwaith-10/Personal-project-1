#!/usr/bin/env python3
"""
Setup script for Health AI Twin project
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "logs",
        "data",
        "ml_models/models",
        "ml_models/preprocessing",
        "backend/tests",
        "frontend/components"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install Python dependencies"""
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    return True

def create_env_file():
    """Create .env file from example"""
    env_example = Path("env.example")
    env_file = Path(".env")
    
    if not env_file.exists():
        if env_example.exists():
            env_file.write_text(env_example.read_text())
            print("✅ Created .env file from env.example")
        else:
            print("⚠️ env.example not found, creating basic .env file")
            basic_env = """# Environment Configuration for Health AI Twin
ENVIRONMENT=development
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=health_ai_twin
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=True
STREAMLIT_HOST=localhost
STREAMLIT_PORT=8501
SECRET_KEY=your-super-secret-key-change-this-in-production
"""
            env_file.write_text(basic_env)
    else:
        print("ℹ️ .env file already exists")

def check_mongodb():
    """Check if MongoDB is available"""
    try:
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("✅ MongoDB connection successful")
        client.close()
        return True
    except Exception as e:
        print(f"⚠️ MongoDB not available: {e}")
        print("💡 Please install and start MongoDB, or update MONGODB_URL in .env file")
        return False

def main():
    """Main setup function"""
    print("🏥 Health AI Twin Setup")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    print("\n📁 Creating project directories...")
    create_directories()
    
    # Create .env file
    print("\n⚙️ Setting up environment...")
    create_env_file()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Check MongoDB
    print("\n🗄️ Checking MongoDB connection...")
    check_mongodb()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Start MongoDB (if not already running)")
    print("2. Run 'python start_backend.py' to start the API server")
    print("3. Run 'python start_frontend.py' to start the web interface")
    print("4. Visit http://localhost:8501 to access the application")
    print("5. API documentation available at http://localhost:8000/docs")
    
    print("\n🔧 Optional:")
    print("- Run 'python train_models.py' to train ML models")
    print("- Update .env file with your specific configuration")

if __name__ == "__main__":
    main()
