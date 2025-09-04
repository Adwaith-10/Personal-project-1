# 🏥 Health AI Twin - Complete Living Guide

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Quick Start Guide](#quick-start-guide)
4. [Detailed Setup Instructions](#detailed-setup-instructions)
5. [Features & Capabilities](#features--capabilities)
6. [API Documentation](#api-documentation)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Maintenance & Updates](#maintenance--updates)
9. [Development Guide](#development-guide)
10. [FAQ](#faq)

---

## 🎯 Project Overview

**Health AI Twin** is a comprehensive health monitoring and AI-powered wellness platform that combines wearable data, food tracking, lab reports, and AI-driven health insights to provide personalized health recommendations.

### 🎯 Key Features
- **📊 Health Data Integration**: Wearable devices, lab reports, food tracking
- **🤖 AI-Powered Analysis**: Virtual doctor consultations, health predictions
- **🍎 Smart Food Recognition**: Image-based food classification with nutrition analysis
- **📈 Predictive Health Models**: ML-powered health metric predictions
- **🔐 Secure Authentication**: JWT-based user management
- **📱 Modern Web Interface**: Streamlit dashboard with real-time visualizations

### 🏗️ Technology Stack
- **Backend**: FastAPI (Python)
- **Frontend**: Streamlit (Python)
- **Database**: MongoDB
- **AI/ML**: XGBoost, LangChain, OpenAI
- **Image Processing**: PIL, NumPy
- **Authentication**: JWT, bcrypt

---

## 🏗️ System Architecture

### 📁 Project Structure
```
Health AITwin/
├── backend/
│   ├── app/
│   │   ├── main_part1.py      # Core Infrastructure
│   │   ├── main_part2.py      # Data Processing Services
│   │   ├── main_part3.py      # ML Pipeline
│   │   ├── main_part4.py      # AI Services
│   │   ├── models/            # Pydantic models
│   │   └── services/          # Business logic
├── frontend/
│   └── dashboard.py           # Streamlit interface
├── data/                      # Sample data and models
├── logs/                      # Application logs
├── requirements.txt           # Dependencies
├── start_part1.py            # Part 1 startup script
├── start_part2.py            # Part 2 startup script
├── start_part3.py            # Part 3 startup script
├── start_part4.py            # Part 4 startup script
└── start_part5.py            # Frontend startup script
```

### 🔄 Modular Design
The system is divided into 5 independent parts, each running on different ports:

| Part | Service | Port | Description |
|------|---------|------|-------------|
| 1 | Core Infrastructure | 8003 | Basic FastAPI + MongoDB setup |
| 2 | Data Processing | 8004 | Lab reports, wearable data, food classification |
| 3 | ML Pipeline | 8005 | Health predictions, model training |
| 4 | AI Services | 8006 | Virtual doctor, authentication |
| 5 | Frontend | 8501 | Streamlit dashboard |

---

## 🚀 Quick Start Guide

### ⚡ 5-Minute Setup

1. **Clone and Navigate**
   ```bash
   cd "Health AITwin"
   ```

2. **Install Dependencies**
   ```bash
   python3 -m pip install -r requirements.txt
   ```

3. **Start All Services**
   ```bash
   # Terminal 1: Core Infrastructure
   python3 start_part1.py
   
   # Terminal 2: Data Processing
   python3 start_part2.py
   
   # Terminal 3: ML Pipeline
   python3 start_part3.py
   
   # Terminal 4: AI Services
   python3 start_part4.py
   
   # Terminal 5: Frontend
   python3 start_part5.py
   ```

4. **Access the Application**
   - **Frontend**: http://localhost:8501
   - **Test Login**: `test@example.com` / `test123`

---

## 📋 Detailed Setup Instructions

### 🔧 Prerequisites
- Python 3.8+
- MongoDB (local or cloud)
- Internet connection (for AI services)

### 📦 Installation Steps

#### Step 1: Environment Setup
```bash
# Create virtual environment (optional but recommended)
python3 -m venv health_ai_twin_env
source health_ai_twin_env/bin/activate  # On Windows: health_ai_twin_env\Scripts\activate

# Install dependencies
python3 -m pip install -r requirements.txt
```

#### Step 2: MongoDB Setup
```bash
# Option 1: Local MongoDB
# Install MongoDB locally or use Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest

# Option 2: MongoDB Atlas (Cloud)
# Create free cluster at https://cloud.mongodb.com
# Update connection string in backend/app/services/database.py
```

#### Step 3: Environment Variables
Create `.env` file in project root:
```env
MONGODB_URL=mongodb://localhost:27017
OPENAI_API_KEY=your_openai_api_key_here
JWT_SECRET_KEY=your_jwt_secret_key_here
```

#### Step 4: Start Services
```bash
# Start each service in separate terminals
python3 start_part1.py  # Port 8003
python3 start_part2.py  # Port 8004
python3 start_part3.py  # Port 8005
python3 start_part4.py  # Port 8006
python3 start_part5.py  # Port 8501
```

---

## 🎯 Features & Capabilities

### 📊 Health Data Management

#### Lab Report Processing
- **Upload PDF lab reports**
- **Automatic biomarker extraction** (LDL, glucose, hemoglobin)
- **Structured data storage** in MongoDB
- **Historical trend analysis**

#### Wearable Data Integration
- **Real-time data ingestion** from wearables
- **Multiple metrics support**:
  - Heart rate & HRV
  - Sleep stages & duration
  - Step count & activity
  - SpO₂ levels
- **Daily aggregation** and trend analysis

#### Food Tracking System
- **Image-based food recognition** with 40+ food types
- **Automatic nutrition calculation** (calories, protein, carbs, fat)
- **Meal categorization** (breakfast, lunch, dinner, snacks)
- **Daily nutrition summaries**

### 🤖 AI-Powered Features

#### Virtual Doctor (Dr. Sarah Chen)
- **Conversational AI interface** (ChatGPT-like)
- **Personalized health advice** based on user data
- **Specialized knowledge areas**:
  - Nutrition & diet
  - Exercise & fitness
  - Sleep & recovery
  - Mental health
  - Weight management
  - Heart health
  - Diabetes management
- **Context-aware responses** using user's health data

#### Health Predictions
- **ML-powered predictions** for:
  - LDL cholesterol
  - Blood glucose
  - Hemoglobin levels
- **Feature engineering** from wearable and lifestyle data
- **Risk assessment** and early warning system

### 📱 User Interface

#### Dashboard Features
- **Real-time health metrics** visualization
- **Interactive charts** using Plotly
- **Food tracking interface** with image upload
- **Virtual doctor chat** interface
- **Health trend analysis**
- **Risk assessment display**

#### Authentication System
- **JWT-based authentication**
- **User registration** and login
- **Secure password hashing**
- **Session management**

---

## 📚 API Documentation

### 🔐 Authentication Endpoints

#### Register User
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword",
  "first_name": "John",
  "last_name": "Doe"
}
```

#### Login User
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword"
}
```

### 📊 Health Data Endpoints

#### Upload Lab Report
```http
POST /api/v1/lab-reports/upload
Content-Type: multipart/form-data

file: [PDF file]
patient_id: "user123"
```

#### Submit Wearable Data
```http
POST /api/v1/wearable-data
Content-Type: application/json

{
  "patient_id": "user123",
  "date": "2024-01-15",
  "heart_rate_avg": 72,
  "steps_count": 8500,
  "sleep_hours": 7.5,
  "spo2_avg": 98
}
```

#### Classify Food Image
```http
POST /api/v1/food-classification/classify
Content-Type: multipart/form-data

file: [Image file]
meal_type: "lunch"
```

### 🤖 AI Services Endpoints

#### Chat with Virtual Doctor
```http
POST /api/v1/virtual-doctor/chat
Content-Type: application/json
Authorization: Bearer <jwt_token>

{
  "question": "How can I improve my diet?",
  "include_food_data": true,
  "recent_food_log": [...]
}
```

#### Get Health Predictions
```http
POST /api/v1/health-prediction/predict
Content-Type: application/json

{
  "patient_id": "user123",
  "age": 35,
  "bmi": 24.5,
  "heart_rate_avg": 72,
  "steps_avg": 8500,
  "sleep_hours_avg": 7.5,
  "calories_avg": 2100
}
```

---

## 🔧 Troubleshooting Guide

### 🚨 Common Issues

#### Port Already in Use
```bash
# Find process using port
lsof -i :8003
lsof -i :8004
lsof -i :8005
lsof -i :8006
lsof -i :8501

# Kill process
kill -9 <PID>
```

#### MongoDB Connection Issues
```bash
# Check MongoDB status
sudo systemctl status mongod

# Start MongoDB
sudo systemctl start mongod

# Check connection string in database.py
```

#### Module Import Errors
```bash
# Reinstall dependencies
python3 -m pip install -r requirements.txt --force-reinstall

# Check Python version
python3 --version  # Should be 3.8+
```

#### Food Recognition Not Working
```bash
# Check image analysis dependencies
python3 -c "import PIL, numpy; print('Dependencies OK')"

# Verify image format (JPG, PNG supported)
# Ensure descriptive filename (e.g., "maggie.jpg")
```

#### AI Doctor Not Responding
```bash
# Check OpenAI API key
echo $OPENAI_API_KEY

# Verify LangChain installation
python3 -c "import langchain; print('LangChain OK')"
```

### 🔍 Debug Information

#### Enable Debug Logging
Add to `.env`:
```env
DEBUG=true
LOG_LEVEL=DEBUG
```

#### Check Service Health
```bash
# Test each service
curl http://localhost:8003/health
curl http://localhost:8004/health
curl http://localhost:8005/health
curl http://localhost:8006/health
```

#### View Application Logs
```bash
# Check terminal output for each service
# Look for DEBUG messages in food classification
# Monitor MongoDB connection status
```

---

## 🔄 Maintenance & Updates

### 📅 Regular Maintenance Tasks

#### Daily
- **Monitor service health** via health check endpoints
- **Check error logs** for any issues
- **Verify database connectivity**

#### Weekly
- **Backup MongoDB data**
- **Update dependencies** if needed
- **Review system performance**

#### Monthly
- **Update AI models** with new training data
- **Review and update food database**
- **Security audit** of authentication system

### 🔄 Update Procedures

#### Update Dependencies
```bash
# Update requirements
python3 -m pip install --upgrade -r requirements.txt

# Test all services after update
python3 -c "import fastapi, streamlit, pymongo; print('Update successful')"
```

#### Update Food Database
Edit `backend/app/main_part2.py`:
```python
# Add new foods to food_database
food_database["grains"].append({
    "name": "new_food",
    "calories": 100,
    "protein": 5,
    "carbs": 20,
    "fat": 2,
    "fiber": 3,
    "colors": ["yellow", "white"]
})
```

#### Update AI Models
```bash
# Retrain ML models
curl -X POST http://localhost:8005/api/v1/health-prediction/train

# Check model status
curl http://localhost:8005/api/v1/health-prediction/status
```

---

## 👨‍💻 Development Guide

### 🛠️ Adding New Features

#### 1. Backend Development
```bash
# Create new endpoint in appropriate part
# Add to main_partX.py

@app.post("/api/v1/new-feature")
async def new_feature():
    # Implementation
    pass
```

#### 2. Frontend Development
```bash
# Add new page to dashboard.py
def new_feature_page():
    st.title("New Feature")
    # Implementation
```

#### 3. Database Schema Updates
```python
# Add new model in models/
class NewFeature(BaseModel):
    field1: str
    field2: int
```

### 🧪 Testing

#### Run Tests
```bash
# Install test dependencies
python3 -m pip install pytest pytest-asyncio pytest-mock

# Run tests
python3 -m pytest tests/
```

#### Manual Testing
```bash
# Test food classification
curl -X POST -F "file=@test_image.jpg" http://localhost:8004/api/v1/food-classification/classify

# Test virtual doctor
curl -X POST -H "Content-Type: application/json" -d '{"question":"How can I improve my diet?"}' http://localhost:8006/api/v1/virtual-doctor/chat
```

### 📝 Code Standards

#### Python Style
- **PEP 8** compliance
- **Type hints** for all functions
- **Docstrings** for all classes and methods
- **Error handling** with proper exceptions

#### API Design
- **RESTful endpoints**
- **Consistent response format**
- **Proper HTTP status codes**
- **Input validation** with Pydantic

---

## ❓ FAQ

### 🤔 General Questions

**Q: How do I reset the system?**
A: Stop all services, clear MongoDB collections, restart services.

**Q: Can I use my own MongoDB instance?**
A: Yes, update the connection string in `backend/app/services/database.py`.

**Q: How do I add new food types?**
A: Edit the `food_database` in `backend/app/main_part2.py`.

**Q: Can I customize the AI doctor's responses?**
A: Yes, modify the conversation handlers in `backend/app/main_part4.py`.

### 🔧 Technical Questions

**Q: Why is food recognition not working?**
A: Check image format, filename, and ensure PIL/NumPy are installed.

**Q: How do I increase prediction accuracy?**
A: Add more training data and retrain models via the ML pipeline.

**Q: Can I run this without OpenAI?**
A: Yes, but virtual doctor features will be limited.

**Q: How do I scale this for production?**
A: Use Docker containers, load balancers, and cloud MongoDB.

### 📊 Data Questions

**Q: How is my data stored?**
A: All data is stored in MongoDB with user-specific collections.

**Q: Is my health data secure?**
A: Yes, JWT authentication and encrypted passwords are used.

**Q: Can I export my data?**
A: Add export endpoints to download user data in JSON format.

**Q: How long is data retained?**
A: Data is retained indefinitely unless manually deleted.

---

## 📞 Support & Contact

### 🆘 Getting Help
1. **Check this guide** for common solutions
2. **Review troubleshooting section** for specific issues
3. **Check application logs** for error details
4. **Test individual services** using health check endpoints

### 🔗 Useful Links
- **MongoDB Documentation**: https://docs.mongodb.com
- **FastAPI Documentation**: https://fastapi.tiangolo.com
- **Streamlit Documentation**: https://docs.streamlit.io
- **OpenAI API Documentation**: https://platform.openai.com/docs

### 📝 Version Information
- **Current Version**: 1.0.0
- **Last Updated**: January 2024
- **Python Version**: 3.8+
- **MongoDB Version**: 4.4+

---

## 📄 License & Legal

### ⚖️ Legal Notice
This system is for educational and demonstration purposes. It is not intended for medical diagnosis or treatment. Always consult healthcare professionals for medical advice.

### 🔒 Privacy & Security
- User data is stored securely in MongoDB
- Passwords are hashed using bcrypt
- JWT tokens are used for authentication
- No data is shared with third parties

### 📋 Compliance
- Follows healthcare data best practices
- Implements secure authentication
- Provides data export capabilities
- Maintains audit trails

---

*This living guide is maintained and updated regularly. For the latest version, check the project repository.*
