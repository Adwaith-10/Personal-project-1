# 🧪 Health AI Twin - Complete Testing Guide

## 📋 Table of Contents
1. [Quick Start Testing](#quick-start-testing)
2. [Manual Testing](#manual-testing)
3. [Automated Testing](#automated-testing)
4. [Component-Specific Testing](#component-specific-testing)
5. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start Testing

### **Step 1: Start All Services**
```bash
# Start all services automatically
python3 start_all_services.py

# Or start manually in separate terminals:
# Terminal 1: python3 start_part1.py
# Terminal 2: python3 start_part2.py
# Terminal 3: python3 start_part3.py
# Terminal 4: python3 start_part4.py
# Terminal 5: python3 start_part5.py
```

### **Step 2: Run Automated Tests**
```bash
# Run comprehensive system tests
python3 test_system.py
```

### **Step 3: Check Results**
- View test results in terminal
- Check `test_results.json` for detailed results
- Access services at provided URLs

---

## 🔧 Manual Testing

### **1. Service Health Checks**

Test if all services are running:
```bash
# Check each service
curl http://localhost:8003/health  # Part 1
curl http://localhost:8004/health  # Part 2
curl http://localhost:8005/health  # Part 3
curl http://localhost:8006/health  # Part 4
```

Expected response:
```json
{
  "status": "healthy",
  "service": "Health AI Twin",
  "timestamp": "2024-01-15T10:30:00"
}
```

### **2. Food Recognition Testing**

#### **Test with Sample Images**
```bash
# Test food classification
curl -X POST \
  -F "file=@test_image.jpg" \
  -F "patient_id=507f1f77bcf86cd799439011" \
  -F "meal_type=lunch" \
  http://localhost:8004/api/v1/food-classification/classify
```

#### **Expected Results**
- **Known foods** (e.g., "maggie.jpg"): >95% accuracy
- **Unknown foods**: Reasonable classification with appropriate confidence
- **Response format**:
```json
{
  "classification": {
    "food_name": "noodles",
    "confidence": 0.95,
    "category": "grains"
  },
  "nutrition": {
    "calories": 138,
    "protein": 4.5,
    "carbs": 26,
    "fat": 1.2,
    "fiber": 1.5
  }
}
```

#### **Test Different Food Types**
1. **Grains**: rice.jpg, bread.jpg, pasta.jpg
2. **Fruits**: apple.jpg, banana.jpg, orange.jpg
3. **Vegetables**: carrot.jpg, broccoli.jpg, tomato.jpg
4. **Proteins**: chicken.jpg, beef.jpg, fish.jpg
5. **Dairy**: milk.jpg, cheese.jpg, yogurt.jpg

### **3. AI Doctor Testing**

#### **Test Authentication**
```bash
# Login
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "test123"}' \
  http://localhost:8006/api/v1/auth/login
```

#### **Test AI Doctor Chat**
```bash
# Chat with AI doctor
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "question": "How can I improve my diet?",
    "include_food_data": true,
    "recent_food_log": [
      {
        "food_name": "apple",
        "nutrition": {"calories": 95, "protein": 0.5, "carbs": 25, "fat": 0.3}
      }
    ]
  }' \
  http://localhost:8006/api/v1/virtual-doctor/chat
```

#### **Expected Results**
- **Conversational responses** (not bullet points)
- **Personalized advice** based on food data
- **Context-aware recommendations**
- **Response format**:
```json
{
  "message": "Health consultation completed",
  "data": {
    "recommendations": [...],
    "health_insights": [...]
  },
  "conversational_response": "Hello! I'm Dr. Sarah Chen..."
}
```

### **4. Health Predictions Testing**

```bash
# Test health predictions
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "507f1f77bcf86cd799439011",
    "age": 35,
    "bmi": 24.5,
    "heart_rate_avg": 72,
    "steps_avg": 8500,
    "sleep_hours_avg": 7.5,
    "calories_avg": 2100
  }' \
  http://localhost:8005/api/v1/health-prediction/predict
```

### **5. Wearable Data Testing**

```bash
# Test wearable data submission
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "507f1f77bcf86cd799439011",
    "date": "2024-01-15",
    "heart_rate_avg": 72,
    "steps_count": 8500,
    "sleep_hours": 7.5,
    "spo2_avg": 98
  }' \
  http://localhost:8004/api/v1/wearable-data
```

---

## 🤖 Automated Testing

### **Run Complete Test Suite**
```bash
python3 test_system.py
```

### **Test Categories**
1. **Service Health**: All services running
2. **Authentication**: User login/logout
3. **Food Recognition**: Image classification
4. **AI Doctor**: Virtual consultations
5. **Health Predictions**: ML model predictions
6. **Wearable Data**: Data processing
7. **Frontend Access**: Dashboard accessibility

### **Test Results**
- **Pass/Fail status** for each test
- **Detailed error messages** for failures
- **Success rate calculation**
- **Results saved** to `test_results.json`

---

## 🎯 Component-Specific Testing

### **Food Recognition Accuracy Testing**

#### **Test Cases**
1. **High Accuracy Foods** (>95%):
   - `maggie.jpg` → noodles
   - `apple.jpg` → apple
   - `chicken.jpg` → chicken

2. **Category Recognition**:
   - Fruits: apple, banana, orange
   - Vegetables: carrot, broccoli, tomato
   - Grains: rice, bread, pasta
   - Proteins: chicken, beef, fish
   - Dairy: milk, cheese, yogurt

3. **Edge Cases**:
   - Unknown foods
   - Blurry images
   - Multiple foods in one image

#### **Validation Criteria**
- **Confidence > 95%** for known foods
- **Reasonable classification** for unknown foods
- **Proper nutrition data** returned
- **Fast response time** (< 5 seconds)

### **AI Doctor Response Testing**

#### **Test Questions**
1. **Diet Questions**:
   - "How can I improve my diet?"
   - "What should I eat for breakfast?"
   - "Is my current diet healthy?"

2. **Exercise Questions**:
   - "How much should I exercise?"
   - "What's the best workout for me?"
   - "How can I increase my activity?"

3. **Health Questions**:
   - "How can I lower my cholesterol?"
   - "What affects my blood sugar?"
   - "How can I sleep better?"

#### **Validation Criteria**
- **Conversational tone** (not bullet points)
- **Personalized advice** based on user data
- **Actionable recommendations**
- **Context awareness** (uses food data)

### **Health Predictions Testing**

#### **Test Scenarios**
1. **Normal Health**:
   - Age: 30, BMI: 22, Active lifestyle
   - Expected: Normal predictions

2. **Risk Factors**:
   - Age: 50, BMI: 28, Sedentary lifestyle
   - Expected: Elevated risk predictions

3. **Extreme Cases**:
   - Very high BMI, very low activity
   - Expected: High risk warnings

#### **Validation Criteria**
- **Realistic predictions** within expected ranges
- **Risk assessment** provided
- **Recommendations** included
- **Confidence scores** provided

---

## 🔍 Troubleshooting

### **Common Issues**

#### **1. Port Already in Use**
```bash
# Find and kill processes
lsof -i :8003
lsof -i :8004
lsof -i :8005
lsof -i :8006
lsof -i :8501

# Kill specific process
kill -9 <PID>
```

#### **2. MongoDB Connection Issues**
```bash
# Check MongoDB status
sudo systemctl status mongod

# Start MongoDB
sudo systemctl start mongod

# Check connection string
cat backend/app/services/database.py
```

#### **3. Food Recognition Not Working**
```bash
# Check dependencies
python3 -c "import torch, torchvision, PIL, numpy; print('Dependencies OK')"

# Check model loading
python3 -c "from services.food_classifier_service import AdvancedFoodClassifier; f = AdvancedFoodClassifier(); print('Model loaded')"
```

#### **4. AI Doctor Not Responding**
```bash
# Check OpenAI API key
echo $OPENAI_API_KEY

# Check LangChain installation
python3 -c "import langchain; print('LangChain OK')"
```

#### **5. Authentication Issues**
```bash
# Check JWT secret
echo $JWT_SECRET_KEY

# Test login endpoint
curl -X POST -H "Content-Type: application/json" -d '{"email": "test@example.com", "password": "test123"}' http://localhost:8006/api/v1/auth/login
```

### **Debug Information**

#### **Enable Debug Logging**
Add to `.env`:
```env
DEBUG=true
LOG_LEVEL=DEBUG
```

#### **Check Service Logs**
```bash
# Monitor service logs in real-time
tail -f logs/app.log

# Check specific service
ps aux | grep uvicorn
```

#### **Database Issues**
```bash
# Check MongoDB collections
mongo health_ai_twin --eval "db.getCollectionNames()"

# Check data
mongo health_ai_twin --eval "db.patients.find().pretty()"
```

---

## 📊 Performance Testing

### **Response Time Benchmarks**
- **Health Check**: < 1 second
- **Food Recognition**: < 5 seconds
- **AI Doctor Chat**: < 10 seconds
- **Health Predictions**: < 3 seconds
- **Wearable Data**: < 2 seconds

### **Load Testing**
```bash
# Test with multiple requests
for i in {1..10}; do
  curl http://localhost:8004/health &
done
wait
```

### **Memory Usage**
```bash
# Monitor memory usage
ps aux | grep python | grep -v grep
```

---

## ✅ Success Criteria

### **System Requirements**
- [ ] All 5 services start successfully
- [ ] Health checks return 200 OK
- [ ] Authentication works
- [ ] Food recognition >95% accuracy
- [ ] AI doctor provides conversational responses
- [ ] Health predictions generate realistic values
- [ ] Frontend dashboard accessible

### **Performance Requirements**
- [ ] Response times within benchmarks
- [ ] No memory leaks
- [ ] Stable under load
- [ ] Graceful error handling

### **User Experience**
- [ ] Intuitive interface
- [ ] Fast response times
- [ ] Accurate results
- [ ] Helpful error messages

---

## 🎉 Testing Complete!

Once all tests pass:
1. **System is ready** for production use
2. **All components** are working correctly
3. **Performance** meets requirements
4. **User experience** is optimal

**Your Health AI Twin system is fully operational!** 🏥✨



