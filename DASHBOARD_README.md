# 🏥 Health AI Twin Dashboard

A comprehensive Streamlit dashboard for health monitoring and virtual doctor consultations, featuring real-time health metrics visualization, trend analysis, and AI-powered health advice.

## 🚀 Features

### 📊 Health Overview
- **Real-time Health Monitoring**: Track overall health status, risk levels, and deviations
- **Health Predictions**: Visualize predicted LDL, glucose, and hemoglobin trends
- **Automated Warnings**: Get alerts for unhealthy trends and concerning patterns
- **Health Recommendations**: Receive personalized health advice based on your data

### 💓 Heart Rate Analysis
- **Trend Visualization**: Interactive charts showing heart rate patterns over time
- **Resting vs Average HR**: Compare resting and average heart rate trends
- **Normal Range Indicators**: Visual markers for healthy heart rate ranges (60-100 bpm)
- **Statistical Summary**: Average, minimum, and maximum heart rate metrics

### 😴 Sleep Analysis
- **Sleep Duration Tracking**: Monitor total sleep hours with recommended ranges (7-9 hours)
- **Sleep Quality Scoring**: Track sleep quality scores over time
- **Sleep Stage Analysis**: Visualize deep, REM, and light sleep patterns
- **Sleep Trend Warnings**: Alerts for insufficient or excessive sleep

### 🍎 Nutrition Analysis
- **Daily Nutrition Tracking**: Monitor calories, protein, carbs, and fat intake
- **Nutrition Goals**: Visual indicators for recommended daily intake levels
- **Food Log Integration**: Connect with food classification system
- **Nutrition Trends**: Track nutritional patterns over time

### 👩‍⚕️ Virtual Doctor
- **AI-Powered Consultations**: Chat with Dr. Sarah Chen, a lifestyle medicine specialist
- **Health Data Integration**: Doctor analyzes your real-time health data
- **Personalized Advice**: Get tailored recommendations based on your health profile
- **Urgency Assessment**: Receive guidance on when to seek medical attention

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- FastAPI backend running on localhost:8000
- MongoDB database with health data

### Setup
1. **Install Dependencies**:
   ```bash
   pip install streamlit plotly pandas requests numpy
   ```

2. **Start the Backend API**:
   ```bash
   cd backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Launch the Dashboard**:
   ```bash
   # Option 1: Use the launcher script
   python run_dashboard.py
   
   # Option 2: Run directly with Streamlit
   streamlit run frontend/health_dashboard.py --server.port 8501
   ```

## 📱 Usage

### Getting Started
1. **Access the Dashboard**: Open http://localhost:8501 in your browser
2. **Select Patient**: Choose a patient from the sidebar dropdown
3. **Set Date Range**: Adjust the time period for data visualization (7-90 days)
4. **Explore Tabs**: Navigate through different health monitoring sections

### Dashboard Navigation

#### 📊 Health Overview Tab
- View overall health status and risk assessment
- See health predictions and trend analysis
- Get automated health warnings and recommendations
- Monitor critical health deviations

#### 💓 Heart Rate Tab
- Analyze heart rate trends over time
- Compare resting vs average heart rate
- View statistical summaries
- Identify concerning patterns

#### 😴 Sleep Tab
- Track sleep duration and quality
- Monitor sleep stage distribution
- View sleep trend analysis
- Get sleep improvement recommendations

#### 🍎 Nutrition Tab
- Monitor daily nutrition intake
- Track macronutrient balance
- View nutrition trends over time
- Compare with recommended daily values

#### 👩‍⚕️ Virtual Doctor Tab
- Chat with AI-powered virtual doctor
- Ask health questions and get personalized advice
- Receive health insights and recommendations
- Get guidance on lifestyle improvements

### Virtual Doctor Features
- **Real-time Health Analysis**: Doctor analyzes your current health data
- **Personalized Recommendations**: Get advice tailored to your health profile
- **Urgency Assessment**: Understand when to seek medical attention
- **Lifestyle Guidance**: Receive tips for improving health and wellness

## 🔧 Configuration

### API Configuration
The dashboard connects to the FastAPI backend at `http://localhost:8000`. To change this:

1. Edit `frontend/health_dashboard.py`
2. Modify the `API_BASE_URL` variable:
   ```python
   API_BASE_URL = "http://your-api-server:port"
   ```

### Dashboard Customization
- **Date Range**: Adjust the default date range in the sidebar
- **Charts**: Modify chart colors and styling in the CSS section
- **Warnings**: Customize warning thresholds in the `analyze_trends` method

## 📊 Data Sources

The dashboard integrates with multiple data sources:

### Wearable Data
- Heart rate (average, resting, min, max)
- Sleep data (duration, quality, stages)
- Activity levels and steps
- Heart rate variability (HRV)
- Blood oxygen levels (SpO2)

### Health Predictions
- LDL cholesterol predictions
- Glucose level forecasts
- Hemoglobin predictions
- Confidence scores and model versions

### Nutrition Data
- Food classification results
- Calorie and macronutrient tracking
- Daily nutrition summaries
- Food log history

### Lab Reports
- Actual biomarker values
- Lab report history
- Comparison with predictions
- Deviation analysis

## 🚨 Health Warnings

The dashboard automatically detects and alerts users to:

### Heart Rate Warnings
- **Elevated Heart Rate**: Average HR > 100 bpm
- **Low Heart Rate**: Average HR < 50 bpm
- **Irregular Patterns**: Sudden changes in heart rate

### Sleep Warnings
- **Insufficient Sleep**: < 6 hours average
- **Excessive Sleep**: > 10 hours average
- **Poor Sleep Quality**: Low quality scores

### Nutrition Warnings
- **Calorie Imbalance**: Significant deviations from recommended intake
- **Macronutrient Imbalance**: Protein, carb, or fat deficiencies
- **Inconsistent Eating**: Irregular meal patterns

## 🔒 Security & Privacy

- **Local Deployment**: Dashboard runs locally for data privacy
- **Patient Selection**: Secure patient data access
- **Session Management**: Chat sessions are maintained securely
- **Data Encryption**: All API communications use HTTPS (in production)

## 🐛 Troubleshooting

### Common Issues

1. **Dashboard Won't Load**:
   - Ensure FastAPI backend is running on port 8000
   - Check that all dependencies are installed
   - Verify MongoDB connection

2. **No Data Displayed**:
   - Confirm patient has health data in the database
   - Check API endpoints are responding correctly
   - Verify date range selection

3. **Virtual Doctor Not Responding**:
   - Ensure LangChain and OpenAI are configured
   - Check API key configuration
   - Verify patient data is available for analysis

### Debug Mode
Enable debug logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance

### Optimization Tips
- **Data Caching**: Dashboard caches API responses for better performance
- **Lazy Loading**: Charts load data on-demand
- **Efficient Queries**: Optimized database queries for large datasets
- **Responsive Design**: Dashboard adapts to different screen sizes

### Scalability
- **Multiple Patients**: Dashboard supports multiple patient profiles
- **Large Datasets**: Efficiently handles large amounts of health data
- **Real-time Updates**: Supports real-time data updates
- **Concurrent Users**: Multiple users can access simultaneously

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Add type hints to functions
- Include docstrings for classes and methods
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the troubleshooting section
- Review the API documentation
- Open an issue on GitHub
- Contact the development team

## 🔄 Updates

### Version History
- **v1.0.0**: Initial release with basic health monitoring
- **v1.1.0**: Added virtual doctor integration
- **v1.2.0**: Enhanced trend analysis and warnings
- **v1.3.0**: Improved nutrition tracking and visualization

### Future Features
- **Mobile App**: Native mobile application
- **Advanced Analytics**: Machine learning insights
- **Integration APIs**: Third-party health device integration
- **Telemedicine**: Video consultation features
- **Health Goals**: Goal setting and tracking
- **Social Features**: Family health sharing

---

**🏥 Health AI Twin Dashboard** - Empowering health monitoring with AI-driven insights and personalized care.
