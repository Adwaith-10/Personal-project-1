# 🍽️ Food Vision Pro

**AI-powered food analysis and nutrition tracking across web, mobile, and backend**

Food Vision Pro is a comprehensive nutrition tracking application that uses computer vision and machine learning to automatically detect, classify, and estimate the nutritional content of food from images. Built with modern technologies and designed for both personal and professional use.

## ✨ Features

### 🎯 Core Functionality
- **AI Food Detection**: YOLOv8 segmentation for multi-food detection
- **Smart Classification**: Food-101 fine-tuned classifier with ONNX runtime
- **Portion Estimation**: Volume calculation using density priors and shape analysis
- **Nutrition Calculation**: USDA FDC and Open Food Facts integration
- **Barcode Scanning**: Quick nutrition lookup via product barcodes

### 📱 Multi-Platform Support
- **Mobile App**: React Native (Expo) with camera, gallery, and barcode scanning
- **Web Portal**: Streamlit dashboard for analytics and meal management
- **Backend API**: FastAPI with PostgreSQL, Redis, and S3-compatible storage

### 🔒 Security & Privacy
- JWT authentication with refresh tokens
- GDPR compliance features
- Secure image storage with optional face redaction
- Role-based access control

### 💳 Optional Billing
- Stripe integration for subscription management
- Free tier with usage limits
- Premium features for advanced users

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mobile App    │    │   Web Portal    │    │   Backend API   │
│  (React Native) │    │   (Streamlit)   │    │    (FastAPI)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   PostgreSQL    │
                    │   + Redis       │
                    │   + MinIO/S3    │
                    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Node.js 18+ (for mobile development)
- Expo CLI (`npm install -g @expo/cli`)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd food-vision-pro
cp env.example .env
# Edit .env with your configuration
```

### 2. Start Services
```bash
# Start all services
docker-compose up --build

# Or start individual services
docker-compose up db cache minio backend web
```

### 3. Access Applications
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Web Portal**: http://localhost:8501
- **MinIO Console**: http://localhost:9001

### 4. Mobile App Development
```bash
cd mobile
npm install
npx expo start
```

## 📋 Environment Configuration

Copy `env.example` to `.env` and configure:

```bash
# Security
JWT_SECRET=your-super-secret-jwt-key-change-in-production

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/foodvision

# Storage
STORAGE_TYPE=local  # or minio, s3
S3_BUCKET=foodvision-images
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin123

# External APIs (optional)
USDA_FDC_API_KEY=your-usda-api-key
STRIPE_ENABLED=false
```

## 🧠 Machine Learning Models

### Training Custom Models
```bash
# Train food classifier
cd backend/ml
python train_classifier.py \
  --data_dir /path/to/food101 \
  --epochs 50 \
  --batch_size 32 \
  --model efficientnet_b0

# Export to ONNX
python export_onnx.py \
  --model_path ./trained_models/efficientnet_b0_best.pth \
  --optimize \
  --validate
```

### Model Requirements
- **Segmentation**: YOLOv8-seg (auto-downloaded)
- **Classification**: Custom ONNX model (Food-101 fine-tuned)
- **Density Priors**: YAML configuration file
- **Class Index**: JSON mapping of food classes

## 📱 Mobile App Development

### Features
- Camera capture with real-time food detection
- Gallery import with batch processing
- Barcode scanning for quick nutrition lookup
- Offline meal tracking
- Push notifications for meal reminders

### Development Commands
```bash
cd mobile

# Install dependencies
npm install

# Start development server
npx expo start

# Run on specific platform
npx expo start --android
npx expo start --ios
npx expo start --web

# Build for production
npx expo build:android
npx expo build:ios
```

## 🌐 Web Portal Features

### Dashboard
- Real-time nutrition overview
- Meal history and analytics
- Food database search
- User management (admin)

### Analytics
- Daily/weekly/monthly nutrition trends
- Macronutrient breakdowns
- Meal frequency analysis
- Export functionality (CSV)

## 🔌 API Endpoints

### Authentication
- `POST /api/v1/auth/signup` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Token refresh
- `POST /api/v1/auth/logout` - User logout

### Food Analysis
- `POST /api/v1/analyze` - Analyze food image
- `GET /api/v1/analyze/health` - Analysis service health

### Meals
- `POST /api/v1/meals` - Create meal
- `GET /api/v1/meals` - List user meals
- `GET /api/v1/meals/{id}` - Get specific meal
- `PATCH /api/v1/meals/{id}` - Update meal
- `DELETE /api/v1/meals/{id}` - Delete meal

### Food Database
- `GET /api/v1/foods/search` - Search foods
- `GET /api/v1/foods/barcode/{code}` - Barcode lookup
- `GET /api/v1/foods/popular` - Popular foods

## 🧪 Testing

### Backend Tests
```bash
cd backend
pip install -r requirements.txt
pytest tests/ -v
```

### API Tests
```bash
# Test authentication
python -m pytest tests/test_auth.py -v

# Test food analysis
python -m pytest tests/test_analyze.py -v

# Test meal management
python -m pytest tests/test_meals.py -v
```

### Mobile Tests
```bash
cd mobile
npm test
```

## 📊 Performance & Monitoring

### Health Checks
- Database connectivity
- Redis cache status
- ML model availability
- Storage service health

### Metrics
- Request response times
- Analysis processing times
- Model inference latency
- Error rates and types

### Logging
- Structured logging with JSON format
- Request/response logging
- Error tracking and alerting
- Performance monitoring

## 🔧 Development

### Code Quality
```bash
# Backend
cd backend
black app/
ruff check app/
mypy app/

# Mobile
cd mobile
npm run lint
npm run type-check
```

### Database Migrations
```bash
cd backend
alembic revision --autogenerate -m "Description"
alembic upgrade head
```

### Docker Development
```bash
# Development mode with hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Production mode
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

## 🚀 Deployment

### Production Considerations
- Use strong, unique secrets
- Enable HTTPS with proper certificates
- Configure proper CORS origins
- Set up monitoring and alerting
- Implement backup strategies
- Use production-grade databases

### Cloud Deployment
- **AWS**: ECS/Fargate, RDS, ElastiCache, S3
- **Google Cloud**: Cloud Run, Cloud SQL, Memorystore, Cloud Storage
- **Azure**: Container Instances, Azure SQL, Redis Cache, Blob Storage

### Environment Variables
```bash
# Production settings
DEBUG=false
LOG_LEVEL=WARNING
STORAGE_TYPE=s3
STRIPE_ENABLED=true
GDPR_ENABLED=true
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Development Guidelines
- Follow PEP 8 for Python code
- Use TypeScript for mobile app
- Write comprehensive tests
- Update documentation
- Follow semantic versioning

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Documentation
- [API Documentation](http://localhost:8000/docs) (when running)
- [User Guide](docs/USER_GUIDE.md)
- [Developer Guide](docs/DEVELOPER_GUIDE.md)
- [API Reference](docs/API_REFERENCE.md)

### Community
- [GitHub Issues](https://github.com/your-repo/issues)
- [Discussions](https://github.com/your-repo/discussions)
- [Wiki](https://github.com/your-repo/wiki)

### Professional Support
For enterprise support and custom development, contact:
- Email: support@foodvisionpro.com
- Website: https://foodvisionpro.com

## 🙏 Acknowledgments

- **Food-101 Dataset**: ETH Zurich for the food classification dataset
- **YOLOv8**: Ultralytics for the segmentation model
- **USDA FDC**: FoodData Central API for nutrition data
- **Open Food Facts**: Community-driven food database
- **Open Source Community**: All the amazing libraries and tools

## 📈 Roadmap

### v1.1 (Q1 2024)
- [ ] Advanced analytics dashboard
- [ ] Meal planning and recipes
- [ ] Social features and sharing
- [ ] Multi-language support

### v1.2 (Q2 2024)
- [ ] Wearable device integration
- [ ] AI-powered meal recommendations
- [ ] Advanced portion estimation
- [ ] Export to health apps

### v2.0 (Q3 2024)
- [ ] Real-time collaboration
- [ ] Advanced ML models
- [ ] Enterprise features
- [ ] Mobile app store release

---

**Made with ❤️ by the Food Vision Pro Team**

*Empowering healthier eating through AI-powered nutrition tracking*
