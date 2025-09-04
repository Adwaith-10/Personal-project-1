# Food Vision Pro - Development Makefile

.PHONY: help install start stop clean test lint format build mobile web backend db migrate logs

# Default target
help:
	@echo "🍽️ Food Vision Pro - Development Commands"
	@echo ""
	@echo "📋 Setup & Installation:"
	@echo "  install     Install all dependencies"
	@echo "  setup       Initial setup (copy env, create dirs)"
	@echo ""
	@echo "🚀 Running Services:"
	@echo "  start       Start all services with Docker Compose"
	@echo "  stop        Stop all services"
	@echo "  restart     Restart all services"
	@echo "  logs        View logs from all services"
	@echo ""
	@echo "🔧 Development:"
	@echo "  backend     Start only backend service"
	@echo "  web         Start only web portal"
	@echo "  db          Start only database services"
	@echo "  mobile      Start mobile development server"
	@echo ""
	@echo "🧪 Testing & Quality:"
	@echo "  test        Run all tests"
	@echo "  test-backend Run backend tests"
	@echo "  test-mobile Run mobile tests"
	@echo "  lint        Run linting checks"
	@echo "  format      Format code with black and ruff"
	@echo "  type-check  Run type checking with mypy"
	@echo ""
	@echo "🏗️ Building & Deployment:"
	@echo "  build       Build all Docker images"
	@echo "  build-backend Build backend image"
	@echo "  build-web   Build web portal image"
	@echo "  deploy      Deploy to production (placeholder)"
	@echo ""
	@echo "🗄️ Database:"
	@echo "  migrate     Run database migrations"
	@echo "  db-reset    Reset database (WARNING: destroys data)"
	@echo "  db-seed     Seed database with sample data"
	@echo ""
	@echo "🧠 Machine Learning:"
	@echo "  train       Train food classifier model"
	@echo "  export      Export model to ONNX format"
	@echo "  download-models Download pre-trained models"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  clean       Clean up Docker resources and cache"
	@echo "  clean-models Clean up ML model files"
	@echo "  update      Update dependencies and Docker images"

# Setup and installation
install:
	@echo "📦 Installing dependencies..."
	cd backend && pip install -r requirements.txt
	cd web && pip install -r requirements.txt
	cd mobile && npm install
	@echo "✅ Dependencies installed!"

setup:
	@echo "🔧 Setting up Food Vision Pro..."
	cp env.example .env
	@echo "📝 Environment file created. Please edit .env with your configuration."
	mkdir -p backend/uploads backend/ml_models logs
	@echo "📁 Directories created."
	@echo "✅ Setup complete! Edit .env and run 'make start'"

# Service management
start:
	@echo "🚀 Starting Food Vision Pro services..."
	docker-compose up --build -d
	@echo "✅ Services started!"
	@echo "🌐 Backend API: http://localhost:8000"
	@echo "📊 Web Portal: http://localhost:8501"
	@echo "🗄️ MinIO Console: http://localhost:9001"

stop:
	@echo "🛑 Stopping services..."
	docker-compose down
	@echo "✅ Services stopped!"

restart: stop start

logs:
	@echo "📋 Viewing logs..."
	docker-compose logs -f

# Individual services
backend:
	@echo "🔧 Starting backend service..."
	docker-compose up --build backend -d

web:
	@echo "🌐 Starting web portal..."
	docker-compose up --build web -d

db:
	@echo "🗄️ Starting database services..."
	docker-compose up --build db cache minio -d

mobile:
	@echo "📱 Starting mobile development server..."
	cd mobile && npx expo start

# Testing
test: test-backend test-mobile

test-backend:
	@echo "🧪 Running backend tests..."
	cd backend && python -m pytest tests/ -v

test-mobile:
	@echo "🧪 Running mobile tests..."
	cd mobile && npm test

# Code quality
lint:
	@echo "🔍 Running linting checks..."
	cd backend && ruff check app/
	cd mobile && npm run lint

format:
	@echo "✨ Formatting code..."
	cd backend && black app/ && ruff format app/
	cd mobile && npm run format

type-check:
	@echo "🔍 Running type checks..."
	cd backend && mypy app/
	cd mobile && npm run type-check

# Building
build: build-backend build-web

build-backend:
	@echo "🏗️ Building backend image..."
	docker-compose build backend

build-web:
	@echo "🏗️ Building web portal image..."
	docker-compose build web

# Database operations
migrate:
	@echo "🗄️ Running database migrations..."
	cd backend && alembic upgrade head

db-reset:
	@echo "⚠️  WARNING: This will destroy all data!"
	@read -p "Are you sure? Type 'yes' to continue: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		docker-compose down -v; \
		docker-compose up db -d; \
		echo "✅ Database reset complete!"; \
	else \
		echo "❌ Database reset cancelled."; \
	fi

db-seed:
	@echo "🌱 Seeding database with sample data..."
	cd backend && python scripts/seed_data.py

# Machine Learning
train:
	@echo "🧠 Training food classifier model..."
	cd backend/ml && python train_classifier.py --help

export:
	@echo "📤 Exporting model to ONNX..."
	cd backend/ml && python export_onnx.py --help

download-models:
	@echo "⬇️  Downloading pre-trained models..."
	cd backend/ml && python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"
	@echo "✅ Models downloaded!"

# Maintenance
clean:
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	docker volume prune -f
	@echo "✅ Cleanup complete!"

clean-models:
	@echo "🧹 Cleaning up ML model files..."
	rm -rf backend/ml_models/*
	@echo "✅ Model files cleaned!"

update:
	@echo "🔄 Updating dependencies and images..."
	cd backend && pip install -r requirements.txt --upgrade
	cd web && pip install -r requirements.txt --upgrade
	cd mobile && npm update
	docker-compose pull
	@echo "✅ Updates complete!"

# Development helpers
dev-backend:
	@echo "🔧 Starting backend in development mode..."
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-web:
	@echo "🌐 Starting web portal in development mode..."
	cd web && streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# Production helpers
deploy:
	@echo "🚀 Deploying to production..."
	@echo "⚠️  This is a placeholder. Implement your deployment strategy."
	@echo "Consider using:"
	@echo "  - Docker Swarm"
	@echo "  - Kubernetes"
	@echo "  - AWS ECS/Fargate"
	@echo "  - Google Cloud Run"
	@echo "  - Azure Container Instances"

# Health checks
health:
	@echo "🏥 Checking service health..."
	@echo "Backend API:" && curl -s http://localhost:8000/health || echo "❌ Backend not responding"
	@echo "Web Portal:" && curl -s http://localhost:8501/_stcore/health || echo "❌ Web portal not responding"
	@echo "Database:" && docker-compose exec -T db pg_isready -U postgres || echo "❌ Database not responding"

# Quick development workflow
dev: setup start
	@echo "🎉 Development environment ready!"
	@echo "Run 'make logs' to view service logs"
	@echo "Run 'make mobile' to start mobile development"

# Show service status
status:
	@echo "📊 Service Status:"
	docker-compose ps
	@echo ""
	@echo "🌐 Service URLs:"
	@echo "  Backend API: http://localhost:8000"
	@echo "  Web Portal: http://localhost:8501"
	@echo "  MinIO Console: http://localhost:9001"
	@echo "  Database: localhost:5432"
	@echo "  Redis: localhost:6379"
