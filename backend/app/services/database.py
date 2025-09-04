from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.database import Database
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
client: AsyncIOMotorClient = None
db: Database = None

async def connect_to_mongo():
    """Create database connection"""
    global client, db
    try:
        client = AsyncIOMotorClient(MONGODB_URL)
        db = client.health_ai_twin
        # Test the connection
        await client.admin.command('ping')
        print("✅ Connected to MongoDB")
        return client, db
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        raise e

async def close_mongo_connection():
    """Close database connection"""
    global client
    if client:
        client.close()
        print("🔌 Disconnected from MongoDB")

def get_database() -> Database:
    """Get database instance"""
    return db

async def get_collection(collection_name: str):
    """Get a specific collection from the database"""
    if db is None:
        raise Exception("Database not connected")
    return db[collection_name]

# Database initialization functions
async def init_db():
    """Initialize database with indexes and initial data"""
    if db is None:
        raise Exception("Database not connected")
    
    # Create indexes for better performance
    await db.patients.create_index("email", unique=True)
    await db.patients.create_index("first_name")
    await db.patients.create_index("last_name")
    await db.health_metrics.create_index("patient_id")
    await db.health_metrics.create_index("timestamp")
    
    print("✅ Database indexes created")

async def seed_sample_data():
    """Seed database with sample data for testing"""
    if db is None:
        raise Exception("Database not connected")
    
    # Check if sample data already exists
    patient_count = await db.patients.count_documents({})
    if patient_count > 0:
        print("ℹ️ Sample data already exists, skipping seed")
        return
    
    # Sample patients
    sample_patients = [
        {
            "first_name": "John",
            "last_name": "Doe",
            "date_of_birth": "1990-01-15T00:00:00",
            "gender": "male",
            "email": "john.doe@example.com",
            "phone": "+1234567890",
            "blood_type": "A+",
            "height_cm": 175.0,
            "weight_kg": 70.0,
            "emergency_contact": "+1234567891",
            "medical_history": [],
            "current_medications": [],
            "allergies": ["Penicillin"],
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00"
        },
        {
            "first_name": "Jane",
            "last_name": "Smith",
            "date_of_birth": "1985-05-20T00:00:00",
            "gender": "female",
            "email": "jane.smith@example.com",
            "phone": "+1234567892",
            "blood_type": "O+",
            "height_cm": 165.0,
            "weight_kg": 60.0,
            "emergency_contact": "+1234567893",
            "medical_history": [],
            "current_medications": [],
            "allergies": [],
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00"
        }
    ]
    
    # Insert sample patients
    result = await db.patients.insert_many(sample_patients)
    print(f"✅ Inserted {len(result.inserted_ids)} sample patients")
    
    # Sample health metrics
    from datetime import datetime, timedelta
    import random
    
    sample_metrics = []
    for patient_id in result.inserted_ids:
        for i in range(10):  # 10 days of data
            date = datetime.now() - timedelta(days=i)
            sample_metrics.append({
                "patient_id": str(patient_id),
                "timestamp": date,
                "heart_rate": random.randint(60, 100),
                "blood_pressure_systolic": random.randint(110, 140),
                "blood_pressure_diastolic": random.randint(70, 90),
                "temperature": round(random.uniform(36.5, 37.5), 1),
                "oxygen_saturation": round(random.uniform(95.0, 99.0), 1),
                "respiratory_rate": random.randint(12, 20),
                "glucose_level": round(random.uniform(80.0, 120.0), 1),
                "notes": f"Sample reading {i+1}"
            })
    
    # Insert sample metrics
    metrics_result = await db.health_metrics.insert_many(sample_metrics)
    print(f"✅ Inserted {len(metrics_result.inserted_ids)} sample health metrics")
