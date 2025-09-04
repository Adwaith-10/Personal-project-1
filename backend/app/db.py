from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.config import settings
from app.models import Base
import redis.asyncio as redis
import logging

logger = logging.getLogger(__name__)

# Database engine
engine = None
SessionLocal = None

# Redis client
redis_client = None


def get_database_url():
    """Convert PostgreSQL URL to async format"""
    if settings.DATABASE_URL.startswith("postgresql://"):
        return settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    return settings.DATABASE_URL


async def init_db():
    """Initialize database connection and create tables"""
    global engine, SessionLocal, redis_client
    
    try:
        # Initialize PostgreSQL
        database_url = get_database_url()
        engine = create_async_engine(
            database_url,
            echo=settings.DEBUG,
            pool_pre_ping=True,
            pool_recycle=300,
        )
        
        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Create session factory
        SessionLocal = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        
        logger.info("✅ Database initialized successfully")
        
        # Initialize Redis
        redis_client = redis.from_url(settings.REDIS_URL)
        await redis_client.ping()
        logger.info("✅ Redis connected successfully")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise e


async def get_db():
    """Get database session"""
    if not SessionLocal:
        raise RuntimeError("Database not initialized")
    
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_redis():
    """Get Redis client"""
    if not redis_client:
        raise RuntimeError("Redis not initialized")
    return redis_client


async def close_db():
    """Close database connections"""
    global engine, redis_client
    
    if engine:
        await engine.dispose()
        logger.info("Database connections closed")
    
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")
