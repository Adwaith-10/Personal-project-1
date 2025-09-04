from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
from contextlib import asynccontextmanager

from app.config import settings
from app.db import init_db
from app.routers import auth, analyze, meals, foods, billing
from app.services.logging_mw import LoggingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    yield
    # Shutdown


app = FastAPI(
    title="Food Vision Pro API",
    description="AI-powered food analysis and nutrition tracking",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(analyze.router, prefix="/api/v1", tags=["Analysis"])
app.include_router(meals.router, prefix="/api/v1/meals", tags=["Meals"])
app.include_router(foods.router, prefix="/api/v1/foods", tags=["Foods"])
app.include_router(billing.router, prefix="/api/v1/billing", tags=["Billing"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Food Vision Pro API"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
