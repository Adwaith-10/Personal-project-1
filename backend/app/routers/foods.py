from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.db import get_db
from app.models import User, FoodCache
from app.schemas import FoodSearchRequest, FoodSearchResponse
from app.services.nutrition import nutrition_service
from app.deps import get_current_user
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/search", response_model=FoodSearchResponse)
async def search_foods(
    q: str = Query(..., min_length=1, max_length=100, description="Search query"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Search for foods by name or barcode"""
    try:
        # First, search in local cache
        cached_results = await _search_local_cache(db, q)
        
        # If we have good results from cache, return them
        if len(cached_results) >= 5:
            logger.info(f"✅ Found {len(cached_results)} cached results for query: {q}")
            return FoodSearchResponse(
                foods=cached_results,
                total=len(cached_results)
            )
        
        # Otherwise, search external APIs and cache results
        external_results = await _search_external_apis(q)
        
        # Cache external results
        if external_results:
            await _cache_external_results(db, external_results)
        
        # Combine results (cached first, then external)
        all_results = cached_results + external_results
        
        # Remove duplicates based on name
        seen_names = set()
        unique_results = []
        for result in all_results:
            if result.get("name") not in seen_names:
                seen_names.add(result["name"])
                unique_results.append(result)
        
        logger.info(f"✅ Search completed for '{q}': {len(unique_results)} results")
        
        return FoodSearchResponse(
            foods=unique_results[:20],  # Limit to 20 results
            total=len(unique_results)
        )
        
    except Exception as e:
        logger.error(f"❌ Food search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Food search failed"
        )


@router.get("/barcode/{barcode}")
async def get_food_by_barcode(
    barcode: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get food information by barcode"""
    try:
        # First check local cache
        result = await db.execute(
            select(FoodCache).where(FoodCache.barcode == barcode)
        )
        cached_food = result.scalar_one_or_none()
        
        if cached_food:
            logger.info(f"✅ Found cached food for barcode: {barcode}")
            return _food_cache_to_dict(cached_food)
        
        # Search external APIs
        external_results = await _search_external_apis("", barcode)
        
        if external_results:
            # Cache the result
            await _cache_external_results(db, external_results)
            return external_results[0]
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Food not found for this barcode"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Barcode lookup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Barcode lookup failed"
        )


@router.get("/popular")
async def get_popular_foods(
    limit: int = Query(10, ge=1, le=50, description="Number of popular foods to return"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get popular foods based on usage"""
    try:
        # This is a simple implementation - in production you'd track actual usage
        # For now, return some common foods
        common_foods = [
            {
                "name": "Chicken Breast",
                "calories_per_100g": 165,
                "protein_per_100g": 31,
                "carbs_per_100g": 0,
                "fat_per_100g": 3.6,
                "source": "default"
            },
            {
                "name": "Brown Rice",
                "calories_per_100g": 111,
                "protein_per_100g": 2.6,
                "carbs_per_100g": 23,
                "fat_per_100g": 0.9,
                "source": "default"
            },
            {
                "name": "Broccoli",
                "calories_per_100g": 34,
                "protein_per_100g": 2.8,
                "carbs_per_100g": 7,
                "fat_per_100g": 0.4,
                "source": "default"
            },
            {
                "name": "Salmon",
                "calories_per_100g": 208,
                "protein_per_100g": 25,
                "carbs_per_100g": 0,
                "fat_per_100g": 12,
                "source": "default"
            },
            {
                "name": "Sweet Potato",
                "calories_per_100g": 86,
                "protein_per_100g": 1.6,
                "carbs_per_100g": 20,
                "fat_per_100g": 0.1,
                "source": "default"
            }
        ]
        
        return {"foods": common_foods[:limit]}
        
    except Exception as e:
        logger.error(f"❌ Failed to get popular foods: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve popular foods"
        )


async def _search_local_cache(db: AsyncSession, query: str) -> List[dict]:
    """Search for foods in local cache"""
    try:
        # Simple text search in cached foods
        result = await db.execute(
            select(FoodCache).where(
                func.lower(FoodCache.name).contains(func.lower(query))
            ).limit(10)
        )
        cached_foods = result.scalars().all()
        
        return [_food_cache_to_dict(food) for food in cached_foods]
        
    except Exception as e:
        logger.warning(f"Local cache search failed: {e}")
        return []


async def _search_external_apis(query: str, barcode: str = None) -> List[dict]:
    """Search external APIs for food information"""
    try:
        if barcode:
            # Barcode lookup
            nutrition = await nutrition_service.get_nutrition_info("", barcode)
            if nutrition:
                return [nutrition]
        else:
            # Name search
            nutrition = await nutrition_service.get_nutrition_info(query)
            if nutrition:
                return [nutrition]
        
        return []
        
    except Exception as e:
        logger.warning(f"External API search failed: {e}")
        return []


async def _cache_external_results(db: AsyncSession, foods: List[dict]) -> None:
    """Cache external food results in database"""
    try:
        for food_data in foods:
            # Check if already exists
            existing = None
            if food_data.get("barcode"):
                result = await db.execute(
                    select(FoodCache).where(FoodCache.barcode == food_data["barcode"])
                )
                existing = result.scalar_one_or_none()
            elif food_data.get("fdc_id"):
                result = await db.execute(
                    select(FoodCache).where(FoodCache.fdc_id == food_data["fdc_id"])
                )
                existing = result.scalar_one_or_none()
            
            if not existing:
                # Create new cache entry
                cache_entry = FoodCache(
                    name=food_data.get("name", "Unknown"),
                    barcode=food_data.get("barcode"),
                    fdc_id=food_data.get("fdc_id"),
                    openfoodfacts_id=food_data.get("openfoodfacts_id"),
                    calories_per_100g=food_data.get("calories_per_100g"),
                    protein_per_100g=food_data.get("protein_per_100g"),
                    carbs_per_100g=food_data.get("carbs_per_100g"),
                    fat_per_100g=food_data.get("fat_per_100g"),
                    fiber_per_100g=food_data.get("fiber_per_100g"),
                    sugar_per_100g=food_data.get("sugar_per_100g"),
                    sodium_per_100g=food_data.get("sodium_per_100g"),
                    source=food_data.get("source", "external")
                )
                
                db.add(cache_entry)
        
        await db.commit()
        logger.info(f"✅ Cached {len(foods)} external food results")
        
    except Exception as e:
        logger.warning(f"Failed to cache external results: {e}")


def _food_cache_to_dict(food: FoodCache) -> dict:
    """Convert FoodCache model to dictionary"""
    return {
        "id": food.id,
        "name": food.name,
        "barcode": food.barcode,
        "fdc_id": food.fdc_id,
        "openfoodfacts_id": food.openfoodfacts_id,
        "calories_per_100g": food.calories_per_100g,
        "protein_per_100g": food.protein_per_100g,
        "carbs_per_100g": food.carbs_per_100g,
        "fat_per_100g": food.fat_per_100g,
        "fiber_per_100g": food.fiber_per_100g,
        "sugar_per_100g": food.sugar_per_100g,
        "sodium_per_100g": food.sodium_per_100g,
        "source": food.source,
        "last_updated": food.last_updated.isoformat() if food.last_updated else None
    }
