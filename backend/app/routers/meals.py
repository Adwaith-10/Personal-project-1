from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.db import get_db
from app.models import User, Meal, MealItem
from app.schemas import MealCreate, MealResponse, MealUpdate, PaginationParams, PaginatedResponse
from app.deps import get_current_user
from datetime import datetime, timedelta
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=MealResponse, status_code=status.HTTP_201_CREATED)
async def create_meal(
    meal_data: MealCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new meal"""
    try:
        # Calculate totals from items
        total_calories = sum(item.calories or 0 for item in meal_data.items)
        total_protein = sum(item.protein_g or 0 for item in meal_data.items)
        total_carbs = sum(item.carbs_g or 0 for item in meal_data.items)
        total_fat = sum(item.fat_g or 0 for item in meal_data.items)
        
        # Create meal
        meal = Meal(
            user_id=current_user.id,
            name=meal_data.name,
            plate_diameter_cm=meal_data.plate_diameter_cm,
            total_calories=total_calories,
            total_protein=total_protein,
            total_carbs=total_carbs,
            total_fat=total_fat,
            notes=meal_data.notes
        )
        
        db.add(meal)
        await db.commit()
        await db.refresh(meal)
        
        # Create meal items
        for item_data in meal_data.items:
            meal_item = MealItem(
                meal_id=meal.id,
                label=item_data.label,
                confidence=item_data.confidence,
                grams_estimated=item_data.grams_estimated,
                grams_actual=item_data.grams_actual,
                calories=item_data.calories,
                protein_g=item_data.protein_g,
                carbs_g=item_data.carbs_g,
                fat_g=item_data.fat_g,
                mask_polygon=item_data.mask_polygon
            )
            db.add(meal_item)
        
        await db.commit()
        await db.refresh(meal)
        
        logger.info(f"✅ Meal created for user {current_user.email}: {meal.id}")
        
        # Return meal with items
        return await _get_meal_with_items(db, meal.id)
        
    except Exception as e:
        logger.error(f"❌ Meal creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create meal"
        )


@router.get("/", response_model=PaginatedResponse)
async def get_meals(
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's meals with pagination and date filtering"""
    try:
        # Build query
        query = select(Meal).where(Meal.user_id == current_user.id)
        
        # Apply date filters
        if from_date:
            try:
                from_dt = datetime.strptime(from_date, "%Y-%m-%d")
                query = query.where(Meal.created_at >= from_dt)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid from_date format. Use YYYY-MM-DD"
                )
        
        if to_date:
            try:
                to_dt = datetime.strptime(to_date, "%Y-%m-%d") + timedelta(days=1)
                query = query.where(Meal.created_at < to_dt)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid to_date format. Use YYYY-MM-DD"
                )
        
        # Order by creation date (newest first)
        query = query.order_by(Meal.created_at.desc())
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        offset = (page - 1) * size
        query = query.offset(offset).limit(size)
        
        # Execute query
        result = await db.execute(query)
        meals = result.scalars().all()
        
        # Get meals with items
        meals_with_items = []
        for meal in meals:
            meal_with_items = await _get_meal_with_items(db, meal.id)
            meals_with_items.append(meal_with_items)
        
        # Calculate pagination info
        pages = (total + size - 1) // size
        
        logger.info(f"✅ Retrieved {len(meals_with_items)} meals for user {current_user.email}")
        
        return PaginatedResponse(
            items=meals_with_items,
            total=total,
            page=page,
            size=size,
            pages=pages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get meals: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve meals"
        )


@router.get("/{meal_id}", response_model=MealResponse)
async def get_meal(
    meal_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific meal by ID"""
    try:
        # Get meal
        result = await db.execute(
            select(Meal).where(Meal.id == meal_id, Meal.user_id == current_user.id)
        )
        meal = result.scalar_one_or_none()
        
        if not meal:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Meal not found"
            )
        
        return await _get_meal_with_items(db, meal.id)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get meal {meal_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve meal"
        )


@router.patch("/{meal_id}", response_model=MealResponse)
async def update_meal(
    meal_id: str,
    meal_update: MealUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a meal"""
    try:
        # Get meal
        result = await db.execute(
            select(Meal).where(Meal.id == meal_id, Meal.user_id == current_user.id)
        )
        meal = result.scalar_one_or_none()
        
        if not meal:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Meal not found"
            )
        
        # Update meal fields
        if meal_update.name is not None:
            meal.name = meal_update.name
        if meal_update.notes is not None:
            meal.notes = meal_update.notes
        
        # Update items if provided
        if meal_update.items:
            # Delete existing items
            await db.execute(
                select(MealItem).where(MealItem.meal_id == meal_id)
            )
            existing_items = result.scalars().all()
            for item in existing_items:
                await db.delete(item)
            
            # Create new items
            for item_data in meal_update.items:
                meal_item = MealItem(
                    meal_id=meal.id,
                    label=item_data.label,
                    confidence=item_data.confidence,
                    grams_estimated=item_data.grams_estimated,
                    grams_actual=item_data.grams_actual,
                    calories=item_data.calories,
                    protein_g=item_data.protein_g,
                    carbs_g=item_data.carbs_g,
                    fat_g=item_data.fat_g,
                    mask_polygon=item_data.mask_polygon
                )
                db.add(meal_item)
            
            # Recalculate totals
            total_calories = sum(item.calories or 0 for item in meal_update.items)
            total_protein = sum(item.protein_g or 0 for item in meal_update.items)
            total_carbs = sum(item.carbs_g or 0 for item in meal_update.items)
            total_fat = sum(item.fat_g or 0 for item in meal_update.items)
            
            meal.total_calories = total_calories
            meal.total_protein = total_protein
            meal.total_carbs = total_carbs
            meal.total_fat = total_fat
        
        meal.updated_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(meal)
        
        logger.info(f"✅ Meal {meal_id} updated for user {current_user.email}")
        
        return await _get_meal_with_items(db, meal.id)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update meal {meal_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update meal"
        )


@router.delete("/{meal_id}")
async def delete_meal(
    meal_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a meal"""
    try:
        # Get meal
        result = await db.execute(
            select(Meal).where(Meal.id == meal_id, Meal.user_id == current_user.id)
        )
        meal = result.scalar_one_or_none()
        
        if not meal:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Meal not found"
            )
        
        # Delete meal (items will be deleted due to cascade)
        await db.delete(meal)
        await db.commit()
        
        logger.info(f"✅ Meal {meal_id} deleted for user {current_user.email}")
        
        return {"message": "Meal deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete meal {meal_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete meal"
        )


@router.get("/daily-totals/{date}")
async def get_daily_totals(
    date: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get nutrition totals for a specific date"""
    try:
        # Parse date
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
            next_date = target_date + timedelta(days=1)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Use YYYY-MM-DD"
            )
        
        # Get meals for the date
        result = await db.execute(
            select(Meal).where(
                Meal.user_id == current_user.id,
                Meal.created_at >= target_date,
                Meal.created_at < next_date
            )
        )
        meals = result.scalars().all()
        
        # Calculate totals
        total_calories = sum(meal.total_calories or 0 for meal in meals)
        total_protein = sum(meal.total_protein or 0 for meal in meals)
        total_carbs = sum(meal.total_carbs or 0 for meal in meals)
        total_fat = sum(meal.total_fat or 0 for meal in meals)
        
        logger.info(f"✅ Daily totals retrieved for {date}: {len(meals)} meals")
        
        return {
            "date": date,
            "meals_count": len(meals),
            "totals": {
                "calories": round(total_calories, 1),
                "protein_g": round(total_protein, 1),
                "carbs_g": round(total_carbs, 1),
                "fat_g": round(total_fat, 1)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get daily totals for {date}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve daily totals"
        )


async def _get_meal_with_items(db: AsyncSession, meal_id: str) -> MealResponse:
    """Helper function to get meal with items"""
    # Get meal
    result = await db.execute(select(Meal).where(Meal.id == meal_id))
    meal = result.scalar_one()
    
    # Get meal items
    result = await db.execute(select(MealItem).where(MealItem.meal_id == meal_id))
    items = result.scalars().all()
    
    # Convert to response schemas
    from app.schemas import MealItemResponse
    item_responses = [
        MealItemResponse(
            id=item.id,
            meal_id=item.meal_id,
            label=item.label,
            confidence=item.confidence,
            grams_estimated=item.grams_estimated,
            grams_actual=item.grams_actual,
            calories=item.calories,
            protein_g=item.protein_g,
            carbs_g=item.carbs_g,
            fat_g=item.fat_g,
            mask_polygon=item.mask_polygon,
            created_at=item.created_at
        )
        for item in items
    ]
    
    return MealResponse(
        id=meal.id,
        user_id=meal.user_id,
        name=meal.name,
        image_path=meal.image_path,
        plate_diameter_cm=meal.plate_diameter_cm,
        total_calories=meal.total_calories,
        total_protein=meal.total_protein,
        total_carbs=meal.total_carbs,
        total_fat=meal.total_fat,
        notes=meal.notes,
        created_at=meal.created_at,
        updated_at=meal.updated_at,
        items=item_responses
    )
