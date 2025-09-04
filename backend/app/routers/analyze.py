from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import get_db
from app.models import User, AnalysisLog
from app.schemas import AnalysisRequest, AnalysisResponse, AnalysisItem
from app.services.segmentation import segmentation_service
from app.services.classify import classification_service
from app.services.portion import portion_service
from app.services.nutrition import nutrition_service
from app.services.storage import storage_service
from app.deps import get_current_user
import cv2
import numpy as np
import io
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_food_image(
    image: UploadFile = File(...),
    plate_diameter_cm: Optional[float] = Form(None),
    barcode: Optional[str] = Form(None),
    overrides: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze a food image to detect, classify, and estimate nutrition
    
    Args:
        image: Food image file
        plate_diameter_cm: Plate diameter in centimeters (optional)
        barcode: Barcode for nutrition lookup (optional)
        overrides: JSON string with manual overrides (optional)
        current_user: Authenticated user
        db: Database session
    """
    start_time = time.time()
    
    try:
        # Validate image file
        if not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
        
        # Read and process image
        image_content = await image.read()
        image_array = await _process_image(image_content)
        
        # Set default plate diameter if not provided
        if plate_diameter_cm is None:
            plate_diameter_cm = 26.0  # Standard dinner plate
        
        # Parse overrides if provided
        manual_overrides = {}
        if overrides:
            try:
                import json
                manual_overrides = json.loads(overrides)
            except json.JSONDecodeError:
                logger.warning("Invalid overrides JSON format")
        
        # Detect and segment foods
        detections = await segmentation_service.detect_foods(image_array)
        
        if not detections:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No food items detected in the image"
            )
        
        # Process each detection
        analysis_items = []
        total_calories = 0.0
        total_protein = 0.0
        total_carbs = 0.0
        total_fat = 0.0
        
        for detection in detections:
            try:
                # Classify food
                classification = await classification_service.classify_food(
                    image_array, detection["mask"]
                )
                
                # Get food label (use override if provided)
                mask_id = detection["mask_id"]
                food_label = classification["label"]
                
                if mask_id in manual_overrides and "label" in manual_overrides[mask_id]:
                    food_label = manual_overrides[mask_id]["label"]
                
                # Calculate area and estimate grams
                area_cm2 = portion_service.calculate_area_from_mask(
                    detection["mask"],
                    plate_diameter_cm,
                    image_array.shape[1],  # width
                    image_array.shape[0]   # height
                )
                
                # Estimate grams (use override if provided)
                grams_est = portion_service.estimate_grams(area_cm2, food_label, plate_diameter_cm)
                
                if mask_id in manual_overrides and "grams" in manual_overrides[mask_id]:
                    grams_est = float(manual_overrides[mask_id]["grams"])
                
                # Get nutrition information
                nutrition = await nutrition_service.get_nutrition_info(food_label, barcode)
                
                # Calculate nutrition for estimated grams
                nutrition_for_grams = nutrition_service.calculate_nutrition_for_grams(
                    nutrition, grams_est
                )
                
                # Create analysis item
                analysis_item = AnalysisItem(
                    mask_id=mask_id,
                    label=food_label,
                    candidates=classification["candidates"],
                    grams_est=grams_est,
                    calories=nutrition_for_grams.get("calories", 0.0),
                    protein_g=nutrition_for_grams.get("protein", 0.0),
                    carb_g=nutrition_for_grams.get("carbs", 0.0),
                    fat_g=nutrition_for_grams.get("fat", 0.0),
                    polygon=detection["polygon"]
                )
                
                analysis_items.append(analysis_item)
                
                # Accumulate totals
                total_calories += analysis_item.calories
                total_protein += analysis_item.protein_g
                total_carbs += analysis_item.carb_g
                total_fat += analysis_item.fat_g
                
            except Exception as e:
                logger.warning(f"Failed to process detection {detection['mask_id']}: {e}")
                continue
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Log analysis
        await _log_analysis(
            db, current_user.id, len(analysis_items), processing_time_ms, True
        )
        
        # Create response
        totals = {
            "kcal": round(total_calories, 1),
            "protein_g": round(total_protein, 1),
            "carb_g": round(total_carbs, 1),
            "fat_g": round(total_fat, 1)
        }
        
        logger.info(f"✅ Analysis completed: {len(analysis_items)} items, {processing_time_ms}ms")
        
        return AnalysisResponse(
            items=analysis_items,
            totals=totals,
            processing_time_ms=processing_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Log failed analysis
        processing_time_ms = int((time.time() - start_time) * 1000)
        await _log_analysis(
            db, current_user.id, 0, processing_time_ms, False, str(e)
        )
        
        logger.error(f"❌ Food analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed"
        )


async def _process_image(image_content: bytes) -> np.ndarray:
    """Process uploaded image content to numpy array"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Resize if too large (for performance)
        max_dimension = 1024
        height, width = image.shape[:2]
        
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        return image
        
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        raise ValueError("Invalid image format")


async def _log_analysis(
    db: AsyncSession,
    user_id: str,
    items_detected: int,
    processing_time_ms: int,
    success: bool,
    error_message: str = None
):
    """Log analysis attempt to database"""
    try:
        log_entry = AnalysisLog(
            user_id=user_id,
            items_detected=items_detected,
            processing_time_ms=processing_time_ms,
            success=success,
            error_message=error_message
        )
        
        db.add(log_entry)
        await db.commit()
        
    except Exception as e:
        logger.warning(f"Failed to log analysis: {e}")


@router.get("/health")
async def analysis_health():
    """Check analysis service health"""
    try:
        # Check if ML models are loaded
        models_status = {
            "segmentation": segmentation_service.model is not None,
            "classification": classification_service.session is not None,
            "portion": len(portion_service.density_priors) > 0,
            "nutrition": True  # Always available (has fallbacks)
        }
        
        return {
            "status": "healthy",
            "models": models_status,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }
