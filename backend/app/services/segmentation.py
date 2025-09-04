import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any
import logging
from app.config import settings
import os

logger = logging.getLogger(__name__)


class SegmentationService:
    def __init__(self):
        self.model = None
        self.model_path = settings.YOLO_MODEL_PATH
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 segmentation model"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                logger.info(f"✅ YOLOv8 model loaded from {self.model_path}")
            else:
                # Load pretrained model if custom model doesn't exist
                self.model = YOLO("yolov8n-seg.pt")
                logger.info("✅ YOLOv8n-seg pretrained model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLOv8 model: {e}")
            raise e
    
    async def detect_foods(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect and segment foods in the image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detected food items with masks and bounding boxes
        """
        try:
            # Run inference
            results = self.model(image, verbose=False)
            
            detections = []
            
            for result in results:
                if result.masks is not None:
                    # Get masks, boxes, and confidences
                    masks = result.masks.data.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    for i, (mask, box, conf, class_id) in enumerate(
                        zip(masks, boxes, confidences, class_ids)
                    ):
                        if conf > 0.3:  # Confidence threshold
                            # Convert mask to polygon
                            polygon = self._mask_to_polygon(mask)
                            
                            if len(polygon) > 0:
                                detection = {
                                    "mask_id": f"m{i}",
                                    "mask": mask,
                                    "box": box.tolist(),
                                    "confidence": float(conf),
                                    "class_id": int(class_id),
                                    "polygon": polygon,
                                    "area_pixels": int(np.sum(mask))
                                }
                                detections.append(detection)
            
            # Sort by confidence
            detections.sort(key=lambda x: x["confidence"], reverse=True)
            
            logger.info(f"Detected {len(detections)} food items")
            return detections
            
        except Exception as e:
            logger.error(f"❌ Food detection failed: {e}")
            raise e
    
    def _mask_to_polygon(self, mask: np.ndarray) -> List[List[float]]:
        """
        Convert binary mask to polygon coordinates
        
        Args:
            mask: Binary mask (0 or 1)
            
        Returns:
            List of [x, y] coordinates forming the polygon
        """
        try:
            # Find contours in the mask
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return []
            
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Simplify the contour
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Convert to list of [x, y] coordinates
            polygon = []
            for point in simplified:
                x, y = point[0]
                polygon.append([float(x), float(y)])
            
            return polygon
            
        except Exception as e:
            logger.warning(f"Failed to convert mask to polygon: {e}")
            return []
    
    def get_mask_area_cm2(self, mask: np.ndarray, plate_diameter_cm: float, 
                          image_width: int, image_height: int) -> float:
        """
        Calculate mask area in square centimeters
        
        Args:
            mask: Binary mask
            plate_diameter_cm: Plate diameter in centimeters
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Area in square centimeters
        """
        try:
            # Estimate plate diameter in pixels (assume plate is roughly circular)
            # This is a simple heuristic - in production you'd want more sophisticated plate detection
            plate_diameter_pixels = min(image_width, image_height) * 0.8
            
            # Calculate pixels per cm
            pixels_per_cm = plate_diameter_pixels / plate_diameter_cm
            
            # Calculate mask area in pixels
            mask_area_pixels = np.sum(mask)
            
            # Convert to square centimeters
            mask_area_cm2 = mask_area_pixels / (pixels_per_cm ** 2)
            
            return mask_area_cm2
            
        except Exception as e:
            logger.warning(f"Failed to calculate mask area: {e}")
            return 0.0
    
    def cleanup(self):
        """Clean up resources"""
        if self.model:
            del self.model
            self.model = None


# Global instance
segmentation_service = SegmentationService()
