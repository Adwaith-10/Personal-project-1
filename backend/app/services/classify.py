import onnxruntime as ort
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple
import json
import logging
from app.config import settings
import os

logger = logging.getLogger(__name__)


class ClassificationService:
    def __init__(self):
        self.session = None
        self.class_names = []
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model and class index"""
        try:
            # Load class index
            if os.path.exists(settings.CLASS_INDEX_PATH):
                with open(settings.CLASS_INDEX_PATH, 'r') as f:
                    class_data = json.load(f)
                    self.class_names = class_data.get("classes", [])
                logger.info(f"✅ Loaded {len(self.class_names)} classes")
            else:
                # Fallback to Food-101 classes
                self.class_names = self._get_food101_classes()
                logger.info("✅ Using Food-101 classes")
            
            # Load ONNX model
            if os.path.exists(settings.CLASSIFIER_MODEL_PATH):
                self.session = ort.InferenceSession(
                    settings.CLASSIFIER_MODEL_PATH,
                    providers=['CPUExecutionProvider']
                )
                
                # Get input/output details
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                self.input_shape = self.session.get_inputs()[0].shape
                
                logger.info(f"✅ ONNX model loaded from {settings.CLASSIFIER_MODEL_PATH}")
                logger.info(f"Input shape: {self.input_shape}")
            else:
                logger.warning("❌ ONNX model not found, using mock classification")
                self.session = None
                
        except Exception as e:
            logger.error(f"❌ Failed to load classification model: {e}")
            # Fallback to mock classification
            self.session = None
            self.class_names = self._get_food101_classes()
    
    def _get_food101_classes(self) -> List[str]:
        """Get Food-101 class names"""
        return [
            "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
            "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
            "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
            "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
            "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
            "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
            "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
            "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
            "french_fries", "french_onion_soup", "french_toast", "fried_calamari",
            "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad",
            "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger",
            "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream",
            "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese",
            "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings",
            "oysters", "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
            "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich",
            "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi",
            "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese",
            "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
            "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"
        ]
    
    async def classify_food(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        Classify a food item in the given image region
        
        Args:
            image: Full image as numpy array
            mask: Binary mask for the food item
            
        Returns:
            Classification results with top-k predictions
        """
        try:
            if self.session is None:
                return self._mock_classification()
            
            # Extract masked region
            masked_image = self._extract_masked_region(image, mask)
            
            # Preprocess image
            preprocessed = self._preprocess_image(masked_image)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: preprocessed})
            
            # Process results
            predictions = self._process_predictions(outputs[0][0])
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Classification failed: {e}")
            return self._mock_classification()
    
    def _extract_masked_region(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract the masked region from the image"""
        # Apply mask to image
        masked = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
        
        # Find bounding box of mask
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return image
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Extract region with some padding
        padding = 10
        y_min = max(0, y_min - padding)
        y_max = min(image.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(image.shape[1], x_max + padding)
        
        return masked[y_min:y_max, x_min:x_max]
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for the model"""
        try:
            # Resize to model input size
            if len(self.input_shape) == 4:
                target_height, target_width = self.input_shape[2], self.input_shape[3]
            else:
                target_height, target_width = 224, 224
            
            resized = cv2.resize(image, (target_width, target_height))
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            normalized = rgb.astype(np.float32) / 255.0
            
            # Add batch dimension
            batched = np.expand_dims(normalized, axis=0)
            
            return batched
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            # Return dummy input
            return np.zeros((1, 224, 224, 3), dtype=np.float32)
    
    def _process_predictions(self, logits: np.ndarray) -> Dict[str, Any]:
        """Process model outputs to get top-k predictions"""
        try:
            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            
            # Get top-5 predictions
            top_indices = np.argsort(probabilities)[::-1][:5]
            
            candidates = []
            for idx in top_indices:
                if idx < len(self.class_names):
                    candidates.append({
                        "label": self.class_names[idx],
                        "confidence": float(probabilities[idx])
                    })
            
            # Get top prediction
            top_label = candidates[0]["label"] if candidates else "unknown_food"
            
            return {
                "label": top_label,
                "candidates": [c["label"] for c in candidates],
                "confidence": candidates[0]["confidence"] if candidates else 0.0
            }
            
        except Exception as e:
            logger.error(f"Prediction processing failed: {e}")
            return {
                "label": "unknown_food",
                "candidates": ["unknown_food"],
                "confidence": 0.0
            }
    
    def _mock_classification(self) -> Dict[str, Any]:
        """Mock classification when model is not available"""
        import random
        
        mock_foods = ["grilled_chicken", "rice", "broccoli", "salmon", "pasta"]
        selected = random.choice(mock_foods)
        
        return {
            "label": selected,
            "candidates": [selected, "unknown_food"],
            "confidence": 0.8
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.session:
            del self.session
            self.session = None


# Global instance
classification_service = ClassificationService()
