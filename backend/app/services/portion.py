import numpy as np
import yaml
from typing import Dict, Any, List, Tuple
import logging
from app.config import settings
import os

logger = logging.getLogger(__name__)


class PortionService:
    def __init__(self):
        self.density_priors = {}
        self.shape_priors = {}
        self._load_density_priors()
    
    def _load_density_priors(self):
        """Load density and shape priors from YAML file"""
        try:
            if os.path.exists(settings.DENSITY_PRIORS_PATH):
                with open(settings.DENSITY_PRIORS_PATH, 'r') as f:
                    data = yaml.safe_load(f)
                    self.density_priors = data.get("densities", {})
                    self.shape_priors = data.get("shapes", {})
                logger.info(f"✅ Loaded density priors for {len(self.density_priors)} food categories")
            else:
                # Default density priors
                self.density_priors = {
                    "grilled_chicken": 1.65,
                    "rice": 1.20,
                    "broccoli": 0.91,
                    "salmon": 1.02,
                    "pasta": 1.20,
                    "beef": 1.20,
                    "fish": 1.02,
                    "vegetables": 0.91,
                    "fruits": 0.95,
                    "bread": 0.30,
                    "cheese": 1.10,
                    "eggs": 1.03,
                    "milk": 1.03,
                    "yogurt": 1.05,
                    "soup": 1.00,
                    "sauce": 1.10,
                    "dessert": 1.20,
                    "unknown_food": 1.00
                }
                
                # Default shape priors
                self.shape_priors = {
                    "grilled_chicken": "mound",
                    "rice": "mound",
                    "broccoli": "mound",
                    "salmon": "slab",
                    "pasta": "mound",
                    "beef": "slab",
                    "fish": "slab",
                    "vegetables": "mound",
                    "fruits": "sphere",
                    "bread": "slab",
                    "cheese": "slab",
                    "eggs": "sphere",
                    "milk": "liquid",
                    "yogurt": "liquid",
                    "soup": "liquid",
                    "sauce": "liquid",
                    "dessert": "mound",
                    "unknown_food": "mound"
                }
                
                logger.info("✅ Using default density and shape priors")
                
        except Exception as e:
            logger.error(f"❌ Failed to load density priors: {e}")
            # Use defaults
            self._load_density_priors()
    
    def estimate_volume(self, area_cm2: float, shape: str, height_cm: float = None) -> float:
        """
        Estimate volume from area and shape
        
        Args:
            area_cm2: Area in square centimeters
            shape: Shape type (mound, cylinder, slab, sphere, liquid)
            height_cm: Height for slab/cylinder (optional)
            
        Returns:
            Volume in cubic centimeters
        """
        try:
            if shape == "mound":
                # Assume conical mound
                # V = (1/3) * π * r² * h, where h ≈ r for typical food mounds
                radius_cm = np.sqrt(area_cm2 / np.pi)
                height = radius_cm * 0.8  # Typical height for food mounds
                volume = (1/3) * np.pi * radius_cm**2 * height
                
            elif shape == "cylinder":
                # V = π * r² * h
                radius_cm = np.sqrt(area_cm2 / np.pi)
                height = height_cm if height_cm else radius_cm * 0.5
                volume = np.pi * radius_cm**2 * height
                
            elif shape == "slab":
                # V = area * height
                height = height_cm if height_cm else 1.0  # Default 1cm height
                volume = area_cm2 * height
                
            elif shape == "sphere":
                # V = (4/3) * π * r³, where r = sqrt(area/π)
                radius_cm = np.sqrt(area_cm2 / np.pi)
                volume = (4/3) * np.pi * radius_cm**3
                
            elif shape == "liquid":
                # For liquids, assume shallow depth
                height = height_cm if height_cm else 0.5
                volume = area_cm2 * height
                
            else:
                # Default to mound
                radius_cm = np.sqrt(area_cm2 / np.pi)
                height = radius_cm * 0.8
                volume = (1/3) * np.pi * radius_cm**2 * height
            
            return max(volume, 0.1)  # Minimum volume of 0.1 cm³
            
        except Exception as e:
            logger.warning(f"Volume estimation failed: {e}")
            # Fallback: assume 1cm height
            return area_cm2 * 1.0
    
    def estimate_grams(self, area_cm2: float, food_label: str, 
                      plate_diameter_cm: float = 26.0) -> float:
        """
        Estimate food weight in grams from area and food type
        
        Args:
            area_cm2: Area in square centimeters
            food_label: Food classification label
            plate_diameter_cm: Plate diameter for scale reference
            
        Returns:
            Estimated weight in grams
        """
        try:
            # Get density and shape for this food
            density = self.density_priors.get(food_label, 1.0)
            shape = self.shape_priors.get(food_label, "mound")
            
            # Estimate volume
            volume_cm3 = self.estimate_volume(area_cm2, shape)
            
            # Convert to grams using density (g/cm³)
            grams = volume_cm3 * density
            
            # Apply reasonable bounds
            grams = max(grams, 5.0)  # Minimum 5g
            grams = min(grams, 1000.0)  # Maximum 1kg
            
            return round(grams, 1)
            
        except Exception as e:
            logger.warning(f"Gram estimation failed: {e}")
            # Fallback: assume 1g per cm²
            return round(area_cm2, 1)
    
    def calculate_area_from_mask(self, mask: np.ndarray, 
                                plate_diameter_cm: float,
                                image_width: int, 
                                image_height: int) -> float:
        """
        Calculate area in square centimeters from binary mask
        
        Args:
            mask: Binary mask (0 or 1)
            plate_diameter_cm: Plate diameter in centimeters
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Area in square centimeters
        """
        try:
            # Count mask pixels
            mask_area_pixels = np.sum(mask)
            
            # Estimate plate diameter in pixels
            # Assume plate takes up roughly 80% of the smaller image dimension
            plate_diameter_pixels = min(image_width, image_height) * 0.8
            
            # Calculate pixels per cm
            pixels_per_cm = plate_diameter_pixels / plate_diameter_cm
            
            # Convert mask area to square centimeters
            area_cm2 = mask_area_pixels / (pixels_per_cm ** 2)
            
            return area_cm2
            
        except Exception as e:
            logger.warning(f"Area calculation failed: {e}")
            return 0.0
    
    def get_food_density(self, food_label: str) -> float:
        """Get density for a specific food type"""
        return self.density_priors.get(food_label, 1.0)
    
    def get_food_shape(self, food_label: str) -> str:
        """Get shape for a specific food type"""
        return self.shape_priors.get(food_label, "mound")
    
    def update_density_prior(self, food_label: str, density: float):
        """Update density prior for a food type"""
        self.density_priors[food_label] = density
        logger.info(f"Updated density for {food_label}: {density} g/cm³")
    
    def update_shape_prior(self, food_label: str, shape: str):
        """Update shape prior for a food type"""
        valid_shapes = ["mound", "cylinder", "slab", "sphere", "liquid"]
        if shape in valid_shapes:
            self.shape_priors[food_label] = shape
            logger.info(f"Updated shape for {food_label}: {shape}")
        else:
            logger.warning(f"Invalid shape: {shape}. Must be one of {valid_shapes}")


# Global instance
portion_service = PortionService()
