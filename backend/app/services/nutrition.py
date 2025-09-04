import aiohttp
import asyncio
from typing import Dict, Any, List, Optional
import logging
from app.config import settings
from app.db import get_redis
import json
import time

logger = logging.getLogger(__name__)


class NutritionService:
    def __init__(self):
        self.usda_api_key = settings.USDA_FDC_API_KEY
        self.openfoodfacts_url = settings.OPENFOODFACTS_BASE_URL
        self.cache_ttl = 3600 * 24 * 7  # 7 days
    
    async def get_nutrition_info(self, food_name: str, barcode: str = None) -> Dict[str, Any]:
        """
        Get nutrition information for a food item
        
        Args:
            food_name: Name of the food
            barcode: Optional barcode for lookup
            
        Returns:
            Nutrition information dictionary
        """
        try:
            # Try barcode lookup first if available
            if barcode:
                nutrition = await self._lookup_by_barcode(barcode)
                if nutrition:
                    return nutrition
            
            # Fallback to name search
            nutrition = await self._search_by_name(food_name)
            return nutrition
            
        except Exception as e:
            logger.error(f"❌ Nutrition lookup failed: {e}")
            return self._get_default_nutrition(food_name)
    
    async def _lookup_by_barcode(self, barcode: str) -> Optional[Dict[str, Any]]:
        """Look up nutrition by barcode using Open Food Facts"""
        try:
            # Try Open Food Facts first
            nutrition = await self._openfoodfacts_lookup(barcode)
            if nutrition:
                return nutrition
            
            # Fallback to USDA FDC if available
            if self.usda_api_key:
                nutrition = await self._usda_barcode_lookup(barcode)
                if nutrition:
                    return nutrition
            
            return None
            
        except Exception as e:
            logger.warning(f"Barcode lookup failed: {e}")
            return None
    
    async def _openfoodfacts_lookup(self, barcode: str) -> Optional[Dict[str, Any]]:
        """Look up nutrition in Open Food Facts"""
        try:
            url = f"{self.openfoodfacts_url}/api/v0/product/{barcode}.json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get("status") == 1:  # Product found
                            product = data.get("product", {})
                            
                            # Extract nutrition information
                            nutriments = product.get("nutriments", {})
                            
                            nutrition = {
                                "name": product.get("product_name", "Unknown"),
                                "barcode": barcode,
                                "source": "openfoodfacts",
                                "calories_per_100g": nutriments.get("energy-kcal_100g"),
                                "protein_per_100g": nutriments.get("proteins_100g"),
                                "carbs_per_100g": nutriments.get("carbohydrates_100g"),
                                "fat_per_100g": nutriments.get("fat_100g"),
                                "fiber_per_100g": nutriments.get("fiber_100g"),
                                "sugar_per_100g": nutriments.get("sugars_100g"),
                                "sodium_per_100g": nutriments.get("salt_100g"),
                                "brand": product.get("brands"),
                                "ingredients": product.get("ingredients_text")
                            }
                            
                            # Clean up None values
                            nutrition = {k: v for k, v in nutrition.items() if v is not None}
                            
                            return nutrition
            
            return None
            
        except Exception as e:
            logger.warning(f"Open Food Facts lookup failed: {e}")
            return None
    
    async def _usda_barcode_lookup(self, barcode: str) -> Optional[Dict[str, Any]]:
        """Look up nutrition by barcode using USDA FDC"""
        if not self.usda_api_key:
            return None
            
        try:
            url = "https://api.nal.usda.gov/fdc/v1/foods/search"
            params = {
                "api_key": self.usda_api_key,
                "query": barcode,
                "dataType": ["Foundation", "SR Legacy"],
                "pageSize": 1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get("foods"):
                            food = data["foods"][0]
                            
                            # Extract nutrition information
                            nutrition = {
                                "name": food.get("description", "Unknown"),
                                "barcode": barcode,
                                "source": "usda",
                                "fdc_id": food.get("fdcId"),
                                "calories_per_100g": self._extract_nutrient(food, "Energy"),
                                "protein_per_100g": self._extract_nutrient(food, "Protein"),
                                "carbs_per_100g": self._extract_nutrient(food, "Carbohydrate, by difference"),
                                "fat_per_100g": self._extract_nutrient(food, "Total lipid (fat)"),
                                "fiber_per_100g": self._extract_nutrient(food, "Fiber, total dietary"),
                                "sugar_per_100g": self._extract_nutrient(food, "Sugars, total including NLEA"),
                                "sodium_per_100g": self._extract_nutrient(food, "Sodium, Na")
                            }
                            
                            # Clean up None values
                            nutrition = {k: v for k, v in nutrition.items() if v is not None}
                            
                            return nutrition
            
            return None
            
        except Exception as e:
            logger.warning(f"USDA barcode lookup failed: {e}")
            return None
    
    async def _search_by_name(self, food_name: str) -> Dict[str, Any]:
        """Search for nutrition by food name"""
        try:
            # Try USDA FDC first if API key is available
            if self.usda_api_key:
                nutrition = await self._usda_name_search(food_name)
                if nutrition:
                    return nutrition
            
            # Fallback to default nutrition based on food type
            return self._get_default_nutrition(food_name)
            
        except Exception as e:
            logger.warning(f"Name search failed: {e}")
            return self._get_default_nutrition(food_name)
    
    async def _usda_name_search(self, food_name: str) -> Optional[Dict[str, Any]]:
        """Search USDA FDC by food name"""
        try:
            url = "https://api.nal.usda.gov/fdc/v1/foods/search"
            params = {
                "api_key": self.usda_api_key,
                "query": food_name,
                "dataType": ["Foundation", "SR Legacy"],
                "pageSize": 5
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get("foods"):
                            # Find the best match
                            best_match = None
                            best_score = 0
                            
                            for food in data["foods"]:
                                score = self._calculate_name_similarity(
                                    food_name.lower(), 
                                    food.get("description", "").lower()
                                )
                                
                                if score > best_score:
                                    best_score = score
                                    best_match = food
                            
                            if best_match and best_score > 0.7:  # Good match threshold
                                food = best_match
                                
                                nutrition = {
                                    "name": food.get("description", "Unknown"),
                                    "source": "usda",
                                    "fdc_id": food.get("fdcId"),
                                    "calories_per_100g": self._extract_nutrient(food, "Energy"),
                                    "protein_per_100g": self._extract_nutrient(food, "Protein"),
                                    "carbs_per_100g": self._extract_nutrient(food, "Carbohydrate, by difference"),
                                    "fat_per_100g": self._extract_nutrient(food, "Total lipid (fat)"),
                                    "fiber_per_100g": self._extract_nutrient(food, "Fiber, total dietary"),
                                    "sugar_per_100g": self._extract_nutrient(food, "Sugars, total including NLEA"),
                                    "sodium_per_100g": self._extract_nutrient(food, "Sodium, Na")
                                }
                                
                                # Clean up None values
                                nutrition = {k: v for k, v in nutrition.items() if v is not None}
                                
                                return nutrition
            
            return None
            
        except Exception as e:
            logger.warning(f"USDA name search failed: {e}")
            return None
    
    def _extract_nutrient(self, food: Dict[str, Any], nutrient_name: str) -> Optional[float]:
        """Extract nutrient value from USDA food data"""
        try:
            for nutrient in food.get("foodNutrients", []):
                if nutrient.get("nutrientName") == nutrient_name:
                    value = nutrient.get("value")
                    if value is not None:
                        return float(value)
            return None
        except Exception:
            return None
    
    def _calculate_name_similarity(self, query: str, food_name: str) -> float:
        """Calculate similarity between query and food name"""
        try:
            query_words = set(query.split())
            food_words = set(food_name.split())
            
            if not query_words or not food_words:
                return 0.0
            
            intersection = query_words.intersection(food_words)
            union = query_words.union(food_words)
            
            return len(intersection) / len(union)
            
        except Exception:
            return 0.0
    
    def _get_default_nutrition(self, food_name: str) -> Dict[str, Any]:
        """Get default nutrition values based on food type"""
        food_name_lower = food_name.lower()
        
        # Default nutrition values for common food types
        defaults = {
            "chicken": {"calories_per_100g": 165, "protein_per_100g": 31, "carbs_per_100g": 0, "fat_per_100g": 3.6},
            "rice": {"calories_per_100g": 130, "protein_per_100g": 2.7, "carbs_per_100g": 28, "fat_per_100g": 0.3},
            "broccoli": {"calories_per_100g": 34, "protein_per_100g": 2.8, "carbs_per_100g": 7, "fat_per_100g": 0.4},
            "salmon": {"calories_per_100g": 208, "protein_per_100g": 25, "carbs_per_100g": 0, "fat_per_100g": 12},
            "pasta": {"calories_per_100g": 131, "protein_per_100g": 5, "carbs_per_100g": 25, "fat_per_100g": 1.1},
            "beef": {"calories_per_100g": 250, "protein_per_100g": 26, "carbs_per_100g": 0, "fat_per_100g": 15},
            "fish": {"calories_per_100g": 100, "protein_per_100g": 20, "carbs_per_100g": 0, "fat_per_100g": 2.5},
            "vegetables": {"calories_per_100g": 25, "protein_per_100g": 2, "carbs_per_100g": 5, "fat_per_100g": 0.2},
            "fruits": {"calories_per_100g": 60, "protein_per_100g": 0.5, "carbs_per_100g": 15, "fat_per_100g": 0.2},
            "bread": {"calories_per_100g": 265, "protein_per_100g": 9, "carbs_per_100g": 49, "fat_per_100g": 3.2},
            "cheese": {"calories_per_100g": 402, "protein_per_100g": 25, "carbs_per_100g": 1.3, "fat_per_100g": 33},
            "eggs": {"calories_per_100g": 155, "protein_per_100g": 13, "carbs_per_100g": 1.1, "fat_per_100g": 11},
            "milk": {"calories_per_100g": 42, "protein_per_100g": 3.4, "carbs_per_100g": 5, "fat_per_100g": 1},
            "yogurt": {"calories_per_100g": 59, "protein_per_100g": 10, "carbs_per_100g": 3.6, "fat_per_100g": 0.4},
            "soup": {"calories_per_100g": 50, "protein_per_100g": 3, "carbs_per_100g": 8, "fat_per_100g": 1},
            "sauce": {"calories_per_100g": 80, "protein_per_100g": 2, "carbs_per_100g": 12, "fat_per_100g": 3},
            "dessert": {"calories_per_100g": 300, "protein_per_100g": 5, "carbs_per_100g": 45, "fat_per_100g": 12}
        }
        
        # Find best matching default
        best_match = "unknown_food"
        best_score = 0
        
        for food_type in defaults.keys():
            if food_type in food_name_lower:
                score = len(food_type) / len(food_name_lower)
                if score > best_score:
                    best_score = score
                    best_match = food_type
        
        default_nutrition = defaults.get(best_match, {
            "calories_per_100g": 100,
            "protein_per_100g": 5,
            "carbs_per_100g": 15,
            "fat_per_100g": 2
        })
        
        return {
            "name": food_name,
            "source": "default",
            **default_nutrition
        }
    
    def calculate_nutrition_for_grams(self, nutrition_per_100g: Dict[str, Any], grams: float) -> Dict[str, float]:
        """Calculate nutrition for a specific amount in grams"""
        try:
            factor = grams / 100.0
            
            result = {}
            for nutrient, value in nutrition_per_100g.items():
                if isinstance(value, (int, float)) and "per_100g" in nutrient:
                    # Convert per 100g to actual amount
                    base_nutrient = nutrient.replace("_per_100g", "")
                    result[f"{base_nutrient}"] = round(value * factor, 1)
            
            return result
            
        except Exception as e:
            logger.warning(f"Nutrition calculation failed: {e}")
            return {}


# Global instance
nutrition_service = NutritionService()
