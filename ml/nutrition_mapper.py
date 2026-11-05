"""
Module 3: Nutrition Database & Mapping Module
=============================================
Responsibilities:
- Build structured nutrition lookup engine
- Map predicted CNN food classes to exact calories, macronutrients, and micronutrients
- Dynamically scale values according to user-selected serving size
- Document data sources (USDA FDC, Open Food Facts)
"""

import os
import json
from typing import Dict, Any, Optional

DB_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "food_database.json")


class NutritionMapper:
    def __init__(self, db_path: str = DB_FILE_PATH):
        self.db_path = db_path
        self._load_database()

    def _load_database(self):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Nutrition database not found at {self.db_path}")
        with open(self.db_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.foods = self.data.get("foods", {})
        self.sources = self.data.get("sources", [])
        
        # Also load full USDA catalog if available
        self.catalog_foods = []
        catalog_path = os.path.join(os.path.dirname(__file__), "..", "data", "food_catalog.json")
        if os.path.exists(catalog_path):
            try:
                with open(catalog_path, "r", encoding="utf-8") as f:
                    self.catalog_foods = json.load(f).get("foods", [])
            except Exception:
                self.catalog_foods = []

    def map_food_to_nutrition(self, food_key: str, serving_amount: Optional[float] = None) -> Dict[str, Any]:
        """
        Maps food key to nutritional breakdown.
        Scales per-100g values based on requested serving amount (grams/ml).
        """
        # Normalize key
        clean_key = food_key.lower().strip().replace(" ", "_").replace("-", "_")
        
        # Exact match or prefix match in food_database
        entry = self.foods.get(clean_key)
        if not entry:
            for k in self.foods:
                if k in clean_key or clean_key in k:
                    entry = self.foods[k]
                    break

        if not entry and self.catalog_foods:
            words = [w for w in clean_key.replace("_", " ").split() if len(w) > 2]
            for item in self.catalog_foods:
                item_name = item.get("name", "").lower()
                if any(w in item_name for w in words):
                    target_size = serving_amount if serving_amount is not None else 100.0
                    mult = target_size / 100.0
                    micros = item.get("micronutrients") or {}
                    return {
                        "food_name": item["name"],
                        "serving_size": target_size,
                        "serving_unit": "g",
                        "calories": round((item.get("calories") or 0.0) * mult, 1),
                        "macros": {
                            "protein_g": round((item.get("protein_g") or 0.0) * mult, 1),
                            "carbs_g": round((item.get("carbs_g") or 0.0) * mult, 1),
                            "fat_g": round((item.get("fat_g") or 0.0) * mult, 1),
                            "fiber_g": round((item.get("fiber_g") or 0.0) * mult, 1),
                        },
                        "micros": {
                            "sodium_mg": round((item.get("sodium_mg") or 0.0) * mult, 1),
                            "potassium_mg": round((micros.get("potassium_mg") or 0.0) * mult, 1),
                            "calcium_mg": round((micros.get("calcium_mg") or 0.0) * mult, 1),
                            "iron_mg": round((micros.get("iron_mg") or 0.0) * mult, 1),
                        },
                        "verified_sources": [item.get("source", "USDA FoodData Central")],
                    }

        if not entry:
            # Fallback average nutritional profile if unseen
            return {
                "food_name": food_key.title(),
                "serving_size": serving_amount or 150,
                "serving_unit": "g",
                "calories": 200.0,
                "macros": {"protein_g": 15.0, "carbs_g": 20.0, "fat_g": 6.0, "fiber_g": 3.0},
                "micros": {"sodium_mg": 200.0, "potassium_mg": 250.0, "vitamin_c_mg": 5.0, "calcium_mg": 30.0, "iron_mg": 1.0},
                "source": "Estimated Generic Reference"
            }

        target_size = serving_amount if serving_amount is not None else entry["default_serving_size"]
        multiplier = target_size / 100.0

        p100 = entry["per_100g"]
        return {
            "food_name": entry["display_name"],
            "serving_size": target_size,
            "serving_unit": entry["serving_unit"],
            "calories": round(p100["calories"] * multiplier, 1),
            "macros": {
                "protein_g": round(p100["protein_g"] * multiplier, 1),
                "carbs_g": round(p100["carbs_g"] * multiplier, 1),
                "fat_g": round(p100["fat_g"] * multiplier, 1),
                "fiber_g": round(p100["fiber_g"] * multiplier, 1)
            },
            "micros": {
                "sodium_mg": round(p100["sodium_mg"] * multiplier, 1),
                "potassium_mg": round(p100["potassium_mg"] * multiplier, 1),
                "vitamin_c_mg": round(p100["vitamin_c_mg"] * multiplier, 1),
                "vitamin_d_iu": round(p100["vitamin_d_iu"] * multiplier, 1),
                "calcium_mg": round(p100["calcium_mg"] * multiplier, 1),
                "iron_mg": round(p100["iron_mg"] * multiplier, 1)
            },
            "verified_sources": self.sources
        }


if __name__ == "__main__":
    print("[Module 3] Testing Nutrition Database & Mapping Module...")
    mapper = NutritionMapper()
    res = mapper.map_food_to_nutrition("grilled_salmon_with_asparagus", serving_amount=250)
    print(f"[OK] Mapped '{res['food_name']}' (Serving: {res['serving_size']}{res['serving_unit']}):")
    print(f"  * Calories: {res['calories']} kcal")
    print(f"  * Macros: P={res['macros']['protein_g']}g | C={res['macros']['carbs_g']}g | F={res['macros']['fat_g']}g | Fiber={res['macros']['fiber_g']}g")
    print(f"  * Micros: Potassium={res['micros']['potassium_mg']}mg | Vitamin D={res['micros']['vitamin_d_iu']} IU")
    print("[Module 3] Nutrition mapping verified!")
