"""Production FastAPI entrypoint and legacy educational pipeline helpers.

The authenticated API is created in backend.api. Legacy CNN/ANN/RNN helpers below
are retained for educational tests and are not exposed as production predictions.
"""

import os
import sys
import json
import base64
import io
from datetime import date
from typing import Dict, Any, Optional
from PIL import Image

# Path setups
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from data.nutrition_data_store import NutritionDataStore

data_store = NutritionDataStore()

# Prepopulate realistic demo users and foods on fresh startup
try:
    with data_store._get_connection() as _conn:
        if _conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            data_store.seed_initial_data()
except Exception:
    pass

# -------------------------------------------------------------
# CORE PIPELINE CONTROLLER
# -------------------------------------------------------------
def process_food_image_pipeline(image_bytes: bytes, serving_size_g: Optional[float] = None, user_id: str = "user_demo_01") -> Dict[str, Any]:
    """
    End-to-End Pipeline:
    Image Bytes -> Preprocessing (M1) -> CNN Classification (M2) ->
    Nutrition Mapping (M3) -> ANN Nutritional Balance (M4) -> GenAI Explanation (M6)
    """
    # Educational simulation only; the production API does not expose this pipeline.
    from ml.preprocessing import FoodImagePreprocessor
    from ml.cnn_classifier import FoodClassifierCNN
    from ml.nutrition_mapper import NutritionMapper
    from ml.ann_evaluator import NutritionEvaluationANN
    from ml.genai_coach import GenAINutritionCoach
    preprocessor = FoodImagePreprocessor()
    cnn_model = FoodClassifierCNN()
    nutrition_mapper = NutritionMapper()
    ann_evaluator = NutritionEvaluationANN()
    genai_coach = GenAINutritionCoach()

    # 1. Load image
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 2. Preprocess
    tensor = preprocessor.preprocess_image(pil_img)

    # 3. CNN Classification
    predictions = cnn_model.predict_top_k(tensor, top_k=3)
    top_pred = predictions[0]

    # 4. Nutrition Mapping
    nutrition = nutrition_mapper.map_food_to_nutrition(top_pred["class_id"], serving_amount=serving_size_g)

    # 5. ANN Nutritional Assessment
    ann_result = ann_evaluator.evaluate_meal_balance(nutrition)

    # 6. GenAI Explanation
    user = data_store.get_user(user_id) or {"primary_goal": "Fat Loss"}
    goal_str = user.get("primary_goal") or "Fat Loss"
    explanation = genai_coach.generate_meal_explanation(
        food_name=nutrition["food_name"],
        calories=nutrition["calories"],
        macros=nutrition["macros"],
        ann_verdict=ann_result["verdict"],
        user_goal=goal_str.replace("_", " ").title()
    )

    return {
        "status": "success",
        "top_prediction": top_pred,
        "all_predictions": predictions,
        "nutrition": nutrition,
        "ann_assessment": ann_result,
        "genai_explanation": explanation
    }


def analyze_eating_pattern_pipeline(user_id: str) -> Dict[str, Any]:
    """Return recorded daily totals; no untrained forecast is presented."""
    from datetime import timedelta
    history = []
    for offset in range(6, -1, -1):
        day = (date.today() - timedelta(days=offset)).isoformat()
        summary = data_store.get_daily_summary(user_id, day)
        if summary["meals_list"]:
            history.append({"date": day, "calories": summary["macros"]["calories"]["consumed"]})
    return {"status": "descriptive_only", "daily_calories": history}


from backend.api import create_app

app = create_app(data_store)
HAS_FASTAPI = True

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
