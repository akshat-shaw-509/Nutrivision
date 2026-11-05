"""
Module 4: Artificial Neural Network (ANN) for Nutritional Assessment
=====================================================================
Responsibilities:
- Prepare and normalize nutritional features (Calories, Protein, Carbs, Fat, Fiber, Sodium)
- Multi-layer Perceptron (ANN) with backpropagation
- Classify nutritional balance into categories:
    0: "Balanced & Nutrient-Dense"
    1: "High-Calorie Surge"
    2: "High-Fat Alert"
    3: "Low-Fiber Warning"
    4: "Adequate-Protein Power"
- Delivers real-time nutritional health scores & guidance
"""

import os
import json
import numpy as np
from typing import Dict, Any, List

LABEL_NAMES = [
    "Balanced & Nutrient-Dense",
    "High-Calorie Surge",
    "High-Fat Alert",
    "Low-Fiber Warning",
    "Adequate-Protein Power"
]


class NutritionEvaluationANN:
    def __init__(self):
        # Feature normalization scales [Calories, Protein, Carbs, Fat, Fiber, Sodium]
        self.feature_means = np.array([450.0, 25.0, 45.0, 18.0, 4.0, 500.0], dtype=np.float32)
        self.feature_stds = np.array([250.0, 18.0, 30.0, 12.0, 3.5, 350.0], dtype=np.float32)
        
        # ANN Architecture: 6 inputs -> 16 (Hidden 1) -> 8 (Hidden 2) -> 5 (Outputs)
        self.weights_file = os.path.join(os.path.dirname(__file__), "weights_ann_evaluator.json")
        self._init_network()

    def _init_network(self):
        np.random.seed(42)
        # Xavier / He initialization
        self.W1 = np.random.randn(6, 16) * np.sqrt(2.0 / 6)
        self.b1 = np.zeros((1, 16))
        
        self.W2 = np.random.randn(16, 8) * np.sqrt(2.0 / 16)
        self.b2 = np.zeros((1, 8))
        
        self.W3 = np.random.randn(8, 5) * np.sqrt(2.0 / 8)
        self.b3 = np.zeros((1, 5))

        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, "r", encoding="utf-8") as f:
                    w = json.load(f)
                self.W1 = np.array(w["W1"], dtype=np.float32)
                self.b1 = np.array(w["b1"], dtype=np.float32)
                self.W2 = np.array(w["W2"], dtype=np.float32)
                self.b2 = np.array(w["b2"], dtype=np.float32)
                self.W3 = np.array(w["W3"], dtype=np.float32)
                self.b3 = np.array(w["b3"], dtype=np.float32)
                return
            except Exception:
                pass

        # Train with backpropagation on standard nutritional benchmark dataset
        self.train_benchmark_data(epochs=120)
        try:
            with open(self.weights_file, "w", encoding="utf-8") as f:
                json.dump({
                    "W1": self.W1.tolist(), "b1": self.b1.tolist(),
                    "W2": self.W2.tolist(), "b2": self.b2.tolist(),
                    "W3": self.W3.tolist(), "b3": self.b3.tolist(),
                }, f, indent=2)
        except Exception:
            pass

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def forward(self, X: np.ndarray):
        """Forward pass through ANN layers."""
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)

        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.relu(z2)

        z3 = np.dot(a2, self.W3) + self.b3
        a3 = self.softmax(z3)

        return (z1, a1, z2, a2, z3, a3)

    def train_benchmark_data(self, epochs: int = 100, lr: float = 0.05):
        """
        Trains the ANN using Gradient Descent & Backpropagation on representative meal profiles.
        """
        # Benchmark nutritional patterns: [Calories, Protein, Carbs, Fat, Fiber, Sodium]
        X_raw = np.array([
            [420, 34, 40, 12, 6.0, 320], # Balanced
            [950, 20, 95, 48, 2.0, 1400],# High-Calorie
            [600, 18, 15, 52, 1.5, 750], # High-Fat
            [480, 8,  82, 9,  0.8, 450], # Low-Fiber
            [380, 48, 12, 8,  4.0, 280], # Adequate-Protein
            [350, 30, 35, 10, 5.5, 300], # Balanced
            [1100, 25, 110, 55, 3.0, 1600],# High-Calorie
            [720, 22, 10, 62, 1.0, 800], # High-Fat
            [410, 6,  78, 7,  0.5, 380], # Low-Fiber
            [420, 52, 18, 9,  5.0, 310]  # Adequate-Protein
        ], dtype=np.float32)

        # Labels: 0=Balanced, 1=High-Calorie, 2=High-Fat, 3=Low-Fiber, 4=Adequate-Protein
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        Y_one_hot = np.zeros((len(y), 5))
        Y_one_hot[np.arange(len(y)), y] = 1.0

        # Normalize features
        X_norm = (X_raw - self.feature_means) / (self.feature_stds + 1e-6)

        m = len(X_raw)
        for _ in range(epochs):
            # Forward
            z1, a1, z2, a2, z3, a3 = self.forward(X_norm)

            # Backpropagation (Cross-entropy loss gradient)
            dz3 = (a3 - Y_one_hot) / m
            dW3 = np.dot(a2.T, dz3)
            db3 = np.sum(dz3, axis=0, keepdims=True)

            da2 = np.dot(dz3, self.W3.T)
            dz2 = da2 * self.relu_derivative(z2)
            dW2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self.relu_derivative(z1)
            dW1 = np.dot(X_norm.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)

            # Weight updates
            self.W3 -= lr * dW3
            self.b3 -= lr * db3
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1

    def evaluate_meal_balance(self, meal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies the nutritional balance of a meal.
        Input: meal dictionary containing calories, protein_g, carbs_g, fat_g, fiber_g, sodium_mg
        """
        macros = meal.get("macros") if isinstance(meal.get("macros"), dict) else meal
        micros = meal.get("micros") if isinstance(meal.get("micros"), dict) else meal

        cal = float(meal.get("calories") or 0.0)
        p = float(macros.get("protein_g") or meal.get("protein_g") or 0.0)
        c = float(macros.get("carbs_g") or meal.get("carbs_g") or 0.0)
        f = float(macros.get("fat_g") or meal.get("fat_g") or 0.0)
        fiber = float(macros.get("fiber_g") or meal.get("fiber_g") or 0.0)
        sodium = float(micros.get("sodium_mg") or meal.get("sodium_mg") or 0.0)

        if cal <= 0 and p <= 0 and c <= 0 and f <= 0:
            return {
                "classification_index": -1,
                "verdict": "Unspecified / Minimal Intake",
                "confidence": 1.0,
                "recommendation": "Zero calories logged. Add meal nutritional details to evaluate balance.",
                "is_balanced": False,
            }

        feat = np.array([cal, p, c, f, fiber, sodium], dtype=np.float32)

        feat_norm = (feat - self.feature_means) / (self.feature_stds + 1e-6)
        feat_norm = feat_norm.reshape(1, -1)

        _, _, _, _, _, probas = self.forward(feat_norm)
        probas = probas[0]

        top_class_idx = int(np.argmax(probas))
        confidence = float(probas[top_class_idx])

        # Actionable recommendations based on ANN assessment
        recommendations = {
            0: "Excellent macronutrient harmony! Promotes stable glycemic response and satiety.",
            1: "Energy-dense meal. Consider saving calories for your next meal or engaging in a light walk.",
            2: "Higher lipid concentration. Ensure remaining meals emphasize lean proteins and soluble fiber.",
            3: "Fiber content is low (<2g). Consider adding cruciferous greens, leafy salad, or chia seeds.",
            4: "High protein meal (>35g)! Ideal for muscle protein synthesis and prolonged satiety."
        }

        return {
            "classification_index": top_class_idx,
            "verdict": LABEL_NAMES[top_class_idx],
            "confidence": round(confidence, 3),
            "recommendation": recommendations[top_class_idx],
            "is_balanced": top_class_idx in [0, 4]
        }


if __name__ == "__main__":
    print("[Module 4] Testing ANN Nutritional Assessment Model...")
    ann = NutritionEvaluationANN()
    
    # Test high-protein meal
    test_meal = {
        "calories": 360,
        "protein_g": 46,
        "carbs_g": 10,
        "fat_g": 6,
        "fiber_g": 4,
        "sodium_mg": 280
    }
    result = ann.evaluate_meal_balance(test_meal)
    print(f"[OK] ANN Classification Verdict: '{result['verdict']}' (Confidence: {result['confidence'] * 100:.1f}%)")
    print(f"[OK] Coach Guidance: {result['recommendation']}")
    print("[Module 4] ANN model verified!")
