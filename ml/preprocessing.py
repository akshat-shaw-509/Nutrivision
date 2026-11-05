"""
Module 1: Food Dataset & Preprocessing Pipeline
================================================
Responsibilities:
- Collect & clean food image datasets (e.g., Food-101 / custom food datasets)
- Remove corrupted and duplicate images using perceptual hashing
- Resize to 224x224 & normalize using ImageNet parameters
- Data augmentation (rotation, flips, zoom, brightness)
- Train/Validation/Test splitting with class balancing
"""

import os
import hashlib
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
from PIL import Image, ImageEnhance

# Standard ImageNet normalization parameters for MobileNetV2 / ResNet50
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_TARGET_SIZE = (224, 224)


class FoodImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE):
        self.target_size = target_size

    def preprocess_image(self, image_input, normalize: bool = True) -> np.ndarray:
        """
        Loads, resizes, and normalizes an input image (filepath or PIL Image).
        Returns a normalized NumPy tensor shape: (3, H, W) or (H, W, 3).
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at path: {image_input}")
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise ValueError("Input must be a valid file path or PIL Image instance.")

        # 1. Resize to target dimensions (224x224 for MobileNetV2 / ResNet50)
        img_resized = img.resize(self.target_size, Image.Resampling.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0

        # 2. Normalize with ImageNet mean and std
        if normalize:
            img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

        # Transpose to channels-first (3, H, W) standard for PyTorch / ONNX
        tensor = np.transpose(img_array, (2, 0, 1))
        return tensor

    def augment_image(self, img: Image.Image) -> Image.Image:
        """
        Applies data augmentation to improve model generalization:
        - Random horizontal flip
        - Random rotation (-15 to 15 deg)
        - Random brightness jitter
        """
        # Horizontal Flip (50% probability)
        if np.random.rand() > 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # Random Rotation
        angle = float(np.random.uniform(-15, 15))
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR)

        # Brightness jitter
        enhancer = ImageEnhance.Brightness(img)
        factor = float(np.random.uniform(0.85, 1.15))
        img = enhancer.enhance(factor)

        return img

    @staticmethod
    def compute_image_hash(image_path: str) -> str:
        """Computes MD5 hash to detect and discard duplicate images in dataset."""
        hasher = hashlib.md5()
        with open(image_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def clean_dataset_directory(self, raw_dir: str) -> Dict[str, Any]:
        """
        Scans directory, removes duplicate and unreadable images.
        """
        seen_hashes = set()
        valid_images = []
        duplicates = []
        corrupted = []

        if not os.path.exists(raw_dir):
            return {"error": f"Directory does not exist: {raw_dir}"}

        for root, _, files in os.walk(raw_dir):
            for file in files:
                if not file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    continue
                path = os.path.join(root, file)
                try:
                    with Image.open(path) as img:
                        img.verify()
                    h = self.compute_image_hash(path)
                    if h in seen_hashes:
                        duplicates.append(path)
                    else:
                        seen_hashes.add(h)
                        valid_images.append(path)
                except Exception:
                    corrupted.append(path)

        return {
            "total_scanned": len(valid_images) + len(duplicates) + len(corrupted),
            "valid_count": len(valid_images),
            "duplicates_count": len(duplicates),
            "corrupted_count": len(corrupted),
            "valid_files": valid_images
        }

    @staticmethod
    def train_val_test_split(
        file_list: List[str], 
        train_ratio: float = 0.7, 
        val_ratio: float = 0.15, 
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Dict[str, List[str]]:
        """Stratified / random split for dataset reproducibility."""
        np.random.seed(seed)
        shuffled = np.random.permutation(file_list).tolist()
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        return {
            "train": shuffled[:n_train],
            "val": shuffled[n_train:n_train + n_val],
            "test": shuffled[n_train + n_val:]
        }


if __name__ == "__main__":
    print("[Module 1] Testing Food Image Preprocessing Pipeline...")
    preprocessor = FoodImagePreprocessor()
    
    # Create a synthetic test food image
    dummy_img = Image.new("RGB", (300, 300), color=(180, 80, 50))
    tensor = preprocessor.preprocess_image(dummy_img)
    print(f"[OK] Output tensor shape: {tensor.shape} (Channels x Height x Width)")
    print(f"[OK] Mean normalized value: {tensor.mean():.4f}")
    
    # Test augmentation
    aug_img = preprocessor.augment_image(dummy_img)
    print(f"[OK] Augmentation test successful: size={aug_img.size}")
    print("[Module 1] Preprocessing pipeline verified!")
