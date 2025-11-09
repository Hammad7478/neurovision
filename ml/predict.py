"""
Inference script for NeuroVision MRI tumor classification.
Loads a trained model and predicts the class of a single MRI image.
"""

import argparse
import json
import sys
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import Grad-CAM utility
# Add parent directory to path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent))
from gradcam import generate_gradcam_visualization

# Class names (must match training)
CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
IMAGE_SIZE = (224, 224)


def preprocess_image(image_path: str) -> tuple:
    """
    Load and preprocess an image for model prediction.
    
    Returns:
        Tuple of (preprocessed_array, original_image_array)
        - preprocessed_array: (1, 224, 224, 3) array ready for model
        - original_image_array: (224, 224, 3) RGB array in [0, 255]
    """
    # Load image
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    
    # Keep original for visualization (RGB, uint8)
    original_img = img_array.astype(np.uint8)
    
    # Preprocess for model (ImageNet normalization)
    img_array = preprocess_input(img_array)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, original_img


def predict_image(model: keras.Model, image_path: str) -> dict:
    """
    Predict class probabilities for an image.
    
    Returns:
        Dictionary with prediction results
    """
    # Preprocess image
    img_array, _ = preprocess_image(image_path)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    probs = predictions[0]
    
    # Get predicted class
    pred_idx = np.argmax(probs)
    pred_class = CLASSES[pred_idx]
    pred_prob = float(probs[pred_idx])
    
    # Build results dictionary
    results = {
        "prediction": pred_class,
        "probabilities": {
            cls: float(prob) for cls, prob in zip(CLASSES, probs)
        },
        "tumor_detected": pred_class != "notumor"
    }
    
    return results, pred_idx


def save_gradcam_image(
    model: keras.Model,
    image_path: str,
    output_path: str,
    pred_idx: int
) -> str:
    """
    Generate and save Grad-CAM visualization.
    
    Returns:
        Path to saved Grad-CAM image
    """
    # Preprocess image
    img_array, original_img = preprocess_image(image_path)
    
    # Generate Grad-CAM visualization
    try:
        heatmap, overlayed = generate_gradcam_visualization(
            model,
            img_array,
            original_img,
            pred_index=pred_idx,
            alpha=0.4
        )
        
        # Save image
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(overlayed)
        plt.axis("off")
        plt.title("Grad-CAM Visualization", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close()
        
        return output_path
        
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}", file=sys.stderr)
        return None


def main():
    """
    Main inference function.
    """
    parser = argparse.ArgumentParser(
        description="Predict MRI tumor classification from an image"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./model/model.h5",
        help="Path to trained model file"
    )
    parser.add_argument(
        "--gradcam",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Whether to generate Grad-CAM visualization"
    )
    
    args = parser.parse_args()
    
    # Check if image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(
            json.dumps({"error": f"Image file not found: {image_path}"}),
            file=sys.stderr
        )
        sys.exit(1)
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(
            json.dumps({"error": f"Model file not found: {model_path}"}),
            file=sys.stderr
        )
        sys.exit(1)
    
    try:
        # Load model with compile=False to avoid needing the custom loss function
        # For inference, we don't need the loss function - model.predict() works without it
        model = keras.models.load_model(str(model_path), compile=False)
        
        # Predict
        results, pred_idx = predict_image(model, str(image_path))
        
        # Generate Grad-CAM if requested
        if args.gradcam.lower() == "true":
            # Create output path for Grad-CAM image
            import uuid
            gradcam_filename = f"gradcam_{uuid.uuid4().hex[:8]}.png"
            gradcam_dir = Path("./model/tmp")
            gradcam_path = gradcam_dir / gradcam_filename
            
            saved_path = save_gradcam_image(
                model,
                str(image_path),
                str(gradcam_path),
                pred_idx
            )
            
            if saved_path:
                # Use relative path for API response
                results["gradcam_path"] = f"model/tmp/{gradcam_filename}"
        
        # Print JSON results to stdout
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(
            json.dumps({"error": f"Prediction failed: {str(e)}"}),
            file=sys.stderr
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

