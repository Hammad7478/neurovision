"""
Grad-CAM (Gradient-weighted Class Activation Mapping) utility for visualizing
model predictions on MRI images.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from typing import Tuple, Optional


def make_gradcam_heatmap(
    model: keras.Model,
    img_array: np.ndarray,
    pred_index: Optional[int] = None,
    last_conv_layer_name: Optional[str] = None
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for the given image.
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed image array (batch dimension included)
        pred_index: Index of the predicted class (if None, uses top prediction)
        last_conv_layer_name: Name of the last convolutional layer (if None, auto-detects)
    
    Returns:
        Heatmap array (H x W) with values in [0, 1]
    """
    # Find the last convolutional layer if not specified
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
                last_conv_layer_name = layer.name
                break
    
    if last_conv_layer_name is None:
        raise ValueError("Could not find a convolutional layer in the model")
    
    # Create a model that maps the input image to the activations of the last conv layer
    # as well as the output predictions
    grad_model = keras.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        class_channel = preds[:, pred_index]
    
    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    return heatmap


def overlay_heatmap(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay a heatmap on the original image.
    
    Args:
        original_img: Original image array (H x W x 3) in RGB format, values in [0, 255]
        heatmap: Heatmap array (H x W) with values in [0, 1]
        alpha: Transparency factor for overlay (0.0 to 1.0)
        colormap: OpenCV colormap (default: JET)
    
    Returns:
        Overlayed image array (H x W x 3) in RGB format
    """
    # Resize heatmap to match image if needed
    if heatmap.shape != original_img.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Convert heatmap to uint8
    heatmap = np.uint8(255 * heatmap)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    
    # Convert BGR to RGB (OpenCV uses BGR)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlayed = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed


def generate_gradcam_visualization(
    model: keras.Model,
    img_array: np.ndarray,
    original_img: np.ndarray,
    pred_index: Optional[int] = None,
    last_conv_layer_name: Optional[str] = None,
    alpha: float = 0.4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM visualization for an image.
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed image array (with batch dimension) ready for model input
        original_img: Original image array (H x W x 3) in RGB, values in [0, 255]
        pred_index: Index of class to visualize (if None, uses top prediction)
        last_conv_layer_name: Name of last conv layer (if None, auto-detects)
        alpha: Overlay transparency
    
    Returns:
        Tuple of (heatmap, overlayed_image)
        - heatmap: (H x W) array with values in [0, 1]
        - overlayed_image: (H x W x 3) RGB image array
    """
    # Generate heatmap
    heatmap = make_gradcam_heatmap(
        model, img_array, pred_index, last_conv_layer_name
    )
    
    # Overlay on original image
    overlayed = overlay_heatmap(original_img, heatmap, alpha)
    
    return heatmap, overlayed

