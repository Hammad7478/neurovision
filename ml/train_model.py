"""
Training script for NeuroVision MRI tumor classification model.
Uses transfer learning with ResNet50 to classify brain MRI images into:
- glioma
- meningioma
- pituitary
- notumor
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
DATA_DIR = Path("./data")
MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / "model.h5"
CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
NUM_CLASSES = len(CLASSES)

# Create model directory if it doesn't exist
MODEL_DIR.mkdir(exist_ok=True)


class StopAtValAccuracy(Callback):
    """
    Custom callback that stops training when validation accuracy reaches a target threshold.
    This provides explicit early stopping based on a target accuracy value.
    The callback works across training phases - if 90% is reached in phase 1, training stops;
    otherwise it continues to phase 2 and stops when the target is reached.
    """
    def __init__(self, target=0.90):
        super().__init__()
        self.target = target
        self.stopped_epoch = 0
    
    def on_train_begin(self, logs=None):
        """Reset stopped_epoch at the start of each training phase."""
        self.stopped_epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get("val_accuracy")
        
        if val_acc is not None and val_acc >= self.target:
            print(f"\n✓ Validation accuracy ({val_acc:.4f}) reached target ({self.target:.4f})")
            print(f"Stopping training early at epoch {epoch + 1}")
            self.model.stop_training = True
            self.stopped_epoch = epoch + 1
    
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"\nTraining stopped early at epoch {self.stopped_epoch} due to reaching {self.target:.0%} validation accuracy")


def load_dataset(data_dir: Path):
    """
    Load image paths and labels from class subdirectories.
    Returns: (image_paths, labels) where labels are class indices.
    """
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist")
            continue
        
        # Get all image files
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        for img_path in image_files:
            image_paths.append(str(img_path))
            labels.append(class_idx)
        
        print(f"Loaded {len(image_files)} images from {class_name}")
    
    return np.array(image_paths), np.array(labels)


def create_model():
    """
    Create a transfer learning model based on ResNet50.
    """
    # Load pre-trained ResNet50 without top layers
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    
    # Note: Preprocessing (ImageNet normalization) is applied in the data generator
    # and in predict.py, not here, to ensure consistency between training and inference
    
    # Base model
    x = base_model(inputs, training=False)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dropout for regularization
    x = layers.Dropout(0.5)(x)
    
    # Dense layers
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model


def compute_class_weights(labels):
    """
    Compute class weights to handle imbalance.
    """
    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(labels),
        y=labels
    )
    return dict(enumerate(class_weights))


def weighted_categorical_crossentropy(class_weights):
    """
    Create a weighted categorical crossentropy loss function.
    """
    def loss(y_true, y_pred):
        # Convert class weights to tensor
        weights = tf.constant(list(class_weights.values()), dtype=tf.float32)
        
        # Calculate standard categorical crossentropy
        ce = keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Get class indices from one-hot encoded labels
        class_indices = tf.argmax(y_true, axis=1)
        
        # Get weights for each sample in batch
        sample_weights = tf.gather(weights, class_indices)
        
        # Apply weights
        weighted_ce = ce * sample_weights
        
        return tf.reduce_mean(weighted_ce)
    
    return loss


def create_data_generators(train_paths, train_labels, val_paths, val_labels):
    """
    Create data generators with augmentation for training and validation.
    """
    def load_image(img_path, augment=False):
        """Load and optionally augment an image."""
        # Load image as array (values in [0, 255])
        img = keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = keras.utils.img_to_array(img)
        
        # Apply augmentation before preprocessing (on [0, 255] range)
        if augment:
            # Convert to tensor for augmentation
            img_tensor = tf.expand_dims(img_array, 0)
            
            # Random horizontal flip (50% chance)
            if tf.random.uniform([]) > 0.5:
                img_tensor = tf.image.flip_left_right(img_tensor)
            
            # Random brightness adjustment
            img_tensor = tf.image.random_brightness(img_tensor, max_delta=0.2)
            
            # Random contrast adjustment
            img_tensor = tf.image.random_contrast(img_tensor, lower=0.8, upper=1.2)
            
            # Convert back to numpy and clip to valid range
            img_array = img_tensor[0].numpy()
            img_array = np.clip(img_array, 0, 255)
        
        # Preprocess for ResNet50 (ImageNet normalization)
        img_array = preprocess_input(img_array)
        
        return img_array
    
    def data_generator(paths, labels, batch_size, shuffle=True, augment=False):
        """Create a generator that yields batches from file paths."""
        while True:
            # Shuffle indices if needed
            indices = np.arange(len(paths))
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, len(indices), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_paths = paths[batch_indices]
                batch_labels = labels[batch_indices]
                
                # Load and preprocess images
                batch_images = []
                batch_y = []
                
                for img_path, label in zip(batch_paths, batch_labels):
                    try:
                        img_array = load_image(img_path, augment=augment)
                        batch_images.append(img_array)
                        
                        # One-hot encode label
                        one_hot = keras.utils.to_categorical(label, NUM_CLASSES)
                        batch_y.append(one_hot)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue
                
                if len(batch_images) == 0:
                    continue
                
                batch_images = np.array(batch_images)
                batch_y = np.array(batch_y)
                
                yield batch_images, batch_y
    
    train_gen = data_generator(
        train_paths, train_labels, BATCH_SIZE, shuffle=True, augment=True
    )
    val_gen = data_generator(
        val_paths, val_labels, BATCH_SIZE, shuffle=False, augment=False
    )
    
    return train_gen, val_gen


def calculate_metrics(model, val_paths, val_labels):
    """
    Calculate comprehensive metrics on validation set.
    """
    predictions = []
    true_labels = []
    
    # Process validation set in batches
    batch_size = 32
    for start_idx in range(0, len(val_paths), batch_size):
        batch_paths = val_paths[start_idx:start_idx + batch_size]
        batch_labels = val_labels[start_idx:start_idx + batch_size]
        
        batch_images = []
        for img_path in batch_paths:
            try:
                img = keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
                img_array = keras.utils.img_to_array(img)
                img_array = preprocess_input(img_array)
                batch_images.append(img_array)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        if len(batch_images) == 0:
            continue
        
        batch_images = np.array(batch_images)
        batch_pred = model.predict(batch_images, verbose=0)
        
        predictions.extend(np.argmax(batch_pred, axis=1))
        true_labels.extend(batch_labels[:len(batch_images)])
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate metrics
    report = classification_report(
        true_labels, predictions,
        target_names=CLASSES,
        output_dict=True
    )
    
    f1_macro = f1_score(true_labels, predictions, average="macro")
    f1_per_class = f1_score(true_labels, predictions, average=None)
    
    cm = confusion_matrix(true_labels, predictions)
    
    return report, f1_macro, f1_per_class, cm, predictions, true_labels


def plot_confusion_matrix(cm, save_path):
    """
    Plot and save confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """
    Main training pipeline.
    """
    print("=" * 60)
    print("NeuroVision Model Training")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset...")
    image_paths, labels = load_dataset(DATA_DIR)
    print(f"Total images: {len(image_paths)}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Split dataset (stratified split)
    print("\nSplitting dataset...")
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Compute class weights
    class_weights = compute_class_weights(train_labels)
    print(f"\nClass weights: {class_weights}")
    
    # Create data generators
    print("\nCreating data generators...")
    train_gen, val_gen = create_data_generators(
        train_paths, train_labels, val_paths, val_labels
    )
    
    # Create model
    print("\nCreating model...")
    model, base_model = create_model()
    
    # Create weighted loss function
    weighted_loss = weighted_categorical_crossentropy(class_weights)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=weighted_loss,
        metrics=["accuracy"]
    )
    
    print(f"Model parameters: {model.count_params():,}")
    
    # Callbacks
    # StopAtValAccuracy: Stops training when val_accuracy >= 0.90
    # EarlyStopping: Safety net - stops if validation doesn't improve for 10 epochs
    # ModelCheckpoint: Saves the best model based on val_accuracy
    callbacks = [
        StopAtValAccuracy(target=0.90),
        EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]
    
    # Calculate steps per epoch
    train_steps = len(train_paths) // BATCH_SIZE
    val_steps = len(val_paths) // BATCH_SIZE
    
    # Train model (phase 1: frozen base)
    print("\n" + "=" * 60)
    print("Phase 1: Training with frozen base model")
    print("=" * 60)
    
    history1 = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning (phase 2: unfreeze some layers)
    print("\n" + "=" * 60)
    print("Phase 2: Fine-tuning (unfreezing top layers)")
    print("=" * 60)
    
    # Unfreeze top layers of base model
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile with lower learning rate (using same weighted loss)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=weighted_loss,
        metrics=["accuracy"]
    )
    
    history2 = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=EPOCHS - 20,
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model
    # Note: We load with compile=False to avoid issues with custom loss function
    # For evaluation, we don't need the loss function - we can use predict() directly
    print("\nLoading best model...")
    try:
        # Try loading normally first (in case it was saved without custom loss)
        model = keras.models.load_model(str(MODEL_PATH))
    except:
        # If that fails (due to custom loss), load without compilation
        # We'll need to provide the custom loss function
        weighted_loss = weighted_categorical_crossentropy(class_weights)
        model = keras.models.load_model(
            str(MODEL_PATH),
            custom_objects={'loss': weighted_loss},
            compile=False
        )
    
    # Calculate final metrics
    print("\n" + "=" * 60)
    print("Calculating final metrics...")
    print("=" * 60)
    
    report, f1_macro, f1_per_class, cm, predictions, true_labels = calculate_metrics(
        model, val_paths, val_labels
    )
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=CLASSES))
    
    print(f"\nMacro F1 Score: {f1_macro:.4f}")
    print(f"Per-class F1 Scores:")
    for class_name, f1 in zip(CLASSES, f1_per_class):
        print(f"  {class_name}: {f1:.4f}")
    
    # Save metrics
    metrics = {
        "macro_f1": float(f1_macro),
        "per_class_f1": {cls: float(f1) for cls, f1 in zip(CLASSES, f1_per_class)},
        "classification_report": report
    }
    
    metrics_path = MODEL_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Plot and save confusion matrix
    cm_path = MODEL_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    # Save final model without compilation state for easier loading in predict.py
    # The ModelCheckpoint already saved the best model, but we'll save again without compilation
    # This ensures predict.py can load it without needing the custom loss function
    # Note: save_format argument is deprecated in Keras 3, format is inferred from file extension
    model.save(str(MODEL_PATH))
    print(f"\nModel saved to {MODEL_PATH}")
    
    # Also save weights only as a backup (can be loaded into a model with standard loss)
    # Note: In Keras 3, weights files must end with .weights.h5
    weights_path = MODEL_DIR / "model_weights.weights.h5"
    model.save_weights(str(weights_path))
    print(f"Weights also saved to {weights_path}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

