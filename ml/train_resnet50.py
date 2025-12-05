"""
Training script for NeuroVision MRI tumor classification using ResNet50.
Uses transfer learning with ResNet50 to classify brain MRI images into:
- glioma
- meningioma
- pituitary
- notumor
"""

import os
import json
import argparse
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
from PIL import Image

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
DATA_DIR = Path("./data")
MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / "resnet50_model.h5"
CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
NUM_CLASSES = len(CLASSES)

# Create model directory if it doesn't exist
MODEL_DIR.mkdir(exist_ok=True)


class StopAtValAccuracy(Callback):
    """
    Custom callback that stops training when validation accuracy reaches a target threshold.
    This provides explicit early stopping based on a target accuracy value.
    The callback works across training phases - if 95% is reached in phase 1, training stops;
    otherwise it continues to phase 2 and stops when the target is reached.
    """
    def __init__(self, target=0.95):
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
            print(f"\nValidation accuracy ({val_acc:.4f}) reached target ({self.target:.4f})")
            print(f"Stopping training early at epoch {epoch + 1}")
            self.model.stop_training = True
            self.stopped_epoch = epoch + 1
    
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"\nTraining stopped early at epoch {self.stopped_epoch} due to reaching {self.target:.0%} validation accuracy")


class MetricsLoggerCallback(Callback):
    """
    Logs training metrics to metrics.json after each epoch so the frontend can display live charts.
    """

    def __init__(self, metrics_path: Path):
        super().__init__()
        self.metrics_path = metrics_path
        self.history = {
            "accuracy": [],
            "val_accuracy": [],
            "loss": [],
            "val_loss": []
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def append_metric(log_key: str, history_key: str):
            value = logs.get(log_key)
            if value is None:
                return
            try:
                self.history[history_key].append(float(value))
            except (TypeError, ValueError):
                pass

        append_metric("accuracy", "accuracy")
        append_metric("val_accuracy", "val_accuracy")
        append_metric("loss", "loss")
        append_metric("val_loss", "val_loss")

        latest_val_acc = self.history["val_accuracy"][-1] if self.history["val_accuracy"] else None
        latest_train_acc = self.history["accuracy"][-1] if self.history["accuracy"] else None

        metrics_snapshot = {
            "accuracy": latest_val_acc if latest_val_acc is not None else latest_train_acc,
            "precision": latest_val_acc if latest_val_acc is not None else latest_train_acc,
            "recall": latest_val_acc if latest_val_acc is not None else latest_train_acc,
            "f1_score": latest_val_acc if latest_val_acc is not None else latest_train_acc,
            "train_accuracy": self.history["accuracy"],
            "val_accuracy": self.history["val_accuracy"],
            "train_loss": self.history["loss"],
            "val_loss": self.history["val_loss"]
        }

        existing = {}
        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, "r") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}

        existing.update({k: v for k, v in metrics_snapshot.items() if v is not None})

        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "w") as f:
            json.dump(existing, f, indent=2)


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


def create_data_generators(train_paths, train_labels, val_paths, val_labels, 
                          augmentation="on", color_mode="rgb"):
    """
    Create data generators with augmentation for training and validation.
    
    Args:
        train_paths: Training image paths
        train_labels: Training labels
        val_paths: Validation image paths
        val_labels: Validation labels
        augmentation: "on" or "off" - whether to apply augmentation to training data
        color_mode: "rgb" or "grayscale" - input color mode
    """
    def load_image(img_path, augment=False, color_mode="rgb"):
        """Load and optionally augment an image."""
        # Load image based on color mode
        if color_mode == "grayscale":
            # Load as grayscale
            img = Image.open(img_path).convert("L")
            img = img.resize(IMAGE_SIZE)
            img_array = np.array(img, dtype=np.float32)
            # Stack to 3 channels for ResNet50 input
            img_array = np.stack([img_array, img_array, img_array], axis=-1)
        else:
            # Load as RGB (default)
            img = keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
            img_array = keras.utils.img_to_array(img)
        
        # Apply augmentation before preprocessing (on [0, 255] range)
        if augment:
            # Convert to tensor for augmentation
            img_tensor = tf.expand_dims(img_array, 0)
            
            # Random horizontal flip (50% chance)
            if tf.random.uniform([]) > 0.5:
                img_tensor = tf.image.flip_left_right(img_tensor)

            # Random small rotation (~±15 degrees)
            rotation_layer = layers.RandomRotation(
                factor=(-0.083, 0.083),
                fill_mode="reflect"
            )
            img_tensor = rotation_layer(img_tensor, training=True)
            
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
                        img_array = load_image(img_path, augment=augment, color_mode=color_mode)
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
    
    # Training generator: use augmentation if enabled
    train_augment = (augmentation == "on")
    train_gen = data_generator(
        train_paths, train_labels, BATCH_SIZE, shuffle=True, augment=train_augment
    )
    # Validation generator: never use augmentation
    val_gen = data_generator(
        val_paths, val_labels, BATCH_SIZE, shuffle=False, augment=False
    )
    
    return train_gen, val_gen


def calculate_metrics(model, val_paths, val_labels, color_mode="rgb"):
    """
    Calculate comprehensive metrics on validation set.
    
    Args:
        model: Trained model
        val_paths: Validation image paths
        val_labels: Validation labels
        color_mode: "rgb" or "grayscale" - must match training color mode
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
                if color_mode == "grayscale":
                    # Load as grayscale
                    img = Image.open(img_path).convert("L")
                    img = img.resize(IMAGE_SIZE)
                    img_array = np.array(img, dtype=np.float32)
                    # Stack to 3 channels for ResNet50 input
                    img_array = np.stack([img_array, img_array, img_array], axis=-1)
                else:
                    # Load as RGB
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


def parse_args():
    """Parse command-line arguments for ablation experiments."""
    parser = argparse.ArgumentParser(
        description="Train ResNet-50 model for MRI brain tumor classification with ablation options"
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        choices=["on", "off"],
        default="on",
        help="Enable data augmentation for training (default: on)"
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        choices=["on", "off"],
        default="on",
        help="Use class weights for imbalanced data (default: on)"
    )
    parser.add_argument(
        "--color-mode",
        type=str,
        choices=["rgb", "grayscale"],
        default="rgb",
        help="Input color mode: rgb or grayscale (default: rgb)"
    )
    parser.add_argument(
        "--finetune-mode",
        type=str,
        choices=["partial", "frozen"],
        default="partial",
        help="Fine-tuning mode: partial (unfreeze top layers) or frozen (keep backbone frozen) (default: partial)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing training data (default: ./data)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./model",
        help="Directory to save model and metrics (default: ./model)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Maximum number of epochs (default: 25)"
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.95,
        help="Target validation accuracy for early stopping (default: 0.95)"
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Optional suffix for output files (for ablation experiments)"
    )
    
    return parser.parse_args()


def main(args=None):
    """
    Main training pipeline.
    
    Args:
        args: Optional argparse.Namespace. If None, will parse from command line.
    """
    if args is None:
        args = parse_args()
    
    # Update global paths if specified
    global DATA_DIR, MODEL_DIR, MODEL_PATH, BATCH_SIZE, EPOCHS
    DATA_DIR = Path(args.data_dir)
    MODEL_DIR = Path(args.model_dir)
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Set output file names with optional suffix
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    MODEL_PATH = MODEL_DIR / f"resnet50_model{suffix}.h5"
    
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    
    print("=" * 60)
    print("NeuroVision Model Training")
    print("=" * 60)
    print(f"\nAblation Configuration:")
    print(f"  Augmentation: {args.augmentation}")
    print(f"  Class Weights: {args.class_weights}")
    print(f"  Color Mode: {args.color_mode}")
    print(f"  Fine-tune Mode: {args.finetune_mode}")
    print(f"  Output Suffix: {args.output_suffix if args.output_suffix else '(none)'}")
    
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
    
    # Compute class weights (if enabled)
    class_weights = None
    if args.class_weights == "on":
        class_weights = compute_class_weights(train_labels)
        print(f"\nClass weights: {class_weights}")
    else:
        print("\nClass weights: disabled (using unweighted loss)")
    
    # Create data generators
    print("\nCreating data generators...")
    train_gen, val_gen = create_data_generators(
        train_paths, train_labels, val_paths, val_labels,
        augmentation=args.augmentation,
        color_mode=args.color_mode
    )
    
    # Create model
    print("\nCreating model...")
    model, base_model = create_model()
    
    # Select loss function based on class weights setting
    if args.class_weights == "on" and class_weights is not None:
        loss_fn = weighted_categorical_crossentropy(class_weights)
        print("Using weighted categorical cross-entropy loss")
    else:
        loss_fn = keras.losses.CategoricalCrossentropy()
        print("Using standard categorical cross-entropy loss")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=["accuracy"]
    )
    
    print(f"Model parameters: {model.count_params():,}")
    
    # Set metrics file path with suffix
    metrics_path = MODEL_DIR / f"resnet50_metrics{suffix}.json"
    metrics_logger = MetricsLoggerCallback(metrics_path)

    # Callbacks
    # StopAtValAccuracy: Stops training when val_accuracy >= target
    # EarlyStopping: Safety net - stops if validation doesn't improve for 10 epochs
    # ModelCheckpoint: Saves the best model based on val_accuracy
    callbacks = [
        StopAtValAccuracy(target=args.target_accuracy),
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
        ),
        metrics_logger
    ]
    
    # Calculate steps per epoch
    train_steps = len(train_paths) // BATCH_SIZE
    val_steps = len(val_paths) // BATCH_SIZE
    
    # Determine phase distribution based on total epochs
    # Phase 1: frozen base (use up to 20 epochs, or all epochs if total < 20)
    # Phase 2: fine-tuning (remaining epochs)
    phase1_epochs = min(20, EPOCHS)
    phase2_epochs = max(0, EPOCHS - 20)
    
    # Train model (phase 1: frozen base)
    print("\n" + "=" * 60)
    print(f"Phase 1: Training with frozen base model ({phase1_epochs} epochs)")
    print("=" * 60)
    
    history1 = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=phase1_epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning (phase 2: conditionally unfreeze layers)
    history2 = None
    if args.finetune_mode == "partial" and phase2_epochs > 0:
        print("\n" + "=" * 60)
        print(f"Phase 2: Fine-tuning (unfreezing top layers) ({phase2_epochs} epochs)")
        print("=" * 60)
        
        # Unfreeze top layers of base model
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        # Recompile with lower learning rate (using same loss function)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss=loss_fn,
            metrics=["accuracy"]
        )
        
        history2 = model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=phase2_epochs,
            callbacks=callbacks,
            verbose=1
        )
    elif args.finetune_mode == "partial" and phase2_epochs == 0:
        print("\n" + "=" * 60)
        print("Phase 2: Skipped (total epochs <= 20, only Phase 1 was run)")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Phase 2: Skipped (frozen mode - backbone remains frozen)")
        print("=" * 60)

    # Merge training history from both phases for analytics
    history_data = {
        "accuracy": [],
        "val_accuracy": [],
        "loss": [],
        "val_loss": []
    }

    def extend_history(history_obj):
        if not history_obj:
            return
        for key in history_data.keys():
            values = history_obj.history.get(key, [])
            if values:
                history_data[key].extend([float(v) for v in values])

    extend_history(history1)
    if history2 is not None:
        extend_history(history2)

    # Load best model
    # Note: We load with compile=False to avoid issues with custom loss function
    # For evaluation, we don't need the loss function - we can use predict() directly
    print("\nLoading best model...")
    try:
        # Try loading normally first (in case it was saved without custom loss)
        model = keras.models.load_model(str(MODEL_PATH))
    except:
        # If that fails (due to custom loss), load without compilation
        # We'll need to provide the custom loss function if it was used
        if args.class_weights == "on" and class_weights is not None:
            custom_loss = weighted_categorical_crossentropy(class_weights)
            model = keras.models.load_model(
                str(MODEL_PATH),
                custom_objects={'loss': custom_loss},
                compile=False
            )
        else:
            # Try loading without custom objects
            model = keras.models.load_model(str(MODEL_PATH), compile=False)
    
    # Calculate final metrics
    print("\n" + "=" * 60)
    print("Calculating final metrics...")
    print("=" * 60)
    
    report, f1_macro, f1_per_class, cm, predictions, true_labels = calculate_metrics(
        model, val_paths, val_labels, color_mode=args.color_mode
    )
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=CLASSES))
    
    print(f"\nMacro F1 Score: {f1_macro:.4f}")
    print(f"Per-class F1 Scores:")
    for class_name, f1 in zip(CLASSES, f1_per_class):
        print(f"  {class_name}: {f1:.4f}")
    
    # Save metrics
    accuracy = float(report.get("accuracy", 0.0))
    macro_avg = report.get("macro avg", {})

    history_data = metrics_logger.history

    metrics = {
        "accuracy": accuracy,
        "precision": float(macro_avg.get("precision", 0.0)),
        "recall": float(macro_avg.get("recall", 0.0)),
        "f1_score": float(macro_avg.get("f1-score", f1_macro)),
        "train_accuracy": history_data["accuracy"],
        "val_accuracy": history_data["val_accuracy"],
        "train_loss": history_data["loss"],
        "val_loss": history_data["val_loss"],
        "confusion_matrix": cm.tolist(),
        "per_class_f1": {cls: float(f1) for cls, f1 in zip(CLASSES, f1_per_class)},
        # Store ablation configuration
        "config": {
            "augmentation": args.augmentation,
            "class_weights": args.class_weights,
            "color_mode": args.color_mode,
            "finetune_mode": args.finetune_mode
        }
    }
    
    # Use the same metrics path as the logger
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Plot and save confusion matrix
    cm_path = MODEL_DIR / f"resnet50_confusion_matrix{suffix}.png"
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
    weights_path = MODEL_DIR / "resnet50_weights.weights.h5"
    model.save_weights(str(weights_path))
    print(f"Weights also saved to {weights_path}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(args)

