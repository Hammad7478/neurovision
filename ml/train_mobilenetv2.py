"""
MobileNetV2 baseline training for NeuroVision MRI tumor classification.
Uses ImageNet-pretrained MobileNetV2 with a lightweight classification head.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight

# Configuration mirroring train_resnet50.py
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
DEFAULT_BATCH_SIZE = 32
DEFAULT_HEAD_EPOCHS = 12
DATA_DIR = Path("./data")
MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / "mobilenetv2_model.h5"
METRICS_PATH = MODEL_DIR / "mobilenetv2_metrics.json"
CM_PATH = MODEL_DIR / "mobilenetv2_confusion_matrix.png"
CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
NUM_CLASSES = len(CLASSES)

MODEL_DIR.mkdir(exist_ok=True)


class StopAtValAccuracy(keras.callbacks.Callback):
    """
    Stop training when validation accuracy reaches a target threshold.
    """

    def __init__(self, target: float = 0.95):
        super().__init__()
        self.target = target
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get("val_accuracy")
        if val_acc is not None and val_acc >= self.target:
            print(
                f"\nValidation accuracy ({val_acc:.4f}) reached target ({self.target:.4f})."
            )
            print(f"Stopping training early at epoch {epoch + 1}")
            self.model.stop_training = True
            self.stopped_epoch = epoch + 1

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(
                f"\nTraining stopped early at epoch {self.stopped_epoch} "
                f"due to reaching {self.target:.0%} validation accuracy"
            )


def compute_class_weights(labels):
    """
    Compute class weights to handle imbalance.
    """
    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(labels),
        y=labels,
    )
    return dict(enumerate(class_weights))


def weighted_categorical_crossentropy(class_weights):
    """
    Create weighted categorical crossentropy loss.
    """

    def loss(y_true, y_pred):
        weights = tf.constant(list(class_weights.values()), dtype=tf.float32)
        ce = keras.losses.categorical_crossentropy(y_true, y_pred)
        class_indices = tf.argmax(y_true, axis=1)
        sample_weights = tf.gather(weights, class_indices)
        weighted_ce = ce * sample_weights
        return tf.reduce_mean(weighted_ce)

    return loss


def load_dataset(data_dir: Path):
    """
    Load image paths and labels from class subdirectories.
    """
    image_paths = []
    labels = []

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist")
            continue

        files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        for img_path in files:
            image_paths.append(str(img_path))
            labels.append(class_idx)

        print(f"Loaded {len(files)} images from {class_name}")

    return np.array(image_paths), np.array(labels)


def create_data_generators(
    train_paths,
    train_labels,
    val_paths,
    val_labels,
    image_size: Tuple[int, int],
    batch_size: int,
):
    """
    Create data generators with augmentation for training and validation.
    """

    def load_image(img_path, augment=False):
        img = keras.utils.load_img(img_path, target_size=image_size)
        img_array = keras.utils.img_to_array(img)

        if augment:
            img_tensor = tf.expand_dims(img_array, 0)
            if tf.random.uniform([]) > 0.5:
                img_tensor = tf.image.flip_left_right(img_tensor)
            img_tensor = tf.image.random_brightness(img_tensor, max_delta=0.2)
            img_tensor = tf.image.random_contrast(img_tensor, lower=0.8, upper=1.2)
            img_array = img_tensor[0].numpy()
            img_array = np.clip(img_array, 0, 255)

        img_array = preprocess_input(img_array)
        return img_array

    def data_generator(paths, labels, shuffle=True, augment=False):
        while True:
            indices = np.arange(len(paths))
            if shuffle:
                np.random.shuffle(indices)

            for start_idx in range(0, len(indices), batch_size):
                batch_indices = indices[start_idx : start_idx + batch_size]
                batch_paths = paths[batch_indices]
                batch_labels = labels[batch_indices]

                batch_images = []
                batch_y = []

                for img_path, label in zip(batch_paths, batch_labels):
                    try:
                        img_array = load_image(img_path, augment=augment)
                        batch_images.append(img_array)
                        one_hot = keras.utils.to_categorical(label, NUM_CLASSES)
                        batch_y.append(one_hot)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue

                if not batch_images:
                    continue

                yield np.array(batch_images), np.array(batch_y)

    train_gen = data_generator(
        train_paths, train_labels, shuffle=True, augment=True
    )
    val_gen = data_generator(
        val_paths, val_labels, shuffle=False, augment=False
    )

    return train_gen, val_gen


def build_model(image_size: Tuple[int, int]):
    """
    Build MobileNetV2 model with custom classification head.
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size[0], image_size[1], 3),
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(image_size[0], image_size[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model, base_model


def evaluate_model(model, val_paths, val_labels, image_size: Tuple[int, int], batch_size: int):
    """
    Predict on validation set and compute metrics.
    """
    predictions = []
    true_labels = []

    for start_idx in range(0, len(val_paths), batch_size):
        batch_paths = val_paths[start_idx : start_idx + batch_size]
        batch_labels = val_labels[start_idx : start_idx + batch_size]

        batch_images = []
        for img_path in batch_paths:
            try:
                img = keras.utils.load_img(img_path, target_size=image_size)
                img_array = keras.utils.img_to_array(img)
                img_array = preprocess_input(img_array)
                batch_images.append(img_array)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

        if not batch_images:
            continue

        batch_images = np.array(batch_images)
        batch_pred = model.predict(batch_images, verbose=0)
        predictions.extend(np.argmax(batch_pred, axis=1))
        true_labels.extend(batch_labels[: len(batch_images)])

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predictions,
        labels=range(NUM_CLASSES),
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    accuracy = accuracy_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions, labels=range(NUM_CLASSES))

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "f1_score": float(macro_f1),
        "per_class": {
            class_name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, class_name in enumerate(CLASSES)
        },
    }

    return metrics, cm, predictions, true_labels


def plot_confusion_matrix(cm: np.ndarray, class_names, output_path: Path):
    """
    Plot and save confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train a MobileNetV2 baseline for NeuroVision."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help="Path to dataset directory containing class subfolders.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--head-epochs",
        type=int,
        default=DEFAULT_HEAD_EPOCHS,
        help="Epochs for training the classification head (frozen base).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split fraction.",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.95,
        help="Validation accuracy target for early stopping.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for head training.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=IMAGE_SIZE,
        metavar=("HEIGHT", "WIDTH"),
        help="Input image size (height width).",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    image_size = (args.image_size[0], args.image_size[1])
    batch_size = args.batch_size

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print("=" * 60)
    print("MobileNetV2 Baseline Training")
    print("=" * 60)

    print("\nLoading dataset...")
    image_paths, labels = load_dataset(data_dir)
    print(f"Total images: {len(image_paths)}")
    print(f"Class distribution: {np.bincount(labels)}")

    print("\nSplitting dataset...")
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=args.val_split,
        random_state=42,
        stratify=labels,
    )
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")

    print("\nComputing class weights...")
    class_weights = compute_class_weights(train_labels)
    print(f"Class weights: {class_weights}")

    print("\nCreating data generators...")
    train_gen, val_gen = create_data_generators(
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        image_size,
        batch_size,
    )

    print("\nBuilding model...")
    model, base_model = build_model(image_size)
    weighted_loss = weighted_categorical_crossentropy(class_weights)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.learning_rate),
        loss=weighted_loss,
        metrics=["accuracy"],
    )

    callbacks = [
        StopAtValAccuracy(target=args.target_accuracy),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]

    train_steps = max(1, len(train_paths) // batch_size)
    val_steps = max(1, len(val_paths) // batch_size)

    print("\nTraining classifier head (base frozen)")
    model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=args.head_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nLoading best model checkpoint for evaluation...")
    try:
        best_model = keras.models.load_model(str(MODEL_PATH))
    except Exception:
        best_model = keras.models.load_model(
            str(MODEL_PATH),
            custom_objects={"loss": weighted_loss},
            compile=False,
        )

    print("\nEvaluating on validation set...")
    metrics, cm, predictions, true_labels = evaluate_model(
        best_model, val_paths, val_labels, image_size, batch_size
    )

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    plot_confusion_matrix(cm, CLASSES, CM_PATH)
    print(f"Confusion matrix saved to {CM_PATH}")

    # Save final model with latest weights for convenience
    best_model.save(str(MODEL_PATH))
    weights_path = MODEL_DIR / "mobilenetv2_weights.weights.h5"
    best_model.save_weights(str(weights_path))
    print(f"Model saved to {MODEL_PATH}")
    print(f"Weights also saved to {weights_path}")

    print("\nTraining completed.")


if __name__ == "__main__":
    main()
