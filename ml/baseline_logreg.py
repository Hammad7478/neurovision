"""
Logistic regression baseline for NeuroVision MRI tumor classification.
Trains a multinomial logistic regression model on flattened grayscale images.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Configuration mirrors train_model.py
CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
IMAGE_SIZE = (224, 224)
MODEL_DIR = Path("./model")


def collect_image_label_pairs(data_dir: Path):
    """
    Walk a directory and collect image paths and labels based on class subfolders.
    """
    image_paths = []
    labels = []

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist, skipping.")
            continue

        files = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            files.extend(class_dir.glob(pattern))

        files = sorted(files)
        for img_path in files:
            image_paths.append(img_path)
            labels.append(class_idx)

        print(f"Found {len(files)} images for class '{class_name}' in {class_dir}")

    return image_paths, labels


def load_and_flatten(image_paths, labels, image_size):
    """
    Load images, convert to grayscale, resize, and flatten into 1D vectors.
    Returns arrays of features and aligned labels (skipping unreadable images).
    """
    features = []
    valid_labels = []

    for img_path, label in zip(image_paths, labels):
        try:
            with Image.open(img_path) as img:
                img = img.convert("L")
                img = img.resize(image_size, resample=Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                features.append(arr.flatten())
                valid_labels.append(label)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

    return np.array(features), np.array(valid_labels)


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


def compute_metrics(y_true, y_pred):
    """
    Compute accuracy, macro metrics, and per-class metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=range(len(CLASSES)),
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
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

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Baseline multinomial logistic regression for NeuroVision."
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="./data",
        help="Path to training data directory (class subfolders).",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help=(
            "Optional path to test data directory (class subfolders). "
            "If omitted or the same as train-dir, a stratified split is used."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=IMAGE_SIZE,
        metavar=("HEIGHT", "WIDTH"),
        help="Image size to resize to before flattening (default: 224 224).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction when no separate test directory is provided.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for LogisticRegression (use 1 to avoid joblib temp files).",
    )
    parser.add_argument(
        "--joblib-temp",
        type=str,
        default=None,
        help="Optional temp directory for joblib (must exist or be creatable).",
    )

    args = parser.parse_args()

    if args.joblib_temp:
        joblib_tmp = Path(args.joblib_temp)
        joblib_tmp.mkdir(parents=True, exist_ok=True)
        os.environ["JOBLIB_TEMP_FOLDER"] = str(joblib_tmp.resolve())
        print(f"Using joblib temp directory: {joblib_tmp.resolve()}")

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir) if args.test_dir else None
    image_size = (args.image_size[0], args.image_size[1])

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting training data...")
    train_paths, train_labels = collect_image_label_pairs(train_dir)

    if test_dir and test_dir.exists() and test_dir.resolve() != train_dir.resolve():
        print("Collecting test data...")
        test_paths, test_labels = collect_image_label_pairs(test_dir)
    else:
        print("No separate test directory provided; performing stratified split.")
        (
            train_paths,
            test_paths,
            train_labels,
            test_labels,
        ) = train_test_split(
            train_paths,
            train_labels,
            test_size=args.test_size,
            random_state=42,
            stratify=train_labels,
        )

    print("Loading and flattening training images...")
    X_train, y_train = load_and_flatten(train_paths, train_labels, image_size)
    print(f"Training samples after loading: {len(X_train)}")

    print("Loading and flattening test images...")
    X_test, y_test = load_and_flatten(test_paths, test_labels, image_size)
    print(f"Test samples after loading: {len(X_test)}")

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("No training or test samples available after preprocessing.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=args.n_jobs,
    )

    print("Training logistic regression model...")
    logreg.fit(X_train_scaled, y_train)

    print("Evaluating on test set...")
    y_pred = logreg.predict(X_test_scaled)

    cm = confusion_matrix(
        y_test, y_pred, labels=list(range(len(CLASSES)))
    )
    metrics = compute_metrics(y_test, y_pred)

    metrics_path = MODEL_DIR / "baseline_logreg_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    cm_path = MODEL_DIR / "baseline_logreg_confusion_matrix.png"
    plot_confusion_matrix(cm, CLASSES, cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    print("Baseline logistic regression complete.")


if __name__ == "__main__":
    main()
