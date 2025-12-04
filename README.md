# NeuroVision

A single-app solution for brain MRI tumor classification using deep learning. NeuroVision allows users to upload a brain MRI image and get predictions about whether a tumor exists and, if so, its type (glioma, meningioma, pituitary, or notumor).

## Features

- **Deep Learning Model**: Transfer learning with ResNet50 for accurate tumor classification
- **Early Stopping**: Training automatically stops when validation accuracy reaches 95%
- **Grad-CAM Visualization**: Visualize which regions of the MRI the model focuses on for predictions
- **Modern Web Interface**: Built with Next.js 14, React 18, TypeScript, and Tailwind CSS
- **Single-App Architecture**: No separate Flask server - everything runs in Next.js with Python scripts called via child processes

## Quick Start

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   npm install
   ```

2. **Train the ResNet-50 model** (stops early at 95% validation accuracy):

   ```bash
   python ml/train_resnet50.py
   ```

3. **Start the development server:**

   ```bash
   npm run dev
   ```

4. **Upload an MRI image** → Get prediction with tumor classification

## Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.8+
- **TensorFlow** 2.15+ (or tensorflow-macos for Apple Silicon)

## Usage

### Using a Pre-trained Model (Recommended)

**Train once locally, then reuse the model:**

1. **Train the model once** (takes 10-30 minutes):

   ```bash
   python3 ml/train_resnet50.py
   ```

2. **The model is saved to `model/resnet50_model.h5`** and persists on disk

3. **Use the model for predictions** - no need to retrain!

### Training the Model

**Note**: You only need to train once! The model persists in `model/resnet50_model.h5`.

#### Option 1: Manual Training

```bash
python3 ml/train_resnet50.py
```

The training script will:

- Load images from `data/` directory
- Split data into training and validation sets (80/20)
- Apply data augmentation
- Train a ResNet50-based model with transfer learning
- **Stop automatically when validation accuracy reaches 95%** (early stopping)
- Save the best model to `model/resnet50_model.h5`
- Generate metrics and confusion matrix

**Early Stopping**: The training process uses a custom callback that automatically stops training when validation accuracy reaches 95% (`val_accuracy >= 0.95`). This prevents overtraining and reduces training time. The model will still save the best weights based on validation accuracy, and all metrics (precision, recall, F1 scores, confusion matrix) are computed and saved after training completes.

**Training Outputs (ResNet-50):**

- `model/resnet50_model.h5` - Trained model file (~100-200MB)
- `model/resnet50_metrics.json` - Classification metrics (F1 scores, precision, recall)
- `model/resnet50_confusion_matrix.png` - Confusion matrix visualization

**MobileNetV2 baseline:**

- Train with `python3 ml/train_mobilenetv2.py`
- Outputs: `model/mobilenetv2_model.h5`, `model/mobilenetv2_metrics.json`, `model/mobilenetv2_confusion_matrix.png`

## Model Architecture

The model uses **transfer learning** with ResNet50 as the base architecture:

- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Input Size**: 224×224 pixels
- **Output**: 4 classes (glioma, meningioma, pituitary, notumor)
- **Training Strategy**:
  1. Phase 1: Freeze base model, train classifier head (up to 20 epochs)
  2. Phase 2: Unfreeze top layers, fine-tune with lower learning rate (up to 30 epochs)
- **Early Stopping**: Training stops automatically when validation accuracy reaches 95% (`val_accuracy >= 0.95`)
- **Callbacks**:
  - Custom `StopAtValAccuracy` callback: Stops at 95% validation accuracy
  - `EarlyStopping`: Safety net - stops if validation doesn't improve for 10 epochs
  - `ModelCheckpoint`: Saves the best model based on validation accuracy

## API Endpoints

### POST `/api/predict`

Upload an image and get predictions.

**Request:**

- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (image file)

**Response:**

```json
{
  "prediction": "glioma",
  "probabilities": {
    "glioma": 0.87,
    "meningioma": 0.04,
    "pituitary": 0.07,
    "notumor": 0.02
  },
  "tumor_detected": true,
  "gradcam_path": "model/tmp/gradcam_abc123.png"
}
```

### GET `/api/serve-gradcam/[filename]`

Serve Grad-CAM visualization images.

### Project Structure Notes

- **ML Code**: All machine learning code is in the `ml/` directory
- **API Routes**: Next.js API routes handle file uploads and call Python scripts
- **Frontend**: React components in `app/page.tsx` provide the UI
- **Static Serving**: Grad-CAM images are served via API route (`/api/serve-gradcam`)
