# NeuroVision

A single-app solution for brain MRI tumor classification using deep learning. NeuroVision allows users to upload a brain MRI image and get predictions about whether a tumor exists and, if so, its type (glioma, meningioma, pituitary, or notumor).

## Features

- **Deep Learning Model**: Transfer learning with ResNet50 for accurate tumor classification
- **Grad-CAM Visualization**: Visualize which regions of the MRI the model focuses on for predictions
- **Modern Web Interface**: Built with Next.js 14, React 18, TypeScript, and Tailwind CSS
- **Single-App Architecture**: No separate Flask server - everything runs in Next.js with Python scripts called via child processes

## Project Structure

```
neurovision/
├── app/
│   ├── api/
│   │   ├── predict/
│   │   │   └── route.ts          # API route for predictions
│   │   └── serve-gradcam/
│   │       └── [...path]/
│   │           └── route.ts      # API route to serve Grad-CAM images
│   ├── page.tsx                  # Main UI component
│   └── layout.tsx                # Root layout
├── ml/
│   ├── train_model.py            # Training script
│   ├── predict.py                # Inference script
│   └── gradcam.py                # Grad-CAM utility
├── data/                         # Training data (4 class folders)
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
├── model/                        # Saved models and outputs
│   ├── model.h5                  # Trained model (generated)
│   ├── metrics.json              # Training metrics (generated)
│   ├── confusion_matrix.png      # Confusion matrix (generated)
│   └── tmp/                      # Temporary files for predictions
├── scripts/
│   └── run-train.sh              # Convenience script for training
├── requirements.txt              # Python dependencies
└── package.json                  # Node.js dependencies
```

## Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.8+
- **TensorFlow** 2.15+ (or tensorflow-macos for Apple Silicon)

## Installation

### 1. Install Node.js Dependencies

```bash
npm install
```

### 2. Install Python Dependencies

For standard systems:
```bash
pip install -r requirements.txt
```

For Apple Silicon (M1/M2/M3 Macs):
```bash
pip install tensorflow-macos tensorflow-metal
pip install -r requirements.txt
```

**Note**: On Apple Silicon, you may need to install TensorFlow with Metal support for GPU acceleration:
```bash
pip install tensorflow-macos tensorflow-metal
```

### 3. Prepare Training Data

Ensure your training data is organized in the `data/` directory with the following structure:
```
data/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/
```

Each folder should contain MRI images in JPG or PNG format.

## Usage

### Using a Pre-trained Model (Recommended)

**Train once locally, then reuse the model:**

1. **Train the model once** (takes 10-30 minutes):
   ```bash
   python3 ml/train_model.py
   ```

2. **The model is saved to `model/model.h5`** and persists on disk

3. **Use the model for predictions** - no need to retrain!

**For deployment with a pre-trained model:**

#### Option A: Download from Cloud Storage

Set the `MODEL_URL` environment variable and download:

```bash
# Using Python script
MODEL_URL=https://your-storage.com/model.h5 python3 ml/download_model.py

# Using shell script
MODEL_URL=https://your-storage.com/model.h5 ./scripts/download-model.sh
```

#### Option B: Include in Deployment

Simply include the `model/model.h5` file in your deployment. The model file (~100-200MB) will be used directly.

### Training the Model

**Note**: You only need to train once! The model persists in `model/model.h5`.

#### Option 1: Using the Convenience Script

```bash
./scripts/run-train.sh
```

This script will:
- Create a virtual environment (if needed)
- Install Python dependencies
- Train the model
- Save the model to `model/model.h5`

#### Option 2: Manual Training

```bash
python3 ml/train_model.py
```

The training script will:
- Load images from `data/` directory
- Split data into training and validation sets (80/20)
- Apply data augmentation
- Train a ResNet50-based model with transfer learning
- Save the best model to `model/model.h5`
- Generate metrics and confusion matrix

**Training Outputs:**
- `model/model.h5` - Trained model file (~100-200MB)
- `model/metrics.json` - Classification metrics (F1 scores, precision, recall)
- `model/confusion_matrix.png` - Confusion matrix visualization

### Automatic Model Handling

The application automatically handles model availability:

1. **If model exists**: Uses it immediately for predictions
2. **If model doesn't exist and `MODEL_URL` is set**: Downloads the model
3. **If model doesn't exist and no `MODEL_URL`**: Automatically starts training (takes 10-30 minutes)

See `MODEL_DISTRIBUTION.md` for detailed information on distributing pre-trained models.

### Running the Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Testing Prediction (Without UI)

You can test the prediction script directly:

```bash
python ml/predict.py --image path/to/image.jpg --model ./model/model.h5 --gradcam true
```

This will output JSON with predictions:
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

### Using the Web Interface

1. Start the development server: `npm run dev`
2. Open the application in your browser
3. Upload a brain MRI image (JPG or PNG)
4. Click "Classify MRI"
5. View the results:
   - Tumor detection status (Yes/No)
   - Predicted tumor type
   - Class probabilities
   - Grad-CAM visualization (if enabled)

## Model Architecture

The model uses **transfer learning** with ResNet50 as the base architecture:

- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Input Size**: 224×224 pixels
- **Output**: 4 classes (glioma, meningioma, pituitary, notumor)
- **Training Strategy**:
  1. Phase 1: Freeze base model, train classifier head
  2. Phase 2: Unfreeze top layers, fine-tune with lower learning rate

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
