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

### Training the Model

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
python ml/train_model.py
```

The training script will:
- Load images from `data/` directory
- Split data into training and validation sets (80/20)
- Apply data augmentation
- Train a ResNet50-based model with transfer learning
- Save the best model to `model/model.h5`
- Generate metrics and confusion matrix

**Training Outputs:**
- `model/model.h5` - Trained model file
- `model/metrics.json` - Classification metrics (F1 scores, precision, recall)
- `model/confusion_matrix.png` - Confusion matrix visualization

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

## Troubleshooting

### Apple Silicon (M1/M2/M3 Macs)

If you encounter issues with TensorFlow on Apple Silicon:

1. **Install TensorFlow for macOS:**
   ```bash
   pip install tensorflow-macos tensorflow-metal
   ```

2. **Verify installation:**
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

3. **If you see Metal device**, GPU acceleration is enabled.

### Model Not Found Error

If you get "Model not found" error:
- Ensure you've trained the model first: `python ml/train_model.py`
- Check that `model/model.h5` exists

### Python Script Execution Errors

If the API route fails to execute Python scripts:
- Ensure Python 3 is installed and accessible as `python3`
- Check that all Python dependencies are installed
- Verify the model file exists

### Out of Memory Errors

If you encounter memory errors during training:
- Reduce `BATCH_SIZE` in `ml/train_model.py`
- Reduce image size (though this may affect accuracy)
- Close other applications using GPU/memory

### Grad-CAM Not Generating

If Grad-CAM visualization is not generated:
- Check that `model/tmp/` directory exists and is writable
- Verify matplotlib is installed correctly
- Check console logs for error messages

## Deployment Considerations

### Local Development / VPS

This setup works well for:
- Local development
- VPS deployments (with Node.js and Python installed)
- Docker containers (with both Node.js and Python)

### Node-Only Platforms (e.g., Vercel)

**Current limitation**: This architecture requires Python to be available on the server. Platforms like Vercel (Node.js-only) cannot run Python scripts directly.

**Future options:**
1. **Separate Python microservice**: Deploy the prediction logic as a separate Flask/FastAPI service
2. **TensorFlow.js**: Convert the model to TensorFlow.js format and run inference in the browser/Node.js
3. **VPS deployment**: Deploy to a VPS with both Node.js and Python support

## Development

### Project Structure Notes

- **ML Code**: All machine learning code is in the `ml/` directory
- **API Routes**: Next.js API routes handle file uploads and call Python scripts
- **Frontend**: React components in `app/page.tsx` provide the UI
- **Static Serving**: Grad-CAM images are served via API route (`/api/serve-gradcam`)

### Code Style

- **Python**: Follows PEP 8 style guidelines
- **TypeScript**: Uses strict type checking
- **React**: Functional components with hooks

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- Transfer learning approach inspired by modern deep learning best practices
- ResNet50 architecture from Keras Applications
- Grad-CAM implementation based on the original paper: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style guidelines
- All tests pass
- Documentation is updated

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review error messages in console/logs
3. Ensure all dependencies are correctly installed
