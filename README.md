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

2. **Train the ResNet-50 model**

   - Default: up to 25 epochs, early stop when `val_accuracy >= 0.95`
   - CLI flags (see below) let you change augmentation, class weights, color mode, fine-tuning, epochs, and target accuracy.

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

1. **Train the model once** (default: 25 epochs cap, early stop at 0.95):

   ```bash
   python ml/train_resnet50.py
   ```

2. **The model is saved to `model/resnet50_model.h5`** (or with suffix if specified) and persists on disk.

3. **Use the model for predictions** - no need to retrain unless you want to run ablations.

### Training the Model (ResNet-50)

**Note**: Default behavior matches the main model in the report (25-epoch cap, early stop at 0.95). You can override with CLI flags for ablations.

#### Command

```bash
python ml/train_resnet50.py [--augmentation {on,off}] [--class-weights {on,off}] \
  [--color-mode {rgb,grayscale}] [--finetune-mode {partial,frozen}] \
  [--epochs N] [--target-accuracy X.XX] \
  [--data-dir ./data] [--model-dir ./model] \
  [--batch-size 32] [--output-suffix SUFFIX]
```

#### What it does

- Loads images from `data/<class>` directories.
- Splits into train/validation (80/20 stratified) in code.
- Applies augmentation (flip/rotation/brightness/contrast) when `--augmentation on`.
- Supports RGB or grayscale inputs (`--color-mode`).
- Supports weighted or unweighted loss (`--class-weights`).
- Fine-tuning modes: `partial` (two-phase; unfreeze top ~30 layers) or `frozen` (head only).
- Early stopping: custom `StopAtValAccuracy` at `--target-accuracy` (default 0.95) plus Keras `EarlyStopping` (patience 10).
- Saves best model and metrics to `model/`, optionally suffixed.

#### Key outputs (suffix honored if provided)

- `model/resnet50_model[_{suffix}].h5`
- `model/resnet50_metrics[_{suffix}].json`
- `model/resnet50_confusion_matrix[_{suffix}].png`
- `model/resnet50_weights.weights.h5`

### Ablation sweeps (batch runner)

We provide a helper to run the key ablation configs quickly (8-epoch cap, 0.85 target accuracy by default):

```bash
python -m ml.run_ablation_sweeps
```

It runs:

- Baseline: aug on, class weights on, RGB, partial FT
- No augmentation
- No class weights
- Grayscale inputs
- Frozen backbone

Outputs per run:

- `model/resnet50_model_{config}.h5`
- `model/resnet50_metrics_{config}.json`
- `model/resnet50_confusion_matrix_{config}.png`
  Summary:
- `ablations/ablation_*.json` per run
- `ablations/ablation_summary.json`

### MobileNetV2 baseline

```bash
python ml/train_mobilenetv2.py
```

- Uses ImageNet-pretrained MobileNetV2, classifier head for 4 classes.
- Default: 80/20 split, augmentation on, class weights on.
- Outputs: `model/mobilenetv2_model.h5`, `model/mobilenetv2_metrics.json`, `model/mobilenetv2_confusion_matrix.png`

### Logistic regression baseline

```bash
python ml/baseline_logreg.py
```

- Flattens grayscale images, trains a linear classifier.
- Outputs metrics to stdout/logs (no deep model files).

## Model Architecture (ResNet-50)

- Base: ResNet50 (ImageNet pretrained), input 224×224×3.
- Head: GAP → Dense 512 (ReLU) → Dropout 0.5 → Dense 256 (ReLU) → Dropout 0.3 → Dense 4 (softmax) → Dropout 0.2 after 256.
- Two-phase fine-tuning (when `--finetune-mode partial`): Phase 1 frozen base; Phase 2 unfreeze top ~30 layers, lower LR.
- Loss: categorical cross-entropy (weighted or unweighted).
- Optimizer: Adam (default 0.001/0.0001 across phases; 0.0001 in default single compile).
- Early stopping: target accuracy (default 0.95) + patience 10 safety net.

## Frontend / API

- Next.js 14 + React 18 + Tailwind.
- UI in `app/page.tsx` with model selector (ResNet-50, MobileNetV2), upload, probabilities, Grad-CAM image, metrics charts.
- API routes:
  - `POST /api/predict` → calls Python prediction script, returns class probabilities and optional Grad-CAM.
  - `GET /api/serve-gradcam/[filename]` → serves Grad-CAM images.
  - Training/metrics endpoints to poll/start training (see `app/api/*`).

## Data layout

- Place data under `data/<class>` with folders: `glioma`, `meningioma`, `pituitary`, `notumor`.
- Scripts perform an in-code 80/20 stratified split for train/validation.

## Flags reference (ml/train_resnet50.py)

- `--augmentation {on,off}` (default on)
- `--class-weights {on,off}` (default on)
- `--color-mode {rgb,grayscale}` (default rgb)
- `--finetune-mode {partial,frozen}` (default partial)
- `--epochs N` (default 25)
- `--target-accuracy X.XX` (default 0.95)
- `--data-dir PATH` (default ./data)
- `--model-dir PATH` (default ./model)
- `--batch-size N` (default 32)
- `--output-suffix SUFFIX` (adds `_SUFFIX` to outputs)

## Typical validation results (8-epoch ablation, target 0.85)

- ResNet-50 grayscale: ~90.0% val accuracy
- Frozen backbone: ~88.6% val accuracy
- No augmentation: ~88.5% val accuracy
- No class weights: ~86.8% val accuracy
- Baseline config (aug on, class weights on, RGB, partial FT): ~86.2% val accuracy

Earlier higher numbers (95%+) were from longer full-length training runs (25-epoch cap, target 0.95).

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
