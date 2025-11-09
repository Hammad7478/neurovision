#!/bin/bash

# Script to download pre-trained model from a URL
# Usage: MODEL_URL=https://... ./scripts/download-model.sh

set -e

MODEL_DIR="./model"
MODEL_PATH="$MODEL_DIR/model.h5"

# Check if model URL is set
if [ -z "$MODEL_URL" ]; then
    echo "Error: MODEL_URL environment variable not set"
    echo "Usage: MODEL_URL=https://your-url.com/model.h5 ./scripts/download-model.sh"
    exit 1
fi

# Check if model already exists
if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists at $MODEL_PATH"
    ls -lh "$MODEL_PATH"
    exit 0
fi

# Create model directory
mkdir -p "$MODEL_DIR"

# Download model
echo "Downloading model from $MODEL_URL..."
echo "Destination: $MODEL_PATH"

if command -v curl &> /dev/null; then
    curl -L -o "$MODEL_PATH" "$MODEL_URL"
elif command -v wget &> /dev/null; then
    wget -O "$MODEL_PATH" "$MODEL_URL"
else
    echo "Error: Neither curl nor wget is available"
    exit 1
fi

# Verify download
if [ -f "$MODEL_PATH" ]; then
    file_size=$(ls -lh "$MODEL_PATH" | awk '{print $5}')
    echo "✓ Model downloaded successfully!"
    echo "  Location: $MODEL_PATH"
    echo "  Size: $file_size"
else
    echo "✗ Download failed"
    exit 1
fi

