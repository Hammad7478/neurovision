#!/bin/bash

# Convenience script to train the NeuroVision model

set -e

echo "=========================================="
echo "NeuroVision Model Training Script"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade requirements
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Error: data directory not found"
    exit 1
fi

# Create model directory if it doesn't exist
mkdir -p model

# Run training
echo ""
echo "Starting training..."
echo "=========================================="
python3 ml/train_model.py

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="

