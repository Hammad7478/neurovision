"""
Download pre-trained model from a URL if it doesn't exist locally.
This is useful for deployment scenarios where you want to use a pre-trained model
without training on the server.
"""

import os
import sys
import urllib.request
from pathlib import Path
import hashlib

MODEL_DIR = Path("./model")
MODEL_PATH = MODEL_DIR / "model.h5"

# Get model URL from environment variable or use default
MODEL_URL = os.getenv("MODEL_URL", "")
MODEL_CHECKSUM = os.getenv("MODEL_CHECKSUM", "")  # Optional: MD5 or SHA256 checksum


def calculate_file_hash(filepath: Path, algorithm: str = "md5") -> str:
    """Calculate file hash for verification."""
    hash_obj = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def download_model(url: str, destination: Path, verify_checksum: bool = False, expected_checksum: str = ""):
    """
    Download model from URL to destination.
    
    Args:
        url: URL to download model from
        destination: Path to save the model
        verify_checksum: Whether to verify file checksum
        expected_checksum: Expected checksum value (MD5 or SHA256)
    """
    print(f"Downloading model from {url}...")
    print(f"Destination: {destination}")
    
    # Create model directory if it doesn't exist
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download with progress
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\rProgress: {percent:.1f}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, destination, show_progress)
        print("\nDownload completed!")
        
        # Verify checksum if provided
        if verify_checksum and expected_checksum:
            print("Verifying file integrity...")
            file_size = destination.stat().st_size
            print(f"File size: {file_size / (1024*1024):.2f} MB")
            
            # Determine hash algorithm from checksum length
            if len(expected_checksum) == 32:
                algorithm = "md5"
            elif len(expected_checksum) == 64:
                algorithm = "sha256"
            else:
                print("Warning: Unknown checksum format, skipping verification")
                return
            
            actual_checksum = calculate_file_hash(destination, algorithm)
            if actual_checksum.lower() == expected_checksum.lower():
                print(f"✓ Checksum verified ({algorithm})")
            else:
                print(f"✗ Checksum mismatch!")
                print(f"  Expected: {expected_checksum}")
                print(f"  Actual:   {actual_checksum}")
                destination.unlink()  # Delete corrupted file
                raise ValueError("Model file checksum verification failed")
        
        print(f"Model saved to {destination}")
        
    except Exception as e:
        print(f"\nError downloading model: {e}")
        if destination.exists():
            destination.unlink()  # Clean up partial download
        raise


def main():
    """
    Main function to download model if it doesn't exist.
    """
    # Check if model already exists
    if MODEL_PATH.exists():
        print(f"Model already exists at {MODEL_PATH}")
        file_size = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"File size: {file_size:.2f} MB")
        return
    
    # Check if URL is provided
    if not MODEL_URL:
        print("No MODEL_URL environment variable set.")
        print("Set MODEL_URL to download a pre-trained model, or train locally with:")
        print("  python3 ml/train_model.py")
        sys.exit(1)
    
    # Download model
    verify = bool(MODEL_CHECKSUM)
    download_model(MODEL_URL, MODEL_PATH, verify_checksum=verify, expected_checksum=MODEL_CHECKSUM)
    print("Model is ready to use!")


if __name__ == "__main__":
    main()

