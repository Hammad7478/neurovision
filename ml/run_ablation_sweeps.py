"""
Ablation experiment driver script for NeuroVision ResNet-50 training.

This script runs a set of ablation experiments by training models with different
configurations and saving the results to ./ablations/ directory.

Each ablation configuration tests one dimension:
- Baseline: augmentation=on, class_weights=on, color_mode=rgb, finetune_mode=partial
- No augmentation: augmentation=off, class_weights=on, color_mode=rgb, finetune_mode=partial
- No class weights: augmentation=on, class_weights=off, color_mode=rgb, finetune_mode=partial
- Grayscale inputs: augmentation=on, class_weights=on, color_mode=grayscale, finetune_mode=partial
- Fully frozen ResNet: augmentation=on, class_weights=on, color_mode=rgb, finetune_mode=frozen
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any
import time

# Directory for ablation results
ABLATION_DIR = Path("./ablations")
ABLATION_DIR.mkdir(exist_ok=True)


def get_config_name(config: Dict[str, str]) -> str:
    """Generate a short name for a configuration."""
    aug = "aug-on" if config["augmentation"] == "on" else "aug-off"
    cw = "class-on" if config["class_weights"] == "on" else "class-off"
    color = config["color_mode"]
    ft = config["finetune_mode"]
    return f"{aug}_{cw}_{color}_{ft}"


def run_training(config: Dict[str, str], output_suffix: str) -> bool:
    """
    Run training with a specific configuration.
    
    Args:
        config: Dictionary with ablation configuration
        output_suffix: Suffix for output files
        
    Returns:
        True if training succeeded, False otherwise
    """
    print("\n" + "=" * 80)
    print(f"Running ablation: {get_config_name(config)}")
    print("=" * 80)
    
    # Build command
    # For ablation experiments, use lower target accuracy (85%) and fewer epochs (8)
    # This speeds up experiments while still getting meaningful comparisons
    cmd = [
        sys.executable,
        "-m", "ml.train_resnet50",
        "--augmentation", config["augmentation"],
        "--class-weights", config["class_weights"],
        "--color-mode", config["color_mode"],
        "--finetune-mode", config["finetune_mode"],
        "--epochs", "8",
        "--target-accuracy", "0.85",
        "--output-suffix", output_suffix
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run training
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Training failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"\nERROR: Exception during training: {e}")
        return False


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Load metrics from a JSON file."""
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load metrics from {metrics_path}: {e}")
        return {}


def save_ablation_result(config: Dict[str, str], metrics: Dict[str, Any], 
                        output_suffix: str):
    """Save ablation result to JSON file."""
    config_name = get_config_name(config)
    result = {
        "config": config,
        "config_name": config_name,
        "metrics": {
            "accuracy": metrics.get("accuracy", None),
            "precision": metrics.get("precision", None),
            "recall": metrics.get("recall", None),
            "f1_score": metrics.get("f1_score", None),
            "per_class_f1": metrics.get("per_class_f1", {}),
        },
        "training_history": {
            "train_accuracy": metrics.get("train_accuracy", []),
            "val_accuracy": metrics.get("val_accuracy", []),
            "train_loss": metrics.get("train_loss", []),
            "val_loss": metrics.get("val_loss", []),
        }
    }
    
    # Save to ablation directory
    output_file = ABLATION_DIR / f"ablation_{config_name}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nAblation result saved to: {output_file}")
    return output_file


def main():
    """Run all ablation experiments."""
    print("=" * 80)
    print("NeuroVision Ablation Experiments")
    print("=" * 80)
    print(f"\nResults will be saved to: {ABLATION_DIR.absolute()}")
    
    # Define ablation configurations
    # Each configuration tests one dimension against the baseline
    ablation_configs = [
        {
            "name": "baseline",
            "config": {
                "augmentation": "on",
                "class_weights": "on",
                "color_mode": "rgb",
                "finetune_mode": "partial"
            }
        },
        {
            "name": "no_augmentation",
            "config": {
                "augmentation": "off",
                "class_weights": "on",
                "color_mode": "rgb",
                "finetune_mode": "partial"
            }
        },
        {
            "name": "no_class_weights",
            "config": {
                "augmentation": "on",
                "class_weights": "off",
                "color_mode": "rgb",
                "finetune_mode": "partial"
            }
        },
        {
            "name": "grayscale",
            "config": {
                "augmentation": "on",
                "class_weights": "on",
                "color_mode": "grayscale",
                "finetune_mode": "partial"
            }
        },
        {
            "name": "frozen_backbone",
            "config": {
                "augmentation": "on",
                "class_weights": "on",
                "color_mode": "rgb",
                "finetune_mode": "frozen"
            }
        },
    ]
    
    print(f"\nTotal ablation experiments: {len(ablation_configs)}")
    print("\nConfigurations:")
    for i, ablation in enumerate(ablation_configs, 1):
        config = ablation["config"]
        print(f"  {i}. {ablation['name']}: {get_config_name(config)}")
    
    # Ask for confirmation
    print("\n" + "-" * 80)
    response = input("Start ablation experiments? (y/n): ").strip().lower()
    if response != "y":
        print("Aborted.")
        return
    
    # Run each ablation
    results = []
    start_time = time.time()
    
    for i, ablation in enumerate(ablation_configs, 1):
        config = ablation["config"]
        config_name = ablation["name"]
        # Note: train_resnet50.py adds underscore prefix, so we pass without it
        output_suffix = config_name
        
        print(f"\n{'=' * 80}")
        print(f"Experiment {i}/{len(ablation_configs)}: {ablation['name']}")
        print(f"{'=' * 80}")
        
        # Run training
        success = run_training(config, output_suffix)
        
        if success:
            # Load metrics from the training output
            # The metrics file will be saved as resnet50_metrics_{output_suffix}.json
            # train_resnet50.py adds underscore prefix to suffix
            metrics_path = Path("./model") / f"resnet50_metrics_{output_suffix}.json"
            
            if metrics_path.exists():
                metrics = load_metrics(metrics_path)
                result_file = save_ablation_result(config, metrics, output_suffix)
                results.append({
                    "name": ablation["name"],
                    "config": config,
                    "success": True,
                    "result_file": str(result_file),
                    "metrics": metrics.get("accuracy", None)
                })
                print(f"✓ Completed: {ablation['name']} (Accuracy: {metrics.get('accuracy', 'N/A')})")
            else:
                print(f"⚠ Warning: Metrics file not found at {metrics_path}")
                results.append({
                    "name": ablation["name"],
                    "config": config,
                    "success": False,
                    "error": "Metrics file not found"
                })
        else:
            print(f"✗ Failed: {ablation['name']}")
            results.append({
                "name": ablation["name"],
                "config": config,
                "success": False,
                "error": "Training failed"
            })
    
    # Save summary
    elapsed_time = time.time() - start_time
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_experiments": len(ablation_configs),
        "successful": sum(1 for r in results if r.get("success", False)),
        "failed": sum(1 for r in results if not r.get("success", False)),
        "elapsed_time_seconds": elapsed_time,
        "results": results
    }
    
    summary_file = ABLATION_DIR / "ablation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Ablation Experiments Summary")
    print("=" * 80)
    print(f"Total experiments: {len(ablation_configs)}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Elapsed time: {elapsed_time / 60:.1f} minutes")
    print(f"\nSummary saved to: {summary_file}")
    
    # Print results table
    print("\nResults:")
    print("-" * 80)
    print(f"{'Experiment':<25} {'Status':<12} {'Accuracy':<12}")
    print("-" * 80)
    for result in results:
        name = result["name"]
        status = "✓ Success" if result.get("success", False) else "✗ Failed"
        accuracy = result.get("metrics")
        if accuracy is not None:
            acc_str = f"{accuracy:.4f}"
        else:
            acc_str = "N/A"
        print(f"{name:<25} {status:<12} {acc_str:<12}")
    print("-" * 80)


if __name__ == "__main__":
    main()

