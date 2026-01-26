"""
YOLO Training Studio - Helper Functions
=======================================

Common utility functions used throughout the application.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml
import os


SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}
SUPPORTED_VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def ensure_directory(path: Path | str) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_image_files(directory: Path | str, recursive: bool = False) -> List[Path]:
    """Get all image files in a directory."""
    directory = Path(directory)
    if not directory.exists():
        return []

    if recursive:
        files = []
        for ext in SUPPORTED_IMAGE_FORMATS:
            files.extend(directory.glob(f"**/*{ext}"))
            files.extend(directory.glob(f"**/*{ext.upper()}"))
    else:
        files = [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_FORMATS
        ]

    return sorted(files)


def get_video_files(directory: Path | str, recursive: bool = False) -> List[Path]:
    """Get all video files in a directory."""
    directory = Path(directory)
    if not directory.exists():
        return []

    if recursive:
        files = []
        for ext in SUPPORTED_VIDEO_FORMATS:
            files.extend(directory.glob(f"**/*{ext}"))
            files.extend(directory.glob(f"**/*{ext.upper()}"))
    else:
        files = [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_FORMATS
        ]

    return sorted(files)


def validate_dataset(dataset_path: Path | str) -> Dict[str, Any]:
    """
    Validate a YOLO dataset structure and return validation results.

    Returns:
        Dictionary with validation results including:
        - valid: bool
        - errors: list of error messages
        - warnings: list of warning messages
        - stats: dataset statistics
    """
    dataset_path = Path(dataset_path)
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {
            "train_images": 0,
            "val_images": 0,
            "test_images": 0,
            "total_labels": 0,
            "classes": [],
        }
    }

    # Check if directory exists
    if not dataset_path.exists():
        result["valid"] = False
        result["errors"].append(f"Dataset directory not found: {dataset_path}")
        return result

    # Check for data.yaml
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        result["valid"] = False
        result["errors"].append("data.yaml not found")
        return result

    # Load and validate data.yaml
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        if not data:
            result["valid"] = False
            result["errors"].append("data.yaml is empty")
            return result

        # Check required fields
        if "names" not in data:
            result["errors"].append("'names' field missing in data.yaml")
            result["valid"] = False

        if "nc" not in data:
            result["warnings"].append("'nc' field missing in data.yaml")

        # Extract classes
        names = data.get("names", {})
        if isinstance(names, dict):
            result["stats"]["classes"] = list(names.values())
        elif isinstance(names, list):
            result["stats"]["classes"] = names

    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Error parsing data.yaml: {str(e)}")
        return result

    # Check image and label directories
    for split in ["train", "val", "test"]:
        images_dir = dataset_path / "images" / split
        labels_dir = dataset_path / "labels" / split

        if images_dir.exists():
            images = get_image_files(images_dir)
            result["stats"][f"{split}_images"] = len(images)

            if labels_dir.exists():
                labels = list(labels_dir.glob("*.txt"))
                result["stats"]["total_labels"] += len(labels)

                # Check for missing labels
                image_stems = {img.stem for img in images}
                label_stems = {lbl.stem for lbl in labels}

                missing_labels = image_stems - label_stems
                if missing_labels and split != "test":
                    result["warnings"].append(
                        f"{len(missing_labels)} images in {split} missing labels"
                    )

    # Check if we have any training data
    if result["stats"]["train_images"] == 0:
        result["warnings"].append("No training images found")

    if result["stats"]["val_images"] == 0:
        result["warnings"].append("No validation images found")

    return result


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_gpu_info() -> List[Dict[str, Any]]:
    """Get information about available GPUs."""
    gpus = []

    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "id": i,
                    "name": props.name,
                    "memory_total": props.total_memory,
                    "memory_total_formatted": format_size(props.total_memory),
                })
    except ImportError:
        pass

    return gpus


def count_model_parameters(model_path: Path | str) -> Optional[int]:
    """Count the number of parameters in a PyTorch model."""
    try:
        import torch
        model = torch.load(model_path, map_location="cpu")
        if "model" in model:
            model = model["model"]
        total_params = sum(p.numel() for p in model.parameters())
        return total_params
    except Exception:
        return None
