"""
YOLO Training Studio - Model Utilities
=====================================

Model-related utilities and wrappers.
"""

# Model architecture information for supported YOLO versions
YOLO_ARCHITECTURES = {
    "YOLO26": {
        "release_year": 2026,
        "description": "Latest Ultralytics model with NMS-free inference and optimized edge deployment",
        "features": [
            "End-to-end NMS-free inference",
            "43% faster CPU inference",
            "MuSGD optimizer",
            "Progressive Loss Balancing",
            "Small-Target-Aware Label Assignment",
        ],
        "tasks": ["detect", "segment", "classify", "pose", "obb"],
        "sizes": {
            "n": {"params": "2.6M", "flops": "6.5G", "map": "40.3%"},
            "s": {"params": "9.4M", "flops": "21.5G", "map": "46.2%"},
            "m": {"params": "20.1M", "flops": "67.0G", "map": "51.4%"},
            "l": {"params": "43.7M", "flops": "135.0G", "map": "53.5%"},
            "x": {"params": "68.2M", "flops": "258.1G", "map": "54.7%"},
        },
    },
    "YOLO11": {
        "release_year": 2024,
        "description": "Improved backbone and neck architecture with enhanced feature extraction",
        "features": [
            "22% fewer parameters than YOLOv8",
            "Improved feature extraction",
            "Multi-platform deployment",
            "Enhanced accuracy",
        ],
        "tasks": ["detect", "segment", "classify", "pose", "obb"],
        "sizes": {
            "n": {"params": "2.6M", "flops": "6.5G", "map": "39.5%"},
            "s": {"params": "9.4M", "flops": "21.5G", "map": "47.0%"},
            "m": {"params": "20.1M", "flops": "68.0G", "map": "51.5%"},
            "l": {"params": "25.3M", "flops": "86.9G", "map": "53.4%"},
            "x": {"params": "56.9M", "flops": "194.9G", "map": "54.7%"},
        },
    },
    "YOLOv8": {
        "release_year": 2023,
        "description": "Anchor-free detection with decoupled head and mosaic augmentation",
        "features": [
            "Anchor-free detection",
            "Decoupled head",
            "Advanced augmentations",
            "Widely adopted",
        ],
        "tasks": ["detect", "segment", "classify", "pose", "obb"],
        "sizes": {
            "n": {"params": "3.2M", "flops": "8.7G", "map": "37.3%"},
            "s": {"params": "11.2M", "flops": "28.6G", "map": "44.9%"},
            "m": {"params": "25.9M", "flops": "78.9G", "map": "50.2%"},
            "l": {"params": "43.7M", "flops": "165.2G", "map": "52.9%"},
            "x": {"params": "68.2M", "flops": "257.8G", "map": "53.9%"},
        },
    },
    "YOLOv5": {
        "release_year": 2020,
        "description": "Classic YOLO architecture with extensive community support",
        "features": [
            "Mature and stable",
            "Extensive documentation",
            "Large community",
            "Many pre-trained models",
        ],
        "tasks": ["detect", "segment", "classify"],
        "sizes": {
            "n": {"params": "1.9M", "flops": "4.5G", "map": "28.0%"},
            "s": {"params": "7.2M", "flops": "16.5G", "map": "37.4%"},
            "m": {"params": "21.2M", "flops": "49.0G", "map": "45.4%"},
            "l": {"params": "46.5M", "flops": "109.1G", "map": "49.0%"},
            "x": {"params": "86.7M", "flops": "205.7G", "map": "50.7%"},
        },
    },
}


def get_model_info(model_family: str, size: str = "n") -> dict:
    """Get information about a specific model variant."""
    if model_family not in YOLO_ARCHITECTURES:
        return {}

    arch = YOLO_ARCHITECTURES[model_family]
    size_info = arch["sizes"].get(size, {})

    return {
        "family": model_family,
        "size": size,
        "release_year": arch["release_year"],
        "description": arch["description"],
        "features": arch["features"],
        "tasks": arch["tasks"],
        **size_info,
    }


def get_recommended_model(task: str = "detect", priority: str = "balanced") -> str:
    """
    Get recommended model based on task and priority.

    Args:
        task: Task type (detect, segment, classify, pose, obb)
        priority: One of 'speed', 'accuracy', or 'balanced'

    Returns:
        Recommended model name (e.g., 'yolo26n')
    """
    if priority == "speed":
        return "yolo26n"
    elif priority == "accuracy":
        return "yolo26x"
    else:
        return "yolo26s"
