"""
YOLO Training Studio - UI Components
=====================================

Modular UI components for the training studio.
"""

from .dashboard import create_dashboard_tab
from .dataset_manager import create_dataset_tab
from .labeling_tool import create_labeling_tab
from .training_manager import create_training_tab
from .model_export import create_export_tab
from .inference import create_inference_tab

__all__ = [
    "create_dashboard_tab",
    "create_dataset_tab",
    "create_labeling_tab",
    "create_training_tab",
    "create_export_tab",
    "create_inference_tab",
]
