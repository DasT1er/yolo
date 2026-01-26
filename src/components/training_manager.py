"""
YOLO Training Studio - Training Manager Component
=================================================

Handles model training configuration, execution, and monitoring.
Supports YOLO v5, v8, v11, and v26.
"""

import gradio as gr
from pathlib import Path
import yaml
import json
import threading
import time
from datetime import datetime
from typing import Optional
import subprocess
import sys


DATASETS_DIR = Path("datasets")
EXPORTS_DIR = Path("exports")
RUNS_DIR = Path("runs")


# Available YOLO models
YOLO_MODELS = {
    "YOLO26": {
        "variants": ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"],
        "description": "Latest (2026) - NMS-free, fastest CPU inference",
        "recommended": True,
    },
    "YOLO11": {
        "variants": ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
        "description": "2024 - Balanced accuracy and speed",
        "recommended": True,
    },
    "YOLOv8": {
        "variants": ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        "description": "2023 - Widely adopted, stable",
        "recommended": False,
    },
    "YOLOv5": {
        "variants": ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"],
        "description": "2020 - Legacy support",
        "recommended": False,
    },
}

# Model size descriptions
SIZE_DESCRIPTIONS = {
    "n": ("Nano", "Fastest, lowest accuracy, best for edge devices"),
    "s": ("Small", "Fast, good for real-time applications"),
    "m": ("Medium", "Balanced speed and accuracy"),
    "l": ("Large", "High accuracy, slower inference"),
    "x": ("Extra Large", "Highest accuracy, slowest, for server deployment"),
}


class TrainingManager:
    """Manages training sessions."""

    def __init__(self):
        self.current_training: Optional[subprocess.Popen] = None
        self.training_log: list = []
        self.is_training: bool = False
        self.training_config: dict = {}

    def start_training(
        self,
        dataset_name: str,
        model_variant: str,
        epochs: int,
        batch_size: int,
        img_size: int,
        learning_rate: float,
        optimizer: str,
        augmentation: bool,
        pretrained: bool,
        device: str,
        project_name: str,
    ) -> str:
        """Start a training session."""
        if self.is_training:
            return "Training already in progress!"

        # Validate inputs
        if not dataset_name:
            return "Error: Please select a dataset."

        dataset_path = DATASETS_DIR / dataset_name
        data_yaml = dataset_path / "data.yaml"

        if not data_yaml.exists():
            return f"Error: Dataset '{dataset_name}' not found or invalid."

        # Store config
        self.training_config = {
            "dataset": dataset_name,
            "model": model_variant,
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": img_size,
            "lr0": learning_rate,
            "optimizer": optimizer,
            "augment": augmentation,
            "pretrained": pretrained,
            "device": device,
            "project": project_name,
            "started_at": datetime.now().isoformat(),
        }

        self.training_log = []
        self.is_training = True

        # Build training command
        cmd = [
            sys.executable, "-m", "ultralytics",
            "detect", "train",
            f"data={data_yaml}",
            f"model={model_variant}.pt" if pretrained else f"model={model_variant}.yaml",
            f"epochs={epochs}",
            f"batch={batch_size}",
            f"imgsz={img_size}",
            f"lr0={learning_rate}",
            f"optimizer={optimizer}",
            f"augment={str(augmentation).lower()}",
            f"device={device}",
            f"project={RUNS_DIR / project_name}",
            f"name=train",
            "exist_ok=True",
        ]

        try:
            # Start training in subprocess
            self.current_training = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Start log reader thread
            log_thread = threading.Thread(target=self._read_training_log, daemon=True)
            log_thread.start()

            return f"Training started with {model_variant} on {dataset_name}!"

        except Exception as e:
            self.is_training = False
            return f"Error starting training: {str(e)}"

    def _read_training_log(self):
        """Read training output in background."""
        if self.current_training:
            for line in self.current_training.stdout:
                self.training_log.append(line.strip())
                if len(self.training_log) > 1000:
                    self.training_log = self.training_log[-500:]

            self.current_training.wait()
            self.is_training = False

    def stop_training(self) -> str:
        """Stop the current training session."""
        if not self.is_training or not self.current_training:
            return "No training in progress."

        try:
            self.current_training.terminate()
            self.current_training.wait(timeout=10)
            self.is_training = False
            return "Training stopped."
        except Exception as e:
            return f"Error stopping training: {str(e)}"

    def get_training_status(self) -> tuple:
        """Get current training status and logs."""
        if self.is_training:
            status = "🟢 Training in Progress"
            logs = "\n".join(self.training_log[-50:])
        else:
            status = "⚪ Idle"
            logs = "\n".join(self.training_log[-50:]) if self.training_log else "No training logs."

        return status, logs


# Global training manager
training_manager = TrainingManager()


def get_existing_datasets() -> list:
    """Get list of existing datasets."""
    if not DATASETS_DIR.exists():
        return []
    datasets = []
    for item in DATASETS_DIR.iterdir():
        if item.is_dir() and (item / "data.yaml").exists():
            datasets.append(item.name)
    return datasets


def get_model_variants(model_family: str) -> list:
    """Get available variants for a model family."""
    if model_family in YOLO_MODELS:
        return YOLO_MODELS[model_family]["variants"]
    return []


def create_model_selector_html() -> str:
    """Create HTML for model family selection."""
    cards = []
    for family, info in YOLO_MODELS.items():
        badge = '<span style="background: #dcfce7; color: #166534; padding: 0.125rem 0.5rem; border-radius: 9999px; font-size: 0.625rem; margin-left: 0.5rem;">RECOMMENDED</span>' if info["recommended"] else ''
        cards.append(f"""
            <div style="background: white; border: 2px solid #e2e8f0; border-radius: 8px;
                        padding: 1rem; cursor: pointer; transition: all 0.2s ease;"
                 onmouseover="this.style.borderColor='#667eea'"
                 onmouseout="this.style.borderColor='#e2e8f0'">
                <div style="font-weight: 600; color: #1e293b;">{family}{badge}</div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">{info["description"]}</div>
            </div>
        """)

    return f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem;">
        {''.join(cards)}
    </div>
    """


def start_training(
    dataset_name: str,
    model_family: str,
    model_size: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    learning_rate: float,
    optimizer: str,
    augmentation: bool,
    pretrained: bool,
    device: str,
    project_name: str,
):
    """Start training with the given configuration."""
    # Construct model variant name
    size_map = {"Nano": "n", "Small": "s", "Medium": "m", "Large": "l", "Extra Large": "x"}
    size_suffix = size_map.get(model_size, "n")

    if model_family == "YOLO26":
        model_variant = f"yolo26{size_suffix}"
    elif model_family == "YOLO11":
        model_variant = f"yolo11{size_suffix}"
    elif model_family == "YOLOv8":
        model_variant = f"yolov8{size_suffix}"
    else:
        model_variant = f"yolov5{size_suffix}"

    return training_manager.start_training(
        dataset_name=dataset_name,
        model_variant=model_variant,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        learning_rate=learning_rate,
        optimizer=optimizer,
        augmentation=augmentation,
        pretrained=pretrained,
        device=device,
        project_name=project_name,
    )


def stop_training():
    """Stop the current training."""
    return training_manager.stop_training()


def get_training_status():
    """Get current training status."""
    return training_manager.get_training_status()


def create_training_tab():
    """Create the training manager tab."""

    with gr.Column():
        gr.HTML("""
            <div style="margin-bottom: 1rem;">
                <h2 style="color: #1e293b; margin: 0;">Training Manager</h2>
                <p style="color: #64748b; margin-top: 0.5rem;">
                    Configure and monitor YOLO model training sessions.
                </p>
            </div>
        """)

        with gr.Row():
            # Left Column - Configuration
            with gr.Column(scale=2):
                gr.HTML("<h3 style='margin: 0 0 1rem 0; color: #1e293b;'>Training Configuration</h3>")

                # Dataset Selection
                dataset_select = gr.Dropdown(
                    label="Dataset",
                    choices=get_existing_datasets(),
                    info="Select the dataset to train on",
                )

                refresh_datasets_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm")

                gr.HTML("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #e2e8f0;'>")

                # Model Selection
                gr.HTML("<h4 style='margin: 0 0 0.75rem 0; color: #475569;'>Model Selection</h4>")

                gr.HTML(create_model_selector_html())

                model_family = gr.Dropdown(
                    label="Model Family",
                    choices=list(YOLO_MODELS.keys()),
                    value="YOLO26",
                    info="Select the YOLO version",
                )

                model_size = gr.Radio(
                    label="Model Size",
                    choices=["Nano", "Small", "Medium", "Large", "Extra Large"],
                    value="Nano",
                    info="Larger = more accurate but slower",
                )

                gr.HTML("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #e2e8f0;'>")

                # Training Parameters
                gr.HTML("<h4 style='margin: 0 0 0.75rem 0; color: #475569;'>Training Parameters</h4>")

                with gr.Row():
                    epochs = gr.Slider(
                        label="Epochs",
                        minimum=1,
                        maximum=500,
                        value=100,
                        step=1,
                        info="Number of training epochs",
                    )

                    batch_size = gr.Slider(
                        label="Batch Size",
                        minimum=1,
                        maximum=128,
                        value=16,
                        step=1,
                        info="Images per batch",
                    )

                with gr.Row():
                    img_size = gr.Dropdown(
                        label="Image Size",
                        choices=[320, 416, 512, 640, 768, 1024, 1280],
                        value=640,
                        info="Input image resolution",
                    )

                    learning_rate = gr.Number(
                        label="Learning Rate",
                        value=0.01,
                        minimum=0.0001,
                        maximum=0.1,
                        step=0.001,
                    )

                with gr.Row():
                    optimizer = gr.Dropdown(
                        label="Optimizer",
                        choices=["SGD", "Adam", "AdamW", "RMSProp"],
                        value="SGD",
                    )

                    device = gr.Dropdown(
                        label="Device",
                        choices=["0", "cpu", "0,1", "0,1,2,3"],
                        value="0",
                        info="GPU ID or 'cpu'",
                    )

                with gr.Row():
                    augmentation = gr.Checkbox(
                        label="Data Augmentation",
                        value=True,
                        info="Enable training augmentations",
                    )

                    pretrained = gr.Checkbox(
                        label="Use Pretrained Weights",
                        value=True,
                        info="Start from COCO pretrained weights",
                    )

                project_name = gr.Textbox(
                    label="Project Name",
                    value="yolo_training",
                    info="Name for the training run",
                )

                gr.HTML("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #e2e8f0;'>")

                with gr.Row():
                    start_btn = gr.Button("🚀 Start Training", variant="primary", scale=2)
                    stop_btn = gr.Button("⏹️ Stop", variant="stop", scale=1)

            # Right Column - Monitoring
            with gr.Column(scale=1):
                gr.HTML("<h3 style='margin: 0 0 1rem 0; color: #1e293b;'>Training Monitor</h3>")

                training_status = gr.Textbox(
                    label="Status",
                    value="⚪ Idle",
                    interactive=False,
                )

                refresh_status_btn = gr.Button("🔄 Refresh Status", variant="secondary", size="sm")

                training_logs = gr.Textbox(
                    label="Training Logs",
                    lines=20,
                    max_lines=30,
                    interactive=False,
                )

                # Training Tips
                gr.HTML("""
                    <div class="card" style="margin-top: 1rem; background: #f0fdf4; border-color: #bbf7d0;">
                        <h4 style="margin: 0 0 0.5rem 0; color: #166534; font-size: 0.875rem;">
                            💡 Training Tips
                        </h4>
                        <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.75rem; color: #15803d;">
                            <li>Start with Nano model for quick experiments</li>
                            <li>Use pretrained weights for faster convergence</li>
                            <li>Larger batch sizes need more VRAM</li>
                            <li>100 epochs is a good starting point</li>
                            <li>Monitor val/loss to detect overfitting</li>
                        </ul>
                    </div>
                """)

        start_status = gr.Textbox(label="", interactive=False, visible=False)

        # Event handlers
        refresh_datasets_btn.click(
            fn=lambda: gr.update(choices=get_existing_datasets()),
            outputs=[dataset_select]
        )

        start_btn.click(
            fn=start_training,
            inputs=[
                dataset_select, model_family, model_size, epochs, batch_size,
                img_size, learning_rate, optimizer, augmentation, pretrained,
                device, project_name
            ],
            outputs=[start_status]
        )

        stop_btn.click(
            fn=stop_training,
            outputs=[training_status]
        )

        refresh_status_btn.click(
            fn=get_training_status,
            outputs=[training_status, training_logs]
        )
