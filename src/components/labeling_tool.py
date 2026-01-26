"""
YOLO Training Studio - Labeling Tool Component
==============================================

Interactive image labeling tool for creating bounding box annotations.
"""

import gradio as gr
from pathlib import Path
import yaml
import json
from PIL import Image
import numpy as np
from typing import Optional


DATASETS_DIR = Path("datasets")
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class LabelingSession:
    """Manages the current labeling session state."""

    def __init__(self):
        self.dataset_path: Optional[Path] = None
        self.current_split: str = "train"
        self.images: list = []
        self.current_index: int = 0
        self.classes: list = []
        self.annotations: dict = {}  # filename -> list of boxes

    def load_dataset(self, dataset_name: str, split: str) -> str:
        """Load a dataset for labeling."""
        if not dataset_name:
            return "Please select a dataset."

        self.dataset_path = DATASETS_DIR / dataset_name
        self.current_split = split

        # Load classes from data.yaml
        yaml_path = self.dataset_path / "data.yaml"
        if yaml_path.exists():
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
                names = data.get("names", {})
                if isinstance(names, dict):
                    self.classes = list(names.values())
                else:
                    self.classes = names
        else:
            self.classes = ["object"]

        # Load images
        images_dir = self.dataset_path / "images" / split
        if images_dir.exists():
            self.images = sorted([
                f for f in images_dir.iterdir()
                if f.suffix.lower() in SUPPORTED_IMAGE_FORMATS
            ])
        else:
            self.images = []

        # Load existing annotations
        self.annotations = {}
        labels_dir = self.dataset_path / "labels" / split
        if labels_dir.exists():
            for label_file in labels_dir.glob("*.txt"):
                img_name = label_file.stem
                boxes = []
                with open(label_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            boxes.append({
                                "class_id": class_id,
                                "x_center": x_center,
                                "y_center": y_center,
                                "width": width,
                                "height": height
                            })
                self.annotations[img_name] = boxes

        self.current_index = 0
        return f"Loaded {len(self.images)} images with {len(self.classes)} classes."

    def get_current_image(self) -> tuple:
        """Get the current image and its annotations."""
        if not self.images:
            return None, [], "No images loaded."

        img_path = self.images[self.current_index]
        img = Image.open(img_path)

        # Get annotations for this image
        boxes = self.annotations.get(img_path.stem, [])

        status = f"Image {self.current_index + 1} of {len(self.images)}: {img_path.name}"
        return img, boxes, status

    def save_annotation(self, boxes: list) -> str:
        """Save annotations for the current image."""
        if not self.images:
            return "No images loaded."

        img_path = self.images[self.current_index]
        self.annotations[img_path.stem] = boxes

        # Save to file
        labels_dir = self.dataset_path / "labels" / self.current_split
        labels_dir.mkdir(parents=True, exist_ok=True)

        label_file = labels_dir / f"{img_path.stem}.txt"
        with open(label_file, "w") as f:
            for box in boxes:
                f.write(f"{box['class_id']} {box['x_center']:.6f} {box['y_center']:.6f} "
                       f"{box['width']:.6f} {box['height']:.6f}\n")

        return f"Saved {len(boxes)} annotations for {img_path.name}"

    def navigate(self, direction: int) -> tuple:
        """Navigate to the next or previous image."""
        if not self.images:
            return None, [], "No images loaded."

        self.current_index = (self.current_index + direction) % len(self.images)
        return self.get_current_image()


# Global session instance
labeling_session = LabelingSession()


def get_existing_datasets() -> list:
    """Get list of existing datasets."""
    if not DATASETS_DIR.exists():
        return []
    datasets = []
    for item in DATASETS_DIR.iterdir():
        if item.is_dir() and (item / "data.yaml").exists():
            datasets.append(item.name)
    return datasets


def load_dataset_for_labeling(dataset_name: str, split: str):
    """Load dataset and return initial state."""
    status = labeling_session.load_dataset(dataset_name, split)
    img, boxes, img_status = labeling_session.get_current_image()
    classes = labeling_session.classes

    return (
        status,
        img,
        gr.update(choices=classes, value=classes[0] if classes else None),
        img_status,
        format_annotations(boxes, classes)
    )


def navigate_image(direction: int):
    """Navigate images."""
    img, boxes, status = labeling_session.navigate(direction)
    return img, status, format_annotations(boxes, labeling_session.classes)


def format_annotations(boxes: list, classes: list) -> str:
    """Format annotations as HTML table."""
    if not boxes:
        return "<p style='color: #64748b; text-align: center;'>No annotations yet</p>"

    rows = []
    for i, box in enumerate(boxes):
        class_name = classes[box['class_id']] if box['class_id'] < len(classes) else f"Class {box['class_id']}"
        rows.append(f"""
            <tr>
                <td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0;">{i + 1}</td>
                <td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0;">
                    <span style="background: #e0e7ff; color: #3730a3; padding: 0.125rem 0.5rem;
                                 border-radius: 4px; font-size: 0.75rem;">{class_name}</span>
                </td>
                <td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0; font-family: monospace; font-size: 0.75rem;">
                    ({box['x_center']:.3f}, {box['y_center']:.3f})
                </td>
                <td style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0; font-family: monospace; font-size: 0.75rem;">
                    {box['width']:.3f} x {box['height']:.3f}
                </td>
            </tr>
        """)

    return f"""
    <table style="width: 100%; border-collapse: collapse; font-size: 0.875rem;">
        <thead>
            <tr style="background: #f8fafc;">
                <th style="padding: 0.5rem; text-align: left; border-bottom: 2px solid #e2e8f0;">#</th>
                <th style="padding: 0.5rem; text-align: left; border-bottom: 2px solid #e2e8f0;">Class</th>
                <th style="padding: 0.5rem; text-align: left; border-bottom: 2px solid #e2e8f0;">Center</th>
                <th style="padding: 0.5rem; text-align: left; border-bottom: 2px solid #e2e8f0;">Size</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def add_annotation(
    selected_class: str,
    x_center: float,
    y_center: float,
    width: float,
    height: float
):
    """Add a new annotation."""
    if not labeling_session.images:
        return "No images loaded.", ""

    classes = labeling_session.classes
    if selected_class not in classes:
        return "Invalid class selected.", ""

    class_id = classes.index(selected_class)

    # Validate values
    if not all(0 <= v <= 1 for v in [x_center, y_center, width, height]):
        return "Values must be between 0 and 1.", ""

    img_path = labeling_session.images[labeling_session.current_index]
    boxes = labeling_session.annotations.get(img_path.stem, [])
    boxes.append({
        "class_id": class_id,
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height
    })
    labeling_session.annotations[img_path.stem] = boxes

    return f"Added annotation for {selected_class}", format_annotations(boxes, classes)


def save_current_annotations():
    """Save annotations for the current image."""
    if not labeling_session.images:
        return "No images loaded."

    img_path = labeling_session.images[labeling_session.current_index]
    boxes = labeling_session.annotations.get(img_path.stem, [])
    return labeling_session.save_annotation(boxes)


def clear_annotations():
    """Clear annotations for the current image."""
    if not labeling_session.images:
        return "No images loaded.", ""

    img_path = labeling_session.images[labeling_session.current_index]
    labeling_session.annotations[img_path.stem] = []

    return "Cleared all annotations.", ""


def create_labeling_tab():
    """Create the labeling tool tab."""

    with gr.Column():
        gr.HTML("""
            <div style="margin-bottom: 1rem;">
                <h2 style="color: #1e293b; margin: 0;">Labeling Tool</h2>
                <p style="color: #64748b; margin-top: 0.5rem;">
                    Annotate images with bounding boxes for object detection training.
                </p>
            </div>
        """)

        # Dataset Selection
        with gr.Row():
            with gr.Column(scale=2):
                dataset_select = gr.Dropdown(
                    label="Select Dataset",
                    choices=get_existing_datasets(),
                    interactive=True,
                )
            with gr.Column(scale=1):
                split_select = gr.Radio(
                    label="Split",
                    choices=["train", "val", "test"],
                    value="train",
                )
            with gr.Column(scale=1):
                load_btn = gr.Button("📂 Load Dataset", variant="primary")

        load_status = gr.Textbox(label="Status", interactive=False)

        gr.HTML("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #e2e8f0;'>")

        with gr.Row():
            # Left Column - Image Display
            with gr.Column(scale=2):
                image_display = gr.Image(
                    label="Current Image",
                    type="pil",
                    interactive=False,
                    height=500,
                )

                with gr.Row():
                    prev_btn = gr.Button("⬅️ Previous", variant="secondary")
                    image_status = gr.Textbox(
                        label="",
                        interactive=False,
                        show_label=False,
                    )
                    next_btn = gr.Button("Next ➡️", variant="secondary")

            # Right Column - Annotation Controls
            with gr.Column(scale=1):
                gr.HTML("""
                    <h3 style="margin: 0 0 1rem 0; color: #1e293b;">
                        Add Annotation
                    </h3>
                """)

                class_select = gr.Dropdown(
                    label="Class",
                    choices=[],
                    interactive=True,
                )

                gr.HTML("""
                    <p style="font-size: 0.75rem; color: #64748b; margin: 0.5rem 0;">
                        Enter normalized coordinates (0-1):
                    </p>
                """)

                with gr.Row():
                    x_center = gr.Number(label="X Center", value=0.5, minimum=0, maximum=1, step=0.01)
                    y_center = gr.Number(label="Y Center", value=0.5, minimum=0, maximum=1, step=0.01)

                with gr.Row():
                    box_width = gr.Number(label="Width", value=0.2, minimum=0, maximum=1, step=0.01)
                    box_height = gr.Number(label="Height", value=0.2, minimum=0, maximum=1, step=0.01)

                add_btn = gr.Button("➕ Add Box", variant="primary")
                add_status = gr.Textbox(label="", interactive=False, show_label=False)

                gr.HTML("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #e2e8f0;'>")

                gr.HTML("""
                    <h3 style="margin: 0 0 1rem 0; color: #1e293b;">
                        Current Annotations
                    </h3>
                """)

                annotations_display = gr.HTML(
                    value="<p style='color: #64748b; text-align: center;'>No annotations yet</p>"
                )

                with gr.Row():
                    save_btn = gr.Button("💾 Save", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear All", variant="stop")

                save_status = gr.Textbox(label="", interactive=False, show_label=False)

        # Keyboard shortcuts info
        gr.HTML("""
            <div class="card" style="margin-top: 1rem; background: #f8fafc;">
                <h4 style="margin: 0 0 0.5rem 0; color: #1e293b; font-size: 0.875rem;">
                    Tips & Shortcuts
                </h4>
                <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.75rem; color: #64748b;">
                    <li>Coordinates are normalized (0-1) relative to image dimensions</li>
                    <li>X/Y Center: Center point of the bounding box</li>
                    <li>Width/Height: Size of the bounding box relative to image</li>
                    <li>For precise labeling, consider using external tools like LabelImg or CVAT</li>
                </ul>
            </div>
        """)

        # Event handlers
        load_btn.click(
            fn=load_dataset_for_labeling,
            inputs=[dataset_select, split_select],
            outputs=[load_status, image_display, class_select, image_status, annotations_display]
        )

        prev_btn.click(
            fn=lambda: navigate_image(-1),
            outputs=[image_display, image_status, annotations_display]
        )

        next_btn.click(
            fn=lambda: navigate_image(1),
            outputs=[image_display, image_status, annotations_display]
        )

        add_btn.click(
            fn=add_annotation,
            inputs=[class_select, x_center, y_center, box_width, box_height],
            outputs=[add_status, annotations_display]
        )

        save_btn.click(
            fn=save_current_annotations,
            outputs=[save_status]
        )

        clear_btn.click(
            fn=clear_annotations,
            outputs=[save_status, annotations_display]
        )
