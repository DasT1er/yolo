"""
YOLO Training Studio - Inference/Testing Component
=================================================

Test trained models with images, videos, or webcam.
Visualize detection results and performance metrics.
"""

import gradio as gr
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile
import os
from typing import Optional, Tuple
import time


RUNS_DIR = Path("runs")
EXPORTS_DIR = Path("exports")


def get_available_models() -> list:
    """Get list of available models for inference."""
    models = []

    # PyTorch models from training runs
    if RUNS_DIR.exists():
        for pt_file in RUNS_DIR.glob("**/weights/*.pt"):
            models.append(str(pt_file))

    # Exported models
    if EXPORTS_DIR.exists():
        for model_file in EXPORTS_DIR.glob("**/*"):
            if model_file.suffix in [".pt", ".onnx", ".engine", ".tflite"]:
                models.append(str(model_file))

    # Add pretrained models option
    pretrained = [
        "yolo26n.pt", "yolo26s.pt", "yolo26m.pt",
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
    ]
    models.extend(pretrained)

    return sorted(set(models))


def run_inference(
    model_path: str,
    image: Image.Image,
    confidence: float,
    iou_threshold: float,
    max_detections: int,
    show_labels: bool,
    show_confidence: bool,
    line_width: int,
) -> Tuple[Optional[Image.Image], str, str]:
    """
    Run inference on an image using the selected model.

    Returns:
        Tuple of (annotated_image, detection_results, performance_metrics)
    """
    if not model_path:
        return None, "Error: Please select a model.", ""

    if image is None:
        return None, "Error: Please upload an image.", ""

    try:
        from ultralytics import YOLO

        # Load model
        start_load = time.time()
        model = YOLO(model_path)
        load_time = time.time() - start_load

        # Run inference
        start_inference = time.time()
        results = model.predict(
            source=image,
            conf=confidence,
            iou=iou_threshold,
            max_det=max_detections,
            verbose=False,
        )
        inference_time = time.time() - start_inference

        # Get the result
        result = results[0]

        # Draw annotations on image
        annotated = result.plot(
            labels=show_labels,
            conf=show_confidence,
            line_width=line_width,
        )

        # Convert to PIL Image
        annotated_pil = Image.fromarray(annotated[..., ::-1])  # BGR to RGB

        # Format detection results
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detections.append({
                    "class": cls_name,
                    "confidence": f"{conf:.2%}",
                    "bbox": [round(x, 1) for x in xyxy],
                })

        # Format results text
        if detections:
            results_text = f"Found {len(detections)} object(s):\n\n"
            for i, det in enumerate(detections, 1):
                results_text += f"{i}. {det['class']} ({det['confidence']})\n"
                results_text += f"   Box: {det['bbox']}\n"
        else:
            results_text = "No objects detected."

        # Format metrics
        metrics_text = f"""
Performance Metrics:
-------------------
Model Load Time: {load_time*1000:.1f} ms
Inference Time: {inference_time*1000:.1f} ms
Total Time: {(load_time + inference_time)*1000:.1f} ms
FPS (inference only): {1/inference_time:.1f}

Image Size: {image.size[0]} x {image.size[1]}
Detections: {len(detections)}
        """.strip()

        return annotated_pil, results_text, metrics_text

    except ImportError:
        return None, "Error: Ultralytics not installed. Run: pip install ultralytics", ""
    except Exception as e:
        return None, f"Error during inference: {str(e)}", ""


def run_batch_inference(
    model_path: str,
    images: list,
    confidence: float,
    iou_threshold: float,
) -> Tuple[list, str]:
    """Run inference on multiple images."""
    if not model_path:
        return [], "Error: Please select a model."

    if not images:
        return [], "Error: No images uploaded."

    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        results_gallery = []
        total_detections = 0

        for img_file in images:
            if img_file is None:
                continue

            img_path = img_file.name if hasattr(img_file, 'name') else img_file
            results = model.predict(
                source=img_path,
                conf=confidence,
                iou=iou_threshold,
                verbose=False,
            )

            result = results[0]
            annotated = result.plot()
            annotated_pil = Image.fromarray(annotated[..., ::-1])

            results_gallery.append(annotated_pil)
            total_detections += len(result.boxes) if result.boxes is not None else 0

        summary = f"Processed {len(results_gallery)} images with {total_detections} total detections."
        return results_gallery, summary

    except Exception as e:
        return [], f"Error: {str(e)}"


def create_inference_tab():
    """Create the inference/testing tab."""

    with gr.Column():
        gr.HTML("""
            <div style="margin-bottom: 1rem;">
                <h2 style="color: #1e293b; margin: 0;">Model Testing</h2>
                <p style="color: #64748b; margin-top: 0.5rem;">
                    Test your trained models on images and visualize detection results.
                </p>
            </div>
        """)

        with gr.Tabs():
            # Single Image Tab
            with gr.Tab("Single Image"):
                with gr.Row():
                    # Left Column - Input
                    with gr.Column(scale=1):
                        gr.HTML("<h3 style='margin: 0 0 1rem 0; color: #1e293b;'>Input</h3>")

                        model_select = gr.Dropdown(
                            label="Select Model",
                            choices=get_available_models(),
                            info="Choose a trained model or pretrained weights",
                        )

                        refresh_btn = gr.Button("🔄 Refresh Models", variant="secondary", size="sm")

                        input_image = gr.Image(
                            label="Upload Image",
                            type="pil",
                            height=300,
                        )

                        gr.HTML("<h4 style='margin: 1rem 0 0.5rem 0; color: #475569;'>Detection Settings</h4>")

                        confidence = gr.Slider(
                            label="Confidence Threshold",
                            minimum=0.01,
                            maximum=1.0,
                            value=0.25,
                            step=0.01,
                            info="Minimum confidence for detections",
                        )

                        iou_threshold = gr.Slider(
                            label="IoU Threshold (NMS)",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.45,
                            step=0.05,
                            info="Intersection over Union threshold for NMS",
                        )

                        max_detections = gr.Slider(
                            label="Max Detections",
                            minimum=1,
                            maximum=300,
                            value=100,
                            step=1,
                        )

                        gr.HTML("<h4 style='margin: 1rem 0 0.5rem 0; color: #475569;'>Visualization</h4>")

                        with gr.Row():
                            show_labels = gr.Checkbox(label="Show Labels", value=True)
                            show_confidence = gr.Checkbox(label="Show Confidence", value=True)

                        line_width = gr.Slider(
                            label="Box Line Width",
                            minimum=1,
                            maximum=10,
                            value=2,
                            step=1,
                        )

                        detect_btn = gr.Button("🔍 Run Detection", variant="primary", size="lg")

                    # Right Column - Output
                    with gr.Column(scale=1):
                        gr.HTML("<h3 style='margin: 0 0 1rem 0; color: #1e293b;'>Results</h3>")

                        output_image = gr.Image(
                            label="Detection Results",
                            type="pil",
                            height=400,
                        )

                        with gr.Row():
                            with gr.Column():
                                detection_results = gr.Textbox(
                                    label="Detections",
                                    lines=8,
                                    interactive=False,
                                )

                            with gr.Column():
                                performance_metrics = gr.Textbox(
                                    label="Performance",
                                    lines=8,
                                    interactive=False,
                                )

            # Batch Processing Tab
            with gr.Tab("Batch Processing"):
                gr.HTML("""
                    <p style="color: #64748b; margin-bottom: 1rem;">
                        Process multiple images at once for quick validation.
                    </p>
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        batch_model_select = gr.Dropdown(
                            label="Select Model",
                            choices=get_available_models(),
                        )

                        batch_images = gr.File(
                            label="Upload Images",
                            file_count="multiple",
                            file_types=["image"],
                        )

                        batch_confidence = gr.Slider(
                            label="Confidence Threshold",
                            minimum=0.01,
                            maximum=1.0,
                            value=0.25,
                            step=0.01,
                        )

                        batch_iou = gr.Slider(
                            label="IoU Threshold",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.45,
                            step=0.05,
                        )

                        batch_btn = gr.Button("🚀 Process Batch", variant="primary")
                        batch_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column(scale=2):
                        batch_gallery = gr.Gallery(
                            label="Results",
                            columns=3,
                            height=500,
                            object_fit="contain",
                        )

            # Webcam Tab
            with gr.Tab("Webcam (Live)"):
                gr.HTML("""
                    <div style="background: #fef3c7; border: 1px solid #fbbf24; border-radius: 8px;
                                padding: 1rem; margin-bottom: 1rem;">
                        <strong style="color: #92400e;">⚠️ Webcam Mode</strong>
                        <p style="color: #a16207; margin: 0.5rem 0 0 0; font-size: 0.875rem;">
                            Real-time webcam detection requires a running Python environment.
                            For best performance, run detection in a loop locally.
                        </p>
                    </div>
                """)

                webcam_model = gr.Dropdown(
                    label="Select Model",
                    choices=get_available_models(),
                )

                webcam_input = gr.Image(
                    label="Webcam Feed",
                    sources=["webcam"],
                    type="pil",
                    streaming=True,
                )

                webcam_output = gr.Image(
                    label="Detection Output",
                    type="pil",
                )

                gr.HTML("""
                    <div class="card" style="margin-top: 1rem;">
                        <h4 style="margin: 0 0 0.5rem 0; color: #1e293b; font-size: 0.875rem;">
                            💡 For Real-Time Webcam Detection
                        </h4>
                        <p style="color: #64748b; font-size: 0.75rem; margin: 0;">
                            Run the following command in your terminal:
                        </p>
                        <code style="display: block; background: #f1f5f9; padding: 0.5rem;
                                     border-radius: 4px; margin-top: 0.5rem; font-size: 0.75rem;">
                            yolo detect predict model=path/to/model.pt source=0 show=True
                        </code>
                    </div>
                """)

        # Sample images info
        gr.HTML("""
            <div class="card" style="margin-top: 1.5rem; background: #f0fdf4; border-color: #bbf7d0;">
                <h4 style="margin: 0 0 0.5rem 0; color: #166534; font-size: 0.875rem;">
                    🎯 Testing Tips
                </h4>
                <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.75rem; color: #15803d;">
                    <li>Start with a lower confidence threshold (0.25) to see all potential detections</li>
                    <li>Increase confidence to reduce false positives</li>
                    <li>Adjust IoU threshold to control overlapping box suppression</li>
                    <li>Test with images similar to your training data for best results</li>
                    <li>Compare performance between model sizes (n, s, m, l, x)</li>
                </ul>
            </div>
        """)

        # Event handlers
        refresh_btn.click(
            fn=lambda: gr.update(choices=get_available_models()),
            outputs=[model_select]
        )

        detect_btn.click(
            fn=run_inference,
            inputs=[
                model_select, input_image, confidence, iou_threshold,
                max_detections, show_labels, show_confidence, line_width
            ],
            outputs=[output_image, detection_results, performance_metrics]
        )

        batch_btn.click(
            fn=run_batch_inference,
            inputs=[batch_model_select, batch_images, batch_confidence, batch_iou],
            outputs=[batch_gallery, batch_status]
        )
