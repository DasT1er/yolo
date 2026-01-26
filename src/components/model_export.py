"""
YOLO Training Studio - Model Export Component
=============================================

Export trained models to various formats for deployment.
Supports ONNX, TensorRT, CoreML, TFLite, and more.
"""

import gradio as gr
from pathlib import Path
import subprocess
import sys
from typing import Optional


RUNS_DIR = Path("runs")
EXPORTS_DIR = Path("exports")


# Export formats with descriptions
EXPORT_FORMATS = {
    "onnx": {
        "name": "ONNX",
        "description": "Open Neural Network Exchange - Universal format",
        "extension": ".onnx",
        "icon": "🔄",
        "platforms": ["CPU", "GPU", "Edge"],
    },
    "torchscript": {
        "name": "TorchScript",
        "description": "PyTorch serialized model",
        "extension": ".torchscript",
        "icon": "🔥",
        "platforms": ["CPU", "GPU"],
    },
    "engine": {
        "name": "TensorRT",
        "description": "NVIDIA optimized inference engine",
        "extension": ".engine",
        "icon": "⚡",
        "platforms": ["NVIDIA GPU"],
    },
    "coreml": {
        "name": "CoreML",
        "description": "Apple devices (iOS, macOS)",
        "extension": ".mlpackage",
        "icon": "🍎",
        "platforms": ["iOS", "macOS"],
    },
    "tflite": {
        "name": "TensorFlow Lite",
        "description": "Mobile and embedded devices",
        "extension": ".tflite",
        "icon": "📱",
        "platforms": ["Android", "Embedded"],
    },
    "edgetpu": {
        "name": "Edge TPU",
        "description": "Google Coral Edge TPU",
        "extension": "_edgetpu.tflite",
        "icon": "🐚",
        "platforms": ["Coral"],
    },
    "openvino": {
        "name": "OpenVINO",
        "description": "Intel hardware optimization",
        "extension": "_openvino_model",
        "icon": "🔷",
        "platforms": ["Intel CPU/GPU/VPU"],
    },
    "ncnn": {
        "name": "NCNN",
        "description": "Mobile inference framework",
        "extension": "_ncnn_model",
        "icon": "📲",
        "platforms": ["ARM", "Mobile"],
    },
}


def get_trained_models() -> list:
    """Get list of trained model files."""
    models = []

    # Search in runs directory
    if RUNS_DIR.exists():
        for pt_file in RUNS_DIR.glob("**/weights/*.pt"):
            models.append(str(pt_file))

    # Search in exports directory
    if EXPORTS_DIR.exists():
        for pt_file in EXPORTS_DIR.glob("**/*.pt"):
            models.append(str(pt_file))

    return sorted(set(models))


def export_model(
    model_path: str,
    export_format: str,
    img_size: int,
    half_precision: bool,
    dynamic_batch: bool,
    simplify: bool,
    batch_size: int,
) -> tuple:
    """Export a model to the specified format."""
    if not model_path:
        return "Error: Please select a model.", ""

    if not Path(model_path).exists():
        return f"Error: Model file not found: {model_path}", ""

    # Ensure exports directory exists
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build export command
    cmd = [
        sys.executable, "-m", "ultralytics",
        "export",
        f"model={model_path}",
        f"format={export_format}",
        f"imgsz={img_size}",
        f"half={str(half_precision).lower()}",
        f"dynamic={str(dynamic_batch).lower()}",
        f"simplify={str(simplify).lower()}",
        f"batch={batch_size}",
    ]

    try:
        # Run export
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        output = result.stdout + result.stderr

        if result.returncode == 0:
            # Find the exported file
            model_stem = Path(model_path).stem
            format_info = EXPORT_FORMATS.get(export_format, {})
            ext = format_info.get("extension", f".{export_format}")

            status = f"✅ Model exported successfully to {export_format.upper()} format!"
            return status, output
        else:
            return f"Error during export: {result.returncode}", output

    except subprocess.TimeoutExpired:
        return "Error: Export timed out after 10 minutes.", ""
    except Exception as e:
        return f"Error: {str(e)}", ""


def create_export_format_cards() -> str:
    """Create HTML cards for export formats."""
    cards = []
    for fmt_id, info in EXPORT_FORMATS.items():
        platforms = ", ".join(info["platforms"])
        cards.append(f"""
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px;
                        padding: 0.75rem; transition: all 0.2s ease;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.25rem;">{info["icon"]}</span>
                    <div>
                        <div style="font-weight: 600; color: #1e293b; font-size: 0.875rem;">{info["name"]}</div>
                        <div style="font-size: 0.625rem; color: #64748b;">{platforms}</div>
                    </div>
                </div>
            </div>
        """)

    return f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.5rem; margin-bottom: 1rem;">
        {''.join(cards)}
    </div>
    """


def create_export_tab():
    """Create the model export tab."""

    with gr.Column():
        gr.HTML("""
            <div style="margin-bottom: 1rem;">
                <h2 style="color: #1e293b; margin: 0;">Model Export</h2>
                <p style="color: #64748b; margin-top: 0.5rem;">
                    Export trained models to various formats for deployment.
                </p>
            </div>
        """)

        # Format overview
        gr.HTML("<h3 style='margin: 0 0 0.75rem 0; color: #1e293b;'>Available Export Formats</h3>")
        gr.HTML(create_export_format_cards())

        gr.HTML("<hr style='margin: 1rem 0; border: none; border-top: 1px solid #e2e8f0;'>")

        with gr.Row():
            # Left Column - Export Configuration
            with gr.Column(scale=1):
                gr.HTML("<h3 style='margin: 0 0 1rem 0; color: #1e293b;'>Export Configuration</h3>")

                model_select = gr.Dropdown(
                    label="Select Trained Model",
                    choices=get_trained_models(),
                    info="Choose a .pt model file to export",
                )

                refresh_models_btn = gr.Button("🔄 Refresh Models", variant="secondary", size="sm")

                format_select = gr.Dropdown(
                    label="Export Format",
                    choices=list(EXPORT_FORMATS.keys()),
                    value="onnx",
                    info="Target deployment format",
                )

                img_size = gr.Dropdown(
                    label="Image Size",
                    choices=[320, 416, 512, 640, 768, 1024, 1280],
                    value=640,
                    info="Input resolution for exported model",
                )

                batch_size = gr.Number(
                    label="Batch Size",
                    value=1,
                    minimum=1,
                    maximum=32,
                    step=1,
                    info="Batch size for exported model",
                )

                gr.HTML("<h4 style='margin: 1rem 0 0.5rem 0; color: #475569;'>Optimization Options</h4>")

                half_precision = gr.Checkbox(
                    label="Half Precision (FP16)",
                    value=False,
                    info="Reduce model size and improve inference speed (requires GPU support)",
                )

                dynamic_batch = gr.Checkbox(
                    label="Dynamic Batch Size",
                    value=False,
                    info="Allow variable batch sizes at inference",
                )

                simplify = gr.Checkbox(
                    label="Simplify (ONNX)",
                    value=True,
                    info="Simplify ONNX model graph",
                )

                export_btn = gr.Button("📦 Export Model", variant="primary", size="lg")

            # Right Column - Export Status & Info
            with gr.Column(scale=1):
                gr.HTML("<h3 style='margin: 0 0 1rem 0; color: #1e293b;'>Export Status</h3>")

                export_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2,
                )

                export_logs = gr.Textbox(
                    label="Export Logs",
                    lines=15,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True,
                )

                # Platform compatibility info
                gr.HTML("""
                    <div class="card" style="margin-top: 1rem; background: #eff6ff; border-color: #bfdbfe;">
                        <h4 style="margin: 0 0 0.5rem 0; color: #1e40af; font-size: 0.875rem;">
                            📋 Format Recommendations
                        </h4>
                        <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.75rem; color: #1d4ed8;">
                            <li><strong>ONNX:</strong> Best universal format, works everywhere</li>
                            <li><strong>TensorRT:</strong> Best for NVIDIA GPUs (Jetson, RTX)</li>
                            <li><strong>CoreML:</strong> Required for iOS/macOS apps</li>
                            <li><strong>TFLite:</strong> Best for Android and Raspberry Pi</li>
                            <li><strong>OpenVINO:</strong> Best for Intel hardware</li>
                        </ul>
                    </div>
                """)

        # Event handlers
        refresh_models_btn.click(
            fn=lambda: gr.update(choices=get_trained_models()),
            outputs=[model_select]
        )

        export_btn.click(
            fn=export_model,
            inputs=[
                model_select, format_select, img_size,
                half_precision, dynamic_batch, simplify, batch_size
            ],
            outputs=[export_status, export_logs]
        )
