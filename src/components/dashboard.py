"""
YOLO Training Studio - Dashboard Component
==========================================

Main dashboard with overview statistics and quick actions.
"""

import gradio as gr
from pathlib import Path
import json
from datetime import datetime


def get_project_stats():
    """Get current project statistics."""
    datasets_path = Path("datasets")
    exports_path = Path("exports")

    stats = {
        "datasets": 0,
        "images": 0,
        "labels": 0,
        "models": 0,
    }

    if datasets_path.exists():
        stats["datasets"] = len(list(datasets_path.iterdir()))
        for dataset in datasets_path.iterdir():
            if dataset.is_dir():
                images_dir = dataset / "images"
                labels_dir = dataset / "labels"
                if images_dir.exists():
                    stats["images"] += len(list(images_dir.glob("*.*")))
                if labels_dir.exists():
                    stats["labels"] += len(list(labels_dir.glob("*.txt")))

    if exports_path.exists():
        stats["models"] = len(list(exports_path.glob("**/*.pt"))) + \
                         len(list(exports_path.glob("**/*.onnx")))

    return stats


def create_stats_html(stats: dict) -> str:
    """Create HTML for statistics cards."""
    return f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
        <div class="stat-card">
            <div class="stat-value" style="color: #667eea;">{stats['datasets']}</div>
            <div class="stat-label">Datasets</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: #10b981;">{stats['images']}</div>
            <div class="stat-label">Images</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: #f59e0b;">{stats['labels']}</div>
            <div class="stat-label">Labels</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: #8b5cf6;">{stats['models']}</div>
            <div class="stat-label">Trained Models</div>
        </div>
    </div>
    """


def create_model_comparison_html() -> str:
    """Create HTML for YOLO model comparison."""
    return """
    <div class="card">
        <h3 style="margin-top: 0; color: #1e293b; font-size: 1.25rem;">
            YOLO Model Comparison
        </h3>
        <div style="overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.875rem;">
                <thead>
                    <tr style="background: #f8fafc;">
                        <th style="padding: 0.75rem; text-align: left; border-bottom: 2px solid #e2e8f0;">Model</th>
                        <th style="padding: 0.75rem; text-align: center; border-bottom: 2px solid #e2e8f0;">mAP</th>
                        <th style="padding: 0.75rem; text-align: center; border-bottom: 2px solid #e2e8f0;">Speed (CPU)</th>
                        <th style="padding: 0.75rem; text-align: center; border-bottom: 2px solid #e2e8f0;">Params</th>
                        <th style="padding: 0.75rem; text-align: center; border-bottom: 2px solid #e2e8f0;">Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding: 0.75rem; border-bottom: 1px solid #e2e8f0;">
                            <strong>YOLO26n</strong>
                            <span style="background: #dcfce7; color: #166534; padding: 0.125rem 0.5rem;
                                   border-radius: 9999px; font-size: 0.625rem; margin-left: 0.5rem;">
                                LATEST
                            </span>
                        </td>
                        <td style="padding: 0.75rem; text-align: center; border-bottom: 1px solid #e2e8f0;">40.3%</td>
                        <td style="padding: 0.75rem; text-align: center; border-bottom: 1px solid #e2e8f0;">38.9ms</td>
                        <td style="padding: 0.75rem; text-align: center; border-bottom: 1px solid #e2e8f0;">2.6M</td>
                        <td style="padding: 0.75rem; text-align: center; border-bottom: 1px solid #e2e8f0;">
                            <span class="status-badge status-success">Recommended</span>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 0.75rem; border-bottom: 1px solid #e2e8f0;"><strong>YOLO11n</strong></td>
                        <td style="padding: 0.75rem; text-align: center; border-bottom: 1px solid #e2e8f0;">39.5%</td>
                        <td style="padding: 0.75rem; text-align: center; border-bottom: 1px solid #e2e8f0;">56.1ms</td>
                        <td style="padding: 0.75rem; text-align: center; border-bottom: 1px solid #e2e8f0;">2.6M</td>
                        <td style="padding: 0.75rem; text-align: center; border-bottom: 1px solid #e2e8f0;">
                            <span class="status-badge status-success">Stable</span>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 0.75rem; border-bottom: 1px solid #e2e8f0;"><strong>YOLOv8n</strong></td>
                        <td style="padding: 0.75rem; text-align: center; border-bottom: 1px solid #e2e8f0;">37.3%</td>
                        <td style="padding: 0.75rem; text-align: center; border-bottom: 1px solid #e2e8f0;">80.4ms</td>
                        <td style="padding: 0.75rem; text-align: center; border-bottom: 1px solid #e2e8f0;">3.2M</td>
                        <td style="padding: 0.75rem; text-align: center; border-bottom: 1px solid #e2e8f0;">
                            <span class="status-badge" style="background: #e2e8f0; color: #475569;">Legacy</span>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 0.75rem;"><strong>YOLOv5n</strong></td>
                        <td style="padding: 0.75rem; text-align: center;">28.0%</td>
                        <td style="padding: 0.75rem; text-align: center;">73.6ms</td>
                        <td style="padding: 0.75rem; text-align: center;">1.9M</td>
                        <td style="padding: 0.75rem; text-align: center;">
                            <span class="status-badge" style="background: #e2e8f0; color: #475569;">Legacy</span>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
        <p style="margin-top: 1rem; font-size: 0.75rem; color: #64748b;">
            * mAP measured on COCO val2017 dataset at 640px resolution
        </p>
    </div>
    """


def create_quick_actions_html() -> str:
    """Create HTML for quick action buttons."""
    return """
    <div class="card">
        <h3 style="margin-top: 0; color: #1e293b; font-size: 1.25rem;">
            Quick Actions
        </h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 1.25rem; border-radius: 12px; color: white; cursor: pointer;
                        transition: transform 0.2s ease;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📁</div>
                <div style="font-weight: 600;">New Dataset</div>
                <div style="font-size: 0.75rem; opacity: 0.9;">Import or create dataset</div>
            </div>
            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                        padding: 1.25rem; border-radius: 12px; color: white; cursor: pointer;
                        transition: transform 0.2s ease;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🏷️</div>
                <div style="font-weight: 600;">Start Labeling</div>
                <div style="font-size: 0.75rem; opacity: 0.9;">Annotate your images</div>
            </div>
            <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                        padding: 1.25rem; border-radius: 12px; color: white; cursor: pointer;
                        transition: transform 0.2s ease;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🚀</div>
                <div style="font-weight: 600;">Train Model</div>
                <div style="font-size: 0.75rem; opacity: 0.9;">Start training session</div>
            </div>
            <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
                        padding: 1.25rem; border-radius: 12px; color: white; cursor: pointer;
                        transition: transform 0.2s ease;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📦</div>
                <div style="font-weight: 600;">Export Model</div>
                <div style="font-size: 0.75rem; opacity: 0.9;">ONNX, TensorRT, CoreML</div>
            </div>
        </div>
    </div>
    """


def refresh_dashboard():
    """Refresh dashboard statistics."""
    stats = get_project_stats()
    return create_stats_html(stats)


def create_dashboard_tab():
    """Create the dashboard tab content."""

    with gr.Column():
        # Welcome Message
        gr.HTML("""
            <div style="margin-bottom: 1rem;">
                <h2 style="color: #1e293b; margin: 0;">Welcome to YOLO Training Studio</h2>
                <p style="color: #64748b; margin-top: 0.5rem;">
                    Your professional platform for object detection model training.
                    Get started by importing a dataset or creating a new one.
                </p>
            </div>
        """)

        # Statistics
        stats = get_project_stats()
        stats_html = gr.HTML(create_stats_html(stats))

        # Refresh Button
        refresh_btn = gr.Button("🔄 Refresh Statistics", variant="secondary", size="sm")
        refresh_btn.click(fn=refresh_dashboard, outputs=stats_html)

        # Quick Actions
        gr.HTML(create_quick_actions_html())

        # Model Comparison
        gr.HTML(create_model_comparison_html())

        # Recent Activity
        gr.HTML("""
            <div class="card">
                <h3 style="margin-top: 0; color: #1e293b; font-size: 1.25rem;">
                    Getting Started Guide
                </h3>
                <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                    <div style="display: flex; align-items: center; gap: 1rem; padding: 0.75rem;
                                background: #f8fafc; border-radius: 8px;">
                        <div style="background: #667eea; color: white; width: 28px; height: 28px;
                                    border-radius: 50%; display: flex; align-items: center;
                                    justify-content: center; font-weight: 600; font-size: 0.875rem;">1</div>
                        <div>
                            <div style="font-weight: 600; color: #1e293b;">Import or Create Dataset</div>
                            <div style="font-size: 0.75rem; color: #64748b;">
                                Upload images or import from Roboflow
                            </div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 1rem; padding: 0.75rem;
                                background: #f8fafc; border-radius: 8px;">
                        <div style="background: #10b981; color: white; width: 28px; height: 28px;
                                    border-radius: 50%; display: flex; align-items: center;
                                    justify-content: center; font-weight: 600; font-size: 0.875rem;">2</div>
                        <div>
                            <div style="font-weight: 600; color: #1e293b;">Label Your Data</div>
                            <div style="font-size: 0.75rem; color: #64748b;">
                                Use the built-in labeling tool to annotate objects
                            </div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 1rem; padding: 0.75rem;
                                background: #f8fafc; border-radius: 8px;">
                        <div style="background: #f59e0b; color: white; width: 28px; height: 28px;
                                    border-radius: 50%; display: flex; align-items: center;
                                    justify-content: center; font-weight: 600; font-size: 0.875rem;">3</div>
                        <div>
                            <div style="font-weight: 600; color: #1e293b;">Train Your Model</div>
                            <div style="font-size: 0.75rem; color: #64748b;">
                                Configure and start training with YOLO26 or YOLO11
                            </div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 1rem; padding: 0.75rem;
                                background: #f8fafc; border-radius: 8px;">
                        <div style="background: #8b5cf6; color: white; width: 28px; height: 28px;
                                    border-radius: 50%; display: flex; align-items: center;
                                    justify-content: center; font-weight: 600; font-size: 0.875rem;">4</div>
                        <div>
                            <div style="font-weight: 600; color: #1e293b;">Export & Deploy</div>
                            <div style="font-size: 0.75rem; color: #64748b;">
                                Export to ONNX, TensorRT, CoreML for production
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        """)
