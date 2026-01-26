"""
YOLO Training Studio - Main Application
========================================

A professional, modern web-based training studio for YOLO models.
"""

import gradio as gr
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.components.dataset_manager import create_dataset_tab
from src.components.labeling_tool import create_labeling_tab
from src.components.training_manager import create_training_tab
from src.components.model_export import create_export_tab
from src.components.inference import create_inference_tab
from src.components.dashboard import create_dashboard_tab


# Custom CSS for modern, professional look
CUSTOM_CSS = """
/* Global Styles */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    max-width: 1400px !important;
    margin: auto !important;
}

/* Header Styling */
.header-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
}

.header-title {
    color: white !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.header-subtitle {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.1rem !important;
    margin-top: 0.5rem !important;
}

/* Tab Styling */
.tabs {
    border: none !important;
    background: transparent !important;
}

.tab-nav {
    background: #f8fafc !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
    gap: 0.5rem !important;
    border: 1px solid #e2e8f0 !important;
}

.tab-nav button {
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    border: none !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
}

/* Card Styling */
.card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}

/* Button Styling */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    padding: 0.75rem 2rem !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
}

.secondary-btn {
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    color: #475569 !important;
    padding: 0.75rem 2rem !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* Status Badges */
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}

.status-success {
    background: #dcfce7;
    color: #166534;
}

.status-warning {
    background: #fef3c7;
    color: #92400e;
}

.status-error {
    background: #fee2e2;
    color: #991b1b;
}

/* Progress Bar */
.progress-container {
    background: #e2e8f0;
    border-radius: 9999px;
    height: 8px;
    overflow: hidden;
}

.progress-bar {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    height: 100%;
    transition: width 0.3s ease;
}

/* Stats Cards */
.stat-card {
    background: white;
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    border: 1px solid #e2e8f0;
}

.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e293b;
}

.stat-label {
    font-size: 0.875rem;
    color: #64748b;
    margin-top: 0.25rem;
}

/* Model Version Selector */
.model-selector {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.model-option {
    padding: 0.5rem 1rem;
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.model-option.selected {
    border-color: #667eea;
    background: rgba(102, 126, 234, 0.1);
}

/* Responsive Design */
@media (max-width: 768px) {
    .header-title {
        font-size: 1.75rem !important;
    }

    .tab-nav {
        flex-wrap: wrap;
    }
}
"""

# JavaScript for enhanced interactivity
CUSTOM_JS = """
function setupStudio() {
    console.log('YOLO Training Studio initialized');
}
"""


def create_header():
    """Create the application header."""
    return gr.HTML("""
        <div class="header-container">
            <h1 class="header-title">YOLO Training Studio</h1>
            <p class="header-subtitle">
                Professional Object Detection Training Platform |
                Supports YOLO v5, v8, v11 & v26
            </p>
        </div>
    """)


def create_app():
    """Create the main Gradio application."""

    with gr.Blocks(
        css=CUSTOM_CSS,
        js=CUSTOM_JS,
        title="YOLO Training Studio",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as app:

        # Header
        create_header()

        # Main Tabs
        with gr.Tabs() as tabs:

            # Dashboard Tab
            with gr.Tab("Dashboard", id="dashboard"):
                create_dashboard_tab()

            # Dataset Manager Tab
            with gr.Tab("Dataset", id="dataset"):
                create_dataset_tab()

            # Labeling Tool Tab
            with gr.Tab("Labeling", id="labeling"):
                create_labeling_tab()

            # Training Tab
            with gr.Tab("Training", id="training"):
                create_training_tab()

            # Export Tab
            with gr.Tab("Export", id="export"):
                create_export_tab()

            # Testing/Inference Tab
            with gr.Tab("Testing", id="testing"):
                create_inference_tab()

        # Footer
        gr.HTML("""
            <div style="text-align: center; padding: 1.5rem; color: #64748b; font-size: 0.875rem;">
                <p>YOLO Training Studio v1.0.0 | Built with Ultralytics & Gradio</p>
                <p style="margin-top: 0.5rem;">
                    <a href="https://docs.ultralytics.com/" target="_blank" style="color: #667eea;">
                        Documentation
                    </a>
                    &nbsp;|&nbsp;
                    <a href="https://github.com/ultralytics/ultralytics" target="_blank" style="color: #667eea;">
                        GitHub
                    </a>
                </p>
            </div>
        """)

    return app


def main():
    """Main entry point."""
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
    )


if __name__ == "__main__":
    main()
