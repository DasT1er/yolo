# YOLO Training Studio

A professional, modern web-based training studio for YOLO object detection models. Supports YOLO v5, v8, v11, and the latest YOLO26.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.44+-orange.svg)
![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3+-green.svg)

## Features

- **Dashboard** - Overview of your projects, datasets, and models
- **Dataset Manager** - Create, import, and manage datasets (local & Roboflow)
- **Labeling Tool** - Built-in annotation tool for bounding boxes
- **Training Manager** - Configure and monitor training with real-time logs
- **Model Export** - Export to ONNX, TensorRT, CoreML, TFLite, and more
- **Model Testing** - Test your trained models on images, batches, or webcam

## Supported YOLO Versions

| Version | Year | Key Features |
|---------|------|--------------|
| **YOLO26** | 2026 | NMS-free, 43% faster CPU, edge-optimized |
| **YOLO11** | 2024 | 22% fewer params, improved accuracy |
| **YOLOv8** | 2023 | Anchor-free, widely adopted |
| **YOLOv5** | 2020 | Classic, stable, large community |

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/yolo-training-studio.git
cd yolo-training-studio

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Or install with extras
pip install -e ".[all]"
```

## Quick Start

```bash
# Run the application
python -m src.app

# Or use the CLI command (after pip install -e .)
yolo-studio
```

Open your browser at `http://localhost:7860`

## Usage Guide

### 1. Create/Import Dataset

Navigate to the **Dataset** tab to:
- Create a new empty dataset with custom classes
- Upload images directly
- Import from a ZIP file (YOLO format)
- Import from Roboflow

### 2. Label Your Data

Use the **Labeling** tab to:
- View and annotate images
- Add bounding boxes with class labels
- Navigate through your dataset
- Save annotations in YOLO format

### 3. Train Your Model

In the **Training** tab:
- Select your dataset
- Choose YOLO version and size (nano to xlarge)
- Configure training parameters
- Monitor training progress in real-time

### 4. Test Your Model

Use the **Testing** tab to:
- Load your trained model
- Test on single images
- Process batches of images
- Adjust confidence and NMS thresholds
- View detection results with performance metrics

### 5. Export for Deployment

In the **Export** tab:
- Select your trained model
- Choose export format (ONNX, TensorRT, CoreML, etc.)
- Configure optimization options
- Export for your target platform

## Project Structure

```
yolo-training-studio/
├── src/
│   ├── app.py              # Main Gradio application
│   ├── components/         # UI components
│   │   ├── dashboard.py
│   │   ├── dataset_manager.py
│   │   ├── labeling_tool.py
│   │   ├── training_manager.py
│   │   ├── model_export.py
│   │   └── inference.py
│   ├── utils/              # Utility functions
│   │   ├── config.py
│   │   └── helpers.py
│   └── models/             # Model utilities
├── configs/                # Configuration files
├── datasets/               # Your datasets (created automatically)
├── exports/                # Exported models
├── runs/                   # Training runs
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Configuration

Edit `configs/studio_config.yaml` to customize:
- Default training parameters
- Export settings
- UI preferences
- Directory paths

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU training)
- 8GB+ RAM recommended
- GPU with 4GB+ VRAM for training

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the YOLO implementation
- [Gradio](https://gradio.app/) for the web UI framework
- [Roboflow](https://roboflow.com/) for dataset management integration
