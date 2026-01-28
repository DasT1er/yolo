# YOLO Training Studio

Professional desktop application for YOLO object detection training. Supports YOLO v5, v8, v11, and v26.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![CustomTkinter](https://img.shields.io/badge/CustomTkinter-5.2+-purple.svg)
![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3+-green.svg)

## Features

- **Dashboard** - Overview of datasets and models
- **Dataset Manager** - Create, import, and manage datasets
- **Training** - Configure and run YOLO training with live logs
- **Testing** - Test models on images with visualization
- **Export** - Export to ONNX, TensorRT, CoreML, TFLite

## Quick Start

### Windows
```batch
# Double-click start.bat
# OR run manually:
pip install customtkinter ultralytics pillow pyyaml
python app.py
```

### Linux/Mac
```bash
pip install customtkinter ultralytics pillow pyyaml
python app.py
```

## YOLO Models

| Model | mAP | Speed | Status |
|-------|-----|-------|--------|
| YOLO26n | 40.3% | 38.9ms | Latest |
| YOLO11n | 39.5% | 56.1ms | Stable |
| YOLOv8n | 37.3% | 80.4ms | Legacy |

## Project Structure

```
yolo/
├── app.py              # Main application
├── start.bat           # Windows starter
├── start.sh            # Linux/Mac starter
├── requirements.txt    # Dependencies
├── datasets/           # Your datasets
├── modelle/            # Trained models (auto-detected)
└── exports/            # Exported models
```

## Requirements

- Python 3.10+
- 8GB+ RAM
- GPU recommended for training

## License

MIT License
