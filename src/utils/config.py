"""
YOLO Training Studio - Configuration Management
===============================================

Handles loading, saving, and managing application configuration.
"""

import yaml
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List


CONFIG_FILE = Path("configs/studio_config.yaml")


@dataclass
class TrainingDefaults:
    """Default training parameters."""
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    learning_rate: float = 0.01
    optimizer: str = "SGD"
    augmentation: bool = True
    pretrained: bool = True
    device: str = "0"


@dataclass
class ExportDefaults:
    """Default export parameters."""
    format: str = "onnx"
    image_size: int = 640
    half_precision: bool = False
    dynamic_batch: bool = False
    simplify: bool = True


@dataclass
class UISettings:
    """UI preference settings."""
    theme: str = "soft"
    language: str = "en"
    show_tips: bool = True
    auto_save: bool = True
    confirm_delete: bool = True


@dataclass
class Config:
    """Main configuration class."""
    version: str = "1.0.0"
    datasets_dir: str = "datasets"
    exports_dir: str = "exports"
    runs_dir: str = "runs"
    training: TrainingDefaults = field(default_factory=TrainingDefaults)
    export: ExportDefaults = field(default_factory=ExportDefaults)
    ui: UISettings = field(default_factory=UISettings)
    roboflow_api_key: Optional[str] = None
    recent_datasets: List[str] = field(default_factory=list)
    recent_models: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create config from dictionary."""
        training_data = data.pop("training", {})
        export_data = data.pop("export", {})
        ui_data = data.pop("ui", {})

        return cls(
            training=TrainingDefaults(**training_data) if training_data else TrainingDefaults(),
            export=ExportDefaults(**export_data) if export_data else ExportDefaults(),
            ui=UISettings(**ui_data) if ui_data else UISettings(),
            **data
        )


def load_config() -> Config:
    """Load configuration from file."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                data = yaml.safe_load(f)
                if data:
                    return Config.from_dict(data)
        except Exception as e:
            print(f"Error loading config: {e}")

    return Config()


def save_config(config: Config) -> bool:
    """Save configuration to file."""
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
