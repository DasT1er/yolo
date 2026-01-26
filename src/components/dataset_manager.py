"""
YOLO Training Studio - Dataset Manager Component
================================================

Handles dataset creation, import, and management.
Supports local uploads and Roboflow integration.
"""

import gradio as gr
from pathlib import Path
import shutil
import yaml
import json
from typing import Optional
import zipfile
import tempfile


# Constants
DATASETS_DIR = Path("datasets")
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def ensure_datasets_dir():
    """Ensure the datasets directory exists."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def get_existing_datasets() -> list:
    """Get list of existing datasets."""
    ensure_datasets_dir()
    datasets = []
    for item in DATASETS_DIR.iterdir():
        if item.is_dir() and (item / "data.yaml").exists():
            datasets.append(item.name)
    return datasets


def get_dataset_info(dataset_name: str) -> dict:
    """Get information about a specific dataset."""
    dataset_path = DATASETS_DIR / dataset_name
    info = {
        "name": dataset_name,
        "path": str(dataset_path),
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "classes": [],
        "total_labels": 0,
    }

    # Load data.yaml if exists
    yaml_path = dataset_path / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
            info["classes"] = data.get("names", [])

    # Count images
    for split in ["train", "val", "test"]:
        images_dir = dataset_path / "images" / split
        if images_dir.exists():
            count = sum(
                1 for f in images_dir.iterdir()
                if f.suffix.lower() in SUPPORTED_IMAGE_FORMATS
            )
            info[f"{split}_images"] = count

        labels_dir = dataset_path / "labels" / split
        if labels_dir.exists():
            info["total_labels"] += len(list(labels_dir.glob("*.txt")))

    return info


def create_new_dataset(name: str, classes: str) -> str:
    """Create a new empty dataset with the given name and classes."""
    if not name or not name.strip():
        return "Error: Please provide a dataset name."

    name = name.strip().replace(" ", "_")
    dataset_path = DATASETS_DIR / name

    if dataset_path.exists():
        return f"Error: Dataset '{name}' already exists."

    try:
        # Create directory structure
        for split in ["train", "val", "test"]:
            (dataset_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_path / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Parse classes
        class_list = [c.strip() for c in classes.split(",") if c.strip()]
        if not class_list:
            class_list = ["object"]

        # Create data.yaml
        data_yaml = {
            "path": str(dataset_path.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: name for i, name in enumerate(class_list)},
            "nc": len(class_list),
        }

        with open(dataset_path / "data.yaml", "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

        return f"✅ Dataset '{name}' created successfully with {len(class_list)} classes!"

    except Exception as e:
        return f"Error creating dataset: {str(e)}"


def upload_images(dataset_name: str, split: str, files: list) -> str:
    """Upload images to a dataset split."""
    if not dataset_name:
        return "Error: Please select a dataset."

    if not files:
        return "Error: No files selected."

    dataset_path = DATASETS_DIR / dataset_name
    if not dataset_path.exists():
        return f"Error: Dataset '{dataset_name}' not found."

    images_dir = dataset_path / "images" / split
    images_dir.mkdir(parents=True, exist_ok=True)

    uploaded = 0
    for file in files:
        if file is None:
            continue

        file_path = Path(file.name if hasattr(file, 'name') else file)
        if file_path.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
            dest = images_dir / file_path.name
            shutil.copy2(file_path, dest)
            uploaded += 1

    return f"✅ Uploaded {uploaded} images to {split} split."


def import_roboflow_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    target_name: str
) -> str:
    """Import a dataset from Roboflow."""
    if not all([api_key, workspace, project, version]):
        return "Error: Please fill in all Roboflow fields."

    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download(
            "yolov8",
            location=str(DATASETS_DIR / target_name)
        )

        return f"✅ Successfully imported '{project}' from Roboflow!"

    except ImportError:
        return "Error: Roboflow package not installed. Run: pip install roboflow"
    except Exception as e:
        return f"Error importing from Roboflow: {str(e)}"


def import_zip_dataset(zip_file, target_name: str) -> str:
    """Import a dataset from a ZIP file."""
    if not zip_file:
        return "Error: No ZIP file selected."

    if not target_name or not target_name.strip():
        return "Error: Please provide a dataset name."

    target_name = target_name.strip().replace(" ", "_")
    dataset_path = DATASETS_DIR / target_name

    if dataset_path.exists():
        return f"Error: Dataset '{target_name}' already exists."

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP
            zip_path = Path(zip_file.name if hasattr(zip_file, 'name') else zip_file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Find the root directory (might be nested)
            temp_path = Path(temp_dir)
            contents = list(temp_path.iterdir())

            # If there's a single directory, use its contents
            if len(contents) == 1 and contents[0].is_dir():
                source_dir = contents[0]
            else:
                source_dir = temp_path

            # Copy to datasets
            shutil.copytree(source_dir, dataset_path)

            # Check for data.yaml
            if not (dataset_path / "data.yaml").exists():
                # Try to create one based on structure
                classes = []
                if (dataset_path / "classes.txt").exists():
                    with open(dataset_path / "classes.txt") as f:
                        classes = [line.strip() for line in f if line.strip()]

                data_yaml = {
                    "path": str(dataset_path.absolute()),
                    "train": "images/train",
                    "val": "images/val",
                    "names": {i: name for i, name in enumerate(classes)} if classes else {0: "object"},
                    "nc": len(classes) if classes else 1,
                }

                with open(dataset_path / "data.yaml", "w") as f:
                    yaml.dump(data_yaml, f, default_flow_style=False)

        return f"✅ Successfully imported dataset as '{target_name}'!"

    except Exception as e:
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        return f"Error importing ZIP: {str(e)}"


def delete_dataset(dataset_name: str) -> str:
    """Delete a dataset."""
    if not dataset_name:
        return "Error: No dataset selected."

    dataset_path = DATASETS_DIR / dataset_name
    if not dataset_path.exists():
        return f"Error: Dataset '{dataset_name}' not found."

    try:
        shutil.rmtree(dataset_path)
        return f"✅ Dataset '{dataset_name}' deleted successfully."
    except Exception as e:
        return f"Error deleting dataset: {str(e)}"


def format_dataset_info(info: dict) -> str:
    """Format dataset info as HTML."""
    if not info:
        return "Select a dataset to view details."

    return f"""
    <div class="card" style="background: #f8fafc;">
        <h4 style="margin: 0 0 1rem 0; color: #1e293b;">{info['name']}</h4>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1rem;">
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">{info['train_images']}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Train Images</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700; color: #f59e0b;">{info['val_images']}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Val Images</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700; color: #8b5cf6;">{info['test_images']}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Test Images</div>
            </div>
        </div>
        <div style="margin-top: 1rem;">
            <strong style="color: #475569;">Classes ({len(info['classes'])}):</strong>
            <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem;">
                {''.join(f'<span style="background: #e0e7ff; color: #3730a3; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem;">{name}</span>' for name in (list(info['classes'].values()) if isinstance(info['classes'], dict) else info['classes'])[:10])}
                {f'<span style="color: #64748b; font-size: 0.75rem;">+{len(info["classes"]) - 10} more</span>' if len(info['classes']) > 10 else ''}
            </div>
        </div>
        <div style="margin-top: 1rem; font-size: 0.75rem; color: #64748b;">
            Total Labels: {info['total_labels']}
        </div>
    </div>
    """


def create_dataset_tab():
    """Create the dataset management tab."""

    with gr.Column():
        gr.HTML("""
            <div style="margin-bottom: 1rem;">
                <h2 style="color: #1e293b; margin: 0;">Dataset Manager</h2>
                <p style="color: #64748b; margin-top: 0.5rem;">
                    Create, import, and manage your training datasets.
                </p>
            </div>
        """)

        with gr.Row():
            # Left Column - Dataset List & Actions
            with gr.Column(scale=1):
                gr.HTML("<h3 style='margin: 0 0 1rem 0; color: #1e293b;'>Your Datasets</h3>")

                dataset_dropdown = gr.Dropdown(
                    label="Select Dataset",
                    choices=get_existing_datasets(),
                    interactive=True,
                )

                refresh_btn = gr.Button("🔄 Refresh List", variant="secondary", size="sm")

                dataset_info_html = gr.HTML(
                    value="<p style='color: #64748b;'>Select a dataset to view details.</p>"
                )

                delete_btn = gr.Button("🗑️ Delete Selected Dataset", variant="stop", size="sm")
                delete_status = gr.Textbox(label="Status", visible=False)

            # Right Column - Create/Import
            with gr.Column(scale=2):
                with gr.Tabs():
                    # Create New Dataset Tab
                    with gr.Tab("Create New"):
                        gr.HTML("""
                            <p style="color: #64748b; margin-bottom: 1rem;">
                                Create a new empty dataset with your custom classes.
                            </p>
                        """)

                        new_dataset_name = gr.Textbox(
                            label="Dataset Name",
                            placeholder="my_dataset",
                            info="Use alphanumeric characters and underscores"
                        )

                        new_dataset_classes = gr.Textbox(
                            label="Classes",
                            placeholder="person, car, dog, cat",
                            info="Comma-separated list of class names"
                        )

                        create_btn = gr.Button("✨ Create Dataset", variant="primary")
                        create_status = gr.Textbox(label="Status", interactive=False)

                    # Upload Images Tab
                    with gr.Tab("Upload Images"):
                        gr.HTML("""
                            <p style="color: #64748b; margin-bottom: 1rem;">
                                Upload images to an existing dataset.
                            </p>
                        """)

                        upload_dataset = gr.Dropdown(
                            label="Target Dataset",
                            choices=get_existing_datasets(),
                            interactive=True,
                        )

                        upload_split = gr.Radio(
                            label="Split",
                            choices=["train", "val", "test"],
                            value="train",
                        )

                        upload_files = gr.File(
                            label="Select Images",
                            file_count="multiple",
                            file_types=["image"],
                        )

                        upload_btn = gr.Button("📤 Upload Images", variant="primary")
                        upload_status = gr.Textbox(label="Status", interactive=False)

                    # Import from ZIP Tab
                    with gr.Tab("Import ZIP"):
                        gr.HTML("""
                            <p style="color: #64748b; margin-bottom: 1rem;">
                                Import a dataset from a ZIP file (YOLO format).
                            </p>
                        """)

                        zip_file = gr.File(
                            label="Select ZIP File",
                            file_types=[".zip"],
                        )

                        zip_target_name = gr.Textbox(
                            label="Dataset Name",
                            placeholder="imported_dataset",
                        )

                        import_zip_btn = gr.Button("📦 Import ZIP", variant="primary")
                        import_zip_status = gr.Textbox(label="Status", interactive=False)

                    # Import from Roboflow Tab
                    with gr.Tab("Roboflow"):
                        gr.HTML("""
                            <div style="background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
                                        padding: 1rem; border-radius: 8px; color: white; margin-bottom: 1rem;">
                                <strong>Roboflow Integration</strong>
                                <p style="margin: 0.5rem 0 0 0; font-size: 0.875rem; opacity: 0.9;">
                                    Import datasets directly from your Roboflow workspace.
                                </p>
                            </div>
                        """)

                        rf_api_key = gr.Textbox(
                            label="API Key",
                            placeholder="Your Roboflow API key",
                            type="password",
                        )

                        rf_workspace = gr.Textbox(
                            label="Workspace",
                            placeholder="your-workspace",
                        )

                        rf_project = gr.Textbox(
                            label="Project",
                            placeholder="your-project",
                        )

                        rf_version = gr.Number(
                            label="Version",
                            value=1,
                            precision=0,
                        )

                        rf_target_name = gr.Textbox(
                            label="Local Dataset Name",
                            placeholder="roboflow_dataset",
                        )

                        import_rf_btn = gr.Button("🔗 Import from Roboflow", variant="primary")
                        import_rf_status = gr.Textbox(label="Status", interactive=False)

        # Event Handlers
        def update_dataset_info(name):
            if name:
                info = get_dataset_info(name)
                return format_dataset_info(info)
            return "<p style='color: #64748b;'>Select a dataset to view details.</p>"

        def refresh_datasets():
            datasets = get_existing_datasets()
            return gr.update(choices=datasets)

        dataset_dropdown.change(
            fn=update_dataset_info,
            inputs=[dataset_dropdown],
            outputs=[dataset_info_html]
        )

        refresh_btn.click(
            fn=refresh_datasets,
            outputs=[dataset_dropdown]
        )

        create_btn.click(
            fn=create_new_dataset,
            inputs=[new_dataset_name, new_dataset_classes],
            outputs=[create_status]
        ).then(
            fn=refresh_datasets,
            outputs=[dataset_dropdown]
        )

        upload_btn.click(
            fn=upload_images,
            inputs=[upload_dataset, upload_split, upload_files],
            outputs=[upload_status]
        )

        import_zip_btn.click(
            fn=import_zip_dataset,
            inputs=[zip_file, zip_target_name],
            outputs=[import_zip_status]
        ).then(
            fn=refresh_datasets,
            outputs=[dataset_dropdown]
        )

        import_rf_btn.click(
            fn=import_roboflow_dataset,
            inputs=[rf_api_key, rf_workspace, rf_project, rf_version, rf_target_name],
            outputs=[import_rf_status]
        ).then(
            fn=refresh_datasets,
            outputs=[dataset_dropdown]
        )

        delete_btn.click(
            fn=delete_dataset,
            inputs=[dataset_dropdown],
            outputs=[delete_status]
        ).then(
            fn=refresh_datasets,
            outputs=[dataset_dropdown]
        )
