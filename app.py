"""
YOLO Training Studio - Desktop Application
==========================================

Professional YOLO training application with modern GUI.
Supports YOLO v5, v8, v11, and v26.

Usage: python app.py
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import shutil
import yaml


# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class YOLOTrainingStudio(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        # Window setup
        self.title("YOLO Training Studio")
        self.geometry("1400x900")
        self.minsize(1200, 800)

        # Paths
        self.base_path = Path(__file__).parent
        self.datasets_path = self.base_path / "datasets"
        self.runs_path = self.base_path / "runs"
        self.exports_path = self.base_path / "exports"

        # Ensure directories exist
        self.datasets_path.mkdir(exist_ok=True)
        self.runs_path.mkdir(exist_ok=True)
        self.exports_path.mkdir(exist_ok=True)

        # Training state
        self.training_process = None
        self.is_training = False

        # Create UI
        self.create_ui()

    def create_ui(self):
        """Create the main user interface."""
        # Header
        self.create_header()

        # Main content with tabs
        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Add tabs
        self.tab_dashboard = self.tabview.add("Dashboard")
        self.tab_dataset = self.tabview.add("Dataset")
        self.tab_training = self.tabview.add("Training")
        self.tab_testing = self.tabview.add("Testing")
        self.tab_export = self.tabview.add("Export")

        # Fill tabs
        self.create_dashboard_tab()
        self.create_dataset_tab()
        self.create_training_tab()
        self.create_testing_tab()
        self.create_export_tab()

    def create_header(self):
        """Create application header."""
        header = ctk.CTkFrame(self, height=80, corner_radius=0)
        header.pack(fill="x", padx=20, pady=20)
        header.pack_propagate(False)

        # Title
        title = ctk.CTkLabel(
            header,
            text="YOLO Training Studio",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title.pack(side="left", padx=20, pady=20)

        # Subtitle
        subtitle = ctk.CTkLabel(
            header,
            text="Professional Object Detection Training | YOLO v5, v8, v11 & v26",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        subtitle.pack(side="left", padx=10, pady=20)

        # Version
        version = ctk.CTkLabel(
            header,
            text="v1.0.0",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        version.pack(side="right", padx=20, pady=20)

    # ==================== DASHBOARD TAB ====================
    def create_dashboard_tab(self):
        """Create dashboard tab content."""
        # Stats frame
        stats_frame = ctk.CTkFrame(self.tab_dashboard)
        stats_frame.pack(fill="x", padx=20, pady=20)

        # Stats cards
        self.create_stat_card(stats_frame, "Datasets", self.count_datasets(), 0)
        self.create_stat_card(stats_frame, "Images", self.count_images(), 1)
        self.create_stat_card(stats_frame, "Models", self.count_models(), 2)
        self.create_stat_card(stats_frame, "Exports", self.count_exports(), 3)

        # Model comparison
        comparison_frame = ctk.CTkFrame(self.tab_dashboard)
        comparison_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        ctk.CTkLabel(
            comparison_frame,
            text="YOLO Model Comparison",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=20, pady=15)

        # Comparison table
        table_frame = ctk.CTkFrame(comparison_frame, fg_color="transparent")
        table_frame.pack(fill="x", padx=20, pady=(0, 20))

        # Headers
        headers = ["Model", "mAP", "Speed (CPU)", "Parameters", "Status"]
        for i, header in enumerate(headers):
            ctk.CTkLabel(
                table_frame,
                text=header,
                font=ctk.CTkFont(weight="bold"),
                width=150
            ).grid(row=0, column=i, padx=10, pady=10)

        # Data
        models = [
            ("YOLO26n", "40.3%", "38.9ms", "2.6M", "LATEST"),
            ("YOLO11n", "39.5%", "56.1ms", "2.6M", "Stable"),
            ("YOLOv8n", "37.3%", "80.4ms", "3.2M", "Legacy"),
            ("YOLOv5n", "28.0%", "73.6ms", "1.9M", "Legacy"),
        ]

        for row, (model, mAP, speed, params, status) in enumerate(models, 1):
            ctk.CTkLabel(table_frame, text=model, width=150).grid(row=row, column=0, padx=10, pady=8)
            ctk.CTkLabel(table_frame, text=mAP, width=150).grid(row=row, column=1, padx=10, pady=8)
            ctk.CTkLabel(table_frame, text=speed, width=150).grid(row=row, column=2, padx=10, pady=8)
            ctk.CTkLabel(table_frame, text=params, width=150).grid(row=row, column=3, padx=10, pady=8)

            status_color = "#22c55e" if status in ["LATEST", "Stable"] else "gray"
            ctk.CTkLabel(
                table_frame,
                text=status,
                width=150,
                text_color=status_color
            ).grid(row=row, column=4, padx=10, pady=8)

    def create_stat_card(self, parent, title, value, column):
        """Create a statistics card."""
        card = ctk.CTkFrame(parent, width=200, height=100)
        card.grid(row=0, column=column, padx=15, pady=15)
        card.grid_propagate(False)

        ctk.CTkLabel(
            card,
            text=str(value),
            font=ctk.CTkFont(size=36, weight="bold")
        ).pack(pady=(20, 5))

        ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=14),
            text_color="gray"
        ).pack()

    # ==================== DATASET TAB ====================
    def create_dataset_tab(self):
        """Create dataset management tab."""
        # Left panel - Dataset list
        left_frame = ctk.CTkFrame(self.tab_dataset, width=300)
        left_frame.pack(side="left", fill="y", padx=(20, 10), pady=20)
        left_frame.pack_propagate(False)

        ctk.CTkLabel(
            left_frame,
            text="Your Datasets",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=15, pady=15)

        # Dataset listbox
        self.dataset_listbox = ctk.CTkScrollableFrame(left_frame, height=400)
        self.dataset_listbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.refresh_dataset_list()

        # Refresh button
        ctk.CTkButton(
            left_frame,
            text="Refresh",
            command=self.refresh_dataset_list
        ).pack(fill="x", padx=10, pady=10)

        # Right panel - Actions
        right_frame = ctk.CTkFrame(self.tab_dataset)
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 20), pady=20)

        ctk.CTkLabel(
            right_frame,
            text="Create New Dataset",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=20, pady=15)

        # Dataset name
        ctk.CTkLabel(right_frame, text="Dataset Name:").pack(anchor="w", padx=20, pady=(10, 5))
        self.dataset_name_entry = ctk.CTkEntry(right_frame, width=400, placeholder_text="my_dataset")
        self.dataset_name_entry.pack(anchor="w", padx=20)

        # Classes
        ctk.CTkLabel(right_frame, text="Classes (comma-separated):").pack(anchor="w", padx=20, pady=(15, 5))
        self.classes_entry = ctk.CTkEntry(right_frame, width=400, placeholder_text="person, car, dog, cat")
        self.classes_entry.pack(anchor="w", padx=20)

        # Create button
        ctk.CTkButton(
            right_frame,
            text="Create Dataset",
            command=self.create_dataset,
            width=200
        ).pack(anchor="w", padx=20, pady=20)

        # Separator
        ctk.CTkLabel(right_frame, text="─" * 60, text_color="gray").pack(pady=10)

        # Import section
        ctk.CTkLabel(
            right_frame,
            text="Import Dataset",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=20, pady=15)

        button_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        button_frame.pack(anchor="w", padx=20)

        ctk.CTkButton(
            button_frame,
            text="Import from Folder",
            command=self.import_from_folder,
            width=180
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            button_frame,
            text="Import ZIP",
            command=self.import_zip,
            width=180
        ).pack(side="left")

    def refresh_dataset_list(self):
        """Refresh the dataset list."""
        # Clear existing
        for widget in self.dataset_listbox.winfo_children():
            widget.destroy()

        # List datasets
        if self.datasets_path.exists():
            datasets = [d for d in self.datasets_path.iterdir() if d.is_dir()]
            for dataset in sorted(datasets):
                btn = ctk.CTkButton(
                    self.dataset_listbox,
                    text=dataset.name,
                    fg_color="transparent",
                    text_color=("gray10", "gray90"),
                    hover_color=("gray70", "gray30"),
                    anchor="w",
                    command=lambda d=dataset: self.select_dataset(d)
                )
                btn.pack(fill="x", pady=2)

    def create_dataset(self):
        """Create a new dataset."""
        name = self.dataset_name_entry.get().strip().replace(" ", "_")
        classes = self.classes_entry.get().strip()

        if not name:
            messagebox.showerror("Error", "Please enter a dataset name.")
            return

        dataset_path = self.datasets_path / name
        if dataset_path.exists():
            messagebox.showerror("Error", f"Dataset '{name}' already exists.")
            return

        try:
            # Create structure
            for split in ["train", "val", "test"]:
                (dataset_path / "images" / split).mkdir(parents=True)
                (dataset_path / "labels" / split).mkdir(parents=True)

            # Parse classes
            class_list = [c.strip() for c in classes.split(",") if c.strip()] or ["object"]

            # Create data.yaml
            data = {
                "path": str(dataset_path.absolute()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {i: name for i, name in enumerate(class_list)},
                "nc": len(class_list)
            }

            with open(dataset_path / "data.yaml", "w") as f:
                yaml.dump(data, f, sort_keys=False)

            messagebox.showinfo("Success", f"Dataset '{name}' created!")
            self.refresh_dataset_list()
            self.dataset_name_entry.delete(0, "end")
            self.classes_entry.delete(0, "end")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def import_from_folder(self):
        """Import dataset from folder."""
        folder = filedialog.askdirectory(title="Select Dataset Folder")
        if folder:
            folder_path = Path(folder)
            target = self.datasets_path / folder_path.name

            try:
                shutil.copytree(folder_path, target)
                messagebox.showinfo("Success", f"Imported '{folder_path.name}'!")
                self.refresh_dataset_list()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def import_zip(self):
        """Import dataset from ZIP."""
        zip_file = filedialog.askopenfilename(
            title="Select ZIP File",
            filetypes=[("ZIP files", "*.zip")]
        )
        if zip_file:
            import zipfile
            zip_path = Path(zip_file)
            target = self.datasets_path / zip_path.stem

            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(target)
                messagebox.showinfo("Success", f"Imported '{zip_path.stem}'!")
                self.refresh_dataset_list()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def select_dataset(self, dataset_path):
        """Select a dataset."""
        self.selected_dataset = dataset_path
        messagebox.showinfo("Selected", f"Dataset: {dataset_path.name}")

    # ==================== TRAINING TAB ====================
    def create_training_tab(self):
        """Create training configuration tab."""
        # Left - Configuration
        left_frame = ctk.CTkScrollableFrame(self.tab_training, width=500)
        left_frame.pack(side="left", fill="y", padx=(20, 10), pady=20)

        ctk.CTkLabel(
            left_frame,
            text="Training Configuration",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=15, pady=15)

        # Dataset selection
        ctk.CTkLabel(left_frame, text="Dataset:").pack(anchor="w", padx=15, pady=(10, 5))
        self.training_dataset = ctk.CTkComboBox(
            left_frame,
            values=self.get_dataset_names(),
            width=300
        )
        self.training_dataset.pack(anchor="w", padx=15)

        # Model selection
        ctk.CTkLabel(left_frame, text="Model:").pack(anchor="w", padx=15, pady=(15, 5))
        self.model_var = ctk.StringVar(value="yolo11n")
        models = ["yolo26n", "yolo26s", "yolo26m", "yolo11n", "yolo11s", "yolo11m", "yolov8n", "yolov8s", "yolov8m"]

        model_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        model_frame.pack(anchor="w", padx=15)

        for i, model in enumerate(models):
            ctk.CTkRadioButton(
                model_frame,
                text=model,
                variable=self.model_var,
                value=model
            ).grid(row=i // 3, column=i % 3, padx=10, pady=5)

        # Epochs
        ctk.CTkLabel(left_frame, text="Epochs:").pack(anchor="w", padx=15, pady=(15, 5))
        self.epochs_slider = ctk.CTkSlider(left_frame, from_=10, to=500, number_of_steps=49, width=300)
        self.epochs_slider.set(100)
        self.epochs_slider.pack(anchor="w", padx=15)
        self.epochs_label = ctk.CTkLabel(left_frame, text="100")
        self.epochs_label.pack(anchor="w", padx=15)
        self.epochs_slider.configure(command=lambda v: self.epochs_label.configure(text=str(int(v))))

        # Batch size
        ctk.CTkLabel(left_frame, text="Batch Size:").pack(anchor="w", padx=15, pady=(15, 5))
        self.batch_slider = ctk.CTkSlider(left_frame, from_=1, to=64, number_of_steps=63, width=300)
        self.batch_slider.set(16)
        self.batch_slider.pack(anchor="w", padx=15)
        self.batch_label = ctk.CTkLabel(left_frame, text="16")
        self.batch_label.pack(anchor="w", padx=15)
        self.batch_slider.configure(command=lambda v: self.batch_label.configure(text=str(int(v))))

        # Image size
        ctk.CTkLabel(left_frame, text="Image Size:").pack(anchor="w", padx=15, pady=(15, 5))
        self.imgsz_combo = ctk.CTkComboBox(
            left_frame,
            values=["320", "416", "512", "640", "768", "1024"],
            width=150
        )
        self.imgsz_combo.set("640")
        self.imgsz_combo.pack(anchor="w", padx=15)

        # Options
        self.pretrained_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            left_frame,
            text="Use Pretrained Weights",
            variable=self.pretrained_var
        ).pack(anchor="w", padx=15, pady=(15, 5))

        self.augment_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            left_frame,
            text="Data Augmentation",
            variable=self.augment_var
        ).pack(anchor="w", padx=15, pady=5)

        # Device
        ctk.CTkLabel(left_frame, text="Device:").pack(anchor="w", padx=15, pady=(15, 5))
        self.device_combo = ctk.CTkComboBox(
            left_frame,
            values=["0", "cpu", "0,1"],
            width=150
        )
        self.device_combo.set("0")
        self.device_combo.pack(anchor="w", padx=15)

        # Buttons
        btn_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        btn_frame.pack(anchor="w", padx=15, pady=20)

        self.start_btn = ctk.CTkButton(
            btn_frame,
            text="Start Training",
            command=self.start_training,
            width=150,
            fg_color="#22c55e",
            hover_color="#16a34a"
        )
        self.start_btn.pack(side="left", padx=(0, 10))

        self.stop_btn = ctk.CTkButton(
            btn_frame,
            text="Stop",
            command=self.stop_training,
            width=100,
            fg_color="#ef4444",
            hover_color="#dc2626",
            state="disabled"
        )
        self.stop_btn.pack(side="left")

        # Right - Logs
        right_frame = ctk.CTkFrame(self.tab_training)
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 20), pady=20)

        ctk.CTkLabel(
            right_frame,
            text="Training Logs",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=15, pady=15)

        self.training_log = ctk.CTkTextbox(right_frame, height=500)
        self.training_log.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Progress
        self.progress_bar = ctk.CTkProgressBar(right_frame, width=400)
        self.progress_bar.pack(pady=(0, 15))
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(right_frame, text="Ready", text_color="gray")
        self.status_label.pack(pady=(0, 15))

    def start_training(self):
        """Start YOLO training."""
        dataset = self.training_dataset.get()
        if not dataset:
            messagebox.showerror("Error", "Please select a dataset.")
            return

        data_yaml = self.datasets_path / dataset / "data.yaml"
        if not data_yaml.exists():
            messagebox.showerror("Error", f"data.yaml not found in {dataset}")
            return

        self.is_training = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="Training...", text_color="#22c55e")
        self.training_log.delete("1.0", "end")

        # Build command
        model = self.model_var.get()
        epochs = int(self.epochs_slider.get())
        batch = int(self.batch_slider.get())
        imgsz = int(self.imgsz_combo.get())
        device = self.device_combo.get()

        cmd = [
            sys.executable, "-m", "ultralytics",
            "detect", "train",
            f"data={data_yaml}",
            f"model={model}.pt",
            f"epochs={epochs}",
            f"batch={batch}",
            f"imgsz={imgsz}",
            f"device={device}",
            f"project={self.runs_path}",
            f"name={dataset}_{model}",
            "exist_ok=True"
        ]

        # Run in thread
        def run_training():
            try:
                self.training_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                for line in self.training_process.stdout:
                    self.after(0, lambda l=line: self.append_log(l))

                self.training_process.wait()
                self.after(0, self.training_finished)

            except Exception as e:
                self.after(0, lambda: self.append_log(f"Error: {e}\n"))
                self.after(0, self.training_finished)

        threading.Thread(target=run_training, daemon=True).start()

    def append_log(self, text):
        """Append text to training log."""
        self.training_log.insert("end", text)
        self.training_log.see("end")

    def stop_training(self):
        """Stop training."""
        if self.training_process:
            self.training_process.terminate()
            self.append_log("\n[STOPPED BY USER]\n")

    def training_finished(self):
        """Called when training finishes."""
        self.is_training = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="Finished", text_color="gray")
        self.progress_bar.set(1)

    # ==================== TESTING TAB ====================
    def create_testing_tab(self):
        """Create model testing tab."""
        # Left - Settings
        left_frame = ctk.CTkFrame(self.tab_testing, width=350)
        left_frame.pack(side="left", fill="y", padx=(20, 10), pady=20)
        left_frame.pack_propagate(False)

        ctk.CTkLabel(
            left_frame,
            text="Model Testing",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=15, pady=15)

        # Model selection
        ctk.CTkLabel(left_frame, text="Select Model:").pack(anchor="w", padx=15, pady=(10, 5))
        self.test_model_combo = ctk.CTkComboBox(
            left_frame,
            values=self.get_model_files(),
            width=280
        )
        self.test_model_combo.pack(anchor="w", padx=15)

        ctk.CTkButton(
            left_frame,
            text="Browse Model...",
            command=self.browse_model,
            width=150
        ).pack(anchor="w", padx=15, pady=10)

        # Confidence
        ctk.CTkLabel(left_frame, text="Confidence:").pack(anchor="w", padx=15, pady=(15, 5))
        self.conf_slider = ctk.CTkSlider(left_frame, from_=0.01, to=1.0, width=280)
        self.conf_slider.set(0.25)
        self.conf_slider.pack(anchor="w", padx=15)
        self.conf_label = ctk.CTkLabel(left_frame, text="0.25")
        self.conf_label.pack(anchor="w", padx=15)
        self.conf_slider.configure(command=lambda v: self.conf_label.configure(text=f"{v:.2f}"))

        # Buttons
        ctk.CTkButton(
            left_frame,
            text="Select Image",
            command=self.select_test_image,
            width=280
        ).pack(anchor="w", padx=15, pady=(30, 10))

        ctk.CTkButton(
            left_frame,
            text="Run Detection",
            command=self.run_detection,
            width=280,
            fg_color="#22c55e",
            hover_color="#16a34a"
        ).pack(anchor="w", padx=15)

        # Results text
        ctk.CTkLabel(left_frame, text="Results:").pack(anchor="w", padx=15, pady=(20, 5))
        self.results_text = ctk.CTkTextbox(left_frame, height=200)
        self.results_text.pack(fill="x", padx=15, pady=(0, 15))

        # Right - Image display
        right_frame = ctk.CTkFrame(self.tab_testing)
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 20), pady=20)

        self.image_label = ctk.CTkLabel(
            right_frame,
            text="Select an image to test",
            font=ctk.CTkFont(size=16)
        )
        self.image_label.pack(fill="both", expand=True, padx=20, pady=20)

        self.test_image_path = None

    def browse_model(self):
        """Browse for model file."""
        file = filedialog.askopenfilename(
            title="Select Model",
            filetypes=[("PyTorch models", "*.pt"), ("ONNX models", "*.onnx"), ("All files", "*.*")]
        )
        if file:
            self.test_model_combo.set(file)

    def select_test_image(self):
        """Select image for testing."""
        file = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file:
            self.test_image_path = file
            self.display_image(file)

    def display_image(self, path, detections=None):
        """Display image in the testing tab."""
        try:
            img = Image.open(path)
            # Resize to fit
            img.thumbnail((800, 600))
            photo = ctk.CTkImage(img, size=img.size)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            self.image_label.configure(text=f"Error loading image: {e}")

    def run_detection(self):
        """Run detection on selected image."""
        if not self.test_image_path:
            messagebox.showerror("Error", "Please select an image first.")
            return

        model_path = self.test_model_combo.get()
        if not model_path:
            messagebox.showerror("Error", "Please select a model.")
            return

        try:
            from ultralytics import YOLO
            import time

            self.results_text.delete("1.0", "end")
            self.results_text.insert("end", "Running detection...\n")

            # Load model
            start = time.time()
            model = YOLO(model_path)
            load_time = time.time() - start

            # Run inference
            start = time.time()
            results = model.predict(
                self.test_image_path,
                conf=self.conf_slider.get(),
                verbose=False
            )
            inference_time = time.time() - start

            # Get results
            result = results[0]
            detections = []

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    conf = float(box.conf[0])
                    detections.append(f"{cls_name}: {conf:.2%}")

            # Display annotated image
            annotated = result.plot()
            img = Image.fromarray(annotated[..., ::-1])
            img.thumbnail((800, 600))
            photo = ctk.CTkImage(img, size=img.size)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

            # Show results
            self.results_text.delete("1.0", "end")
            self.results_text.insert("end", f"Found {len(detections)} objects\n")
            self.results_text.insert("end", f"Load time: {load_time*1000:.1f}ms\n")
            self.results_text.insert("end", f"Inference: {inference_time*1000:.1f}ms\n\n")
            for det in detections:
                self.results_text.insert("end", f"  - {det}\n")

        except ImportError:
            messagebox.showerror("Error", "Ultralytics not installed.\nRun: pip install ultralytics")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ==================== EXPORT TAB ====================
    def create_export_tab(self):
        """Create model export tab."""
        frame = ctk.CTkFrame(self.tab_export)
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            frame,
            text="Export Model",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=15, pady=15)

        # Model selection
        ctk.CTkLabel(frame, text="Select Model:").pack(anchor="w", padx=15, pady=(10, 5))
        self.export_model_combo = ctk.CTkComboBox(
            frame,
            values=self.get_model_files(),
            width=400
        )
        self.export_model_combo.pack(anchor="w", padx=15)

        ctk.CTkButton(
            frame,
            text="Browse...",
            command=self.browse_export_model,
            width=100
        ).pack(anchor="w", padx=15, pady=5)

        # Format selection
        ctk.CTkLabel(frame, text="Export Format:").pack(anchor="w", padx=15, pady=(20, 5))

        formats = [
            ("ONNX", "onnx"),
            ("TensorRT", "engine"),
            ("CoreML", "coreml"),
            ("TFLite", "tflite"),
            ("OpenVINO", "openvino"),
            ("TorchScript", "torchscript"),
        ]

        self.export_format = ctk.StringVar(value="onnx")
        format_frame = ctk.CTkFrame(frame, fg_color="transparent")
        format_frame.pack(anchor="w", padx=15)

        for i, (name, value) in enumerate(formats):
            ctk.CTkRadioButton(
                format_frame,
                text=name,
                variable=self.export_format,
                value=value
            ).grid(row=i // 3, column=i % 3, padx=15, pady=5)

        # Options
        ctk.CTkLabel(frame, text="Options:").pack(anchor="w", padx=15, pady=(20, 5))

        self.half_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(frame, text="Half Precision (FP16)", variable=self.half_var).pack(anchor="w", padx=15)

        self.dynamic_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(frame, text="Dynamic Batch", variable=self.dynamic_var).pack(anchor="w", padx=15, pady=5)

        # Export button
        ctk.CTkButton(
            frame,
            text="Export Model",
            command=self.export_model,
            width=200,
            fg_color="#22c55e",
            hover_color="#16a34a"
        ).pack(anchor="w", padx=15, pady=30)

        # Status
        self.export_status = ctk.CTkLabel(frame, text="", text_color="gray")
        self.export_status.pack(anchor="w", padx=15)

    def browse_export_model(self):
        """Browse for model to export."""
        file = filedialog.askopenfilename(
            title="Select Model",
            filetypes=[("PyTorch models", "*.pt")]
        )
        if file:
            self.export_model_combo.set(file)

    def export_model(self):
        """Export the model."""
        model_path = self.export_model_combo.get()
        if not model_path:
            messagebox.showerror("Error", "Please select a model.")
            return

        export_format = self.export_format.get()
        self.export_status.configure(text=f"Exporting to {export_format}...", text_color="#22c55e")
        self.update()

        try:
            from ultralytics import YOLO

            model = YOLO(model_path)
            model.export(
                format=export_format,
                half=self.half_var.get(),
                dynamic=self.dynamic_var.get()
            )

            self.export_status.configure(text=f"Exported successfully!", text_color="#22c55e")
            messagebox.showinfo("Success", f"Model exported to {export_format} format!")

        except ImportError:
            messagebox.showerror("Error", "Ultralytics not installed.\nRun: pip install ultralytics")
        except Exception as e:
            self.export_status.configure(text="Export failed", text_color="#ef4444")
            messagebox.showerror("Error", str(e))

    # ==================== HELPER METHODS ====================
    def get_dataset_names(self):
        """Get list of dataset names."""
        if self.datasets_path.exists():
            return [d.name for d in self.datasets_path.iterdir() if d.is_dir()]
        return []

    def get_model_files(self):
        """Get list of model files."""
        models = []
        # Pretrained
        models.extend(["yolo26n.pt", "yolo11n.pt", "yolov8n.pt"])
        # From runs
        if self.runs_path.exists():
            models.extend([str(f) for f in self.runs_path.glob("**/weights/*.pt")])
        return models

    def count_datasets(self):
        if self.datasets_path.exists():
            return len([d for d in self.datasets_path.iterdir() if d.is_dir()])
        return 0

    def count_images(self):
        count = 0
        if self.datasets_path.exists():
            for img in self.datasets_path.glob("**/*.jpg"):
                count += 1
            for img in self.datasets_path.glob("**/*.png"):
                count += 1
        return count

    def count_models(self):
        count = 0
        if self.runs_path.exists():
            count = len(list(self.runs_path.glob("**/weights/*.pt")))
        return count

    def count_exports(self):
        count = 0
        if self.exports_path.exists():
            for ext in ["*.onnx", "*.engine", "*.tflite"]:
                count += len(list(self.exports_path.glob(f"**/{ext}")))
        return count


def main():
    """Main entry point."""
    app = YOLOTrainingStudio()
    app.mainloop()


if __name__ == "__main__":
    main()
