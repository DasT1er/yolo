"""
YOLO Training Studio - Professional Desktop Application
=======================================================

Professional YOLO training application with modern GUI.
Supports YOLO v5, v8, v11, and v26.

Usage: python app.py
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import shutil
import yaml
import time


# Theme Configuration
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Colors
COLORS = {
    "primary": "#6366f1",
    "primary_hover": "#4f46e5",
    "success": "#22c55e",
    "success_hover": "#16a34a",
    "danger": "#ef4444",
    "danger_hover": "#dc2626",
    "warning": "#f59e0b",
    "bg_dark": "#1e1e2e",
    "bg_card": "#2d2d3d",
    "text": "#e2e8f0",
    "text_muted": "#94a3b8",
}

# Model Information
MODEL_INFO = {
    "yolo26n": {"name": "YOLO26 Nano", "size": "Nano", "params": "2.6M", "speed": "38.9ms", "mAP": "40.3%", "desc": "Schnellstes - Edge Devices, Raspberry Pi"},
    "yolo26s": {"name": "YOLO26 Small", "size": "Small", "params": "9.1M", "speed": "52.3ms", "mAP": "45.4%", "desc": "Gute Balance für Echtzeit"},
    "yolo26m": {"name": "YOLO26 Medium", "size": "Medium", "params": "20.1M", "speed": "78.5ms", "mAP": "49.0%", "desc": "Empfohlen - Beste Balance"},
    "yolo11n": {"name": "YOLO11 Nano", "size": "Nano", "params": "2.6M", "speed": "56.1ms", "mAP": "39.5%", "desc": "Schnell und stabil"},
    "yolo11s": {"name": "YOLO11 Small", "size": "Small", "params": "9.4M", "speed": "90.0ms", "mAP": "47.0%", "desc": "Gute Genauigkeit"},
    "yolo11m": {"name": "YOLO11 Medium", "size": "Medium", "params": "20.1M", "speed": "183ms", "mAP": "51.5%", "desc": "Hohe Genauigkeit"},
    "yolov8n": {"name": "YOLOv8 Nano", "size": "Nano", "params": "3.2M", "speed": "80.4ms", "mAP": "37.3%", "desc": "Legacy - weit verbreitet"},
    "yolov8s": {"name": "YOLOv8 Small", "size": "Small", "params": "11.2M", "speed": "128ms", "mAP": "44.9%", "desc": "Legacy - stabil"},
    "yolov8m": {"name": "YOLOv8 Medium", "size": "Medium", "params": "25.9M", "speed": "234ms", "mAP": "50.2%", "desc": "Legacy - bewährt"},
}


class YOLOTrainingStudio(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        # Window setup
        self.title("YOLO Training Studio Pro")
        self.geometry("1500x950")
        self.minsize(1400, 850)

        # Paths
        self.base_path = Path(__file__).parent
        self.datasets_path = self.base_path / "datasets"
        self.runs_path = self.base_path / "runs"
        self.exports_path = self.base_path / "exports"

        # Ensure directories exist
        self.datasets_path.mkdir(exist_ok=True)
        self.runs_path.mkdir(exist_ok=True)
        self.exports_path.mkdir(exist_ok=True)

        # State
        self.training_process = None
        self.is_training = False
        self.selected_model = ctk.StringVar(value="yolo11n")
        self.test_image_path = None

        # Create UI
        self.create_ui()

        # Bind tab change to refresh datasets
        self.tabview.configure(command=self.on_tab_change)

    def on_tab_change(self):
        """Called when tab changes."""
        self.refresh_all_dropdowns()

    def refresh_all_dropdowns(self):
        """Refresh all dataset and model dropdowns."""
        datasets = self.get_dataset_names()
        models = self.get_model_files()

        if hasattr(self, 'training_dataset'):
            self.training_dataset.configure(values=datasets)
            if datasets and not self.training_dataset.get():
                self.training_dataset.set(datasets[0])

        if hasattr(self, 'test_model_combo'):
            self.test_model_combo.configure(values=models)

        if hasattr(self, 'export_model_combo'):
            self.export_model_combo.configure(values=models)

    def create_ui(self):
        """Create the main user interface."""
        # Header
        self.create_header()

        # Main content with tabs
        self.tabview = ctk.CTkTabview(self, corner_radius=15, segmented_button_selected_color=COLORS["primary"])
        self.tabview.pack(fill="both", expand=True, padx=25, pady=(0, 25))

        # Add tabs
        self.tab_dashboard = self.tabview.add("📊 Dashboard")
        self.tab_dataset = self.tabview.add("📁 Dataset")
        self.tab_training = self.tabview.add("🚀 Training")
        self.tab_testing = self.tabview.add("🔍 Testing")
        self.tab_export = self.tabview.add("📦 Export")

        # Fill tabs
        self.create_dashboard_tab()
        self.create_dataset_tab()
        self.create_training_tab()
        self.create_testing_tab()
        self.create_export_tab()

    def create_header(self):
        """Create application header."""
        header = ctk.CTkFrame(self, height=90, corner_radius=15, fg_color=COLORS["bg_card"])
        header.pack(fill="x", padx=25, pady=(25, 15))
        header.pack_propagate(False)

        # Left side - Logo and title
        left = ctk.CTkFrame(header, fg_color="transparent")
        left.pack(side="left", fill="y", padx=20)

        ctk.CTkLabel(
            left,
            text="🎯 YOLO Training Studio",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=COLORS["text"]
        ).pack(side="left", pady=25)

        ctk.CTkLabel(
            left,
            text="Professional Edition",
            font=ctk.CTkFont(size=14),
            text_color=COLORS["primary"]
        ).pack(side="left", padx=15, pady=25)

        # Right side - Status
        right = ctk.CTkFrame(header, fg_color="transparent")
        right.pack(side="right", fill="y", padx=20)

        self.gpu_label = ctk.CTkLabel(
            right,
            text=self.get_gpu_status(),
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_muted"]
        )
        self.gpu_label.pack(side="right", pady=25)

    def get_gpu_status(self):
        """Get GPU status string."""
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                return f"🟢 GPU: {name}"
            return "🟡 CPU Mode"
        except:
            return "🟡 CPU Mode"

    # ==================== DASHBOARD TAB ====================
    def create_dashboard_tab(self):
        """Create dashboard tab content."""
        main = ctk.CTkFrame(self.tab_dashboard, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20, pady=20)

        # Top row - Stats
        stats_frame = ctk.CTkFrame(main, fg_color="transparent")
        stats_frame.pack(fill="x", pady=(0, 20))

        stats = [
            ("📁", "Datasets", self.count_datasets(), COLORS["primary"]),
            ("🖼️", "Bilder", self.count_images(), COLORS["success"]),
            ("🤖", "Modelle", self.count_models(), COLORS["warning"]),
            ("📦", "Exports", self.count_exports(), "#8b5cf6"),
        ]

        for i, (icon, title, value, color) in enumerate(stats):
            card = ctk.CTkFrame(stats_frame, fg_color=COLORS["bg_card"], corner_radius=15, height=120)
            card.pack(side="left", fill="x", expand=True, padx=(0 if i == 0 else 10, 0))
            card.pack_propagate(False)

            ctk.CTkLabel(card, text=icon, font=ctk.CTkFont(size=28)).pack(pady=(20, 5))
            ctk.CTkLabel(card, text=str(value), font=ctk.CTkFont(size=32, weight="bold"), text_color=color).pack()
            ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=13), text_color=COLORS["text_muted"]).pack()

        # Bottom row - Model comparison and Quick start
        bottom = ctk.CTkFrame(main, fg_color="transparent")
        bottom.pack(fill="both", expand=True)

        # Left - Model comparison
        left_card = ctk.CTkFrame(bottom, fg_color=COLORS["bg_card"], corner_radius=15)
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ctk.CTkLabel(
            left_card,
            text="📊 Model-Größen Vergleich",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=25, pady=(20, 15))

        # Size explanation
        sizes_frame = ctk.CTkFrame(left_card, fg_color="transparent")
        sizes_frame.pack(fill="x", padx=25, pady=(0, 15))

        size_info = [
            ("n (Nano)", "Schnellste", "Edge/Mobile", COLORS["success"]),
            ("s (Small)", "Schnell", "Echtzeit", "#22d3ee"),
            ("m (Medium)", "Balance", "Empfohlen", COLORS["primary"]),
            ("l (Large)", "Genau", "Server", COLORS["warning"]),
            ("x (XLarge)", "Maximum", "Cloud", COLORS["danger"]),
        ]

        for size, speed, use, color in size_info:
            row = ctk.CTkFrame(sizes_frame, fg_color="transparent")
            row.pack(fill="x", pady=3)
            ctk.CTkLabel(row, text=size, font=ctk.CTkFont(weight="bold"), width=100, anchor="w", text_color=color).pack(side="left")
            ctk.CTkLabel(row, text=speed, width=100, anchor="w", text_color=COLORS["text_muted"]).pack(side="left")
            ctk.CTkLabel(row, text=use, anchor="w", text_color=COLORS["text_muted"]).pack(side="left")

        # Table
        ctk.CTkLabel(left_card, text="", height=10).pack()  # Spacer

        table = ctk.CTkFrame(left_card, fg_color="transparent")
        table.pack(fill="x", padx=25, pady=(0, 20))

        headers = ["Modell", "mAP", "Speed", "Params", "Empfehlung"]
        for i, h in enumerate(headers):
            ctk.CTkLabel(table, text=h, font=ctk.CTkFont(weight="bold"), width=120, anchor="w").grid(row=0, column=i, pady=5)

        models_data = [
            ("🆕 YOLO26m", "49.0%", "78ms", "20M", "⭐ Beste Wahl"),
            ("🆕 YOLO26n", "40.3%", "39ms", "2.6M", "📱 Edge"),
            ("✓ YOLO11m", "51.5%", "183ms", "20M", "🎯 Genau"),
            ("✓ YOLO11n", "39.5%", "56ms", "2.6M", "⚡ Schnell"),
        ]

        for row, (model, mAP, speed, params, rec) in enumerate(models_data, 1):
            ctk.CTkLabel(table, text=model, width=120, anchor="w").grid(row=row, column=0, pady=3)
            ctk.CTkLabel(table, text=mAP, width=120, anchor="w", text_color=COLORS["success"]).grid(row=row, column=1, pady=3)
            ctk.CTkLabel(table, text=speed, width=120, anchor="w").grid(row=row, column=2, pady=3)
            ctk.CTkLabel(table, text=params, width=120, anchor="w").grid(row=row, column=3, pady=3)
            ctk.CTkLabel(table, text=rec, width=120, anchor="w", text_color=COLORS["warning"]).grid(row=row, column=4, pady=3)

        # Right - Quick Start
        right_card = ctk.CTkFrame(bottom, fg_color=COLORS["bg_card"], corner_radius=15, width=350)
        right_card.pack(side="right", fill="y", padx=(10, 0))
        right_card.pack_propagate(False)

        ctk.CTkLabel(
            right_card,
            text="🚀 Quick Start",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=25, pady=(20, 15))

        steps = [
            ("1️⃣", "Dataset erstellen oder importieren"),
            ("2️⃣", "Bilder und Labels hinzufügen"),
            ("3️⃣", "Modell und Parameter wählen"),
            ("4️⃣", "Training starten"),
            ("5️⃣", "Model testen & exportieren"),
        ]

        for icon, text in steps:
            row = ctk.CTkFrame(right_card, fg_color="transparent")
            row.pack(fill="x", padx=25, pady=8)
            ctk.CTkLabel(row, text=icon, font=ctk.CTkFont(size=16)).pack(side="left")
            ctk.CTkLabel(row, text=text, font=ctk.CTkFont(size=13), text_color=COLORS["text_muted"]).pack(side="left", padx=10)

        # Quick actions
        ctk.CTkLabel(right_card, text="", height=20).pack()

        ctk.CTkButton(
            right_card,
            text="📁 Neues Dataset",
            command=lambda: self.tabview.set("📁 Dataset"),
            width=280,
            height=40,
            fg_color=COLORS["primary"],
            hover_color=COLORS["primary_hover"]
        ).pack(pady=5)

        ctk.CTkButton(
            right_card,
            text="🚀 Training starten",
            command=lambda: self.tabview.set("🚀 Training"),
            width=280,
            height=40,
            fg_color=COLORS["success"],
            hover_color=COLORS["success_hover"]
        ).pack(pady=5)

    # ==================== DATASET TAB ====================
    def create_dataset_tab(self):
        """Create dataset management tab."""
        main = ctk.CTkFrame(self.tab_dataset, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20, pady=20)

        # Left panel - Dataset list
        left = ctk.CTkFrame(main, fg_color=COLORS["bg_card"], corner_radius=15, width=320)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        ctk.CTkLabel(left, text="📁 Deine Datasets", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=20)

        self.dataset_listbox = ctk.CTkScrollableFrame(left, fg_color="transparent")
        self.dataset_listbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.refresh_dataset_list()

        btn_frame = ctk.CTkFrame(left, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkButton(btn_frame, text="🔄 Aktualisieren", command=self.refresh_dataset_list, width=140).pack(side="left", padx=(0, 5))
        ctk.CTkButton(btn_frame, text="🗑️ Löschen", command=self.delete_selected_dataset, width=140, fg_color=COLORS["danger"], hover_color=COLORS["danger_hover"]).pack(side="left")

        # Right panel
        right = ctk.CTkFrame(main, fg_color="transparent")
        right.pack(side="right", fill="both", expand=True)

        # Create section
        create_card = ctk.CTkFrame(right, fg_color=COLORS["bg_card"], corner_radius=15)
        create_card.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(create_card, text="✨ Neues Dataset erstellen", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=25, pady=(20, 15))

        form = ctk.CTkFrame(create_card, fg_color="transparent")
        form.pack(fill="x", padx=25, pady=(0, 20))

        ctk.CTkLabel(form, text="Name:", width=100, anchor="w").grid(row=0, column=0, pady=8)
        self.dataset_name_entry = ctk.CTkEntry(form, width=350, placeholder_text="mein_dataset")
        self.dataset_name_entry.grid(row=0, column=1, pady=8)

        ctk.CTkLabel(form, text="Klassen:", width=100, anchor="w").grid(row=1, column=0, pady=8)
        self.classes_entry = ctk.CTkEntry(form, width=350, placeholder_text="person, auto, hund, katze")
        self.classes_entry.grid(row=1, column=1, pady=8)

        ctk.CTkButton(form, text="✨ Dataset erstellen", command=self.create_dataset, width=200, fg_color=COLORS["success"], hover_color=COLORS["success_hover"]).grid(row=2, column=1, pady=15, sticky="w")

        # Import section
        import_card = ctk.CTkFrame(right, fg_color=COLORS["bg_card"], corner_radius=15)
        import_card.pack(fill="x")

        ctk.CTkLabel(import_card, text="📥 Dataset importieren", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=25, pady=(20, 15))

        btn_row = ctk.CTkFrame(import_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=25, pady=(0, 20))

        ctk.CTkButton(btn_row, text="📂 Ordner importieren", command=self.import_from_folder, width=200).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btn_row, text="📦 ZIP importieren", command=self.import_zip, width=200).pack(side="left")

        # Info
        info_card = ctk.CTkFrame(right, fg_color=COLORS["bg_card"], corner_radius=15)
        info_card.pack(fill="both", expand=True, pady=(15, 0))

        ctk.CTkLabel(info_card, text="ℹ️ Dataset Struktur", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=25, pady=(20, 10))

        structure = """datasets/
└── mein_dataset/
    ├── data.yaml          # Konfiguration
    ├── images/
    │   ├── train/         # Trainingsbilder (80%)
    │   ├── val/           # Validierungsbilder (20%)
    │   └── test/          # Testbilder (optional)
    └── labels/
        ├── train/         # YOLO Format Labels
        ├── val/
        └── test/"""

        ctk.CTkLabel(info_card, text=structure, font=ctk.CTkFont(family="Consolas", size=12), justify="left", text_color=COLORS["text_muted"]).pack(anchor="w", padx=25, pady=(0, 20))

    def refresh_dataset_list(self):
        """Refresh the dataset list."""
        for widget in self.dataset_listbox.winfo_children():
            widget.destroy()

        self.selected_dataset_path = None

        if self.datasets_path.exists():
            datasets = sorted([d for d in self.datasets_path.iterdir() if d.is_dir()])
            for dataset in datasets:
                # Get info
                img_count = len(list(dataset.glob("**/*.jpg"))) + len(list(dataset.glob("**/*.png")))

                btn = ctk.CTkButton(
                    self.dataset_listbox,
                    text=f"📁 {dataset.name}\n     {img_count} Bilder",
                    fg_color="transparent",
                    text_color=COLORS["text"],
                    hover_color=COLORS["primary"],
                    anchor="w",
                    height=50,
                    command=lambda d=dataset: self.select_dataset(d)
                )
                btn.pack(fill="x", pady=3)

        self.refresh_all_dropdowns()

    def select_dataset(self, dataset_path):
        """Select a dataset."""
        self.selected_dataset_path = dataset_path

        # Show info
        yaml_path = dataset_path / "data.yaml"
        info = f"Dataset: {dataset_path.name}\n"

        if yaml_path.exists():
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
                classes = data.get("names", {})
                if isinstance(classes, dict):
                    classes = list(classes.values())
                info += f"Klassen: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}\n"

        img_count = len(list(dataset_path.glob("**/*.jpg"))) + len(list(dataset_path.glob("**/*.png")))
        info += f"Bilder: {img_count}"

        messagebox.showinfo("Dataset ausgewählt", info)

    def delete_selected_dataset(self):
        """Delete the selected dataset."""
        if not hasattr(self, 'selected_dataset_path') or not self.selected_dataset_path:
            messagebox.showerror("Fehler", "Bitte wähle zuerst ein Dataset aus.")
            return

        if messagebox.askyesno("Löschen bestätigen", f"Dataset '{self.selected_dataset_path.name}' wirklich löschen?"):
            try:
                shutil.rmtree(self.selected_dataset_path)
                self.selected_dataset_path = None
                self.refresh_dataset_list()
                messagebox.showinfo("Erfolg", "Dataset gelöscht!")
            except Exception as e:
                messagebox.showerror("Fehler", str(e))

    def create_dataset(self):
        """Create a new dataset."""
        name = self.dataset_name_entry.get().strip().replace(" ", "_")
        classes = self.classes_entry.get().strip()

        if not name:
            messagebox.showerror("Fehler", "Bitte gib einen Dataset-Namen ein.")
            return

        dataset_path = self.datasets_path / name
        if dataset_path.exists():
            messagebox.showerror("Fehler", f"Dataset '{name}' existiert bereits.")
            return

        try:
            for split in ["train", "val", "test"]:
                (dataset_path / "images" / split).mkdir(parents=True)
                (dataset_path / "labels" / split).mkdir(parents=True)

            class_list = [c.strip() for c in classes.split(",") if c.strip()] or ["object"]

            data = {
                "path": str(dataset_path.absolute()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {i: n for i, n in enumerate(class_list)},
                "nc": len(class_list)
            }

            with open(dataset_path / "data.yaml", "w") as f:
                yaml.dump(data, f, sort_keys=False)

            messagebox.showinfo("Erfolg", f"Dataset '{name}' erstellt!")
            self.refresh_dataset_list()
            self.dataset_name_entry.delete(0, "end")
            self.classes_entry.delete(0, "end")

        except Exception as e:
            messagebox.showerror("Fehler", str(e))

    def import_from_folder(self):
        """Import dataset from folder."""
        folder = filedialog.askdirectory(title="Dataset Ordner auswählen")
        if folder:
            folder_path = Path(folder)
            target = self.datasets_path / folder_path.name

            try:
                shutil.copytree(folder_path, target)
                messagebox.showinfo("Erfolg", f"'{folder_path.name}' importiert!")
                self.refresh_dataset_list()
            except Exception as e:
                messagebox.showerror("Fehler", str(e))

    def import_zip(self):
        """Import dataset from ZIP."""
        zip_file = filedialog.askopenfilename(title="ZIP Datei auswählen", filetypes=[("ZIP", "*.zip")])
        if zip_file:
            import zipfile
            zip_path = Path(zip_file)
            target = self.datasets_path / zip_path.stem

            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(target)
                messagebox.showinfo("Erfolg", f"'{zip_path.stem}' importiert!")
                self.refresh_dataset_list()
            except Exception as e:
                messagebox.showerror("Fehler", str(e))

    # ==================== TRAINING TAB ====================
    def create_training_tab(self):
        """Create training configuration tab."""
        main = ctk.CTkFrame(self.tab_training, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20, pady=20)

        # Left - Config
        left = ctk.CTkFrame(main, fg_color=COLORS["bg_card"], corner_radius=15, width=480)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        # Header with start button
        header = ctk.CTkFrame(left, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=20)

        ctk.CTkLabel(header, text="⚙️ Training Konfiguration", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")

        self.start_btn = ctk.CTkButton(
            header,
            text="▶️ Start",
            command=self.start_training,
            width=100,
            height=35,
            fg_color=COLORS["success"],
            hover_color=COLORS["success_hover"],
            font=ctk.CTkFont(weight="bold")
        )
        self.start_btn.pack(side="right")

        self.stop_btn = ctk.CTkButton(
            header,
            text="⏹️ Stop",
            command=self.stop_training,
            width=80,
            height=35,
            fg_color=COLORS["danger"],
            hover_color=COLORS["danger_hover"],
            state="disabled"
        )
        self.stop_btn.pack(side="right", padx=(0, 10))

        # Scrollable content
        content = ctk.CTkScrollableFrame(left, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=10)

        # Dataset
        ctk.CTkLabel(content, text="📁 Dataset:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 5))
        self.training_dataset = ctk.CTkComboBox(content, values=self.get_dataset_names(), width=420)
        self.training_dataset.pack(anchor="w", padx=10)

        # Model selection with info
        ctk.CTkLabel(content, text="🤖 Modell:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(20, 5))

        model_frame = ctk.CTkFrame(content, fg_color="transparent")
        model_frame.pack(fill="x", padx=10)

        models = ["yolo26n", "yolo26s", "yolo26m", "yolo11n", "yolo11s", "yolo11m", "yolov8n", "yolov8s", "yolov8m"]

        for i, model in enumerate(models):
            info = MODEL_INFO.get(model, {})
            btn = ctk.CTkRadioButton(
                model_frame,
                text=f"{model} ({info.get('mAP', 'N/A')})",
                variable=self.selected_model,
                value=model,
                command=self.update_model_info
            )
            btn.grid(row=i // 3, column=i % 3, padx=8, pady=5, sticky="w")

        # Model info display
        self.model_info_label = ctk.CTkLabel(
            content,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_muted"]
        )
        self.model_info_label.pack(anchor="w", padx=10, pady=5)
        self.update_model_info()

        # Parameters
        ctk.CTkLabel(content, text="📊 Parameter:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(20, 10))

        params = ctk.CTkFrame(content, fg_color="transparent")
        params.pack(fill="x", padx=10)

        # Epochs
        ctk.CTkLabel(params, text="Epochs:", width=80).grid(row=0, column=0, pady=5, sticky="w")
        self.epochs_var = ctk.IntVar(value=100)
        self.epochs_slider = ctk.CTkSlider(params, from_=10, to=300, variable=self.epochs_var, width=200)
        self.epochs_slider.grid(row=0, column=1, pady=5, padx=5)
        self.epochs_label = ctk.CTkLabel(params, text="100", width=50)
        self.epochs_label.grid(row=0, column=2, pady=5)
        self.epochs_slider.configure(command=lambda v: self.epochs_label.configure(text=str(int(v))))

        # Batch
        ctk.CTkLabel(params, text="Batch:", width=80).grid(row=1, column=0, pady=5, sticky="w")
        self.batch_var = ctk.IntVar(value=16)
        self.batch_slider = ctk.CTkSlider(params, from_=1, to=64, variable=self.batch_var, width=200)
        self.batch_slider.grid(row=1, column=1, pady=5, padx=5)
        self.batch_label = ctk.CTkLabel(params, text="16", width=50)
        self.batch_label.grid(row=1, column=2, pady=5)
        self.batch_slider.configure(command=lambda v: self.batch_label.configure(text=str(int(v))))

        # Image size
        ctk.CTkLabel(params, text="Img Size:", width=80).grid(row=2, column=0, pady=5, sticky="w")
        self.imgsz_combo = ctk.CTkComboBox(params, values=["320", "416", "512", "640", "768", "1024"], width=100)
        self.imgsz_combo.set("640")
        self.imgsz_combo.grid(row=2, column=1, pady=5, padx=5, sticky="w")

        # Device
        ctk.CTkLabel(params, text="Device:", width=80).grid(row=3, column=0, pady=5, sticky="w")
        self.device_combo = ctk.CTkComboBox(params, values=["0", "cpu", "0,1", "mps"], width=100)
        self.device_combo.set("0")
        self.device_combo.grid(row=3, column=1, pady=5, padx=5, sticky="w")

        # Options
        opts = ctk.CTkFrame(content, fg_color="transparent")
        opts.pack(fill="x", padx=10, pady=10)

        self.pretrained_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(opts, text="Pretrained Weights", variable=self.pretrained_var).pack(side="left", padx=(0, 15))

        self.augment_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(opts, text="Augmentation", variable=self.augment_var).pack(side="left")

        # Right - Logs
        right = ctk.CTkFrame(main, fg_color=COLORS["bg_card"], corner_radius=15)
        right.pack(side="right", fill="both", expand=True)

        log_header = ctk.CTkFrame(right, fg_color="transparent")
        log_header.pack(fill="x", padx=20, pady=15)

        ctk.CTkLabel(log_header, text="📋 Training Logs", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")

        self.status_label = ctk.CTkLabel(log_header, text="⚪ Bereit", text_color=COLORS["text_muted"])
        self.status_label.pack(side="right")

        self.training_log = ctk.CTkTextbox(right, font=ctk.CTkFont(family="Consolas", size=12))
        self.training_log.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Progress
        progress_frame = ctk.CTkFrame(right, fg_color="transparent")
        progress_frame.pack(fill="x", padx=15, pady=(0, 15))

        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=400)
        self.progress_bar.pack(side="left", padx=(0, 15))
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(progress_frame, text="0%", width=50)
        self.progress_label.pack(side="left")

    def update_model_info(self):
        """Update the model info display."""
        model = self.selected_model.get()
        info = MODEL_INFO.get(model, {})
        self.model_info_label.configure(
            text=f"ℹ️ {info.get('desc', '')} | Speed: {info.get('speed', 'N/A')} | Params: {info.get('params', 'N/A')}"
        )

    def start_training(self):
        """Start YOLO training."""
        dataset = self.training_dataset.get()
        if not dataset:
            messagebox.showerror("Fehler", "Bitte wähle ein Dataset aus.")
            return

        data_yaml = self.datasets_path / dataset / "data.yaml"
        if not data_yaml.exists():
            messagebox.showerror("Fehler", f"data.yaml nicht gefunden in {dataset}")
            return

        self.is_training = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="🟢 Training...", text_color=COLORS["success"])
        self.training_log.delete("1.0", "end")
        self.progress_bar.set(0)

        model = self.selected_model.get()
        epochs = int(self.epochs_var.get())
        batch = int(self.batch_var.get())
        imgsz = int(self.imgsz_combo.get())
        device = self.device_combo.get()

        self.training_log.insert("end", f"{'='*50}\n")
        self.training_log.insert("end", f"🚀 Training gestartet\n")
        self.training_log.insert("end", f"📁 Dataset: {dataset}\n")
        self.training_log.insert("end", f"🤖 Model: {model}\n")
        self.training_log.insert("end", f"📊 Epochs: {epochs} | Batch: {batch} | ImgSize: {imgsz}\n")
        self.training_log.insert("end", f"{'='*50}\n\n")

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

                    # Parse progress
                    if "Epoch" in line and "/" in line:
                        try:
                            parts = line.split()
                            for p in parts:
                                if "/" in p:
                                    current, total = p.split("/")
                                    progress = int(current) / int(total)
                                    self.after(0, lambda p=progress: self.update_progress(p))
                                    break
                        except:
                            pass

                self.training_process.wait()
                self.after(0, self.training_finished)

            except Exception as e:
                self.after(0, lambda: self.append_log(f"❌ Fehler: {e}\n"))
                self.after(0, self.training_finished)

        threading.Thread(target=run_training, daemon=True).start()

    def append_log(self, text):
        """Append text to training log."""
        self.training_log.insert("end", text)
        self.training_log.see("end")

    def update_progress(self, value):
        """Update progress bar."""
        self.progress_bar.set(value)
        self.progress_label.configure(text=f"{int(value * 100)}%")

    def stop_training(self):
        """Stop training."""
        if self.training_process:
            self.training_process.terminate()
            self.append_log("\n⏹️ Training vom Benutzer gestoppt\n")

    def training_finished(self):
        """Called when training finishes."""
        self.is_training = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="✅ Fertig", text_color=COLORS["success"])
        self.progress_bar.set(1)
        self.progress_label.configure(text="100%")
        self.append_log("\n✅ Training abgeschlossen!\n")

    # ==================== TESTING TAB ====================
    def create_testing_tab(self):
        """Create model testing tab."""
        main = ctk.CTkFrame(self.tab_testing, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20, pady=20)

        # Left - Controls
        left = ctk.CTkFrame(main, fg_color=COLORS["bg_card"], corner_radius=15, width=380)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        ctk.CTkLabel(left, text="🔍 Model Testing", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=20)

        # Model
        ctk.CTkLabel(left, text="🤖 Modell:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=20, pady=(0, 5))
        self.test_model_combo = ctk.CTkComboBox(left, values=self.get_model_files(), width=320)
        self.test_model_combo.pack(anchor="w", padx=20)

        ctk.CTkButton(left, text="📂 Model laden...", command=self.browse_model, width=320).pack(anchor="w", padx=20, pady=10)

        # Confidence
        ctk.CTkLabel(left, text="🎯 Confidence:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=20, pady=(20, 5))

        conf_frame = ctk.CTkFrame(left, fg_color="transparent")
        conf_frame.pack(fill="x", padx=20)

        self.conf_slider = ctk.CTkSlider(conf_frame, from_=0.01, to=1.0, width=250)
        self.conf_slider.set(0.25)
        self.conf_slider.pack(side="left")

        self.conf_label = ctk.CTkLabel(conf_frame, text="0.25", width=50)
        self.conf_label.pack(side="left", padx=10)
        self.conf_slider.configure(command=lambda v: self.conf_label.configure(text=f"{v:.2f}"))

        # Buttons
        ctk.CTkLabel(left, text="", height=20).pack()

        ctk.CTkButton(
            left,
            text="🖼️ Bild auswählen",
            command=self.select_test_image,
            width=320,
            height=40
        ).pack(anchor="w", padx=20, pady=5)

        ctk.CTkButton(
            left,
            text="▶️ Erkennung starten",
            command=self.run_detection,
            width=320,
            height=45,
            fg_color=COLORS["success"],
            hover_color=COLORS["success_hover"],
            font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", padx=20, pady=10)

        # Results
        ctk.CTkLabel(left, text="📊 Ergebnisse:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=20, pady=(20, 5))

        self.results_text = ctk.CTkTextbox(left, height=200, font=ctk.CTkFont(family="Consolas", size=12))
        self.results_text.pack(fill="x", padx=20, pady=(0, 20))

        # Right - Image
        right = ctk.CTkFrame(main, fg_color=COLORS["bg_card"], corner_radius=15)
        right.pack(side="right", fill="both", expand=True)

        self.image_label = ctk.CTkLabel(
            right,
            text="🖼️\n\nBild hier ablegen oder\n'Bild auswählen' klicken",
            font=ctk.CTkFont(size=18),
            text_color=COLORS["text_muted"]
        )
        self.image_label.pack(fill="both", expand=True, padx=30, pady=30)

    def browse_model(self):
        """Browse for model file."""
        file = filedialog.askopenfilename(
            title="Model auswählen",
            filetypes=[("PyTorch", "*.pt"), ("ONNX", "*.onnx"), ("Alle", "*.*")]
        )
        if file:
            self.test_model_combo.set(file)

    def select_test_image(self):
        """Select image for testing."""
        file = filedialog.askopenfilename(
            title="Bild auswählen",
            filetypes=[("Bilder", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file:
            self.test_image_path = file
            self.display_test_image(file)

    def display_test_image(self, path):
        """Display image in testing tab."""
        try:
            img = Image.open(path)
            img.thumbnail((900, 700))
            photo = ctk.CTkImage(img, size=img.size)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            self.image_label.configure(text=f"❌ Fehler: {e}")

    def run_detection(self):
        """Run detection on selected image."""
        if not self.test_image_path:
            messagebox.showerror("Fehler", "Bitte zuerst ein Bild auswählen.")
            return

        model_path = self.test_model_combo.get()
        if not model_path:
            messagebox.showerror("Fehler", "Bitte ein Model auswählen.")
            return

        try:
            from ultralytics import YOLO

            self.results_text.delete("1.0", "end")
            self.results_text.insert("end", "🔄 Lade Model...\n")
            self.update()

            start = time.time()
            model = YOLO(model_path)
            load_time = time.time() - start

            self.results_text.insert("end", "🔍 Führe Erkennung aus...\n")
            self.update()

            start = time.time()
            results = model.predict(self.test_image_path, conf=self.conf_slider.get(), verbose=False)
            inference_time = time.time() - start

            result = results[0]
            detections = []

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    conf = float(box.conf[0])
                    detections.append((cls_name, conf))

            # Display annotated image
            annotated = result.plot()
            img = Image.fromarray(annotated[..., ::-1])
            img.thumbnail((900, 700))
            photo = ctk.CTkImage(img, size=img.size)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

            # Show results
            self.results_text.delete("1.0", "end")
            self.results_text.insert("end", f"✅ Gefunden: {len(detections)} Objekte\n")
            self.results_text.insert("end", f"⏱️ Load: {load_time*1000:.0f}ms\n")
            self.results_text.insert("end", f"⏱️ Inference: {inference_time*1000:.0f}ms\n")
            self.results_text.insert("end", f"{'─'*30}\n")

            for name, conf in detections:
                self.results_text.insert("end", f"  • {name}: {conf:.1%}\n")

        except ImportError:
            messagebox.showerror("Fehler", "Ultralytics nicht installiert.\nRun: pip install ultralytics")
        except Exception as e:
            messagebox.showerror("Fehler", str(e))

    # ==================== EXPORT TAB ====================
    def create_export_tab(self):
        """Create model export tab."""
        main = ctk.CTkFrame(self.tab_export, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20, pady=20)

        # Left - Export config
        left = ctk.CTkFrame(main, fg_color=COLORS["bg_card"], corner_radius=15, width=450)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        ctk.CTkLabel(left, text="📦 Model Export", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=25, pady=20)

        # Model
        ctk.CTkLabel(left, text="🤖 Source Model:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=25, pady=(0, 5))
        self.export_model_combo = ctk.CTkComboBox(left, values=self.get_model_files(), width=380)
        self.export_model_combo.pack(anchor="w", padx=25)

        ctk.CTkButton(left, text="📂 Model laden...", command=self.browse_export_model, width=180).pack(anchor="w", padx=25, pady=10)

        # Format
        ctk.CTkLabel(left, text="📄 Export Format:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=25, pady=(20, 10))

        formats = [
            ("ONNX", "onnx", "Universal - CPU/GPU"),
            ("TensorRT", "engine", "NVIDIA GPUs"),
            ("CoreML", "coreml", "Apple iOS/macOS"),
            ("TFLite", "tflite", "Mobile/Edge"),
            ("OpenVINO", "openvino", "Intel Hardware"),
            ("TorchScript", "torchscript", "PyTorch Deploy"),
        ]

        self.export_format = ctk.StringVar(value="onnx")
        format_frame = ctk.CTkFrame(left, fg_color="transparent")
        format_frame.pack(fill="x", padx=25)

        for name, value, desc in formats:
            row = ctk.CTkFrame(format_frame, fg_color="transparent")
            row.pack(fill="x", pady=3)
            ctk.CTkRadioButton(row, text=name, variable=self.export_format, value=value, width=120).pack(side="left")
            ctk.CTkLabel(row, text=desc, text_color=COLORS["text_muted"], font=ctk.CTkFont(size=11)).pack(side="left", padx=10)

        # Options
        ctk.CTkLabel(left, text="⚙️ Optionen:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=25, pady=(25, 10))

        self.half_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(left, text="Half Precision (FP16) - Schneller, weniger Speicher", variable=self.half_var).pack(anchor="w", padx=25)

        self.dynamic_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(left, text="Dynamic Batch - Variable Batch-Größe", variable=self.dynamic_var).pack(anchor="w", padx=25, pady=10)

        # Export button
        ctk.CTkButton(
            left,
            text="📦 Exportieren",
            command=self.export_model,
            width=380,
            height=50,
            fg_color=COLORS["success"],
            hover_color=COLORS["success_hover"],
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=25, pady=25)

        self.export_status = ctk.CTkLabel(left, text="", text_color=COLORS["text_muted"])
        self.export_status.pack(anchor="w", padx=25)

        # Right - Info
        right = ctk.CTkFrame(main, fg_color=COLORS["bg_card"], corner_radius=15)
        right.pack(side="right", fill="both", expand=True)

        ctk.CTkLabel(right, text="ℹ️ Format Empfehlungen", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=25, pady=20)

        recommendations = [
            ("🔄 ONNX", "Universell - funktioniert überall, guter Standard", COLORS["primary"]),
            ("⚡ TensorRT", "NVIDIA GPUs - schnellste Inferenz auf RTX/Jetson", COLORS["success"]),
            ("🍎 CoreML", "Apple - für iPhone, iPad, Mac Apps", "#f472b6"),
            ("📱 TFLite", "Mobile - Android, Raspberry Pi, Edge TPU", COLORS["warning"]),
            ("🔷 OpenVINO", "Intel - optimiert für Intel CPUs/GPUs", "#22d3ee"),
        ]

        for title, desc, color in recommendations:
            card = ctk.CTkFrame(right, fg_color="#1e1e2e", corner_radius=10)
            card.pack(fill="x", padx=20, pady=5)

            ctk.CTkLabel(card, text=title, font=ctk.CTkFont(weight="bold"), text_color=color).pack(anchor="w", padx=15, pady=(10, 0))
            ctk.CTkLabel(card, text=desc, text_color=COLORS["text_muted"], font=ctk.CTkFont(size=12)).pack(anchor="w", padx=15, pady=(0, 10))

    def browse_export_model(self):
        """Browse for model to export."""
        file = filedialog.askopenfilename(title="Model auswählen", filetypes=[("PyTorch", "*.pt")])
        if file:
            self.export_model_combo.set(file)

    def export_model(self):
        """Export the model."""
        model_path = self.export_model_combo.get()
        if not model_path:
            messagebox.showerror("Fehler", "Bitte ein Model auswählen.")
            return

        export_format = self.export_format.get()
        self.export_status.configure(text=f"🔄 Exportiere zu {export_format}...", text_color=COLORS["warning"])
        self.update()

        try:
            from ultralytics import YOLO

            model = YOLO(model_path)
            model.export(format=export_format, half=self.half_var.get(), dynamic=self.dynamic_var.get())

            self.export_status.configure(text=f"✅ Export erfolgreich!", text_color=COLORS["success"])
            messagebox.showinfo("Erfolg", f"Model zu {export_format.upper()} exportiert!")

        except ImportError:
            messagebox.showerror("Fehler", "Ultralytics nicht installiert.")
        except Exception as e:
            self.export_status.configure(text="❌ Export fehlgeschlagen", text_color=COLORS["danger"])
            messagebox.showerror("Fehler", str(e))

    # ==================== HELPERS ====================
    def get_dataset_names(self):
        """Get list of dataset names."""
        if self.datasets_path.exists():
            return [d.name for d in self.datasets_path.iterdir() if d.is_dir()]
        return []

    def get_model_files(self):
        """Get list of model files."""
        models = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt"]
        if self.runs_path.exists():
            for f in self.runs_path.glob("**/weights/best.pt"):
                models.append(str(f))
        return models

    def count_datasets(self):
        return len(self.get_dataset_names())

    def count_images(self):
        count = 0
        if self.datasets_path.exists():
            count = len(list(self.datasets_path.glob("**/*.jpg"))) + len(list(self.datasets_path.glob("**/*.png")))
        return count

    def count_models(self):
        if self.runs_path.exists():
            return len(list(self.runs_path.glob("**/weights/best.pt")))
        return 0

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
