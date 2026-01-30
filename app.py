"""
YOLO Training Studio - Professional Edition mit Labeling
=========================================================

Komplettes YOLO Training Tool mit integriertem Labeling.
Workflow: Dataset erstellen → Bilder labeln → Training → Testen

Usage: python app.py
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox, Canvas
from PIL import Image, ImageTk, ImageDraw
import threading
import sys
import os
from pathlib import Path
import shutil
import yaml


# Theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Farben
PRIMARY = "#6366f1"
SUCCESS = "#22c55e"
DANGER = "#ef4444"
WARNING = "#f59e0b"
BG_CARD = "#2d2d3d"
TEXT_MUTED = "#94a3b8"

# YOLO Modelle mit Beschreibung
YOLO_MODELS = {
    "YOLOv8m": {"file": "yolov8m.pt", "desc": "Stabil, gut getestet"},
    "YOLOv10m": {"file": "yolov10m.pt", "desc": "Schneller, NMS-frei"},
    "YOLO11m": {"file": "yolo11m.pt", "desc": "Gute Balance"},
    "YOLO26m": {"file": "yolo26m.pt", "desc": "Neueste Version, 43% schneller"},
}


def get_dataset_paths(dataset_path):
    """
    Erkennt das Dataset-Format und gibt die Pfade zurück.
    Unterstützt:
    - Standard YOLO: images/train, images/val, labels/train, labels/val
    - Roboflow: train/, valid/ oder val/, test/ (Bilder und Labels gemischt)
    """
    dataset_path = Path(dataset_path)

    # Mögliche Ordnernamen für Training und Validierung
    train_names = ["train", "training"]
    val_names = ["val", "valid", "validation"]

    result = {
        "train_images": None,
        "val_images": None,
        "train_labels": None,
        "val_labels": None,
        "format": "unknown"
    }

    # Check Standard YOLO Format: images/train, labels/train
    if (dataset_path / "images" / "train").exists():
        result["train_images"] = dataset_path / "images" / "train"
        result["train_labels"] = dataset_path / "labels" / "train"
        result["format"] = "standard"

        # Val folder
        for val_name in val_names:
            if (dataset_path / "images" / val_name).exists():
                result["val_images"] = dataset_path / "images" / val_name
                result["val_labels"] = dataset_path / "labels" / val_name
                break
        return result

    # Check Roboflow Format: train/, valid/ mit Bildern direkt drin
    for train_name in train_names:
        train_dir = dataset_path / train_name
        if train_dir.exists():
            # Prüfe ob Bilder direkt im Ordner sind
            has_images = any(train_dir.glob("*.jpg")) or any(train_dir.glob("*.png"))
            # Oder in images/ Unterordner
            has_images_subdir = (train_dir / "images").exists()

            if has_images:
                # Roboflow Format: Bilder direkt in train/, Labels auch
                result["train_images"] = train_dir
                result["train_labels"] = train_dir  # Labels sind im gleichen Ordner
                result["format"] = "roboflow"
            elif has_images_subdir:
                result["train_images"] = train_dir / "images"
                result["train_labels"] = train_dir / "labels"
                result["format"] = "roboflow_nested"

            # Val folder suchen
            for val_name in val_names:
                val_dir = dataset_path / val_name
                if val_dir.exists():
                    if (val_dir / "images").exists():
                        result["val_images"] = val_dir / "images"
                        result["val_labels"] = val_dir / "labels"
                    else:
                        result["val_images"] = val_dir
                        result["val_labels"] = val_dir
                    break

            return result

    return result


def count_images(folder):
    """Zählt Bilder in einem Ordner."""
    if not folder or not folder.exists():
        return 0
    count = 0
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        count += len(list(folder.glob(ext)))
    return count


def count_labels(folder):
    """Zählt Label-Dateien in einem Ordner (ignoriert README etc.)."""
    if not folder or not folder.exists():
        return 0
    count = 0
    for txt_file in folder.glob("*.txt"):
        # Ignoriere README und andere Nicht-Label-Dateien
        name_lower = txt_file.stem.lower()
        if name_lower.startswith("readme") or name_lower.startswith("_"):
            continue
        count += 1
    return count


class LabelingWindow(ctk.CTkToplevel):
    """Fenster zum Labeln von Bildern mit Bounding Boxes."""

    def __init__(self, parent, dataset_path, class_names):
        super().__init__(parent)

        self.title("Bild Labeling Tool")
        self.geometry("1400x900")
        self.minsize(1200, 800)

        self.dataset_path = Path(dataset_path)
        self.class_names = class_names
        self.current_image_path = None
        self.current_image = None
        self.current_photo = None
        self.boxes = []  # Liste von (x1, y1, x2, y2, class_id)
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_rect = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.selected_class = 0

        # Bilder laden
        self.image_list = []
        self.current_index = 0
        self.load_image_list()

        self.create_ui()

        if self.image_list:
            self.load_image(0)

    def load_image_list(self):
        """Lädt alle Bilder aus dem Dataset (unterstützt beide Formate)."""
        paths = get_dataset_paths(self.dataset_path)

        # Bilder aus train und val laden
        for img_dir in [paths["train_images"], paths["val_images"]]:
            if img_dir and img_dir.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
                    self.image_list.extend(list(img_dir.glob(ext)))

        self.image_list = sorted(self.image_list)
        self.dataset_format = paths["format"]

    def create_ui(self):
        """Erstellt die UI."""
        # Header
        header = ctk.CTkFrame(self, height=60, fg_color=BG_CARD)
        header.pack(fill="x", padx=10, pady=10)
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="🏷️ Labeling Tool",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(side="left", padx=15, pady=15)

        # Navigation
        nav = ctk.CTkFrame(header, fg_color="transparent")
        nav.pack(side="right", padx=15)

        ctk.CTkButton(nav, text="◀ Zurück", command=self.prev_image, width=100).pack(side="left", padx=5)
        self.img_counter = ctk.CTkLabel(nav, text="0/0")
        self.img_counter.pack(side="left", padx=15)
        ctk.CTkButton(nav, text="Weiter ▶", command=self.next_image, width=100).pack(side="left", padx=5)

        # Main Content
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Linke Seite - Controls
        left = ctk.CTkFrame(main, fg_color=BG_CARD, width=280, corner_radius=10)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        ctk.CTkLabel(left, text="🏷️ Klasse auswählen:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(15, 5))

        # Klassen-Buttons
        self.class_buttons = []
        colors = ["#ef4444", "#f59e0b", "#22c55e", "#3b82f6", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"]
        for i, cls in enumerate(self.class_names):
            color = colors[i % len(colors)]
            btn = ctk.CTkButton(
                left,
                text=f"{i}: {cls}",
                command=lambda idx=i: self.select_class(idx),
                fg_color=color if i == 0 else "transparent",
                hover_color=color,
                border_width=2,
                border_color=color
            )
            btn.pack(fill="x", padx=15, pady=3)
            self.class_buttons.append((btn, color))

        # Trennlinie
        ctk.CTkFrame(left, height=2, fg_color="#404050").pack(fill="x", padx=15, pady=15)

        # Aktionen
        ctk.CTkLabel(left, text="⚡ Aktionen:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(0, 10))

        ctk.CTkButton(
            left, text="💾 Labels speichern",
            command=self.save_labels,
            fg_color=SUCCESS,
            width=240
        ).pack(padx=15, pady=5)

        ctk.CTkButton(
            left, text="🗑️ Alle Boxen löschen",
            command=self.clear_boxes,
            fg_color=DANGER,
            width=240
        ).pack(padx=15, pady=5)

        ctk.CTkButton(
            left, text="↩️ Letzte Box löschen",
            command=self.delete_last_box,
            width=240
        ).pack(padx=15, pady=5)

        # Hilfe
        help_frame = ctk.CTkFrame(left, fg_color="#1e3a5f", corner_radius=10)
        help_frame.pack(fill="x", padx=15, pady=20)

        ctk.CTkLabel(
            help_frame,
            text="💡 Anleitung:",
            font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            help_frame,
            text="1. Klasse oben auswählen\n"
                 "2. Auf dem Bild ziehen um\n"
                 "   eine Box zu zeichnen\n"
                 "3. Labels speichern\n"
                 "4. Weiter zum nächsten Bild",
            text_color=TEXT_MUTED,
            justify="left"
        ).pack(anchor="w", padx=10, pady=(0, 10))

        # Box-Liste
        ctk.CTkLabel(left, text="📦 Boxen:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(10, 5))

        self.box_list = ctk.CTkTextbox(left, height=150)
        self.box_list.pack(fill="x", padx=15, pady=(0, 15))

        # Rechte Seite - Canvas für Bild
        right = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=10)
        right.pack(side="right", fill="both", expand=True)

        # Canvas für Bild und Boxen
        self.canvas = Canvas(right, bg="#1a1a2e", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # Mouse Events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Resize Event
        self.canvas.bind("<Configure>", self.on_resize)

    def select_class(self, idx):
        """Wählt eine Klasse aus."""
        self.selected_class = idx
        for i, (btn, color) in enumerate(self.class_buttons):
            if i == idx:
                btn.configure(fg_color=color)
            else:
                btn.configure(fg_color="transparent")

    def load_image(self, index):
        """Lädt ein Bild."""
        if not self.image_list or index < 0 or index >= len(self.image_list):
            return

        self.current_index = index
        self.current_image_path = self.image_list[index]

        # Bild laden
        self.current_image = Image.open(self.current_image_path)

        # Labels laden falls vorhanden
        self.load_labels()

        # Anzeigen
        self.display_image()
        self.update_counter()
        self.update_box_list()

    def load_labels(self):
        """Lädt vorhandene Labels für das aktuelle Bild."""
        self.boxes = []

        if not self.current_image_path:
            return

        # Label-Pfad berechnen
        label_path = self.get_label_path()

        if label_path.exists():
            img_w, img_h = self.current_image.size
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        # YOLO Format zu Pixel konvertieren
                        x1 = int((cx - w/2) * img_w)
                        y1 = int((cy - h/2) * img_h)
                        x2 = int((cx + w/2) * img_w)
                        y2 = int((cy + h/2) * img_h)
                        self.boxes.append((x1, y1, x2, y2, cls_id))

    def get_label_path(self):
        """Gibt den Pfad zur Label-Datei zurück (unterstützt beide Formate)."""
        if not self.current_image_path:
            return None

        # Roboflow Format: Label ist im gleichen Ordner wie Bild
        if hasattr(self, 'dataset_format') and self.dataset_format in ["roboflow", "roboflow_nested"]:
            # Label im gleichen Ordner wie Bild
            return self.current_image_path.with_suffix(".txt")

        # Standard YOLO Format: images/train/img.jpg -> labels/train/img.txt
        try:
            rel_path = self.current_image_path.relative_to(self.dataset_path / "images")
            label_path = self.dataset_path / "labels" / rel_path.with_suffix(".txt")
            return label_path
        except ValueError:
            # Fallback: Label im gleichen Ordner
            return self.current_image_path.with_suffix(".txt")

    def display_image(self):
        """Zeigt das Bild auf dem Canvas an."""
        if not self.current_image:
            return

        self.canvas.delete("all")

        # Canvas-Größe
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w < 10 or canvas_h < 10:
            return

        # Skalierung berechnen
        img_w, img_h = self.current_image.size
        scale_w = canvas_w / img_w
        scale_h = canvas_h / img_h
        self.scale = min(scale_w, scale_h, 1.0)  # Maximal Originalgröße

        new_w = int(img_w * self.scale)
        new_h = int(img_h * self.scale)

        # Offset für Zentrierung
        self.offset_x = (canvas_w - new_w) // 2
        self.offset_y = (canvas_h - new_h) // 2

        # Bild skalieren und anzeigen
        resized = self.current_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.current_photo)

        # Boxen zeichnen
        colors = ["#ef4444", "#f59e0b", "#22c55e", "#3b82f6", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"]
        for x1, y1, x2, y2, cls_id in self.boxes:
            # Koordinaten skalieren
            sx1 = int(x1 * self.scale) + self.offset_x
            sy1 = int(y1 * self.scale) + self.offset_y
            sx2 = int(x2 * self.scale) + self.offset_x
            sy2 = int(y2 * self.scale) + self.offset_y

            color = colors[cls_id % len(colors)]
            self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=color, width=2)

            # Label
            label = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Klasse {cls_id}"
            self.canvas.create_text(sx1 + 5, sy1 + 5, text=label, fill=color, anchor="nw", font=("Arial", 10, "bold"))

    def on_resize(self, event):
        """Wird aufgerufen wenn Canvas-Größe sich ändert."""
        self.display_image()

    def on_mouse_down(self, event):
        """Maus gedrückt - Start einer Box."""
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y
        self.current_rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="#ffffff", width=2, dash=(4, 4)
        )

    def on_mouse_move(self, event):
        """Maus bewegt - Box vergrößern."""
        if self.drawing and self.current_rect:
            self.canvas.coords(self.current_rect, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        """Maus losgelassen - Box fertig."""
        if not self.drawing:
            return

        self.drawing = False

        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None

        # Koordinaten zurück in Bildpixel umrechnen
        x1 = int((min(self.start_x, event.x) - self.offset_x) / self.scale)
        y1 = int((min(self.start_y, event.y) - self.offset_y) / self.scale)
        x2 = int((max(self.start_x, event.x) - self.offset_x) / self.scale)
        y2 = int((max(self.start_y, event.y) - self.offset_y) / self.scale)

        # Prüfen ob Box groß genug
        if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
            # Clipping auf Bildgrenzen
            img_w, img_h = self.current_image.size
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))

            self.boxes.append((x1, y1, x2, y2, self.selected_class))
            self.display_image()
            self.update_box_list()

    def update_box_list(self):
        """Aktualisiert die Box-Liste."""
        self.box_list.delete("1.0", "end")
        for i, (x1, y1, x2, y2, cls_id) in enumerate(self.boxes):
            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Klasse {cls_id}"
            self.box_list.insert("end", f"{i+1}. {cls_name}\n")

    def update_counter(self):
        """Aktualisiert den Bild-Zähler."""
        self.img_counter.configure(text=f"{self.current_index + 1}/{len(self.image_list)}")

    def save_labels(self):
        """Speichert die Labels im YOLO-Format."""
        if not self.current_image_path:
            return

        label_path = self.get_label_path()
        label_path.parent.mkdir(parents=True, exist_ok=True)

        img_w, img_h = self.current_image.size

        with open(label_path, 'w') as f:
            for x1, y1, x2, y2, cls_id in self.boxes:
                # Pixel zu YOLO Format konvertieren
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        messagebox.showinfo("Gespeichert", f"Labels gespeichert!\n{label_path}")

    def clear_boxes(self):
        """Löscht alle Boxen."""
        self.boxes = []
        self.display_image()
        self.update_box_list()

    def delete_last_box(self):
        """Löscht die letzte Box."""
        if self.boxes:
            self.boxes.pop()
            self.display_image()
            self.update_box_list()

    def prev_image(self):
        """Vorheriges Bild."""
        if self.current_index > 0:
            self.load_image(self.current_index - 1)

    def next_image(self):
        """Nächstes Bild."""
        if self.current_index < len(self.image_list) - 1:
            self.load_image(self.current_index + 1)


class YOLOStudio(ctk.CTk):
    """Hauptfenster der Anwendung."""

    def __init__(self):
        super().__init__()

        self.title("YOLO Training Studio")
        self.geometry("1400x900")
        self.minsize(1200, 800)

        # Pfade
        self.base_path = Path(__file__).parent
        self.datasets_path = self.base_path / "datasets"
        self.modelle_path = self.base_path / "modelle"

        self.datasets_path.mkdir(exist_ok=True)
        self.modelle_path.mkdir(exist_ok=True)

        # Status
        self.training_thread = None
        self.is_training = False
        self.stop_training_flag = False
        self.current_dataset = None
        self.test_image_path = None

        self.create_ui()

    def create_ui(self):
        """Erstellt die Benutzeroberfläche."""
        # Header
        header = ctk.CTkFrame(self, height=80, fg_color=BG_CARD, corner_radius=15)
        header.pack(fill="x", padx=20, pady=20)
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="🎯 YOLO Training Studio",
            font=ctk.CTkFont(size=28, weight="bold")
        ).pack(side="left", padx=25, pady=20)

        # GPU Status
        gpu_text = self.get_gpu_status()
        ctk.CTkLabel(header, text=gpu_text, text_color=TEXT_MUTED).pack(side="right", padx=25, pady=20)

        # Tabs
        self.tabview = ctk.CTkTabview(self, corner_radius=15)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # 4 Tabs inkl. Labeling
        self.tab1 = self.tabview.add("1️⃣ Dataset")
        self.tab2 = self.tabview.add("2️⃣ Labeling")
        self.tab3 = self.tabview.add("3️⃣ Training")
        self.tab4 = self.tabview.add("4️⃣ Testen")

        self.create_dataset_tab()
        self.create_labeling_tab()
        self.create_training_tab()
        self.create_testing_tab()

    def get_gpu_status(self):
        try:
            import torch
            if torch.cuda.is_available():
                return f"🟢 GPU: {torch.cuda.get_device_name(0)}"
        except:
            pass
        return "🟡 CPU Modus"

    # ==================== TAB 1: DATASET ====================
    def create_dataset_tab(self):
        """Dataset Management - Import, erstellen."""
        main = ctk.CTkFrame(self.tab1, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=15, pady=15)

        # Linke Seite - Dataset Liste
        left = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=15, width=350)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        ctk.CTkLabel(left, text="📁 Deine Datasets", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)

        # Dataset Liste
        self.dataset_list = ctk.CTkScrollableFrame(left, fg_color="transparent")
        self.dataset_list.pack(fill="both", expand=True, padx=10)

        # Buttons
        btn_frame = ctk.CTkFrame(left, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=15)

        ctk.CTkButton(
            btn_frame,
            text="🔄 Aktualisieren",
            command=self.refresh_datasets,
            width=150
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame,
            text="📥 ZIP Import",
            command=self.import_zip_dialog,
            width=150,
            fg_color=PRIMARY
        ).pack(side="left", padx=5)

        # Rechte Seite - Dataset Details & Erstellen
        right = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=15)
        right.pack(side="right", fill="both", expand=True)

        # Info
        info_frame = ctk.CTkFrame(right, fg_color="transparent")
        info_frame.pack(fill="x", padx=25, pady=20)

        ctk.CTkLabel(info_frame, text="📊 Dataset Info", font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w")

        self.dataset_info = ctk.CTkLabel(
            info_frame,
            text="Wähle links ein Dataset aus oder erstelle ein neues.",
            text_color=TEXT_MUTED,
            justify="left"
        )
        self.dataset_info.pack(anchor="w", pady=10)

        # Trennlinie
        ctk.CTkFrame(right, height=2, fg_color="#404050").pack(fill="x", padx=25, pady=10)

        # Neues Dataset erstellen
        create_frame = ctk.CTkFrame(right, fg_color="transparent")
        create_frame.pack(fill="x", padx=25, pady=10)

        ctk.CTkLabel(create_frame, text="✨ Neues Dataset erstellen", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w")

        form = ctk.CTkFrame(create_frame, fg_color="transparent")
        form.pack(fill="x", pady=10)

        ctk.CTkLabel(form, text="Name:", width=80).grid(row=0, column=0, sticky="w", pady=5)
        self.new_name = ctk.CTkEntry(form, width=300, placeholder_text="z.B. hamster_dataset")
        self.new_name.grid(row=0, column=1, pady=5, padx=10)

        ctk.CTkLabel(form, text="Klassen:", width=80).grid(row=1, column=0, sticky="w", pady=5)
        self.new_classes = ctk.CTkEntry(form, width=300, placeholder_text="z.B. hamster, futter, käfig")
        self.new_classes.grid(row=1, column=1, pady=5, padx=10)

        ctk.CTkButton(
            form,
            text="✨ Erstellen",
            command=self.create_new_dataset,
            width=150,
            fg_color=PRIMARY
        ).grid(row=2, column=1, pady=15, sticky="w", padx=10)

        # Bilder hinzufügen
        ctk.CTkFrame(right, height=2, fg_color="#404050").pack(fill="x", padx=25, pady=10)

        add_frame = ctk.CTkFrame(right, fg_color="transparent")
        add_frame.pack(fill="x", padx=25, pady=10)

        ctk.CTkLabel(add_frame, text="🖼️ Bilder hinzufügen", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w")

        btn_row = ctk.CTkFrame(add_frame, fg_color="transparent")
        btn_row.pack(fill="x", pady=10)

        self.split_var = ctk.StringVar(value="train")
        ctk.CTkRadioButton(btn_row, text="Training", variable=self.split_var, value="train").pack(side="left", padx=10)
        ctk.CTkRadioButton(btn_row, text="Validierung", variable=self.split_var, value="val").pack(side="left", padx=10)

        ctk.CTkButton(
            add_frame,
            text="🖼️ Bilder hinzufügen (ohne Labels)",
            command=self.add_images_to_dataset,
            width=300,
            height=40,
            fg_color=SUCCESS
        ).pack(anchor="w", pady=5)

        # Hilfe
        help_frame = ctk.CTkFrame(right, fg_color="#1e3a5f", corner_radius=10)
        help_frame.pack(fill="x", padx=25, pady=20)

        ctk.CTkLabel(
            help_frame,
            text="💡 Workflow:",
            font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 5))

        ctk.CTkLabel(
            help_frame,
            text="1. Dataset erstellen mit Klassen\n"
                 "2. Bilder hinzufügen\n"
                 "3. Im 'Labeling' Tab Objekte markieren\n"
                 "4. Training starten",
            text_color=TEXT_MUTED,
            justify="left"
        ).pack(anchor="w", padx=15, pady=(0, 15))

        self.refresh_datasets()

    def refresh_datasets(self):
        """Aktualisiert die Dataset-Liste."""
        for widget in self.dataset_list.winfo_children():
            widget.destroy()

        datasets = []
        if self.datasets_path.exists():
            for d in self.datasets_path.iterdir():
                if d.is_dir() and (d / "data.yaml").exists():
                    datasets.append(d)

        if not datasets:
            ctk.CTkLabel(
                self.dataset_list,
                text="Keine Datasets.\n\nErstelle ein neues\noder importiere eines.",
                text_color=TEXT_MUTED
            ).pack(pady=50)
            return

        for dataset in sorted(datasets):
            self.create_dataset_button(dataset)

    def create_dataset_button(self, path):
        """Erstellt einen Button für ein Dataset."""
        paths = get_dataset_paths(path)
        train_count = count_images(paths["train_images"])
        val_count = count_images(paths["val_images"])

        btn = ctk.CTkButton(
            self.dataset_list,
            text=f"📁 {path.name}\n     {train_count} Train | {val_count} Val",
            fg_color="transparent",
            hover_color=PRIMARY,
            anchor="w",
            height=55,
            command=lambda: self.select_dataset(path)
        )
        btn.pack(fill="x", pady=3)

    def select_dataset(self, path):
        """Wählt ein Dataset aus."""
        self.current_dataset = path

        yaml_path = path / "data.yaml"
        info_text = f"📁 Name: {path.name}\n"

        if yaml_path.exists():
            with open(yaml_path) as f:
                data = yaml.safe_load(f)

            names = data.get("names", {})
            if isinstance(names, dict):
                classes = list(names.values())
            else:
                classes = names

            info_text += f"\n🏷️ Klassen ({len(classes)}):\n"
            for i, cls in enumerate(classes[:10]):
                info_text += f"   {i}: {cls}\n"

        # Dataset-Pfade ermitteln (unterstützt beide Formate)
        paths = get_dataset_paths(path)
        train_imgs = count_images(paths["train_images"])
        val_imgs = count_images(paths["val_images"])
        train_labels = count_labels(paths["train_labels"])
        val_labels = count_labels(paths["val_labels"])

        info_text += f"\n📊 Statistiken:\n"
        info_text += f"   Training: {train_imgs} Bilder, {train_labels} Labels\n"
        info_text += f"   Validierung: {val_imgs} Bilder, {val_labels} Labels\n"
        info_text += f"\n📂 Format: {paths['format']}"

        if train_labels < train_imgs:
            info_text += f"\n⚠️ {train_imgs - train_labels} Bilder noch nicht gelabelt!"
        elif train_imgs > 0 and train_labels > 0:
            info_text += f"\n✅ Bereit für Training!"

        self.dataset_info.configure(text=info_text)

    def import_zip_dialog(self):
        """Öffnet Dialog zum ZIP-Import."""
        zip_file = filedialog.askopenfilename(
            title="ZIP-Datei auswählen",
            filetypes=[("ZIP", "*.zip")]
        )
        if zip_file:
            self.import_zip(Path(zip_file))

    def import_zip(self, zip_path):
        """Importiert ein Dataset aus ZIP."""
        import zipfile

        name = zip_path.stem.replace(" ", "_")
        target = self.datasets_path / name

        if target.exists():
            if not messagebox.askyesno("Überschreiben?", f"'{name}' existiert. Überschreiben?"):
                return
            shutil.rmtree(target)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(target)

            # Unterordner hochschieben wenn nötig
            contents = list(target.iterdir())
            if len(contents) == 1 and contents[0].is_dir():
                sub = contents[0]
                for item in sub.iterdir():
                    shutil.move(str(item), str(target))
                sub.rmdir()

            messagebox.showinfo("Erfolg", f"Dataset '{name}' importiert!")
            self.refresh_datasets()
            self.select_dataset(target)

        except Exception as e:
            messagebox.showerror("Fehler", str(e))

    def create_new_dataset(self):
        """Erstellt ein neues Dataset."""
        name = self.new_name.get().strip().replace(" ", "_")
        classes = self.new_classes.get().strip()

        if not name:
            messagebox.showerror("Fehler", "Bitte Name eingeben.")
            return

        target = self.datasets_path / name
        if target.exists():
            messagebox.showerror("Fehler", f"'{name}' existiert bereits.")
            return

        try:
            for split in ["train", "val"]:
                (target / "images" / split).mkdir(parents=True)
                (target / "labels" / split).mkdir(parents=True)

            class_list = [c.strip() for c in classes.split(",") if c.strip()]
            if not class_list:
                class_list = ["object"]

            data = {
                "path": str(target.absolute()),
                "train": "images/train",
                "val": "images/val",
                "names": {i: c for i, c in enumerate(class_list)},
                "nc": len(class_list)
            }

            with open(target / "data.yaml", "w") as f:
                yaml.dump(data, f, sort_keys=False)

            messagebox.showinfo("Erfolg", f"Dataset '{name}' erstellt!")
            self.new_name.delete(0, "end")
            self.new_classes.delete(0, "end")
            self.refresh_datasets()
            self.select_dataset(target)

        except Exception as e:
            messagebox.showerror("Fehler", str(e))

    def add_images_to_dataset(self):
        """Fügt Bilder zum Dataset hinzu."""
        if not self.current_dataset:
            messagebox.showerror("Fehler", "Zuerst Dataset auswählen.")
            return

        files = filedialog.askopenfilenames(
            title="Bilder auswählen",
            filetypes=[("Bilder", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )

        if not files:
            return

        split = self.split_var.get()
        target = self.current_dataset / "images" / split
        target.mkdir(parents=True, exist_ok=True)

        for file in files:
            shutil.copy2(file, target / Path(file).name)

        messagebox.showinfo("Erfolg", f"{len(files)} Bilder hinzugefügt!\n\nJetzt im 'Labeling' Tab markieren.")
        self.select_dataset(self.current_dataset)

    # ==================== TAB 2: LABELING ====================
    def create_labeling_tab(self):
        """Labeling Tab."""
        main = ctk.CTkFrame(self.tab2, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=15, pady=15)

        # Linke Seite - Manuelles Labeling
        left = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=15)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ctk.CTkLabel(
            left,
            text="🏷️ Manuelles Labeling",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(25, 10))

        ctk.CTkLabel(
            left,
            text="Öffne das Labeling-Tool um Objekte auf\ndeinen Bildern manuell zu markieren.",
            text_color=TEXT_MUTED,
            justify="center"
        ).pack(pady=5)

        # Dataset Auswahl
        ctk.CTkLabel(left, text="Dataset:", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5))
        self.labeling_dataset = ctk.CTkComboBox(left, values=self.get_dataset_names(), width=300)
        self.labeling_dataset.pack(pady=5)

        ctk.CTkButton(
            left,
            text="🔄 Aktualisieren",
            command=self.refresh_labeling_datasets,
            width=200
        ).pack(pady=10)

        ctk.CTkButton(
            left,
            text="🏷️ Labeling Tool öffnen",
            command=self.open_labeling_tool,
            width=300,
            height=50,
            fg_color=PRIMARY,
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=20)

        # Anleitung
        help_frame = ctk.CTkFrame(left, fg_color="#1e3a5f", corner_radius=10)
        help_frame.pack(fill="x", padx=30, pady=(0, 20))

        ctk.CTkLabel(
            help_frame,
            text="💡 So funktioniert's:",
            font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 5))

        ctk.CTkLabel(
            help_frame,
            text="1. Klasse auswählen (z.B. 'hamster')\n"
                 "2. Auf dem Bild ziehen um Box zu zeichnen\n"
                 "3. Labels speichern\n"
                 "4. Nächstes Bild",
            text_color=TEXT_MUTED,
            justify="left"
        ).pack(anchor="w", padx=15, pady=(0, 15))

        # Rechte Seite - Auto-Labeler + Tools
        right = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=15)
        right.pack(side="right", fill="both", expand=True, padx=(10, 0))

        ctk.CTkLabel(
            right,
            text="🤖 Auto-Labeler",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(25, 10))

        ctk.CTkLabel(
            right,
            text="Nutze ein trainiertes Modell um\nBilder automatisch zu labeln.",
            text_color=TEXT_MUTED,
            justify="center"
        ).pack(pady=5)

        # Dataset
        ctk.CTkLabel(right, text="Dataset:", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5))
        self.auto_label_dataset = ctk.CTkComboBox(right, values=self.get_dataset_names(), width=300)
        self.auto_label_dataset.pack(pady=5)

        # Modell
        ctk.CTkLabel(right, text="Modell (.pt):", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 5))
        self.auto_label_model = ctk.CTkComboBox(right, values=[], width=300)
        self.auto_label_model.pack(pady=5)

        model_btn_frame = ctk.CTkFrame(right, fg_color="transparent")
        model_btn_frame.pack(pady=5)
        ctk.CTkButton(model_btn_frame, text="🔄", command=self.refresh_auto_label_models, width=50).pack(side="left", padx=5)
        ctk.CTkButton(model_btn_frame, text="📂 Modell laden...", command=self.browse_auto_label_model, width=150).pack(side="left", padx=5)

        # Confidence
        ctk.CTkLabel(right, text="Confidence:", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 5))
        conf_frame = ctk.CTkFrame(right, fg_color="transparent")
        conf_frame.pack()
        self.auto_conf_slider = ctk.CTkSlider(conf_frame, from_=0.01, to=1.0, width=200)
        self.auto_conf_slider.set(0.25)
        self.auto_conf_slider.pack(side="left")
        self.auto_conf_label = ctk.CTkLabel(conf_frame, text="25%", width=50)
        self.auto_conf_label.pack(side="left")
        self.auto_conf_slider.configure(command=lambda v: self.auto_conf_label.configure(text=f"{int(v*100)}%"))

        ctk.CTkButton(
            right,
            text="🤖 Auto-Label starten",
            command=self.run_auto_label,
            width=300,
            height=50,
            fg_color=WARNING,
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=20)

        self.auto_label_status = ctk.CTkLabel(right, text="", text_color=TEXT_MUTED)
        self.auto_label_status.pack(pady=5)

        # Trennlinie
        ctk.CTkFrame(right, height=2, fg_color="#404050").pack(fill="x", padx=30, pady=10)

        # Aufräumen
        ctk.CTkLabel(
            right,
            text="🧹 Aufräumen",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))

        ctk.CTkButton(
            right,
            text="🗑️ Bilder ohne Labels löschen",
            command=self.delete_unlabeled_images,
            width=300,
            height=40,
            fg_color=DANGER
        ).pack(pady=(10, 5))

        # Label-Klasse löschen
        del_cls_frame = ctk.CTkFrame(right, fg_color="transparent")
        del_cls_frame.pack(fill="x", padx=30, pady=5)

        self.delete_class_entry = ctk.CTkEntry(
            del_cls_frame,
            placeholder_text="Klassen-ID (z.B. 0, 1, 2)",
            width=170
        )
        self.delete_class_entry.pack(side="left", padx=(0, 5))

        ctk.CTkButton(
            del_cls_frame,
            text="🏷️ Klasse löschen",
            command=self.delete_class_labels,
            width=130,
            fg_color=DANGER
        ).pack(side="left")

        ctk.CTkLabel(
            right,
            text="Entfernt eine Klasse aus allen Label-Dateien\noder löscht Bilder ohne Labels.",
            text_color=TEXT_MUTED,
            justify="center"
        ).pack(pady=(5, 20))

        self.refresh_auto_label_models()

    def refresh_labeling_datasets(self):
        """Aktualisiert Labeling Dataset-Liste."""
        self.labeling_dataset.configure(values=self.get_dataset_names())

    def open_labeling_tool(self):
        """Öffnet das Labeling-Fenster."""
        dataset_name = self.labeling_dataset.get()
        if not dataset_name:
            messagebox.showerror("Fehler", "Bitte Dataset auswählen.")
            return

        dataset_path = self.datasets_path / dataset_name
        if not dataset_path.exists():
            messagebox.showerror("Fehler", f"Dataset '{dataset_name}' nicht gefunden.")
            return

        # Klassen laden
        yaml_path = dataset_path / "data.yaml"
        if not yaml_path.exists():
            messagebox.showerror("Fehler", "data.yaml nicht gefunden.")
            return

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        names = data.get("names", {})
        if isinstance(names, dict):
            class_names = list(names.values())
        else:
            class_names = names

        # Labeling-Fenster öffnen
        LabelingWindow(self, dataset_path, class_names)

    def refresh_auto_label_models(self):
        """Aktualisiert die Modell-Liste für Auto-Labeling."""
        models = []
        if self.modelle_path.exists():
            for best_pt in self.modelle_path.glob("**/weights/best.pt"):
                models.append(str(best_pt))
        if models:
            self.auto_label_model.configure(values=models)
            self.auto_label_model.set(models[0])
        else:
            self.auto_label_model.configure(values=["Noch keine Modelle"])
            self.auto_label_model.set("Noch keine Modelle")

    def browse_auto_label_model(self):
        """Modell manuell auswählen."""
        file = filedialog.askopenfilename(
            title="Modell auswählen",
            filetypes=[("PyTorch", "*.pt")]
        )
        if file:
            self.auto_label_model.set(file)

    def run_auto_label(self):
        """Auto-Labeling mit einem trainierten Modell."""
        dataset_name = self.auto_label_dataset.get()
        model_path = self.auto_label_model.get()

        if not dataset_name:
            messagebox.showerror("Fehler", "Bitte Dataset auswählen.")
            return

        if not model_path or "Noch keine" in model_path:
            messagebox.showerror("Fehler", "Bitte Modell auswählen.\n\nDu brauchst ein trainiertes Modell.")
            return

        if not Path(model_path).exists():
            messagebox.showerror("Fehler", f"Modell nicht gefunden:\n{model_path}")
            return

        dataset_path = self.datasets_path / dataset_name
        paths = get_dataset_paths(dataset_path)
        conf = self.auto_conf_slider.get()

        # Alle Bilder sammeln
        all_images = []
        for img_dir in [paths["train_images"], paths["val_images"]]:
            if img_dir and img_dir.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
                    all_images.extend(list(img_dir.glob(ext)))

        if not all_images:
            messagebox.showerror("Fehler", "Keine Bilder gefunden.")
            return

        self.auto_label_status.configure(text=f"⏳ Labele {len(all_images)} Bilder...")
        self.update()

        def do_auto_label():
            try:
                from ultralytics import YOLO

                model = YOLO(model_path)
                labeled = 0
                skipped = 0
                ds_format = paths["format"]

                for img_path in all_images:
                    # Label-Pfad bestimmen
                    if ds_format in ["roboflow", "roboflow_nested"]:
                        label_path = img_path.with_suffix(".txt")
                    else:
                        try:
                            rel = img_path.relative_to(dataset_path / "images")
                            label_path = dataset_path / "labels" / rel.with_suffix(".txt")
                        except ValueError:
                            label_path = img_path.with_suffix(".txt")

                    # Nur labeln wenn noch kein Label existiert
                    if label_path.exists() and label_path.stat().st_size > 0:
                        skipped += 1
                        continue

                    # Erkennung
                    results = model.predict(str(img_path), conf=conf, verbose=False)
                    result = results[0]

                    if result.boxes is not None and len(result.boxes) > 0:
                        label_path.parent.mkdir(parents=True, exist_ok=True)
                        img_w, img_h = result.orig_shape[1], result.orig_shape[0]

                        with open(label_path, 'w') as f:
                            for box in result.boxes:
                                cls_id = int(box.cls[0])
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                cx = ((x1 + x2) / 2) / img_w
                                cy = ((y1 + y2) / 2) / img_h
                                w = (x2 - x1) / img_w
                                h = (y2 - y1) / img_h
                                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

                        labeled += 1

                msg = f"✅ Fertig! {labeled} Bilder gelabelt, {skipped} übersprungen."
                self.after(0, lambda: self.auto_label_status.configure(text=msg))
                self.after(0, lambda: messagebox.showinfo("Auto-Label", msg))

            except Exception as e:
                error = str(e)
                self.after(0, lambda: self.auto_label_status.configure(text=f"❌ Fehler: {error}"))
                self.after(0, lambda: messagebox.showerror("Fehler", error))

        threading.Thread(target=do_auto_label, daemon=True).start()

    def delete_unlabeled_images(self):
        """Löscht alle Bilder ohne zugehöriges Label."""
        dataset_name = self.auto_label_dataset.get()
        if not dataset_name:
            # Versuche labeling_dataset
            dataset_name = self.labeling_dataset.get()

        if not dataset_name:
            messagebox.showerror("Fehler", "Bitte Dataset auswählen.")
            return

        dataset_path = self.datasets_path / dataset_name
        paths = get_dataset_paths(dataset_path)
        ds_format = paths["format"]

        # Alle Bilder sammeln
        unlabeled = []
        for img_dir in [paths["train_images"], paths["val_images"]]:
            if not img_dir or not img_dir.exists():
                continue
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
                for img_path in img_dir.glob(ext):
                    # Label-Pfad bestimmen
                    if ds_format in ["roboflow", "roboflow_nested"]:
                        label_path = img_path.with_suffix(".txt")
                    else:
                        try:
                            rel = img_path.relative_to(dataset_path / "images")
                            label_path = dataset_path / "labels" / rel.with_suffix(".txt")
                        except ValueError:
                            label_path = img_path.with_suffix(".txt")

                    # Prüfe ob Label existiert und nicht leer ist
                    if not label_path.exists() or label_path.stat().st_size == 0:
                        unlabeled.append(img_path)

        if not unlabeled:
            messagebox.showinfo("Aufräumen", "Alle Bilder haben Labels. Nichts zu löschen.")
            return

        if messagebox.askyesno("Löschen?", f"{len(unlabeled)} Bilder ohne Labels gefunden.\n\nWirklich löschen?"):
            for img_path in unlabeled:
                img_path.unlink()
            messagebox.showinfo("Erfolg", f"{len(unlabeled)} Bilder gelöscht!")
            self.refresh_datasets()

    def delete_class_labels(self):
        """Löscht alle Labels einer bestimmten Klasse aus dem Dataset."""
        dataset_name = self.auto_label_dataset.get()
        if not dataset_name:
            dataset_name = self.labeling_dataset.get()

        if not dataset_name:
            messagebox.showerror("Fehler", "Bitte Dataset auswählen.")
            return

        cls_text = self.delete_class_entry.get().strip()
        if not cls_text:
            messagebox.showerror("Fehler", "Bitte Klassen-ID eingeben (z.B. 0, 1, 2).")
            return

        try:
            cls_id = int(cls_text)
        except ValueError:
            messagebox.showerror("Fehler", "Klassen-ID muss eine Zahl sein.")
            return

        dataset_path = self.datasets_path / dataset_name
        paths = get_dataset_paths(dataset_path)

        # Klassen-Name ermitteln
        cls_name = str(cls_id)
        yaml_path = dataset_path / "data.yaml"
        if yaml_path.exists():
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            names = data.get("names", {})
            if isinstance(names, dict) and cls_id in names:
                cls_name = names[cls_id]
            elif isinstance(names, list) and cls_id < len(names):
                cls_name = names[cls_id]

        # Label-Ordner finden
        label_dirs = []
        for lbl_dir in [paths["train_labels"], paths["val_labels"]]:
            if lbl_dir and lbl_dir.exists():
                label_dirs.append(lbl_dir)

        if not label_dirs:
            messagebox.showerror("Fehler", "Keine Label-Ordner gefunden.")
            return

        # Zählen
        affected_files = 0
        removed_lines = 0
        for lbl_dir in label_dirs:
            for txt_file in lbl_dir.glob("*.txt"):
                if txt_file.stem.lower().startswith("readme"):
                    continue
                with open(txt_file) as f:
                    lines = f.readlines()
                new_lines = [l for l in lines if l.strip() and not l.strip().startswith(f"{cls_id} ")]
                if len(new_lines) < len(lines):
                    affected_files += 1
                    removed_lines += len(lines) - len(new_lines)

        if removed_lines == 0:
            messagebox.showinfo("Info", f"Klasse {cls_id} ('{cls_name}') nicht gefunden.")
            return

        if not messagebox.askyesno(
            "Klasse löschen?",
            f"Klasse {cls_id} ('{cls_name}') löschen?\n\n"
            f"{removed_lines} Labels in {affected_files} Dateien werden entfernt."
        ):
            return

        # Löschen
        for lbl_dir in label_dirs:
            for txt_file in lbl_dir.glob("*.txt"):
                if txt_file.stem.lower().startswith("readme"):
                    continue
                with open(txt_file) as f:
                    lines = f.readlines()
                new_lines = [l for l in lines if l.strip() and not l.strip().startswith(f"{cls_id} ")]
                with open(txt_file, 'w') as f:
                    f.writelines(new_lines)

        messagebox.showinfo("Erfolg", f"Klasse {cls_id} ('{cls_name}') gelöscht!\n{removed_lines} Labels entfernt.")
        self.refresh_datasets()

    # ==================== TAB 3: TRAINING ====================
    def create_training_tab(self):
        """Training Tab mit Python API."""
        main = ctk.CTkFrame(self.tab3, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=15, pady=15)

        # Linke Seite
        left = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=15, width=450)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        # Header
        header = ctk.CTkFrame(left, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=20)

        ctk.CTkLabel(header, text="🚀 Training", font=ctk.CTkFont(size=22, weight="bold")).pack(side="left")

        self.start_btn = ctk.CTkButton(
            header,
            text="▶️ Starten",
            command=self.start_training,
            width=120,
            height=40,
            fg_color=SUCCESS,
            font=ctk.CTkFont(weight="bold")
        )
        self.start_btn.pack(side="right")

        self.stop_btn = ctk.CTkButton(
            header,
            text="⏹️",
            command=self.stop_training,
            width=50,
            height=40,
            fg_color=DANGER,
            state="disabled"
        )
        self.stop_btn.pack(side="right", padx=10)

        # Dataset
        ctk.CTkLabel(left, text="📁 Dataset:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=20, pady=(10, 5))
        self.train_dataset = ctk.CTkComboBox(left, values=self.get_dataset_names(), width=380)
        self.train_dataset.pack(anchor="w", padx=20)

        ctk.CTkButton(left, text="🔄 Aktualisieren", command=self.refresh_training_datasets, width=200).pack(anchor="w", padx=20, pady=10)

        # YOLO Version
        ctk.CTkLabel(left, text="🤖 YOLO Version:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=20, pady=(10, 5))

        model_names = [f"{k} - {v['desc']}" for k, v in YOLO_MODELS.items()]
        self.yolo_version = ctk.CTkComboBox(left, values=model_names, width=380)
        self.yolo_version.set(model_names[-1])  # Standard: neueste (v26)
        self.yolo_version.pack(anchor="w", padx=20)

        # Parameter
        ctk.CTkLabel(left, text="⚙️ Einstellungen:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=20, pady=(20, 10))

        params = ctk.CTkFrame(left, fg_color="transparent")
        params.pack(fill="x", padx=20)

        ctk.CTkLabel(params, text="Epochen:", width=100, anchor="w").grid(row=0, column=0, pady=8)
        self.epochs_entry = ctk.CTkEntry(params, width=100)
        self.epochs_entry.insert(0, "100")
        self.epochs_entry.grid(row=0, column=1, pady=8)

        ctk.CTkLabel(params, text="Batch:", width=100, anchor="w").grid(row=1, column=0, pady=8)
        self.batch_entry = ctk.CTkEntry(params, width=100)
        self.batch_entry.insert(0, "8")
        self.batch_entry.grid(row=1, column=1, pady=8)

        ctk.CTkLabel(params, text="Bildgröße:", width=100, anchor="w").grid(row=2, column=0, pady=8)
        self.imgsz_entry = ctk.CTkEntry(params, width=100)
        self.imgsz_entry.insert(0, "640")
        self.imgsz_entry.grid(row=2, column=1, pady=8)

        # Info
        info = ctk.CTkFrame(left, fg_color="#1e3a5f", corner_radius=10)
        info.pack(fill="x", padx=20, pady=20)

        ctk.CTkLabel(info, text="💡 Training Info:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(15, 5))
        ctk.CTkLabel(
            info,
            text="Wähle die YOLO Version oben aus.\n"
                 "v26 = neueste, v8 = stabilste.\n"
                 "Modell wird gespeichert in:\n"
                 "modelle/[dataset]/weights/best.pt",
            text_color=TEXT_MUTED,
            justify="left"
        ).pack(anchor="w", padx=15, pady=(0, 15))

        # Rechte Seite - Log
        right = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=15)
        right.pack(side="right", fill="both", expand=True)

        log_header = ctk.CTkFrame(right, fg_color="transparent")
        log_header.pack(fill="x", padx=20, pady=15)

        ctk.CTkLabel(log_header, text="📋 Training Log", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")
        self.status_label = ctk.CTkLabel(log_header, text="⚪ Bereit", text_color=TEXT_MUTED)
        self.status_label.pack(side="right")

        self.log_text = ctk.CTkTextbox(right, font=ctk.CTkFont(family="Consolas", size=11))
        self.log_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Progress
        prog_frame = ctk.CTkFrame(right, fg_color="transparent")
        prog_frame.pack(fill="x", padx=15, pady=(0, 15))

        self.progress = ctk.CTkProgressBar(prog_frame, width=500)
        self.progress.pack(side="left")
        self.progress.set(0)

        self.progress_label = ctk.CTkLabel(prog_frame, text="0%", width=60)
        self.progress_label.pack(side="left", padx=15)

    def get_dataset_names(self):
        """Gibt Dataset-Namen zurück."""
        names = []
        if self.datasets_path.exists():
            for d in self.datasets_path.iterdir():
                if d.is_dir() and (d / "data.yaml").exists():
                    names.append(d.name)
        return sorted(names)

    def refresh_training_datasets(self):
        """Aktualisiert Training-Datasets."""
        self.train_dataset.configure(values=self.get_dataset_names())

    def start_training(self):
        """Startet Training mit Python API."""
        dataset = self.train_dataset.get()
        if not dataset:
            messagebox.showerror("Fehler", "Bitte Dataset auswählen.")
            return

        data_yaml = self.datasets_path / dataset / "data.yaml"
        if not data_yaml.exists():
            messagebox.showerror("Fehler", "data.yaml nicht gefunden.")
            return

        try:
            epochs = int(self.epochs_entry.get())
            batch = int(self.batch_entry.get())
            imgsz = int(self.imgsz_entry.get())
        except ValueError:
            messagebox.showerror("Fehler", "Ungültige Parameter.")
            return

        # YOLO Version ermitteln
        version_str = self.yolo_version.get().split(" - ")[0]  # z.B. "YOLOv8m"
        model_info = YOLO_MODELS.get(version_str, YOLO_MODELS["YOLO26m"])
        model_file = model_info["file"]

        self.is_training = True
        self.stop_training_flag = False
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="🟢 Training läuft...", text_color=SUCCESS)
        self.log_text.delete("1.0", "end")
        self.progress.set(0)

        self.log_text.insert("end", "=" * 50 + "\n")
        self.log_text.insert("end", f"🚀 Training gestartet!\n")
        self.log_text.insert("end", f"📁 Dataset: {dataset}\n")
        self.log_text.insert("end", f"🤖 Modell: {version_str} ({model_file})\n")
        self.log_text.insert("end", f"📊 Epochen: {epochs} | Batch: {batch} | Größe: {imgsz}\n")
        self.log_text.insert("end", "=" * 50 + "\n\n")

        # Training in Thread
        def train():
            try:
                self.after(0, lambda: self.log_text.insert("end", f"⏳ Lade {version_str}...\n"))

                from ultralytics import YOLO

                self.after(0, lambda: self.log_text.insert("end", "✅ YOLO geladen!\n"))
                self.after(0, lambda: self.log_text.insert("end", f"🤖 Verwende: {model_file}\n\n"))

                # Modell laden
                model = YOLO(model_file)

                # Projekt-Pfad
                project_path = str(self.modelle_path)

                self.after(0, lambda: self.log_text.insert("end", "🏋️ Starte Training...\n\n"))

                # Training starten
                results = model.train(
                    data=str(data_yaml),
                    epochs=epochs,
                    batch=batch,
                    imgsz=imgsz,
                    project=project_path,
                    name=dataset,
                    exist_ok=True,
                    patience=50,
                    save=True,
                    plots=True,
                    verbose=True
                )

                # Erfolg
                model_path = self.modelle_path / dataset / "weights" / "best.pt"

                self.after(0, lambda: self.training_success(str(model_path)))

            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda: self.training_error(error_msg))

        self.training_thread = threading.Thread(target=train, daemon=True)
        self.training_thread.start()

        # Progress-Update
        self.update_training_progress()

    def update_training_progress(self):
        """Aktualisiert Progress während Training."""
        if self.is_training:
            # Scroll Log nach unten
            self.log_text.see("end")
            self.after(1000, self.update_training_progress)

    def training_success(self, model_path):
        """Wird bei erfolgreichem Training aufgerufen."""
        self.is_training = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

        if Path(model_path).exists():
            self.status_label.configure(text="✅ Erfolgreich!", text_color=SUCCESS)
            self.progress.set(1)
            self.progress_label.configure(text="100%")

            self.log_text.insert("end", "\n" + "=" * 50 + "\n")
            self.log_text.insert("end", "✅ Training erfolgreich abgeschlossen!\n")
            self.log_text.insert("end", f"📁 Modell gespeichert: {model_path}\n")
            self.log_text.insert("end", "=" * 50 + "\n")

            messagebox.showinfo("Erfolg", f"Training abgeschlossen!\n\nModell: {model_path}")
        else:
            self.training_error("Modell wurde nicht erstellt.")

        self.refresh_models()

    def training_error(self, error):
        """Wird bei Training-Fehler aufgerufen."""
        self.is_training = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="❌ Fehler!", text_color=DANGER)

        self.log_text.insert("end", "\n" + "=" * 50 + "\n")
        self.log_text.insert("end", f"❌ FEHLER: {error}\n")
        self.log_text.insert("end", "=" * 50 + "\n")

        messagebox.showerror("Training Fehler", f"Training fehlgeschlagen:\n\n{error}")

    def stop_training(self):
        """Stoppt Training."""
        self.stop_training_flag = True
        self.is_training = False
        self.status_label.configure(text="⏹️ Gestoppt", text_color=WARNING)
        self.log_text.insert("end", "\n⏹️ Training wird gestoppt...\n")

    # ==================== TAB 4: TESTEN ====================
    def create_testing_tab(self):
        """Test Tab."""
        main = ctk.CTkFrame(self.tab4, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=15, pady=15)

        # Linke Seite
        left = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=15, width=400)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        ctk.CTkLabel(left, text="🔍 Modell testen", font=ctk.CTkFont(size=22, weight="bold")).pack(pady=20)

        # Modell
        ctk.CTkLabel(left, text="🤖 Trainiertes Modell:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=20, pady=(10, 5))

        self.model_combo = ctk.CTkComboBox(left, values=[], width=340)
        self.model_combo.pack(anchor="w", padx=20)

        btn_frame = ctk.CTkFrame(left, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkButton(btn_frame, text="🔄 Aktualisieren", command=self.refresh_models, width=160).pack(side="left")
        ctk.CTkButton(btn_frame, text="📂 Laden...", command=self.browse_model, width=160).pack(side="left", padx=10)

        # Confidence
        ctk.CTkLabel(left, text="🎯 Confidence:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=20, pady=(20, 5))

        conf_frame = ctk.CTkFrame(left, fg_color="transparent")
        conf_frame.pack(fill="x", padx=20)

        self.conf_slider = ctk.CTkSlider(conf_frame, from_=0.01, to=1.0, width=280)
        self.conf_slider.set(0.25)
        self.conf_slider.pack(side="left")

        self.conf_label = ctk.CTkLabel(conf_frame, text="25%", width=60)
        self.conf_label.pack(side="left")
        self.conf_slider.configure(command=lambda v: self.conf_label.configure(text=f"{int(v*100)}%"))

        # Buttons
        ctk.CTkLabel(left, text="").pack(pady=10)

        ctk.CTkButton(
            left,
            text="🖼️ Bild auswählen",
            command=self.select_image,
            width=340,
            height=45
        ).pack(padx=20, pady=5)

        ctk.CTkButton(
            left,
            text="▶️ Erkennung starten",
            command=self.run_detection,
            width=340,
            height=50,
            fg_color=SUCCESS,
            font=ctk.CTkFont(weight="bold")
        ).pack(padx=20, pady=10)

        # Ergebnisse
        ctk.CTkLabel(left, text="📊 Ergebnisse:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=20, pady=(20, 5))

        self.results_box = ctk.CTkTextbox(left, height=200, font=ctk.CTkFont(family="Consolas", size=12))
        self.results_box.pack(fill="x", padx=20, pady=(0, 20))

        # Rechte Seite - Bild
        right = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=15)
        right.pack(side="right", fill="both", expand=True)

        self.image_label = ctk.CTkLabel(
            right,
            text="🖼️\n\nWähle ein Bild und\nstarte die Erkennung",
            font=ctk.CTkFont(size=18),
            text_color=TEXT_MUTED
        )
        self.image_label.pack(fill="both", expand=True, padx=30, pady=30)

        self.refresh_models()

    def refresh_models(self):
        """Aktualisiert Modell-Liste."""
        models = []

        if self.modelle_path.exists():
            for best_pt in self.modelle_path.glob("**/weights/best.pt"):
                models.append(str(best_pt))

        if models:
            self.model_combo.configure(values=models)
            self.model_combo.set(models[0])
        else:
            self.model_combo.configure(values=["Noch keine Modelle"])
            self.model_combo.set("Noch keine Modelle")

    def browse_model(self):
        """Lädt Modell von Festplatte."""
        file = filedialog.askopenfilename(
            title="Modell auswählen",
            filetypes=[("PyTorch", "*.pt")]
        )
        if file:
            self.model_combo.set(file)

    def select_image(self):
        """Wählt Testbild."""
        file = filedialog.askopenfilename(
            title="Bild auswählen",
            filetypes=[("Bilder", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file:
            self.test_image_path = file
            img = Image.open(file)
            img.thumbnail((900, 700))
            photo = ctk.CTkImage(img, size=img.size)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo

    def run_detection(self):
        """Führt Erkennung aus."""
        if not self.test_image_path:
            messagebox.showerror("Fehler", "Bitte Bild auswählen.")
            return

        model_path = self.model_combo.get()
        if not model_path or "Noch keine" in model_path:
            messagebox.showerror("Fehler", "Bitte trainiertes Modell auswählen.")
            return

        if not Path(model_path).exists():
            messagebox.showerror("Fehler", f"Modell nicht gefunden:\n{model_path}")
            return

        try:
            from ultralytics import YOLO

            self.results_box.delete("1.0", "end")
            self.results_box.insert("end", "⏳ Lade Modell...\n")
            self.update()

            model = YOLO(model_path)

            self.results_box.insert("end", "🔍 Erkenne...\n")
            self.update()

            results = model.predict(
                self.test_image_path,
                conf=self.conf_slider.get(),
                verbose=False
            )

            result = results[0]

            # Bild mit Boxen
            annotated = result.plot()
            img = Image.fromarray(annotated[..., ::-1])
            img.thumbnail((900, 700))
            photo = ctk.CTkImage(img, size=img.size)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

            # Ergebnisse
            self.results_box.delete("1.0", "end")

            if result.boxes is not None and len(result.boxes) > 0:
                self.results_box.insert("end", f"✅ {len(result.boxes)} gefunden:\n")
                self.results_box.insert("end", "-" * 25 + "\n")

                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    conf = float(box.conf[0])
                    self.results_box.insert("end", f"• {cls_name}: {conf:.1%}\n")
            else:
                self.results_box.insert("end", "❌ Nichts gefunden.\n\n")
                self.results_box.insert("end", "Tipps:\n• Confidence senken\n• Mehr trainieren")

        except Exception as e:
            messagebox.showerror("Fehler", str(e))


def main():
    app = YOLOStudio()
    app.mainloop()


if __name__ == "__main__":
    main()
