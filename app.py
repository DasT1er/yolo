"""
YOLO Training Studio - Simplified Professional Edition
======================================================

Einfaches, benutzerfreundliches YOLO Training Tool.
Klarer Workflow: Dataset → Training → Testen

Usage: python app.py
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import threading
import subprocess
import sys
import os
from pathlib import Path
import shutil
import yaml
import time


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
        self.training_process = None
        self.is_training = False
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

        # Nur 3 einfache Tabs
        self.tab1 = self.tabview.add("1️⃣ Dataset")
        self.tab2 = self.tabview.add("2️⃣ Training")
        self.tab3 = self.tabview.add("3️⃣ Testen")

        self.create_dataset_tab()
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
        """Dataset Management - Import, Bilder hinzufügen, Info anzeigen."""
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
            text="📥 Importieren",
            command=self.import_dataset,
            width=150,
            fg_color=PRIMARY
        ).pack(side="left", padx=5)

        # Rechte Seite - Dataset Details
        right = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=15)
        right.pack(side="right", fill="both", expand=True)

        # Oben - Info
        info_frame = ctk.CTkFrame(right, fg_color="transparent")
        info_frame.pack(fill="x", padx=25, pady=20)

        ctk.CTkLabel(info_frame, text="📊 Dataset Info", font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w")

        self.dataset_info = ctk.CTkLabel(
            info_frame,
            text="Wähle links ein Dataset aus oder importiere eines.",
            text_color=TEXT_MUTED,
            justify="left"
        )
        self.dataset_info.pack(anchor="w", pady=10)

        # Trennlinie
        ctk.CTkFrame(right, height=2, fg_color="#404050").pack(fill="x", padx=25, pady=10)

        # Bilder hinzufügen
        add_frame = ctk.CTkFrame(right, fg_color="transparent")
        add_frame.pack(fill="x", padx=25, pady=10)

        ctk.CTkLabel(add_frame, text="➕ Bilder hinzufügen", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w")

        ctk.CTkLabel(
            add_frame,
            text="Füge Bilder zu deinem Dataset hinzu. Die Labels müssen im YOLO-Format vorliegen.",
            text_color=TEXT_MUTED,
            wraplength=600
        ).pack(anchor="w", pady=5)

        btn_row = ctk.CTkFrame(add_frame, fg_color="transparent")
        btn_row.pack(fill="x", pady=10)

        self.split_var = ctk.StringVar(value="train")
        ctk.CTkLabel(btn_row, text="Ziel:").pack(side="left", padx=(0, 10))
        ctk.CTkRadioButton(btn_row, text="Training (80%)", variable=self.split_var, value="train").pack(side="left", padx=10)
        ctk.CTkRadioButton(btn_row, text="Validierung (20%)", variable=self.split_var, value="val").pack(side="left", padx=10)

        ctk.CTkButton(
            add_frame,
            text="🖼️ Bilder auswählen und hinzufügen",
            command=self.add_images_to_dataset,
            width=300,
            height=40,
            fg_color=SUCCESS
        ).pack(anchor="w", pady=10)

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

        # Hilfe-Box
        help_frame = ctk.CTkFrame(right, fg_color="#1e3a5f", corner_radius=10)
        help_frame.pack(fill="x", padx=25, pady=20)

        ctk.CTkLabel(
            help_frame,
            text="💡 Tipp: Roboflow Dataset importieren",
            font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 5))

        ctk.CTkLabel(
            help_frame,
            text="1. Exportiere dein Roboflow Dataset als 'YOLOv8' Format\n"
                 "2. Lade die ZIP-Datei herunter\n"
                 "3. Klicke auf '📥 Importieren' und wähle die ZIP\n"
                 "4. Das Dataset ist sofort einsatzbereit!",
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
                if d.is_dir():
                    # Prüfe ob data.yaml existiert
                    yaml_path = d / "data.yaml"
                    if yaml_path.exists():
                        datasets.append(d)

        if not datasets:
            ctk.CTkLabel(
                self.dataset_list,
                text="Keine Datasets gefunden.\n\nImportiere ein Dataset\noder erstelle ein neues.",
                text_color=TEXT_MUTED
            ).pack(pady=50)
            return

        for dataset in sorted(datasets):
            self.create_dataset_button(dataset)

    def create_dataset_button(self, path):
        """Erstellt einen Button für ein Dataset."""
        # Zähle Bilder
        train_count = len(list((path / "images" / "train").glob("*"))) if (path / "images" / "train").exists() else 0
        val_count = len(list((path / "images" / "val").glob("*"))) if (path / "images" / "val").exists() else 0

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
        """Wählt ein Dataset aus und zeigt Info an."""
        self.current_dataset = path

        # Lade data.yaml
        yaml_path = path / "data.yaml"
        info_text = f"📁 Name: {path.name}\n"

        if yaml_path.exists():
            with open(yaml_path) as f:
                data = yaml.safe_load(f)

            # Klassen
            names = data.get("names", {})
            if isinstance(names, dict):
                classes = list(names.values())
            else:
                classes = names

            info_text += f"\n🏷️ Klassen ({len(classes)}):\n"
            for i, cls in enumerate(classes[:10]):
                info_text += f"   {i}: {cls}\n"
            if len(classes) > 10:
                info_text += f"   ... und {len(classes) - 10} weitere\n"

        # Zähle Bilder
        train_imgs = len(list((path / "images" / "train").glob("*"))) if (path / "images" / "train").exists() else 0
        val_imgs = len(list((path / "images" / "val").glob("*"))) if (path / "images" / "val").exists() else 0
        train_labels = len(list((path / "labels" / "train").glob("*.txt"))) if (path / "labels" / "train").exists() else 0
        val_labels = len(list((path / "labels" / "val").glob("*.txt"))) if (path / "labels" / "val").exists() else 0

        info_text += f"\n📊 Statistiken:\n"
        info_text += f"   Training: {train_imgs} Bilder, {train_labels} Labels\n"
        info_text += f"   Validierung: {val_imgs} Bilder, {val_labels} Labels\n"

        # Prüfe ob bereit für Training
        if train_imgs > 0 and train_labels > 0 and val_imgs > 0:
            info_text += f"\n✅ Dataset ist bereit für Training!"
        else:
            info_text += f"\n⚠️ Dataset unvollständig - Bilder oder Labels fehlen"

        self.dataset_info.configure(text=info_text)

    def import_dataset(self):
        """Importiert ein Dataset (ZIP oder Ordner)."""
        choice = messagebox.askquestion(
            "Import-Methode",
            "ZIP-Datei importieren?\n\n'Ja' = ZIP-Datei\n'Nein' = Ordner",
            icon='question'
        )

        if choice == 'yes':
            # ZIP Import
            zip_file = filedialog.askopenfilename(
                title="ZIP-Datei auswählen",
                filetypes=[("ZIP", "*.zip")]
            )
            if zip_file:
                self.import_zip(Path(zip_file))
        else:
            # Ordner Import
            folder = filedialog.askdirectory(title="Dataset-Ordner auswählen")
            if folder:
                self.import_folder(Path(folder))

    def import_zip(self, zip_path):
        """Importiert ein Dataset aus einer ZIP-Datei."""
        import zipfile

        # Zielname
        name = zip_path.stem.replace(" ", "_")
        target = self.datasets_path / name

        if target.exists():
            if not messagebox.askyesno("Überschreiben?", f"Dataset '{name}' existiert bereits. Überschreiben?"):
                return
            shutil.rmtree(target)

        try:
            # Entpacken
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(target)

            # Prüfe ob Unterordner
            contents = list(target.iterdir())
            if len(contents) == 1 and contents[0].is_dir():
                # Verschiebe Inhalt nach oben
                sub = contents[0]
                for item in sub.iterdir():
                    shutil.move(str(item), str(target))
                sub.rmdir()

            messagebox.showinfo("Erfolg", f"Dataset '{name}' importiert!")
            self.refresh_datasets()
            self.select_dataset(target)

        except Exception as e:
            messagebox.showerror("Fehler", str(e))

    def import_folder(self, folder_path):
        """Importiert ein Dataset aus einem Ordner."""
        name = folder_path.name.replace(" ", "_")
        target = self.datasets_path / name

        if target.exists():
            if not messagebox.askyesno("Überschreiben?", f"Dataset '{name}' existiert bereits. Überschreiben?"):
                return
            shutil.rmtree(target)

        try:
            shutil.copytree(folder_path, target)
            messagebox.showinfo("Erfolg", f"Dataset '{name}' importiert!")
            self.refresh_datasets()
            self.select_dataset(target)
        except Exception as e:
            messagebox.showerror("Fehler", str(e))

    def create_new_dataset(self):
        """Erstellt ein neues leeres Dataset."""
        name = self.new_name.get().strip().replace(" ", "_")
        classes = self.new_classes.get().strip()

        if not name:
            messagebox.showerror("Fehler", "Bitte gib einen Namen ein.")
            return

        target = self.datasets_path / name
        if target.exists():
            messagebox.showerror("Fehler", f"Dataset '{name}' existiert bereits.")
            return

        try:
            # Struktur erstellen
            for split in ["train", "val"]:
                (target / "images" / split).mkdir(parents=True)
                (target / "labels" / split).mkdir(parents=True)

            # Klassen parsen
            class_list = [c.strip() for c in classes.split(",") if c.strip()]
            if not class_list:
                class_list = ["object"]

            # data.yaml erstellen
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
        """Fügt Bilder zum aktuellen Dataset hinzu."""
        if not self.current_dataset:
            messagebox.showerror("Fehler", "Bitte wähle zuerst ein Dataset aus.")
            return

        files = filedialog.askopenfilenames(
            title="Bilder auswählen",
            filetypes=[("Bilder", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )

        if not files:
            return

        split = self.split_var.get()
        target_imgs = self.current_dataset / "images" / split
        target_labels = self.current_dataset / "labels" / split

        target_imgs.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        added = 0
        for file in files:
            file_path = Path(file)
            # Kopiere Bild
            shutil.copy2(file_path, target_imgs / file_path.name)

            # Suche nach Label
            label_path = file_path.with_suffix(".txt")
            if label_path.exists():
                shutil.copy2(label_path, target_labels / label_path.name)

            added += 1

        messagebox.showinfo("Erfolg", f"{added} Bilder zu '{split}' hinzugefügt!")
        self.select_dataset(self.current_dataset)

    # ==================== TAB 2: TRAINING ====================
    def create_training_tab(self):
        """Training Tab - Einfach und übersichtlich."""
        main = ctk.CTkFrame(self.tab2, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=15, pady=15)

        # Linke Seite - Einstellungen
        left = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=15, width=450)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        # Header mit Start-Button
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

        # Dataset Auswahl
        ctk.CTkLabel(left, text="📁 Dataset:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=20, pady=(10, 5))
        self.train_dataset = ctk.CTkComboBox(left, values=self.get_dataset_names(), width=380)
        self.train_dataset.pack(anchor="w", padx=20)

        ctk.CTkButton(
            left,
            text="🔄 Liste aktualisieren",
            command=self.refresh_training_datasets,
            width=200
        ).pack(anchor="w", padx=20, pady=10)

        # Parameter
        ctk.CTkLabel(left, text="⚙️ Einstellungen:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=20, pady=(20, 10))

        params = ctk.CTkFrame(left, fg_color="transparent")
        params.pack(fill="x", padx=20)

        # Epochs
        ctk.CTkLabel(params, text="Epochen:", width=100, anchor="w").grid(row=0, column=0, pady=8)
        self.epochs_entry = ctk.CTkEntry(params, width=100, placeholder_text="100")
        self.epochs_entry.insert(0, "100")
        self.epochs_entry.grid(row=0, column=1, pady=8)
        ctk.CTkLabel(params, text="(Mehr = genauer, länger)", text_color=TEXT_MUTED).grid(row=0, column=2, padx=10)

        # Batch
        ctk.CTkLabel(params, text="Batch:", width=100, anchor="w").grid(row=1, column=0, pady=8)
        self.batch_entry = ctk.CTkEntry(params, width=100, placeholder_text="16")
        self.batch_entry.insert(0, "16")
        self.batch_entry.grid(row=1, column=1, pady=8)
        ctk.CTkLabel(params, text="(Kleiner bei wenig GPU RAM)", text_color=TEXT_MUTED).grid(row=1, column=2, padx=10)

        # Image Size
        ctk.CTkLabel(params, text="Bildgröße:", width=100, anchor="w").grid(row=2, column=0, pady=8)
        self.imgsz_entry = ctk.CTkEntry(params, width=100, placeholder_text="640")
        self.imgsz_entry.insert(0, "640")
        self.imgsz_entry.grid(row=2, column=1, pady=8)
        ctk.CTkLabel(params, text="(640 ist Standard)", text_color=TEXT_MUTED).grid(row=2, column=2, padx=10)

        # Info-Box
        info = ctk.CTkFrame(left, fg_color="#1e3a5f", corner_radius=10)
        info.pack(fill="x", padx=20, pady=20)

        ctk.CTkLabel(
            info,
            text="💡 Was passiert beim Training?",
            font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 5))

        ctk.CTkLabel(
            info,
            text="Das Modell lernt, deine Objekte zu erkennen.\n"
                 "Je mehr Bilder und Epochen, desto besser.\n\n"
                 "Nach dem Training findest du dein Modell unter:\n"
                 "modelle/[dataset]/weights/best.pt",
            text_color=TEXT_MUTED,
            justify="left"
        ).pack(anchor="w", padx=15, pady=(0, 15))

        # Rechte Seite - Logs
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
        """Gibt Liste der Dataset-Namen zurück."""
        names = []
        if self.datasets_path.exists():
            for d in self.datasets_path.iterdir():
                if d.is_dir() and (d / "data.yaml").exists():
                    names.append(d.name)
        return sorted(names)

    def refresh_training_datasets(self):
        """Aktualisiert die Dataset-Dropdown."""
        self.train_dataset.configure(values=self.get_dataset_names())

    def start_training(self):
        """Startet das Training."""
        dataset = self.train_dataset.get()
        if not dataset:
            messagebox.showerror("Fehler", "Bitte wähle ein Dataset aus.")
            return

        data_yaml = self.datasets_path / dataset / "data.yaml"
        if not data_yaml.exists():
            messagebox.showerror("Fehler", f"data.yaml nicht gefunden in {dataset}")
            return

        # Parameter
        try:
            epochs = int(self.epochs_entry.get())
            batch = int(self.batch_entry.get())
            imgsz = int(self.imgsz_entry.get())
        except ValueError:
            messagebox.showerror("Fehler", "Bitte gib gültige Zahlen für die Parameter ein.")
            return

        self.is_training = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="🟢 Training läuft...", text_color=SUCCESS)
        self.log_text.delete("1.0", "end")
        self.progress.set(0)

        # Log starten
        self.log_text.insert("end", "=" * 50 + "\n")
        self.log_text.insert("end", f"🚀 Training gestartet!\n")
        self.log_text.insert("end", f"📁 Dataset: {dataset}\n")
        self.log_text.insert("end", f"📊 Epochen: {epochs} | Batch: {batch} | Größe: {imgsz}\n")
        self.log_text.insert("end", f"🤖 Modell: YOLO11m (beste Balance)\n")
        self.log_text.insert("end", "=" * 50 + "\n\n")

        # Kommando - NUR YOLO11m (beste Balance)
        cmd = [
            sys.executable, "-m", "ultralytics",
            "detect", "train",
            f"data={data_yaml}",
            "model=yolo11m.pt",  # Immer das beste Modell
            f"epochs={epochs}",
            f"batch={batch}",
            f"imgsz={imgsz}",
            f"project={self.modelle_path}",
            f"name={dataset}",
            "exist_ok=True",
            "patience=50",  # Early stopping
            "save=True",
            "plots=True"
        ]

        def run():
            try:
                self.training_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                for line in self.training_process.stdout:
                    self.after(0, lambda l=line: self.log_text.insert("end", l))
                    self.after(0, lambda: self.log_text.see("end"))

                    # Progress parsen
                    if "Epoch" in line:
                        try:
                            # Format: "Epoch 1/100"
                            parts = line.split()
                            for p in parts:
                                if "/" in p and p[0].isdigit():
                                    current, total = map(int, p.split("/"))
                                    progress = current / total
                                    self.after(0, lambda p=progress: self.progress.set(p))
                                    self.after(0, lambda p=progress: self.progress_label.configure(text=f"{int(p*100)}%"))
                                    break
                        except:
                            pass

                self.training_process.wait()
                self.after(0, self.training_done)

            except Exception as e:
                self.after(0, lambda: self.log_text.insert("end", f"\n❌ Fehler: {e}\n"))
                self.after(0, self.training_done)

        threading.Thread(target=run, daemon=True).start()

    def stop_training(self):
        """Stoppt das Training."""
        if self.training_process:
            self.training_process.terminate()
            self.log_text.insert("end", "\n⏹️ Training gestoppt!\n")

    def training_done(self):
        """Wird aufgerufen wenn Training fertig."""
        self.is_training = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="✅ Fertig!", text_color=SUCCESS)
        self.progress.set(1)
        self.progress_label.configure(text="100%")

        self.log_text.insert("end", "\n" + "=" * 50 + "\n")
        self.log_text.insert("end", "✅ Training abgeschlossen!\n")
        self.log_text.insert("end", f"📁 Dein Modell: modelle/{self.train_dataset.get()}/weights/best.pt\n")
        self.log_text.insert("end", "=" * 50 + "\n")

        # Model Liste aktualisieren
        self.refresh_models()

    # ==================== TAB 3: TESTEN ====================
    def create_testing_tab(self):
        """Test Tab - Trainiertes Modell testen."""
        main = ctk.CTkFrame(self.tab3, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=15, pady=15)

        # Linke Seite - Einstellungen
        left = ctk.CTkFrame(main, fg_color=BG_CARD, corner_radius=15, width=400)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        ctk.CTkLabel(left, text="🔍 Modell testen", font=ctk.CTkFont(size=22, weight="bold")).pack(pady=20)

        # Modell Auswahl
        ctk.CTkLabel(left, text="🤖 Dein trainiertes Modell:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=20, pady=(10, 5))

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
            text="🖼️\n\nWähle ein Bild aus\nund klicke 'Erkennung starten'",
            font=ctk.CTkFont(size=18),
            text_color=TEXT_MUTED
        )
        self.image_label.pack(fill="both", expand=True, padx=30, pady=30)

        # Modelle laden
        self.refresh_models()

    def refresh_models(self):
        """Aktualisiert die Liste der trainierten Modelle."""
        models = []

        # Suche trainierte Modelle im modelle/ Ordner
        if self.modelle_path.exists():
            for best_pt in self.modelle_path.glob("**/weights/best.pt"):
                # Relativer Pfad
                rel = best_pt.relative_to(self.modelle_path)
                models.append(str(self.modelle_path / rel))

        if models:
            self.model_combo.configure(values=models)
            self.model_combo.set(models[0])  # Erstes Modell auswählen
        else:
            self.model_combo.configure(values=["Noch keine trainierten Modelle"])
            self.model_combo.set("Noch keine trainierten Modelle")

    def browse_model(self):
        """Lädt ein Modell von der Festplatte."""
        file = filedialog.askopenfilename(
            title="Modell auswählen",
            filetypes=[("PyTorch", "*.pt"), ("Alle", "*.*")]
        )
        if file:
            self.model_combo.set(file)

    def select_image(self):
        """Wählt ein Testbild aus."""
        file = filedialog.askopenfilename(
            title="Bild auswählen",
            filetypes=[("Bilder", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file:
            self.test_image_path = file
            # Bild anzeigen
            img = Image.open(file)
            img.thumbnail((900, 700))
            photo = ctk.CTkImage(img, size=img.size)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo

    def run_detection(self):
        """Führt die Erkennung aus."""
        if not self.test_image_path:
            messagebox.showerror("Fehler", "Bitte wähle zuerst ein Bild aus.")
            return

        model_path = self.model_combo.get()
        if not model_path or "Noch keine" in model_path:
            messagebox.showerror("Fehler", "Bitte wähle ein trainiertes Modell aus.\n\nWenn du noch keins hast, trainiere zuerst im Tab '2️⃣ Training'!")
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

            self.results_box.insert("end", "🔍 Erkenne Objekte...\n")
            self.update()

            results = model.predict(
                self.test_image_path,
                conf=self.conf_slider.get(),
                verbose=False
            )

            result = results[0]

            # Bild mit Erkennungen anzeigen
            annotated = result.plot()
            img = Image.fromarray(annotated[..., ::-1])
            img.thumbnail((900, 700))
            photo = ctk.CTkImage(img, size=img.size)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

            # Ergebnisse
            self.results_box.delete("1.0", "end")

            if result.boxes is not None and len(result.boxes) > 0:
                self.results_box.insert("end", f"✅ {len(result.boxes)} Objekte gefunden:\n")
                self.results_box.insert("end", "-" * 30 + "\n")

                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    conf = float(box.conf[0])
                    self.results_box.insert("end", f"  • {cls_name}: {conf:.1%}\n")
            else:
                self.results_box.insert("end", "❌ Keine Objekte gefunden.\n\n")
                self.results_box.insert("end", "Tipps:\n")
                self.results_box.insert("end", "• Confidence senken\n")
                self.results_box.insert("end", "• Mehr Training-Epochen\n")
                self.results_box.insert("end", "• Mehr Trainingsbilder\n")

        except Exception as e:
            messagebox.showerror("Fehler", str(e))


def main():
    app = YOLOStudio()
    app.mainloop()


if __name__ == "__main__":
    main()
