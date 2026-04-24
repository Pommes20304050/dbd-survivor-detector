"""
ui.py — DBD Survivor Detector GUI
Starten: python src/ui.py
"""

import sys
import threading
import queue
import time
import subprocess
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

sys.path.insert(0, str(Path(__file__).parent))

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def count_files(folder: Path, pattern: str = '*') -> int:
    if not folder.exists():
        return 0
    return len(list(folder.glob(pattern)))


class SurvivorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DBD Survivor Detector")
        self.geometry("950x720")
        self.minsize(850, 600)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.log_q = queue.Queue()

        self._build_header()
        self._build_tabs()
        self._build_log()
        self._poll_log()
        self._refresh_stats()

    # ─── HEADER ──────────────────────────────────────────────────────────

    def _build_header(self):
        h = ctk.CTkFrame(self, height=52, corner_radius=0, fg_color="#0d1b2a")
        h.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(h, text="DBD Survivor Detector",
                     font=("Arial", 22, "bold"),
                     text_color="#4fc3f7").pack(side="left", padx=18, pady=10)
        ctk.CTkLabel(h, text="YOLOv8  |  RTX 4070 SUPER",
                     text_color="#555", font=("Arial", 11)).pack(side="right", padx=18)

    # ─── TABS ────────────────────────────────────────────────────────────

    def _build_tabs(self):
        self.tabs = ctk.CTkTabview(self, corner_radius=8)
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)
        for t in ["Live-Monitor", "Self-Training", "Screenshots", "Labeln", "Trainieren"]:
            self.tabs.add(t)
        self._tab_live()
        self._tab_selftrain()
        self._tab_capture()
        self._tab_label()
        self._tab_train()

    # ─── TAB 1: LIVE ─────────────────────────────────────────────────────

    def _tab_live(self):
        tab = self.tabs.tab("Live-Monitor")
        tab.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(tab, text="Echtzeit Survivor-Erkennung",
                     font=("Arial", 16, "bold")).grid(row=0, column=0, sticky="w", padx=14, pady=14)

        info = ctk.CTkTextbox(tab, height=150, font=("Consolas", 12))
        info.grid(row=1, column=0, sticky="ew", padx=14, pady=6)
        info.insert("0.0",
            "Startet ein Fenster das deinen Bildschirm live analysiert.\n"
            "Erkennt Survivors und zeichnet Boxes um sie.\n\n"
            "Funktioniert SOFORT ohne Training — YOLOv8 erkennt Personen schon\n"
            "aus dem COCO-Dataset (Millionen Trainingsbilder).\n\n"
            "Fuer bessere Ergebnisse: eigene DBD-Screenshots labeln\n"
            "und danach trainieren (siehe Tabs 'Labeln' + 'Trainieren').\n\n"
            "Bedienung im Detektor-Fenster:\n"
            "  ESC = Beenden   |   F = FPS ein/aus"
        )
        info.configure(state="disabled")

        bf = ctk.CTkFrame(tab, fg_color="transparent")
        bf.grid(row=2, column=0, pady=24)
        ctk.CTkButton(bf, text="Live vom Screen", width=200,
                      font=("Arial", 14, "bold"),
                      fg_color="#1b5e20", hover_color="#2e7d32",
                      command=self._start_live).pack(side="left", padx=10)
        ctk.CTkButton(bf, text="Video analysieren", width=200,
                      font=("Arial", 14, "bold"),
                      fg_color="#1565c0", hover_color="#1976d2",
                      command=self._start_live_video).pack(side="left", padx=10)

        self.live_status = ctk.CTkLabel(tab, text="Bereit", text_color="#888")
        self.live_status.grid(row=3, column=0, pady=4)

        # Model Status
        m = ctk.CTkFrame(tab)
        m.grid(row=4, column=0, sticky="ew", padx=14, pady=12)
        self.model_label = ctk.CTkLabel(m, text="", font=("Consolas", 12))
        self.model_label.pack(padx=10, pady=10)

    def _start_live(self):
        self._log("Starte Live-Monitor (Screen) — ESC zum Beenden...")
        self.live_status.configure(text="Läuft", text_color="#4caf50")
        def run():
            subprocess.run([sys.executable, "src/monitor.py", "--screen"], check=False)
            self._ui(self.live_status.configure, text="Beendet.", text_color="#888")
            self._log("Live-Monitor beendet.")
        threading.Thread(target=run, daemon=True).start()

    def _start_live_video(self):
        video = filedialog.askopenfilename(
            title="Video für Live-Monitor auswählen",
            filetypes=[("Videos", "*.mp4 *.webm *.mkv *.avi")]
        )
        if not video:
            return
        self._log(f"Live-Monitor mit Video: {Path(video).name}")
        def run():
            subprocess.run([sys.executable, "src/monitor.py", "--video", video], check=False)
            self._log("Video-Monitor beendet.")
        threading.Thread(target=run, daemon=True).start()

    # ─── TAB: SELF-TRAINING ──────────────────────────────────────────────

    def _tab_selftrain(self):
        tab = self.tabs.tab("Self-Training")
        tab.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(tab, text="Automatisches Self-Training aus Videos",
                     font=("Arial", 16, "bold")).grid(row=0, column=0, sticky="w", padx=14, pady=14)

        info = ctk.CTkTextbox(tab, height=180, font=("Consolas", 12))
        info.grid(row=1, column=0, sticky="ew", padx=14, pady=6)
        info.insert("0.0",
            "DIE EINFACHSTE METHODE — kein Labeln noetig!\n\n"
            "Wie es funktioniert:\n"
            "  1. Videos auswaehlen (dein DBD Gameplay)\n"
            "  2. KI erkennt mit COCO-Basis schon 'Person' = Survivor\n"
            "  3. Nur hoch-confidence Detections werden als Training-Labels gespeichert\n"
            "  4. KI trainiert darauf → wird besser\n"
            "  5. KI labelt neue Frames besser → mehr Trainingsdaten\n"
            "  6. Loop bis keine Verbesserung mehr\n\n"
            "Jede Runde wird die KI strenger (hoehere Confidence) und sampled\n"
            "mehr Frames. So lernt sie automatisch Survivors in DBD zu\n"
            "erkennen — egal welche Map, Skin oder Distanz."
        )
        info.configure(state="disabled")

        # Videos-Ordner
        f = ctk.CTkFrame(tab)
        f.grid(row=2, column=0, sticky="ew", padx=14, pady=8)
        f.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(f, text="Video-Ordner:", width=110).grid(row=0, column=0, padx=8, pady=8)
        self.st_folder = ctk.CTkEntry(f, placeholder_text="Pfad zu Ordner mit Videos...")
        self.st_folder.grid(row=0, column=1, sticky="ew", padx=6)
        ctk.CTkButton(f, text="...", width=40,
                      command=self._st_pick_folder).grid(row=0, column=2, padx=6)

        # Parameter
        g = ctk.CTkFrame(tab)
        g.grid(row=3, column=0, sticky="ew", padx=14, pady=8)
        g.grid_columnconfigure((1,3), weight=1)

        ctk.CTkLabel(g, text="Runden:", width=80).grid(row=0, column=0, padx=8, pady=8)
        self.st_rounds = ctk.CTkSlider(g, from_=1, to=10, number_of_steps=9)
        self.st_rounds.set(5)
        self.st_rounds.grid(row=0, column=1, sticky="ew", padx=6)
        self.st_rounds_lbl = ctk.CTkLabel(g, text="5", width=30, font=("Arial", 13, "bold"))
        self.st_rounds_lbl.grid(row=0, column=2, padx=6)
        self.st_rounds.configure(command=lambda v: self.st_rounds_lbl.configure(text=str(int(v))))

        ctk.CTkLabel(g, text="Epochen/Runde:", width=130).grid(row=0, column=3, padx=8)
        self.st_epochs = ctk.CTkSlider(g, from_=10, to=100, number_of_steps=9)
        self.st_epochs.set(40)
        self.st_epochs.grid(row=0, column=4, sticky="ew", padx=6)
        self.st_epochs_lbl = ctk.CTkLabel(g, text="40", width=30, font=("Arial", 13, "bold"))
        self.st_epochs_lbl.grid(row=0, column=5, padx=6)
        self.st_epochs.configure(command=lambda v: self.st_epochs_lbl.configure(text=str(int(v))))

        # Progress
        self.st_bar = ctk.CTkProgressBar(tab, height=14)
        self.st_bar.set(0)
        self.st_bar.grid(row=4, column=0, sticky="ew", padx=14, pady=12)

        self.st_info = ctk.CTkLabel(tab, text="", font=("Consolas", 13), text_color="#4fc3f7")
        self.st_info.grid(row=5, column=0, pady=4)

        # Buttons
        bf = ctk.CTkFrame(tab, fg_color="transparent")
        bf.grid(row=6, column=0, pady=18)

        self.st_start_btn = ctk.CTkButton(bf, text="Self-Training starten", width=260,
                                          font=("Arial", 15, "bold"),
                                          fg_color="#6a1b9a", hover_color="#7b1fa2",
                                          command=self._start_selftrain)
        self.st_start_btn.pack(side="left", padx=10)

        ctk.CTkButton(bf, text="Historie anzeigen", width=180,
                      command=self._show_history).pack(side="left", padx=10)

    def _st_pick_folder(self):
        from tkinter import filedialog
        folder = filedialog.askdirectory(title="Video-Ordner auswählen")
        if folder:
            self.st_folder.delete(0, 'end')
            self.st_folder.insert(0, folder)

    def _start_selftrain(self):
        folder = self.st_folder.get().strip()
        if not folder or not Path(folder).is_dir():
            self._log("Bitte gültigen Video-Ordner auswählen!")
            return

        vids = list(Path(folder).glob('*.mp4')) + \
               list(Path(folder).glob('*.webm')) + \
               list(Path(folder).glob('*.mkv'))
        if not vids:
            self._log(f"Keine Videos in {folder}")
            return

        rounds = int(self.st_rounds.get())
        epochs = int(self.st_epochs.get())

        self._log(f"\nSelf-Training gestartet — {len(vids)} Videos, "
                  f"{rounds} Runden, {epochs} Epochen pro Runde")
        self._ui(self.st_start_btn.configure, state="disabled")
        self._ui(self.st_bar.set, 0)

        def run():
            import re
            proc = subprocess.Popen(
                [sys.executable, "src/iterative_train.py",
                 "--videos", folder,
                 f"--rounds={rounds}",
                 f"--epochs={epochs}"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1
            )

            current_round = 0
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                self._log(line)

                # RUNDE X/Y parsen
                m = re.search(r'RUNDE\s+(\d+)/(\d+)', line)
                if m:
                    current_round = int(m.group(1))
                    total = int(m.group(2))
                    pct = current_round / total
                    self._ui(self.st_bar.set, pct)
                    self._ui(self.st_info.configure,
                             text=f"Runde {current_round} von {total} laeuft...")

                # mAP Ergebnis parsen
                m2 = re.search(r'mAP@50\s*=\s*([\d.]+)', line)
                if m2:
                    map_val = float(m2.group(1))
                    self._ui(self.st_info.configure,
                             text=f"Runde {current_round}: mAP@50 = {map_val:.3f}",
                             text_color="#4caf50")

            proc.wait()
            self._ui(self.st_start_btn.configure, state="normal")
            self._ui(self.st_bar.set, 1.0)
            self._ui(self.st_info.configure, text="Self-Training fertig!",
                     text_color="#4caf50")
            self._refresh_stats()
            self._log("Self-Training abgeschlossen.")

        threading.Thread(target=run, daemon=True).start()

    def _show_history(self):
        path = Path('data/training_history.txt')
        if not path.exists():
            self._log("Noch keine Historie vorhanden.")
            return
        content = path.read_text()
        self._log("Training Historie:")
        for line in content.splitlines():
            self._log(f"  {line}")

    # ─── TAB 2: SCREENSHOTS ──────────────────────────────────────────────

    def _tab_capture(self):
        tab = self.tabs.tab("Screenshots")
        tab.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(tab, text="Screenshots fuer's Labeling sammeln",
                     font=("Arial", 16, "bold")).grid(row=0, column=0, sticky="w", padx=14, pady=14)

        info = ctk.CTkTextbox(tab, height=100, font=("Consolas", 12))
        info.grid(row=1, column=0, sticky="ew", padx=14, pady=6)
        info.insert("0.0",
            "Option A: Aus Videos extrahieren (schnell, viele Bilder auf einmal)\n"
            "Option B: Live vom Screen (waehrend DBD laeuft)\n\n"
            "Empfehlung: 100-200 Screenshots fuer den Start. Verschiedene Maps,\n"
            "verschiedene Survivor-Skins, verschiedene Distanzen."
        )
        info.configure(state="disabled")

        # Video capture
        f1 = ctk.CTkFrame(tab)
        f1.grid(row=2, column=0, sticky="ew", padx=14, pady=10)
        f1.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(f1, text="Aus Video:", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=6)
        ctk.CTkLabel(f1, text="Anzahl:").grid(row=1, column=0, padx=10, pady=6)
        self.n_video = ctk.CTkEntry(f1, width=80)
        self.n_video.insert(0, "200")
        self.n_video.grid(row=1, column=1, sticky="w", padx=5)
        ctk.CTkButton(f1, text="Video waehlen + Screenshots extrahieren",
                      width=300, command=self._capture_video).grid(row=1, column=2, padx=10, pady=6)

        # Screen capture
        f2 = ctk.CTkFrame(tab)
        f2.grid(row=3, column=0, sticky="ew", padx=14, pady=10)
        f2.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(f2, text="Vom Screen:", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=6)
        ctk.CTkLabel(f2, text="Anzahl:").grid(row=1, column=0, padx=10, pady=6)
        self.n_screen = ctk.CTkEntry(f2, width=80)
        self.n_screen.insert(0, "100")
        self.n_screen.grid(row=1, column=1, sticky="w", padx=5)
        ctk.CTkButton(f2, text="Screen-Capture starten (in DBD)",
                      width=300, command=self._capture_screen).grid(row=1, column=2, padx=10, pady=6)

        # Live Auto-Label (NEU)
        f3 = ctk.CTkFrame(tab, fg_color="#3a1a5f")
        f3.grid(row=4, column=0, sticky="ew", padx=14, pady=10)
        f3.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(f3, text="Live Auto-Labeling (DBD spielen + KI labelt automatisch):",
                     font=("Arial", 12, "bold"),
                     text_color="#e1bee7").grid(row=0, column=0, columnspan=3,
                                                sticky="w", padx=10, pady=6)
        ctk.CTkLabel(f3, text="Interval (Sek):").grid(row=1, column=0, padx=10)
        self.auto_interval = ctk.CTkEntry(f3, width=80)
        self.auto_interval.insert(0, "3")
        self.auto_interval.grid(row=1, column=1, sticky="w", padx=5)
        ctk.CTkButton(f3, text="Live Auto-Label starten",
                      fg_color="#6a1b9a", hover_color="#7b1fa2",
                      font=("Arial", 12, "bold"), width=300,
                      command=self._start_live_label).grid(row=1, column=2, padx=10, pady=6)

        # Stats
        self.stats_raw = ctk.CTkLabel(tab, text="", font=("Consolas", 13))
        self.stats_raw.grid(row=5, column=0, pady=12)

    def _capture_video(self):
        video = filedialog.askopenfilename(
            title="Video auswählen",
            filetypes=[("Videos", "*.mp4 *.avi *.mkv *.mov *.webm"), ("Alle", "*.*")]
        )
        if not video:
            return
        try:
            n = int(self.n_video.get())
        except ValueError:
            n = 200
        self._log(f"Extrahiere {n} Screenshots aus {Path(video).name}...")
        def run():
            subprocess.run([sys.executable, "src/capture.py",
                            "--video", video, "--n", str(n)], check=False)
            self._refresh_stats()
            self._log("Screenshot-Extraktion fertig.")
        threading.Thread(target=run, daemon=True).start()

    def _capture_screen(self):
        try:
            n = int(self.n_screen.get())
        except ValueError:
            n = 100
        self._log(f"Screen-Capture startet — DBD starten! {n} Screenshots.")
        def run():
            subprocess.run([sys.executable, "src/capture.py",
                            "--screen", "--n", str(n)], check=False)
            self._refresh_stats()
            self._log("Screen-Capture beendet.")
        threading.Thread(target=run, daemon=True).start()

    def _start_live_label(self):
        try:
            interval = float(self.auto_interval.get())
        except ValueError:
            interval = 3.0
        self._log(f"Live Auto-Label startet — DBD spielen, KI labelt alle {interval}s.")
        self._log("  Vorschau erscheint auf 2. Monitor (wenn vorhanden). ESC = Beenden.")
        def run():
            subprocess.run([sys.executable, "src/live_auto_label.py",
                            f"--interval={interval}"], check=False)
            self._refresh_stats()
            self._log("Live Auto-Label beendet.")
        threading.Thread(target=run, daemon=True).start()

    # ─── TAB 3: LABELN ───────────────────────────────────────────────────

    def _tab_label(self):
        tab = self.tabs.tab("Labeln")
        tab.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(tab, text="Screenshots labeln",
                     font=("Arial", 16, "bold")).grid(row=0, column=0, sticky="w", padx=14, pady=14)

        info = ctk.CTkTextbox(tab, height=200, font=("Consolas", 12))
        info.grid(row=1, column=0, sticky="ew", padx=14, pady=6)
        info.insert("0.0",
            "Das Label-Tool oeffnet jedes Screenshot. Du ziehst mit der Maus\n"
            "eine Box um jeden Survivor im Bild.\n\n"
            "BEDIENUNG:\n"
            "  Linke Maustaste ziehen  =  Box zeichnen\n"
            "  Rechter Klick auf Box   =  Box loeschen\n"
            "  S                       =  Speichern + naechstes Bild\n"
            "  N                       =  Naechstes Bild (ohne speichern)\n"
            "  P                       =  Vorheriges Bild\n"
            "  D                       =  Letzte Box rueckgaengig\n"
            "  A                       =  Auto-Vorschlag (YOLO Basis)\n"
            "  ESC                     =  Beenden\n\n"
            "TIPP: Taste 'A' schlaegt automatisch Boxes vor. Dann nur noch\n"
            "falsche entfernen / Survivors ergaenzen."
        )
        info.configure(state="disabled")

        ctk.CTkButton(tab, text="Label-Tool starten", width=240,
                      font=("Arial", 15, "bold"),
                      fg_color="#1565c0", hover_color="#1976d2",
                      command=self._start_label).grid(row=2, column=0, pady=20)

        self.stats_label = ctk.CTkLabel(tab, text="", font=("Consolas", 13))
        self.stats_label.grid(row=3, column=0, pady=6)

    def _start_label(self):
        self._log("Label-Tool wird gestartet (neues Fenster)...")
        def run():
            subprocess.run([sys.executable, "src/label_tool.py"], check=False)
            self._refresh_stats()
            self._log("Label-Tool beendet.")
        threading.Thread(target=run, daemon=True).start()

    # ─── TAB 4: TRAINING ─────────────────────────────────────────────────

    def _tab_train(self):
        tab = self.tabs.tab("Trainieren")
        tab.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(tab, text="YOLOv8 Fine-tuning auf deinen Labels",
                     font=("Arial", 16, "bold")).grid(row=0, column=0, sticky="w", padx=14, pady=14)

        info = ctk.CTkTextbox(tab, height=90, font=("Consolas", 12))
        info.grid(row=1, column=0, sticky="ew", padx=14, pady=6)
        info.insert("0.0",
            "Nimmt deine gelabelten Screenshots und trainiert YOLOv8 darauf.\n"
            "Mindestens 10 Bilder noetig, besser 50-100.\n"
            "Nach dem Training laeuft die Live-Erkennung automatisch mit deinem Modell."
        )
        info.configure(state="disabled")

        settings = ctk.CTkFrame(tab)
        settings.grid(row=2, column=0, sticky="ew", padx=14, pady=10)
        settings.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(settings, text="Epochen:", width=80).grid(row=0, column=0, padx=10, pady=8)
        self.ep_slider = ctk.CTkSlider(settings, from_=10, to=200, number_of_steps=19)
        self.ep_slider.set(50)
        self.ep_slider.grid(row=0, column=1, sticky="ew", padx=6)
        self.ep_label = ctk.CTkLabel(settings, text="50", width=45, font=("Arial", 13, "bold"))
        self.ep_label.grid(row=0, column=2, padx=8)
        self.ep_slider.configure(command=lambda v: self.ep_label.configure(text=str(int(v))))

        self.train_bar = ctk.CTkProgressBar(tab, height=14)
        self.train_bar.set(0)
        self.train_bar.grid(row=3, column=0, sticky="ew", padx=14, pady=8)

        self.train_info = ctk.CTkLabel(tab, text="", font=("Consolas", 12), text_color="#888")
        self.train_info.grid(row=4, column=0, pady=4)

        self.train_btn = ctk.CTkButton(tab, text="Training starten", width=240,
                                       font=("Arial", 15, "bold"),
                                       fg_color="#1565c0", hover_color="#1976d2",
                                       command=self._start_train)
        self.train_btn.grid(row=5, column=0, pady=16)

        self.stats_train = ctk.CTkLabel(tab, text="", font=("Consolas", 13))
        self.stats_train.grid(row=6, column=0, pady=6)

    def _start_train(self):
        raw_n    = count_files(Path('data/labeled/labels'), '*.txt')
        if raw_n < 10:
            self._log(f"Nur {raw_n} Labels - mindestens 10 nötig!")
            return

        epochs = int(self.ep_slider.get())
        self._log(f"Training startet ({epochs} Epochen, {raw_n} Labels)...")
        self._ui(self.train_btn.configure, state="disabled")
        self._ui(self.train_bar.set, 0)
        self._ui(self.train_info.configure, text="Läuft...")

        def run():
            proc = subprocess.Popen(
                [sys.executable, "src/train.py", f"--epochs={epochs}"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1
            )
            import re
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                self._log(line)
                # YOLO gibt Fortschritt als "1/50" oder "Epoch 1/50" aus
                m = re.search(r'(\d+)/(\d+)', line)
                if m and 'Epoch' in line or (m and line.startswith(m.group(0))):
                    try:
                        cur, tot = int(m.group(1)), int(m.group(2))
                        if tot == epochs:
                            pct = cur / tot
                            self._ui(self.train_bar.set, pct)
                            self._ui(self.train_info.configure,
                                     text=f"Epoch {cur}/{tot}")
                    except ValueError:
                        pass
            proc.wait()
            self._ui(self.train_btn.configure, state="normal")
            self._ui(self.train_bar.set, 1.0)
            self._ui(self.train_info.configure, text="Fertig!", text_color="#4caf50")
            self._refresh_stats()
            self._log("Training abgeschlossen.")
        threading.Thread(target=run, daemon=True).start()

    # ─── STATS ───────────────────────────────────────────────────────────

    def _refresh_stats(self):
        raw    = count_files(Path('data/raw'), '*.jpg')
        labeled = count_files(Path('data/labeled/labels'), '*.txt')
        model   = Path('models/best.pt').exists()

        stats_text = f"Screenshots: {raw}   |   Gelabelt: {labeled}   |   Eigenes Modell: {'JA' if model else 'nein (COCO-Basis wird genutzt)'}"

        for lbl in (getattr(self, 'stats_raw', None),
                    getattr(self, 'stats_label', None),
                    getattr(self, 'stats_train', None)):
            if lbl:
                self._ui(lbl.configure, text=stats_text)

        model_status = "Eigenes trainiertes Modell" if model else "COCO Basis-Modell (yolov8n)"
        color = "#4caf50" if model else "#ff9800"
        if hasattr(self, 'model_label'):
            self._ui(self.model_label.configure, text=f"Aktives Modell: {model_status}",
                     text_color=color)

    # ─── LOG ─────────────────────────────────────────────────────────────

    def _build_log(self):
        lf = ctk.CTkFrame(self)
        lf.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 10))
        lf.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(lf, text="LOG", font=("Arial", 10, "bold"),
                     text_color="#555").grid(row=0, column=0, sticky="w", padx=8, pady=(5, 0))
        self.log_box = ctk.CTkTextbox(lf, height=110, font=("Consolas", 11),
                                      text_color="#ccc")
        self.log_box.grid(row=1, column=0, sticky="ew", padx=8, pady=(2, 8))
        self.log_box.configure(state="disabled")

    def _log(self, msg: str):
        self.log_q.put(msg)

    def _poll_log(self):
        try:
            while True:
                msg = self.log_q.get_nowait()
                ts = time.strftime("%H:%M:%S")
                self.log_box.configure(state="normal")
                self.log_box.insert("end", f"[{ts}] {msg}\n")
                self.log_box.see("end")
                self.log_box.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    def _ui(self, fn, *args, **kwargs):
        self.after(0, lambda: fn(*args, **kwargs))


if __name__ == '__main__':
    app = SurvivorApp()
    app.mainloop()
