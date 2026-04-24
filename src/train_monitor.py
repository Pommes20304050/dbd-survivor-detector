"""
train_monitor.py — Live Fortschrittsanzeige für YOLO Training
Parst die Output-Datei und zeigt Balken + Metriken in UI
"""

import sys
import time
import re
import threading
from pathlib import Path

import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class TrainMonitor(ctk.CTk):
    def __init__(self, log_path: str):
        super().__init__()
        self.title("YOLO Training Monitor")
        self.geometry("760x520")

        self.log_path = Path(log_path)
        self.grid_columnconfigure(0, weight=1)

        # Header
        h = ctk.CTkFrame(self, height=60, corner_radius=0, fg_color="#0d1b2a")
        h.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(h, text="YOLO Training Monitor",
                     font=("Arial", 20, "bold"),
                     text_color="#4fc3f7").pack(side="left", padx=18, pady=12)

        self.status = ctk.CTkLabel(h, text="Starte...",
                                   font=("Arial", 12), text_color="#aaa")
        self.status.pack(side="right", padx=18)

        # Main
        main = ctk.CTkFrame(self)
        main.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        main.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Epoch Progress
        ctk.CTkLabel(main, text="Epoche-Fortschritt",
                     font=("Arial", 14, "bold"),
                     anchor="w").grid(row=0, column=0, sticky="w", padx=10, pady=(12, 4))

        self.epoch_bar = ctk.CTkProgressBar(main, height=18)
        self.epoch_bar.set(0)
        self.epoch_bar.grid(row=1, column=0, sticky="ew", padx=14, pady=4)

        self.epoch_lbl = ctk.CTkLabel(main, text="Epoche 0/0",
                                       font=("Consolas", 13))
        self.epoch_lbl.grid(row=2, column=0, pady=2)

        # Batch Progress (innerhalb einer Epoche)
        ctk.CTkLabel(main, text="Batch-Fortschritt (aktuelle Epoche)",
                     font=("Arial", 12),
                     text_color="#999",
                     anchor="w").grid(row=3, column=0, sticky="w", padx=10, pady=(16, 4))

        self.batch_bar = ctk.CTkProgressBar(main, height=12,
                                             progress_color="#4fc3f7")
        self.batch_bar.set(0)
        self.batch_bar.grid(row=4, column=0, sticky="ew", padx=14, pady=4)

        self.batch_lbl = ctk.CTkLabel(main, text="—",
                                       font=("Consolas", 11),
                                       text_color="#888")
        self.batch_lbl.grid(row=5, column=0, pady=2)

        # Metrics
        metrics_frame = ctk.CTkFrame(main)
        metrics_frame.grid(row=6, column=0, sticky="ew", padx=10, pady=16)
        for c in range(4):
            metrics_frame.grid_columnconfigure(c, weight=1)

        self.metric_widgets = {}
        for i, (key, label, color) in enumerate([
            ('box_loss',  'Box Loss',  '#ff6b6b'),
            ('cls_loss',  'Class Loss', '#ffd93d'),
            ('dfl_loss',  'DFL Loss',   '#6bcf7f'),
            ('map50',     'mAP@50',     '#4fc3f7'),
        ]):
            f = ctk.CTkFrame(metrics_frame, fg_color="#1a1a2e", corner_radius=8)
            f.grid(row=0, column=i, padx=6, pady=6, sticky="nsew")
            ctk.CTkLabel(f, text=label,
                         text_color="#888",
                         font=("Arial", 11)).pack(pady=(10, 2))
            value_lbl = ctk.CTkLabel(f, text="—",
                                      text_color=color,
                                      font=("Consolas", 18, "bold"))
            value_lbl.pack(pady=(2, 10))
            self.metric_widgets[key] = value_lbl

        # Time
        self.time_lbl = ctk.CTkLabel(main, text="",
                                      font=("Consolas", 12),
                                      text_color="#4caf50")
        self.time_lbl.grid(row=7, column=0, pady=8)

        # Log (letzten paar Zeilen)
        ctk.CTkLabel(main, text="Letzte Ausgabe:",
                     font=("Arial", 11),
                     text_color="#666",
                     anchor="w").grid(row=8, column=0, sticky="w", padx=10, pady=(8, 2))

        self.log_box = ctk.CTkTextbox(main, height=100,
                                        font=("Consolas", 10),
                                        text_color="#aaa")
        self.log_box.grid(row=9, column=0, sticky="ew", padx=14, pady=(0, 12))
        self.log_box.configure(state="disabled")

        # Polling
        self.total_epochs = 50
        self.current_epoch = 0
        self.start_time = None
        self._last_pos = 0
        self._tail_and_update()

    def _tail_and_update(self):
        """Liest Log-Datei (auch bei ANSI/Unicode-Progress-Bars) und aktualisiert UI."""
        if self.log_path.exists():
            try:
                with open(self.log_path, 'rb') as f:
                    f.seek(self._last_pos)
                    raw = f.read()
                    self._last_pos = f.tell()

                if raw:
                    text = raw.decode('utf-8', errors='replace')
                    # YOLO nutzt \r für Progress-Updates — splitten
                    chunks = re.split(r'[\r\n]+', text)
                    for chunk in chunks:
                        # ANSI Escape Codes entfernen
                        clean = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', chunk)
                        clean = clean.replace('\x1b[K', '').replace('[K', '')
                        if clean.strip():
                            self._parse_line(clean)
            except Exception:
                pass

        self.after(300, self._tail_and_update)

    def _parse_line(self, line: str):
        # Log-Zeile anzeigen (komprimiert)
        self.log_box.configure(state="normal")
        if any(k in line for k in ['/50', 'GPU', 'mAP', 'all', 'Starting', 'results', 'Epoch']):
            short = line[:110].rstrip()
            if short:
                self.log_box.insert("end", short + "\n")
                self.log_box.see("end")
        self.log_box.configure(state="disabled")

        # Start erkennen
        if self.start_time is None and 'Starting training' in line:
            self.start_time = time.time()
            self.status.configure(text="Training läuft...", text_color="#4caf50")

        # Epoch-Progress Zeile:
        # "   1/50    16.8G    1.079   5.072   1.482    19   1280: 38% ━━━━━╸ 10/26  4.2it/s 9.3s<15.0s"
        m = re.search(r'(\d+)/(\d+)\s+([\d.]+)G\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+\d+', line)
        if m:
            ep    = int(m.group(1))
            tot   = int(m.group(2))
            box_l = float(m.group(4))
            cls_l = float(m.group(5))
            dfl_l = float(m.group(6))

            if tot == self.total_epochs or ep <= self.total_epochs:
                self.current_epoch = ep
                self.total_epochs = tot
                self.metric_widgets['box_loss'].configure(text=f"{box_l:.3f}")
                self.metric_widgets['cls_loss'].configure(text=f"{cls_l:.3f}")
                self.metric_widgets['dfl_loss'].configure(text=f"{dfl_l:.3f}")

                # Batch aus "10/26" am Ende extrahieren
                bm = re.search(r'(\d+)/(\d+)(?!\d)\s*[\d.]+it/s', line)
                if not bm:
                    bm = re.search(r'(\d+)/(\d+)(?!\d)\s+\d', line[40:])
                if bm:
                    b_cur = int(bm.group(1))
                    b_tot = int(bm.group(2))
                    if b_tot > 0 and b_tot < 500:
                        ep_frac = (ep - 1 + b_cur / b_tot) / tot
                        self.epoch_bar.set(ep_frac)
                        self.epoch_lbl.configure(text=f"Epoche {ep}/{tot}")
                        self.batch_bar.set(b_cur / b_tot)
                        self.batch_lbl.configure(text=f"Batch {b_cur}/{b_tot}")
                else:
                    self.epoch_bar.set(ep / tot)
                    self.epoch_lbl.configure(text=f"Epoche {ep}/{tot}")

                # ETA
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    progress = (ep - 1) / tot if ep > 1 else 0.01
                    if progress > 0:
                        total_est = elapsed / max(progress, 0.01)
                        rem = max(0, total_est - elapsed)
                        self.time_lbl.configure(
                            text=f"Vergangen: {int(elapsed//60)}m {int(elapsed%60)}s   |   "
                                 f"Geschaetzt noch: ~{int(rem//60)}m {int(rem%60)}s"
                        )

        # mAP50 aus Validation-Zeile: "  all  36  42  0.823  0.712  0.654  0.412"
        m = re.search(r'\s+all\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+[\d.]+', line)
        if m:
            map50 = float(m.group(1))
            self.metric_widgets['map50'].configure(text=f"{map50:.3f}")

        if 'results saved to' in line.lower() or 'best.pt' in line.lower():
            self.status.configure(text="Training fertig!", text_color="#4caf50")
            self.epoch_bar.set(1.0)
            self.batch_bar.set(1.0)


if __name__ == '__main__':
    log = sys.argv[1] if len(sys.argv) > 1 else None
    if not log:
        # Standard: aktuelles Temp-Log
        import glob
        candidates = sorted(Path(
            r'C:\Users\l\AppData\Local\Temp\claude'
        ).rglob('tasks/*.output'), key=lambda p: p.stat().st_mtime, reverse=True)
        log = str(candidates[0]) if candidates else None

    if not log:
        print("Kein Log gefunden. Starte mit: python src/train_monitor.py <pfad>")
        sys.exit(1)

    app = TrainMonitor(log)
    app.mainloop()
