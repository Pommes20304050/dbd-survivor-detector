"""
overlay_server.py — DBD Survivor Detector Live System

Startet:
  - Transparentes Overlay-Fenster auf DBD
  - Lokalen Web-Server mit Dashboard
  - YOLO v3 Detection Engine

Nutzung:
  python overlay_server.py     (oder start_v3.bat)
"""

import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

# PyTorch CUDA-DLLs auch fuer ONNX/TRT nutzbar machen
try:
    import torch
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)
except Exception:
    pass

import numpy as np
import cv2
import mss
from ultralytics import YOLO

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QTimer, QRectF, pyqtSignal, QObject
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QBrush, QPainterPath

from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS

# ─── Konfiguration ────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).resolve().parent.parent
MODELS_DIR   = BASE_DIR / 'models'
ENGINE_PATH  = MODELS_DIR / 'best.engine'  # TensorRT (schnellste)
ONNX_PATH    = MODELS_DIR / 'best.onnx'    # ONNX
PT_PATH      = MODELS_DIR / 'best.pt'      # PyTorch (Fallback)

# Prioritaet: TensorRT > PyTorch > ONNX
if ENGINE_PATH.exists():
    MODEL_PATH = ENGINE_PATH
    MODEL_TYPE = 'TensorRT FP16 (2x schneller)'
elif PT_PATH.exists():
    MODEL_PATH = PT_PATH
    MODEL_TYPE = 'PyTorch'
elif ONNX_PATH.exists():
    MODEL_PATH = ONNX_PATH
    MODEL_TYPE = 'ONNX'
else:
    MODEL_PATH = PT_PATH
    MODEL_TYPE = 'COCO Basis (kein eigenes Modell)'

INFER_IMGSZ = 1280
PORT        = 8765

HUD_REGIONS = [
    (0.00, 0.00, 0.18, 0.75),
    (0.85, 0.00, 1.00, 0.20),
    (0.30, 0.70, 0.70, 1.00),
    (0.00, 0.80, 0.25, 1.00),
    (0.82, 0.25, 1.00, 1.00),
]


# ─── Shared State ─────────────────────────────────────────────────────────

class State:
    running          = False
    conf_threshold   = 0.30
    show_hud_regions = False
    show_crosshair   = True
    show_labels      = True
    detection_color  = '#00ff88'
    current_fps      = 0.0
    detection_count  = 0
    total_detections = 0
    frame_count      = 0
    session_start    = None
    fps_history      = deque(maxlen=60)
    current_boxes    = []
    current_confs    = []
    lock             = threading.Lock()


state = State()


# ─── Detection Engine ─────────────────────────────────────────────────────

class DetectionEngine(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.model = None
        self.should_stop = threading.Event()
        self._load_model()

    def _load_model(self):
        if MODEL_PATH.exists():
            print(f"[Engine] Lade Modell: {MODEL_PATH.name}  ({MODEL_TYPE})")
            self.model = YOLO(str(MODEL_PATH), task='detect')
            self.classes = None
        else:
            print(f"[Engine] Kein eigenes Modell — nutze COCO")
            self.model = YOLO('yolov8l.pt')
            self.classes = [0]

    @staticmethod
    def _in_hud(box, w, h):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            return False
        for rx1, ry1, rx2, ry2 in HUD_REGIONS:
            hx1, hy1 = rx1 * w, ry1 * h
            hx2, hy2 = rx2 * w, ry2 * h
            iw = max(0, min(x2, hx2) - max(x1, hx1))
            ih = max(0, min(y2, hy2) - max(y1, hy1))
            if iw * ih / area > 0.7:
                return True
        return False

    def run(self):
        print(f"[Engine] Detection-Thread gestartet, wartet auf start...")
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            self.monitor_w = monitor['width']
            self.monitor_h = monitor['height']
            print(f"[Engine] Monitor: {self.monitor_w}x{self.monitor_h}")

            while not self.should_stop.is_set():
                if not state.running:
                    with state.lock:
                        state.current_boxes = []
                        state.current_confs = []
                    time.sleep(0.1)
                    continue

                t0 = time.perf_counter()
                frame = np.array(sct.grab(monitor))[:, :, :3]

                results = self.model(
                    frame, classes=self.classes,
                    conf=state.conf_threshold,
                    imgsz=INFER_IMGSZ, verbose=False
                )[0]

                boxes = []
                confs = []
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if self._in_hud((x1, y1, x2, y2), self.monitor_w, self.monitor_h):
                        continue
                    boxes.append((x1, y1, x2, y2))
                    confs.append(float(box.conf[0]))

                dt = time.perf_counter() - t0
                fps = 1.0 / max(dt, 1e-6)

                with state.lock:
                    state.current_boxes = boxes
                    state.current_confs = confs
                    state.detection_count = len(boxes)
                    state.total_detections += len(boxes)
                    state.frame_count += 1
                    state.current_fps = fps
                    state.fps_history.append(fps)


# ─── Transparent Overlay ──────────────────────────────────────────────────

class OverlayWindow(QWidget):
    def __init__(self, monitor_w, monitor_h):
        super().__init__()
        self.monitor_w = monitor_w
        self.monitor_h = monitor_h

        self.setWindowFlags(
            Qt.Tool |
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        self.setGeometry(0, 0, monitor_w, monitor_h)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(33)   # ~30 FPS repaint

    def paintEvent(self, event):
        if not state.running:
            return

        with state.lock:
            boxes = list(state.current_boxes)
            confs = list(state.current_confs)
            show_crosshair = state.show_crosshair
            show_labels = state.show_labels
            show_hud = state.show_hud_regions
            color_hex = state.detection_color

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Scale from monitor coords to widget (should be 1:1)
        scale_x = self.width() / self.monitor_w
        scale_y = self.height() / self.monitor_h

        # HUD-Regionen anzeigen (optional)
        if show_hud:
            painter.setPen(QPen(QColor(255, 80, 80, 60), 1))
            painter.setBrush(QBrush(QColor(255, 0, 0, 25)))
            for rx1, ry1, rx2, ry2 in HUD_REGIONS:
                painter.drawRect(QRectF(
                    rx1 * self.width(), ry1 * self.height(),
                    (rx2 - rx1) * self.width(), (ry2 - ry1) * self.height()
                ))

        # Crosshair
        if show_crosshair:
            cx, cy = self.width() // 2, self.height() // 2
            painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
            painter.drawLine(cx - 12, cy, cx + 12, cy)
            painter.drawLine(cx, cy - 12, cx, cy + 12)

        # Detection Boxes
        base_color = QColor(color_hex)
        for (x1, y1, x2, y2), conf in zip(boxes, confs):
            px1, py1 = int(x1 * scale_x), int(y1 * scale_y)
            px2, py2 = int(x2 * scale_x), int(y2 * scale_y)

            # Intensität vom Confidence
            intensity = int(180 + conf * 75)
            col = QColor(base_color.red(), base_color.green(), base_color.blue(), intensity)

            # Haupt-Box
            pen = QPen(col, 3)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(px1, py1, px2 - px1, py2 - py1)

            # Ecken-Marker fuer futuristisches Aussehen
            corner_len = 20
            pen.setWidth(5)
            painter.setPen(pen)
            # Oben-links
            painter.drawLine(px1, py1, px1 + corner_len, py1)
            painter.drawLine(px1, py1, px1, py1 + corner_len)
            # Oben-rechts
            painter.drawLine(px2, py1, px2 - corner_len, py1)
            painter.drawLine(px2, py1, px2, py1 + corner_len)
            # Unten-links
            painter.drawLine(px1, py2, px1 + corner_len, py2)
            painter.drawLine(px1, py2, px1, py2 - corner_len)
            # Unten-rechts
            painter.drawLine(px2, py2, px2 - corner_len, py2)
            painter.drawLine(px2, py2, px2, py2 - corner_len)

            # Label
            if show_labels:
                label_text = f"SURVIVOR  {conf:.0%}"
                font = QFont('Consolas', 11, QFont.Bold)
                painter.setFont(font)
                metrics = painter.fontMetrics()
                tw = metrics.horizontalAdvance(label_text) + 14
                th = metrics.height() + 8

                # Hintergrund
                painter.setBrush(QBrush(col))
                painter.setPen(Qt.NoPen)
                painter.drawRect(px1, py1 - th, tw, th)

                # Text
                painter.setPen(QPen(Qt.black, 1))
                painter.drawText(px1 + 7, py1 - 7, label_text)


# ─── Flask Web Server ─────────────────────────────────────────────────────

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / 'overlay' / 'templates'),
    static_folder=str(BASE_DIR / 'overlay' / 'static'),
)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    with state.lock:
        uptime = 0
        if state.session_start:
            uptime = time.time() - state.session_start
        avg_fps = (sum(state.fps_history) / len(state.fps_history)) if state.fps_history else 0
        return jsonify({
            'running':          state.running,
            'fps':              round(state.current_fps, 1),
            'avg_fps':          round(avg_fps, 1),
            'detections':       state.detection_count,
            'total_detections': state.total_detections,
            'frame_count':      state.frame_count,
            'uptime':           round(uptime, 1),
            'conf':             state.conf_threshold,
            'show_crosshair':   state.show_crosshair,
            'show_labels':      state.show_labels,
            'show_hud_regions': state.show_hud_regions,
            'color':            state.detection_color,
            'model':            MODEL_PATH.name,
            'model_exists':     MODEL_PATH.exists(),
        })


@app.route('/api/start', methods=['POST'])
def api_start():
    with state.lock:
        if not state.running:
            state.running = True
            state.session_start = time.time()
            state.total_detections = 0
            state.frame_count = 0
    return jsonify({'ok': True, 'running': True})


@app.route('/api/stop', methods=['POST'])
def api_stop():
    with state.lock:
        state.running = False
    return jsonify({'ok': True, 'running': False})


@app.route('/api/config', methods=['POST'])
def api_config():
    data = request.json or {}
    with state.lock:
        if 'conf' in data:
            state.conf_threshold = max(0.05, min(0.95, float(data['conf'])))
        if 'show_crosshair' in data:
            state.show_crosshair = bool(data['show_crosshair'])
        if 'show_labels' in data:
            state.show_labels = bool(data['show_labels'])
        if 'show_hud_regions' in data:
            state.show_hud_regions = bool(data['show_hud_regions'])
        if 'color' in data:
            state.detection_color = str(data['color'])
    return jsonify({'ok': True})


@app.route('/api/shutdown', methods=['POST'])
def api_shutdown():
    """Komplettes Herunterfahren."""
    threading.Timer(0.3, lambda: QApplication.instance().quit()).start()
    return jsonify({'ok': True})


def run_flask():
    print(f"[Web] Server läuft auf http://localhost:{PORT}")
    app.run(host='127.0.0.1', port=PORT, debug=False, use_reloader=False, threaded=True)


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DBD Survivor Detector — Live Overlay System")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"WARNUNG: {MODEL_PATH} fehlt — nutze COCO Basis")

    # Flask in separatem Thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    time.sleep(0.5)

    # Detection Engine starten
    engine = DetectionEngine()
    engine.start()

    # Qt Application
    app_qt = QApplication(sys.argv)
    app_qt.setQuitOnLastWindowClosed(False)

    # Bildschirm-Infos
    screen = app_qt.primaryScreen().geometry()
    print(f"[Overlay] Primary Screen: {screen.width()}x{screen.height()}")

    # Overlay-Fenster
    overlay = OverlayWindow(screen.width(), screen.height())
    overlay.show()

    # Browser auto-öffnen
    import webbrowser
    threading.Timer(0.8, lambda: webbrowser.open(f'http://localhost:{PORT}')).start()

    print(f"[Main] Dashboard:  http://localhost:{PORT}")
    print(f"[Main] Fertig — nutze das Dashboard zum Starten/Stoppen")

    try:
        sys.exit(app_qt.exec_())
    except SystemExit:
        engine.should_stop.set()


if __name__ == '__main__':
    main()
