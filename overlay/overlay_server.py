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

# Performance-Profile: Auflösung → geschätzte VRAM + FPS + Qualität
# (basierend auf Benchmarks deiner RTX 4070 Super mit v3 Modell)
# Presets für Anfänger — bundelt alle Settings zusammen
SIMPLE_PRESETS = {
    'standard': {
        'name':    'Standard',
        'emoji':   '◉',
        'desc':    'Empfohlen für alle — ausbalanciert',
        'profile':         'high',
        'conf':            0.35,
        'show_labels':     True,
        'show_crosshair':  True,
        'show_hud_regions': False,
        'color':           '#ff8c00',
        'box_thickness':   3,
        'glow':            True,
    },
    'competitive': {
        'name':    'Competitive',
        'emoji':   '◆',
        'desc':    'Maximale Genauigkeit für serious Gameplay',
        'profile':         'ultra',
        'conf':            0.40,
        'show_labels':     True,
        'show_crosshair':  True,
        'show_hud_regions': False,
        'color':           '#ffcc00',
        'box_thickness':   4,
        'glow':            True,
    },
    'minimal': {
        'name':    'Minimal',
        'emoji':   '○',
        'desc':    'Leichtgewichtig, für schwache PCs',
        'profile':         'fast',
        'conf':            0.30,
        'show_labels':     False,
        'show_crosshair':  False,
        'show_hud_regions': False,
        'color':           '#ffffff',
        'box_thickness':   2,
        'glow':            False,
    },
    'stream': {
        'name':    'Stream',
        'emoji':   '●',
        'desc':    'Clean Look für Streaming & Videos',
        'profile':         'high',
        'conf':            0.50,
        'show_labels':     False,
        'show_crosshair':  False,
        'show_hud_regions': False,
        'color':           '#ffa833',
        'box_thickness':   3,
        'glow':            True,
    },
}


PERFORMANCE_PROFILES = {
    'ultra': {
        'name':        'Ultra',
        'imgsz':       1280,
        'use_engine':  True,    # Nutzt TensorRT Engine wenn 1280
        'vram_mb':     1500,
        'vram_pct':    75,      # % der aktuellen GPU-Auslastung
        'fps':         84,
        'quality_pct': 100,
        'desc':        'Maximale Qualität, TensorRT Engine, beste Distanz-Erkennung',
    },
    'high': {
        'name':        'High',
        'imgsz':       960,
        'use_engine':  False,
        'vram_mb':     1000,
        'vram_pct':    55,
        'fps':         130,
        'quality_pct': 92,
        'desc':        'Gute Qualität, schneller, für meiste Situationen ausreichend',
    },
    'balanced': {
        'name':        'Balanced',
        'imgsz':       768,
        'use_engine':  False,
        'vram_mb':     700,
        'vram_pct':    40,
        'fps':         180,
        'quality_pct': 85,
        'desc':        'Ausgewogen, für autonome Bots, flüssig',
    },
    'fast': {
        'name':        'Fast',
        'imgsz':       640,
        'use_engine':  False,
        'vram_mb':     500,
        'vram_pct':    30,
        'fps':         250,
        'quality_pct': 75,
        'desc':        'Maximum Speed, reduzierte Distanz-Erkennung',
    },
    'extreme': {
        'name':        'Extreme',
        'imgsz':       480,
        'use_engine':  False,
        'vram_mb':     350,
        'vram_pct':    20,
        'fps':         350,
        'quality_pct': 60,
        'desc':        'Nur Nah-Erkennung, extreme FPS für Tests',
    },
}

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
    detection_color  = '#ff8c00'
    current_fps      = 0.0
    detection_count  = 0
    total_detections = 0
    frame_count      = 0
    session_start    = None
    fps_history      = deque(maxlen=60)
    profile          = 'ultra'
    current_frame    = None
    # Erweiterte Optionen
    box_thickness    = 3
    glow             = True
    show_conf        = True
    min_box_size     = 400       # Minimum Box-Area (px²)
    max_detections   = 10
    active_preset    = 'standard'
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
        # Beide Modelle laden wenn verfuegbar
        self.engine_model = None
        self.pt_model = None

        if ENGINE_PATH.exists():
            print(f"[Engine] Lade TensorRT: {ENGINE_PATH.name}")
            self.engine_model = YOLO(str(ENGINE_PATH), task='detect')
        if PT_PATH.exists():
            print(f"[Engine] Lade PyTorch:  {PT_PATH.name}")
            self.pt_model = YOLO(str(PT_PATH), task='detect')

        if not self.engine_model and not self.pt_model:
            print(f"[Engine] Kein eigenes Modell — nutze COCO")
            self.pt_model = YOLO('yolov8l.pt')
            self.classes = [0]
        else:
            self.classes = None

    def _get_model(self):
        """Waehlt Modell basierend auf aktivem Profil."""
        profile = PERFORMANCE_PROFILES.get(state.profile, PERFORMANCE_PROFILES['ultra'])
        if profile['use_engine'] and self.engine_model:
            return self.engine_model, profile['imgsz']
        return self.pt_model, profile['imgsz']

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

                model, imgsz = self._get_model()
                results = model(
                    frame, classes=self.classes,
                    conf=state.conf_threshold,
                    imgsz=imgsz, verbose=False
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
                    state.current_frame = frame   # fuer Live-Stream
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
            box_thick = state.box_thickness
            glow_on = state.glow
            show_conf = state.show_conf
            max_det = state.max_detections

        # Limit
        if len(boxes) > max_det:
            combined = sorted(zip(confs, boxes), reverse=True)[:max_det]
            confs = [c for c, _ in combined]
            boxes = [b for _, b in combined]

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

            # Glow-Effekt (wenn aktiviert)
            if glow_on:
                glow_col = QColor(base_color.red(), base_color.green(), base_color.blue(), 60)
                for glow_w in range(box_thick + 6, box_thick, -2):
                    painter.setPen(QPen(glow_col, glow_w))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(px1, py1, px2 - px1, py2 - py1)

            # Haupt-Box
            pen = QPen(col, box_thick)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(px1, py1, px2 - px1, py2 - py1)

            # Ecken-Marker fuer futuristisches Aussehen
            corner_len = 20
            pen.setWidth(box_thick + 2)
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
                label_text = f"SURVIVOR  {conf:.0%}" if show_conf else "SURVIVOR"
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
            'profile':          state.profile,
            'profiles':         PERFORMANCE_PROFILES,
            'trt_available':    ENGINE_PATH.exists(),
            'box_thickness':    state.box_thickness,
            'glow':             state.glow,
            'show_conf':        state.show_conf,
            'min_box_size':     state.min_box_size,
            'max_detections':   state.max_detections,
            'active_preset':    state.active_preset,
            'presets':          SIMPLE_PRESETS,
        })


@app.route('/api/preset', methods=['POST'])
def api_preset():
    """Wechselt zu einem Simple Preset — wendet ALLE Settings gleichzeitig an."""
    data = request.json or {}
    name = data.get('preset')
    preset = SIMPLE_PRESETS.get(name)
    if not preset:
        return jsonify({'ok': False, 'error': 'unknown preset'}), 400

    with state.lock:
        state.active_preset    = name
        state.profile          = preset['profile']
        state.conf_threshold   = preset['conf']
        state.show_labels      = preset['show_labels']
        state.show_crosshair   = preset['show_crosshair']
        state.show_hud_regions = preset['show_hud_regions']
        state.detection_color  = preset['color']
        state.box_thickness    = preset['box_thickness']
        state.glow             = preset['glow']
    return jsonify({'ok': True, 'preset': name})


@app.route('/api/profile', methods=['POST'])
def api_profile():
    data = request.json or {}
    name = data.get('profile')
    if name not in PERFORMANCE_PROFILES:
        return jsonify({'ok': False, 'error': 'unknown profile'}), 400
    with state.lock:
        state.profile = name
    return jsonify({'ok': True, 'profile': name})


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
        if 'box_thickness' in data:
            state.box_thickness = max(1, min(10, int(data['box_thickness'])))
        if 'glow' in data:
            state.glow = bool(data['glow'])
        if 'show_conf' in data:
            state.show_conf = bool(data['show_conf'])
        if 'min_box_size' in data:
            state.min_box_size = max(0, int(data['min_box_size']))
        if 'max_detections' in data:
            state.max_detections = max(1, min(50, int(data['max_detections'])))
        # Manuelle Änderung = custom
        state.active_preset = 'custom'
    return jsonify({'ok': True})


def mjpeg_generator():
    """Live-Stream mit gezeichneten Detections — MJPEG Format."""
    stream_w, stream_h = 800, 450  # reduzierte Aufloesung fuer Web
    last_frame_id = -1
    idle_frame = np.zeros((stream_h, stream_w, 3), dtype=np.uint8)
    cv2.putText(idle_frame, "NO SIGNAL — Press START",
                (180, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)

    while True:
        with state.lock:
            frame  = state.current_frame
            boxes  = list(state.current_boxes)
            confs  = list(state.current_confs)
            frame_id = state.frame_count
            color_hex = state.detection_color
            running = state.running

        if not running or frame is None:
            out = idle_frame
        else:
            # Nur bei neuem Frame verarbeiten
            if frame_id == last_frame_id:
                time.sleep(0.02)
                continue
            last_frame_id = frame_id

            # Frame verkleinern
            img = cv2.resize(frame, (stream_w, stream_h))
            sx = stream_w / frame.shape[1]
            sy = stream_h / frame.shape[0]

            # Farbe konvertieren (hex -> BGR)
            try:
                rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
                color = (rgb[2], rgb[1], rgb[0])
            except Exception:
                color = (0, 255, 136)

            # Detections zeichnen
            for (x1, y1, x2, y2), conf in zip(boxes, confs):
                px1 = int(x1 * sx); py1 = int(y1 * sy)
                px2 = int(x2 * sx); py2 = int(y2 * sy)
                cv2.rectangle(img, (px1, py1), (px2, py2), color, 2)
                label = f"{conf:.0%}"
                cv2.putText(img, label, (px1 + 3, py1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            out = img

        ok, jpeg = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            time.sleep(0.05)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


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
