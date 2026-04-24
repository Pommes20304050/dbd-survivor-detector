"""
detector.py — YOLO-basierter Survivor-Detektor für DBD
Live-Modus: python src/detector.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import mss
from ultralytics import YOLO

COCO_MODEL       = 'yolov8l.pt'       # Large: beste Distanz-Erkennung
FINETUNED_MODEL  = Path('models/best.pt')
PERSON_CLASS     = 0                   # COCO person
CONF_THRESHOLD   = 0.25                # Niedriger Threshold fuer weit entfernte Survivors
INFER_IMGSZ      = 1280                # Hohe Aufloesung = mehr Pixel fuer Distanz
DISPLAY_W        = 1280
DISPLAY_H        = 720
MONITOR_INDEX    = 1

# HUD-Regionen identisch zu den anderen Scripts
HUD_REGIONS = [
    (0.00, 0.00, 0.18, 0.75),
    (0.85, 0.00, 1.00, 0.20),
    (0.30, 0.70, 0.70, 1.00),
    (0.00, 0.80, 0.25, 1.00),
    (0.82, 0.25, 1.00, 1.00),
]


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


class SurvivorDetector:
    def __init__(self, use_finetuned: bool = True):
        self.is_finetuned = use_finetuned and FINETUNED_MODEL.exists()

        try:
            if self.is_finetuned:
                print(f"[YOLO] Fine-tuned Modell: {FINETUNED_MODEL}")
                self.model = YOLO(str(FINETUNED_MODEL))
                self.classes = None
            else:
                print(f"[YOLO] COCO Basis-Modell: {COCO_MODEL}")
                self.model = YOLO(COCO_MODEL)
                self.classes = [PERSON_CLASS]
        except Exception as e:
            print(f"[YOLO] Fehler beim Laden: {e} — Fallback auf COCO")
            self.model = YOLO(COCO_MODEL)
            self.classes = [PERSON_CLASS]
            self.is_finetuned = False

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        results = self.model(
            frame_bgr,
            classes=self.classes,
            conf=CONF_THRESHOLD,
            imgsz=INFER_IMGSZ,
            augment=True,
            verbose=False
        )[0]

        h, w = frame_bgr.shape[:2]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # HUD-Filter
            if _in_hud((x1, y1, x2, y2), w, h):
                continue
            detections.append({
                'box':    [x1, y1, x2, y2],
                'conf':   float(box.conf[0]),
                'cls':    int(box.cls[0]),
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'area':   (x2 - x1) * (y2 - y1),
            })
        return detections

    def nearest(self, frame_bgr: np.ndarray) -> dict | None:
        """Survivor der am nächsten zur Bildmitte ist."""
        dets = self.detect(frame_bgr)
        if not dets:
            return None
        h, w = frame_bgr.shape[:2]
        cx, cy = w // 2, h // 2
        return min(dets, key=lambda d:
                   (d['center'][0]-cx)**2 + (d['center'][1]-cy)**2)

    def draw(self, frame_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
        out = frame_bgr.copy()
        h, w = out.shape[:2]
        cx, cy = w // 2, h // 2

        # Fadenkreuz
        cv2.line(out, (cx-20, cy), (cx+20, cy), (100, 100, 100), 1)
        cv2.line(out, (cx, cy-20), (cx, cy+20), (100, 100, 100), 1)

        for d in detections:
            x1, y1, x2, y2 = d['box']
            conf = d['conf']
            color = (0, 255, 0) if conf > 0.7 else (0, 200, 255)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"Survivor {conf:.0%}"
            tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            cv2.rectangle(out, (x1, y1-th-8), (x1+tw+8, y1), color, -1)
            cv2.putText(out, label, (x1+4, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)

            # Zentrum + Linie zum Bildzentrum
            tcx, tcy = d['center']
            cv2.circle(out, (tcx, tcy), 5, (0, 0, 255), -1)
            cv2.line(out, (cx, cy), (tcx, tcy), (0, 0, 255), 1)
        return out


def live_loop():
    """Echtzeit-Erkennung auf dem Bildschirm. ESC zum Beenden."""
    print("DBD Survivor Detector — Live")
    print("ESC = Beenden | F = FPS anzeigen/ausblenden")

    detector = SurvivorDetector()

    show_fps = True
    fps = 0.0
    t_last = time.perf_counter()

    with mss.mss() as sct:
        mon_idx = MONITOR_INDEX if len(sct.monitors) > MONITOR_INDEX else 0
        monitor = sct.monitors[mon_idx]

        while True:
            t0 = time.perf_counter()

            frame = np.array(sct.grab(monitor))[:, :, :3]
            dets  = detector.detect(frame)
            vis   = detector.draw(frame, dets)

            # Info-Leiste
            info = f"Survivors: {len(dets)}"
            if show_fps:
                info += f"  |  {fps:.1f} FPS"
            cv2.rectangle(vis, (0, 0), (380, 50), (0, 0, 0), -1)
            cv2.putText(vis, info, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow('DBD Survivor Detector',
                       cv2.resize(vis, (DISPLAY_W, DISPLAY_H)))

            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC
                break
            if key == ord('f'):
                show_fps = not show_fps

            dt = time.perf_counter() - t0
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

    cv2.destroyAllWindows()


if __name__ == '__main__':
    live_loop()
