"""
monitor.py — Live-Monitor: siehst was die AI in Echtzeit verarbeitet
Zwei Modi:
  python src/monitor.py --screen               → live vom Bildschirm
  python src/monitor.py --video "pfad.mp4"     → von Video
"""

import argparse
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import mss
from ultralytics import YOLO

FINETUNED = Path('models/best.pt')
BASE      = 'yolov8l.pt'
CONF      = 0.25
INFER_IMGSZ = 1280

HUD_REGIONS = [
    (0.00, 0.00, 0.18, 0.75),   # Survivor-Portraits OBEN LINKS (bis weit nach unten)
    (0.85, 0.00, 1.00, 0.20),   # Killer-Power/Info OBEN RECHTS
    (0.30, 0.70, 0.70, 1.00),   # Killer-Hand unten mitte
    (0.00, 0.80, 0.25, 1.00),   # Perks unten links
    (0.82, 0.25, 1.00, 1.00),   # Perks RECHTS (bis weit nach oben, wie links)
]


def in_hud(box, img_w, img_h):
    x1, y1, x2, y2 = box
    area = (x2-x1) * (y2-y1)
    if area <= 0:
        return False
    for rx1, ry1, rx2, ry2 in HUD_REGIONS:
        hx1, hy1 = rx1 * img_w, ry1 * img_h
        hx2, hy2 = rx2 * img_w, ry2 * img_h
        iw = max(0, min(x2, hx2) - max(x1, hx1))
        ih = max(0, min(y2, hy2) - max(y1, hy1))
        if iw * ih / area > 0.7:
            return True
    return False
VIEW_W    = 1600
VIEW_H    = 540


class Monitor:
    def __init__(self):
        if FINETUNED.exists():
            self.model = YOLO(str(FINETUNED))
            self.model_name = f"Fine-tuned: {FINETUNED.name}"
        else:
            self.model = YOLO(BASE)
            self.model_name = f"COCO Basis: {BASE}"

        self.fps_hist = deque(maxlen=30)
        self.det_hist = deque(maxlen=30)

    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, list]:
        classes = None if FINETUNED.exists() else [0]  # nur Personen im COCO-Modus
        results = self.model(frame_bgr, classes=classes, conf=CONF,
                             imgsz=INFER_IMGSZ, augment=True, verbose=False)[0]
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]

        dets = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if in_hud((x1, y1, x2, y2), w, h):
                continue
            dets.append({'box': [x1, y1, x2, y2],
                         'conf': float(box.conf[0]),
                         'center': ((x1+x2)//2, (y1+y2)//2)})

        cx, cy = w//2, h//2
        cv2.line(annotated, (cx-15, cy), (cx+15, cy), (80,80,80), 1)
        cv2.line(annotated, (cx, cy-15), (cx, cy+15), (80,80,80), 1)

        # HUD-Regionen visualisieren (rot-transparent)
        overlay = annotated.copy()
        for rx1, ry1, rx2, ry2 in HUD_REGIONS:
            hx1, hy1 = int(rx1 * w), int(ry1 * h)
            hx2, hy2 = int(rx2 * w), int(ry2 * h)
            cv2.rectangle(overlay, (hx1, hy1), (hx2, hy2), (0, 0, 100), -1)
        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

        for d in dets:
            x1,y1,x2,y2 = d['box']
            conf = d['conf']
            color = (0,255,0) if conf > 0.7 else (0,200,255)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 3)
            label = f"SURVIVOR {conf:.0%}"
            tw,th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated, (x1, y1-th-12), (x1+tw+12, y1), color, -1)
            cv2.putText(annotated, label, (x1+6, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            tcx, tcy = d['center']
            cv2.circle(annotated, (tcx, tcy), 7, (0,0,255), -1)
            cv2.line(annotated, (cx,cy), (tcx,tcy), (0,0,255), 2)

        return annotated, dets

    def side_by_side(self, raw: np.ndarray, annotated: np.ndarray,
                     info: dict) -> np.ndarray:
        target_h = VIEW_H - 80    # Platz für Header
        target_w = VIEW_W // 2

        def fit(img):
            h, w = img.shape[:2]
            scale = min(target_w / w, target_h / h)
            nw, nh = int(w*scale), int(h*scale)
            resized = cv2.resize(img, (nw, nh))
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            oy = (target_h - nh) // 2
            ox = (target_w - nw) // 2
            canvas[oy:oy+nh, ox:ox+nw] = resized
            return canvas

        left  = fit(raw)
        right = fit(annotated)

        # Labels auf beide Panels
        cv2.putText(left,  "ROHBILD",    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
        cv2.putText(right, "KI-ERKENNUNG", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        panels = np.hstack([left, right])

        # Header-Leiste oben
        header = np.zeros((80, VIEW_W, 3), dtype=np.uint8)
        header[:] = (20, 20, 30)

        cv2.putText(header, "DBD Survivor Detector — Live Monitor",
                    (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)

        avg_fps = sum(self.fps_hist) / len(self.fps_hist) if self.fps_hist else 0
        avg_det = sum(self.det_hist) / len(self.det_hist) if self.det_hist else 0

        stats = (f"Survivors: {info['survivors']}   "
                 f"Conf Avg: {info['avg_conf']:.0%}   "
                 f"FPS: {avg_fps:.1f}   "
                 f"Modell: {self.model_name}")
        cv2.putText(header, stats,
                    (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

        return np.vstack([header, panels])


def run_screen(monitor: Monitor):
    print("Live-Monitor (Screen) — ESC zum Beenden")
    with mss.mss() as sct:
        mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        while True:
            t0 = time.perf_counter()
            raw = np.array(sct.grab(mon))[:, :, :3]
            annotated, dets = monitor.process(raw)

            dt = time.perf_counter() - t0
            monitor.fps_hist.append(1.0 / max(dt, 1e-6))
            monitor.det_hist.append(len(dets))
            info = {
                'survivors': len(dets),
                'avg_conf':  sum(d['conf'] for d in dets) / max(1, len(dets)),
            }

            view = monitor.side_by_side(raw, annotated, info)
            cv2.imshow('DBD Live Monitor', view)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cv2.destroyAllWindows()


def run_video(monitor: Monitor, video_path: str, speed: float = 1.0):
    print(f"Live-Monitor (Video: {Path(video_path).name})")
    print("ESC = Beenden | SPACE = Pause | +/- = Geschwindigkeit")

    cap = cv2.VideoCapture(video_path)
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    paused = False

    while cap.isOpened():
        if not paused:
            t0 = time.perf_counter()
            ret, raw = cap.read()
            if not ret:
                break
            frame_idx += 1

            annotated, dets = monitor.process(raw)
            dt = time.perf_counter() - t0
            monitor.fps_hist.append(1.0 / max(dt, 1e-6))
            monitor.det_hist.append(len(dets))

            info = {
                'survivors': len(dets),
                'avg_conf':  sum(d['conf'] for d in dets) / max(1, len(dets)),
            }

            view = monitor.side_by_side(raw, annotated, info)

            # Progress-Leiste unten
            bar = np.zeros((8, VIEW_W, 3), dtype=np.uint8)
            prog = int(VIEW_W * frame_idx / total)
            bar[:, :prog] = (100, 200, 255)
            view = np.vstack([view, bar])

            cv2.imshow('DBD Live Monitor', view)

        delay = max(1, int(1000 / (vid_fps * speed)))
        key = cv2.waitKey(delay) & 0xFF
        if key == 27:
            break
        if key == ord(' '):
            paused = not paused
        if key == ord('+'):
            speed = min(4.0, speed * 1.5)
            print(f"Speed: {speed:.1f}x")
        if key == ord('-'):
            speed = max(0.25, speed / 1.5)
            print(f"Speed: {speed:.1f}x")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--screen', action='store_true')
    g.add_argument('--video')
    p.add_argument('--speed', type=float, default=1.0)
    args = p.parse_args()

    m = Monitor()
    if args.screen:
        run_screen(m)
    else:
        run_video(m, args.video, args.speed)
