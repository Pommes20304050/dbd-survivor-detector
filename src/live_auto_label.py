"""
live_auto_label.py — Live während DBD: alle X Sekunden Screenshot + Auto-Label
KI erkennt Survivors → speichert Bild + YOLO-Label automatisch
Du spielst normal, die Daten sammeln sich von alleine.

Starten: python src/live_auto_label.py
Stoppen: ESC im Vorschau-Fenster
"""

import argparse
import threading
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import mss
from ultralytics import YOLO
from pynput import keyboard as pkeyboard

BASE_MODEL     = 'yolov8l.pt'
FINETUNED      = Path('models/best.pt')
IMG_DIR        = Path('data/labeled/images')
LBL_DIR        = Path('data/labeled/labels')
RAW_DIR        = Path('data/raw')    # fuer manuelle Force-Captures
INFER_IMGSZ    = 1280
INTERVAL_SEC   = 3.0
CONF_THRESH    = 0.45
MIN_BOX_AREA   = 1500
CAPTURE_IDX    = 1             # Monitor von dem gecaptured wird (DBD laeuft hier)
PREVIEW_IDX    = 2             # Monitor wo Vorschau angezeigt wird (wenn vorhanden)
PREVIEW_W      = 1600
PREVIEW_H      = 900

HUD_REGIONS = [
    (0.00, 0.00, 0.18, 0.75),   # Survivor-Portraits OBEN LINKS (bis weit nach unten)
    (0.85, 0.00, 1.00, 0.20),   # Killer-Power/Info OBEN RECHTS
    (0.30, 0.70, 0.70, 1.00),   # Killer-Hand unten mitte (original)
    (0.00, 0.80, 0.25, 1.00),   # Perks unten links
    (0.82, 0.25, 1.00, 1.00),   # Perks RECHTS (bis weit nach oben, wie links)
]


def in_hud(box, img_w, img_h):
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
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


def load_model():
    if FINETUNED.exists():
        print(f"[Live-Label] Nutze eigenes Modell: {FINETUNED}")
        return YOLO(str(FINETUNED)), None
    print(f"[Live-Label] Nutze COCO Basis: {BASE_MODEL}")
    return YOLO(BASE_MODEL), [0]


def main(interval: float, conf: float):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)

    model, classes = load_model()

    print(f"\n{'='*60}")
    print(f"  LIVE AUTO-LABELING")
    print(f"{'='*60}")
    print(f"  Interval:        {interval}s")
    print(f"  Confidence:      {conf}")
    print(f"  Ziel:            {IMG_DIR}")
    print(f"  Starte DBD — Bilder werden automatisch gesammelt!")
    print(f"  ESC im Vorschau-Fenster = Beenden")
    print(f"{'='*60}\n")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved       = 0
    skipped     = 0
    forced      = 0
    next_shot   = time.time()
    paused      = False

    # Globaler Hotkey (F9) — funktioniert auch wenn DBD im Fokus ist
    force_flag = threading.Event()

    def _on_global_key(key):
        try:
            if hasattr(key, 'char') and key.char and key.char.lower() == 'f':
                force_flag.set()
        except Exception:
            pass

    hotkey_listener = pkeyboard.Listener(on_press=_on_global_key)
    hotkey_listener.daemon = True
    hotkey_listener.start()
    print(f"[Live-Label] Globaler Hotkey aktiv: F = Force-Capture (auch aus DBD!)")

    with mss.mss() as sct:
        monitor = sct.monitors[CAPTURE_IDX]

        # Fenster-Position auf 2. Monitor wenn verfügbar
        window_name = 'Live Auto-Label'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, PREVIEW_W, PREVIEW_H)

        all_monitors = sct.monitors
        if len(all_monitors) > PREVIEW_IDX:
            mon2 = all_monitors[PREVIEW_IDX]
            target_x = mon2['left'] + 50
            target_y = mon2['top'] + 50
            cv2.moveWindow(window_name, target_x, target_y)
            print(f"[Live-Label] Vorschau auf Monitor {PREVIEW_IDX} ({mon2['width']}x{mon2['height']})")
        else:
            print(f"[Live-Label] Nur 1 Monitor erkannt — Vorschau auf Hauptmonitor")

        while True:
            frame = np.array(sct.grab(monitor))[:, :, :3]
            h, w = frame.shape[:2]

            now = time.time()
            about_to_shoot = (now >= next_shot) and not paused

            # Live-Detection für Vorschau (immer)
            results = model(frame, classes=classes, conf=conf,
                            imgsz=INFER_IMGSZ, verbose=False)[0]

            valid_dets = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                c = float(box.conf[0])
                if c < conf:
                    continue
                if (x2-x1) * (y2-y1) < MIN_BOX_AREA:
                    continue
                if in_hud((x1, y1, x2, y2), w, h):
                    continue
                valid_dets.append((x1, y1, x2, y2, c))

            # Wenn Zeit: speichern
            if about_to_shoot:
                next_shot = now + interval

                if valid_dets:
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    fname = f"live_{ts}.jpg"
                    cv2.imwrite(str(IMG_DIR / fname), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                    with open(LBL_DIR / (Path(fname).stem + '.txt'), 'w') as f:
                        for x1, y1, x2, y2, _ in valid_dets:
                            cx = (x1 + x2) / 2 / w
                            cy = (y1 + y2) / 2 / h
                            bw = (x2 - x1) / w
                            bh = (y2 - y1) / h
                            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    saved += 1
                    print(f"  [{saved}] {len(valid_dets)} Survivor(s) gelabelt → {fname}")
                else:
                    skipped += 1

            # Vorschau-Fenster (groessere Ausgabe)
            preview = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
            ph, pw = preview.shape[:2]

            # Detections auf Preview zeichnen (mit fettem Rahmen + Label)
            for x1, y1, x2, y2, c in valid_dets:
                px1 = int(x1 * pw / w)
                py1 = int(y1 * ph / h)
                px2 = int(x2 * pw / w)
                py2 = int(y2 * ph / h)
                color = (0, 255, 0) if c > 0.7 else (0, 200, 255)

                cv2.rectangle(preview, (px1, py1), (px2, py2), color, 3)

                label = f"SURVIVOR {c:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(preview, (px1, py1 - th - 10),
                              (px1 + tw + 10, py1), color, -1)
                cv2.putText(preview, label, (px1 + 5, py1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

                # Zentrum markieren
                tcx, tcy = (px1 + px2) // 2, (py1 + py2) // 2
                cv2.circle(preview, (tcx, tcy), 6, (0, 0, 255), -1)

            # HUD-Overlay (rot transparent)
            overlay = preview.copy()
            for rx1, ry1, rx2, ry2 in HUD_REGIONS:
                cv2.rectangle(overlay,
                              (int(rx1*pw), int(ry1*ph)),
                              (int(rx2*pw), int(ry2*ph)),
                              (0, 0, 100), -1)
            cv2.addWeighted(overlay, 0.2, preview, 0.8, 0, preview)

            # Stats oben
            bar_bg = (40, 0, 40) if paused else (0, 0, 0)
            cv2.rectangle(preview, (0, 0), (pw, 80), bar_bg, -1)

            status_str = "PAUSIERT" if paused else f"Naechstes: {max(0, int(next_shot - now))}s"
            stats = (f"Gesammelt: {saved}  |  Uebersprungen: {skipped}  |  "
                     f"{status_str}  |  Aktuell: {len(valid_dets)} Survivor")
            status_color = (100, 100, 255) if paused else (0, 255, 0)
            cv2.putText(preview, stats, (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

            # Steuerungs-Hinweis
            force_txt = f"   |   FORCE: {forced}" if forced > 0 else ""
            cv2.putText(preview,
                        f"SPACE=Pause   F=Force-Capture (global!)   ESC=Ende{force_txt}",
                        (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 100), 1)

            # Grosses Pause-Icon wenn pausiert
            if paused:
                cx, cy = pw // 2, ph // 2
                cv2.rectangle(preview, (cx-50, cy-70), (cx-15, cy+70),
                              (100, 100, 255), -1)
                cv2.rectangle(preview, (cx+15, cy-70), (cx+50, cy+70),
                              (100, 100, 255), -1)

            # Gruener Rand bei Aufnahme
            if about_to_shoot and valid_dets:
                cv2.rectangle(preview, (0, 0), (pw-1, ph-1), (0, 255, 0), 8)

            cv2.imshow(window_name, preview)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
            if key == ord(' '):
                paused = not paused
                if not paused:
                    next_shot = time.time() + interval
                print(f"[Live-Label] {'PAUSIERT' if paused else 'LÄUFT'}")

            # Force-Capture — sowohl Fenster-Taste F als auch globaler F9-Hotkey
            trigger_force = force_flag.is_set() or key == ord('f') or key == ord('F')
            if trigger_force:
                force_flag.clear()
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                fname = f"forced_{ts}.jpg"
                cv2.imwrite(str(RAW_DIR / fname), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                forced += 1
                print(f"  [FORCE] {fname} → data/raw/  ({forced} insgesamt)")
                # Kurzer gelber Blink als Feedback
                flash = preview.copy()
                cv2.rectangle(flash, (0, 0), (pw-1, ph-1), (0, 255, 255), 20)
                cv2.putText(flash, "FORCE-CAPTURE",
                            (pw//2 - 200, ph//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
                cv2.imshow(window_name, flash)
                cv2.waitKey(150)

    cv2.destroyAllWindows()
    print(f"\n{'='*60}")
    print(f"  Fertig! {saved} Bilder automatisch gelabelt.")
    print(f"  Ordner: {IMG_DIR}")
    print(f"  Jetzt Training starten!")
    print(f"{'='*60}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--interval', type=float, default=INTERVAL_SEC,
                   help='Sekunden zwischen Screenshots')
    p.add_argument('--conf',     type=float, default=CONF_THRESH,
                   help='Minimum Confidence')
    args = p.parse_args()
    main(args.interval, args.conf)
