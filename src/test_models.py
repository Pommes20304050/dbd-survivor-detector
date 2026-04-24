"""
test_models.py — Modelle nebeneinander vergleichen
Zeigt zwei (oder mehr) Modelle auf denselben Testbildern und vergleicht ihre Erkennung.

Modi:
  python src/test_models.py                         # Test-Bilder aus data/labeled/
  python src/test_models.py --folder data/raw       # Beliebiger Ordner
  python src/test_models.py --video pfad.mp4        # Video
  python src/test_models.py --screen                # Live vom Screen
"""

import argparse
import random
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import mss
from ultralytics import YOLO

MODELS_DIR = Path('models')
DEFAULT_IMG_DIR = Path('data/labeled/images')
INFER_IMGSZ = 1280
CONF        = 0.25

PANEL_W = 900
PANEL_H = 520


def find_models() -> list[Path]:
    """Sucht alle .pt Dateien in models/ UND nutzt COCO als Baseline."""
    models = sorted(MODELS_DIR.glob('*.pt'))
    # COCO-Basis als "Ur-Vergleich" hinzufügen (yolov8l.pt)
    return [Path('yolov8l.pt'), *models]


def label_model(path: Path) -> str:
    """Menschenlesbarer Name für Modell-Datei."""
    if path.name == 'yolov8l.pt':
        return 'COCO Basis'
    return path.stem


def detect(model, classes, frame):
    return model(frame, classes=classes, conf=CONF,
                 imgsz=INFER_IMGSZ, verbose=False)[0]


def draw_panel(frame: np.ndarray, results, title: str,
               color: tuple) -> np.ndarray:
    """Zeichnet Boxes + Info-Header in einem Panel."""
    panel = cv2.resize(frame, (PANEL_W, PANEL_H - 60))
    scale_x = PANEL_W / frame.shape[1]
    scale_y = (PANEL_H - 60) / frame.shape[0]

    n = 0
    confs = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        c = float(box.conf[0])
        px1, py1 = int(x1*scale_x), int(y1*scale_y)
        px2, py2 = int(x2*scale_x), int(y2*scale_y)
        cv2.rectangle(panel, (px1, py1), (px2, py2), color, 2)
        label = f"{c:.0%}"
        cv2.putText(panel, label, (px1+2, py1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        n += 1
        confs.append(c)

    # Header
    header = np.zeros((60, PANEL_W, 3), dtype=np.uint8)
    header[:] = (30, 30, 30)
    cv2.rectangle(header, (0, 0), (PANEL_W, 60), color, 2)
    cv2.putText(header, title, (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    avg_conf = sum(confs)/len(confs) if confs else 0
    sub = f"Detections: {n}   |   Avg Conf: {avg_conf:.0%}"
    cv2.putText(header, sub, (14, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    return np.vstack([header, panel])


def build_comparison(frame: np.ndarray, detectors, names, colors):
    """Baut Side-by-Side View aus N Modellen (max 3)."""
    panels = []
    all_results = []
    for (model, classes), name, color in zip(detectors, names, colors):
        t0 = time.perf_counter()
        res = detect(model, classes, frame)
        dt = (time.perf_counter() - t0) * 1000
        title = f"{name}   [{dt:.0f}ms]"
        panels.append(draw_panel(frame, res, title, color))
        all_results.append((name, len(res.boxes)))
    return np.hstack(panels), all_results


def pick_models() -> list[tuple]:
    """Lädt alle verfügbaren Modelle."""
    available = find_models()
    print(f"\n=== Verfügbare Modelle ===")
    for i, p in enumerate(available):
        print(f"  [{i}] {label_model(p)}  ({p})")

    loaded = []
    for p in available:
        print(f"Lade {label_model(p)}...")
        model = YOLO(str(p))
        # COCO-Basis: nur class 0 (person); eigene: alles
        classes = [0] if p.name == 'yolov8l.pt' else None
        loaded.append((model, classes, label_model(p)))
    return loaded


def test_on_images(detectors_info, img_paths, shuffle=True):
    if shuffle:
        random.shuffle(img_paths)

    colors = [(255, 180, 100), (100, 255, 180), (255, 100, 180), (100, 180, 255)]
    names  = [d[2] for d in detectors_info]
    dets   = [(d[0], d[1]) for d in detectors_info]

    # Gesamtstatistik
    totals = defaultdict(lambda: {'detections': 0, 'images': 0})

    idx = 0
    window = 'Model Comparison'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, PANEL_W * len(names), PANEL_H + 100)

    while 0 <= idx < len(img_paths):
        frame = cv2.imread(str(img_paths[idx]))
        if frame is None:
            idx += 1
            continue

        view, results = build_comparison(frame, dets, names, colors[:len(names)])

        # Gesamt-Statistik updaten
        for name, n in results:
            totals[name]['detections'] += n
            totals[name]['images'] += 1

        # Footer mit Gesamt-Stats
        footer = np.zeros((100, view.shape[1], 3), dtype=np.uint8)
        footer[:] = (15, 15, 25)
        cv2.putText(footer, f"[{idx+1}/{len(img_paths)}]  {img_paths[idx].name}",
                    (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        y_off = 50
        cv2.putText(footer, "Gesamt bisher:", (14, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
        x_off = 170
        for name, color in zip(names, colors[:len(names)]):
            t = totals[name]
            avg = t['detections'] / max(1, t['images'])
            txt = f"{name}: {t['detections']} Det ({avg:.1f}/img)"
            cv2.putText(footer, txt, (x_off, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            x_off += 280

        cv2.putText(footer, "PFEIL LINKS/RECHTS = naechstes Bild   |   R = Random   |   ESC = Ende",
                    (14, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        final = np.vstack([view, footer])
        cv2.imshow(window, final)
        key = cv2.waitKeyEx(0)
        if key == 27:
            break
        elif key == 2555904:  # Right arrow
            idx = (idx + 1) % len(img_paths)
        elif key == 2424832:  # Left arrow
            idx = (idx - 1) % len(img_paths)
        elif key == ord('r') or key == ord('R'):
            random.shuffle(img_paths)
            idx = 0

    cv2.destroyAllWindows()

    # Zusammenfassung
    print(f"\n=== Vergleich Zusammenfassung ===")
    for name in names:
        t = totals[name]
        if t['images'] > 0:
            print(f"  {name:20s}  {t['detections']:4d} Detections  "
                  f"({t['detections']/t['images']:.2f} pro Bild)")


def test_on_screen(detectors_info):
    colors = [(255, 180, 100), (100, 255, 180), (255, 100, 180), (100, 180, 255)]
    names = [d[2] for d in detectors_info]
    dets  = [(d[0], d[1]) for d in detectors_info]

    window = 'Model Comparison - LIVE'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, PANEL_W * len(names), PANEL_H)

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        while True:
            frame = np.array(sct.grab(monitor))[:, :, :3]
            view, _ = build_comparison(frame, dets, names, colors[:len(names)])
            cv2.imshow(window, view)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument('--folder', default=None, help='Ordner mit Test-Bildern')
    g.add_argument('--video',  default=None, help='Video für Frame-Tests')
    g.add_argument('--screen', action='store_true', help='Live vom Screen')
    p.add_argument('--n', type=int, default=30, help='Anzahl Test-Bilder')
    args = p.parse_args()

    detectors = pick_models()
    if len(detectors) < 2:
        print("\nBrauchst mindestens 2 Modelle zum Vergleichen.")
        print("Trainiere erst eins (bekommst COCO-Basis + dein Modell).")
        return

    if args.screen:
        test_on_screen(detectors)
    elif args.video:
        print(f"Video-Modus: extrahiere {args.n} Frames aus {args.video}")
        cap = cv2.VideoCapture(args.video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sorted(random.sample(range(total), min(args.n, total)))
        tmp = []
        tmp_dir = Path('data/_test_frames')
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for i, fi in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                pth = tmp_dir / f'test_{i:04d}.jpg'
                cv2.imwrite(str(pth), frame)
                tmp.append(pth)
        cap.release()
        test_on_images(detectors, tmp, shuffle=False)
    else:
        folder = Path(args.folder) if args.folder else DEFAULT_IMG_DIR
        images = sorted(folder.glob('*.jpg'))
        if len(images) > args.n:
            images = random.sample(images, args.n)
        if not images:
            print(f"Keine Bilder in {folder}")
            return
        print(f"\nTeste auf {len(images)} Bildern aus {folder}")
        test_on_images(detectors, images)


if __name__ == '__main__':
    main()
