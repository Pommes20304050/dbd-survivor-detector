"""
auto_label.py — KI labelt Survivors aus Videos automatisch
YOLO erkennt Personen → High-confidence Detections → als Trainings-Labels speichern.

python src/auto_label.py --videos "videos/*.mp4" --fps 1 --conf 0.65
"""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE_MODEL   = 'yolov8l.pt'
INFER_IMGSZ  = 1280

# HUD-Regionen im DBD — hier erkannte Boxen werden ignoriert
# (x_min, y_min, x_max, y_max) als Prozent der Bildgroesse (0.0 - 1.0)
HUD_REGIONS = [
    (0.00, 0.00, 0.18, 0.75),   # Survivor-Portraits OBEN LINKS (bis weit nach unten)
    (0.85, 0.00, 1.00, 0.20),   # Killer-Power/Info OBEN RECHTS
    (0.30, 0.70, 0.70, 1.00),   # Killer-Hand unten mitte
    (0.00, 0.80, 0.25, 1.00),   # Perks unten links
    (0.82, 0.25, 1.00, 1.00),   # Perks RECHTS (bis weit nach oben, wie links)
]


def is_in_hud(box, img_w, img_h):
    """Prueft ob eine Detection im HUD-Bereich liegt (zu > 70% drin)."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    box_area = bw * bh
    if box_area <= 0:
        return False

    for rx1, ry1, rx2, ry2 in HUD_REGIONS:
        hx1 = rx1 * img_w
        hy1 = ry1 * img_h
        hx2 = rx2 * img_w
        hy2 = ry2 * img_h

        # Schnittflaeche
        ix1 = max(x1, hx1)
        iy1 = max(y1, hy1)
        ix2 = min(x2, hx2)
        iy2 = min(y2, hy2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        intersection = iw * ih

        if intersection / box_area > 0.7:
            return True
    return False
FINETUNED    = Path('models/best.pt')
AUTO_IMG_DIR = Path('data/labeled/images')
AUTO_LBL_DIR = Path('data/labeled/labels')


def get_model():
    if FINETUNED.exists():
        print(f"[AutoLabel] Nutze eigenes Modell: {FINETUNED}")
        return YOLO(str(FINETUNED)), None
    print(f"[AutoLabel] Nutze COCO Basis-Modell: {BASE_MODEL}")
    return YOLO(BASE_MODEL), [0]   # nur 'person'


def label_video(video_path: Path, model, classes, fps_sample: float,
                conf_thresh: float, max_per_video: int):
    AUTO_IMG_DIR.mkdir(parents=True, exist_ok=True)
    AUTO_LBL_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Kann nicht öffnen: {video_path.name}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    import math as _math
    total_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps_val = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps_val or _math.isnan(video_fps_val) or video_fps_val <= 0:
        video_fps_val = 30.0
    total     = int(total_raw) if total_raw and total_raw > 0 else 0
    step      = max(1, int(video_fps_val / max(fps_sample, 0.1)))
    n_candidates = max(1, total // step)

    # Hash im Prefix gegen Filename-Kollisionen bei aehnlichen Video-Namen
    import hashlib
    name_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:6]
    stem_clean = video_path.stem[:30].replace(' ', '_').replace('[', '').replace(']', '')
    prefix = f"{stem_clean}_{name_hash}"

    saved  = 0
    idx    = 0
    skipped_no_det = 0
    read_failures  = 0

    print(f"\n[AutoLabel] {video_path.name}  |  ~{n_candidates:,} Kandidaten  |  step={step}")

    # SEEK statt jeden Frame decodieren → 10-50x schneller
    while saved < max_per_video:
        target = idx * step
        if total and target >= total:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret:
            read_failures += 1
            if read_failures > 5:
                break
            idx += 1
            continue
        idx += 1

        results = model(frame, classes=classes, conf=conf_thresh,
                        imgsz=INFER_IMGSZ, verbose=False)[0]
        if len(results.boxes) == 0:
            skipped_no_det += 1
            continue

        # Alle Boxen mit conf > threshold sammeln
        h, w = frame.shape[:2]
        lines = []
        skipped_hud = 0
        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            c = float(box.conf[0])
            if c < conf_thresh:
                continue
            bw, bh = x2 - x1, y2 - y1
            if bw * bh < 800:   # zu klein
                continue

            # HUD-Filter: Survivor-Portraits + Killer-Hand ignorieren
            if is_in_hud((x1, y1, x2, y2), w, h):
                skipped_hud += 1
                continue

            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            nw = bw / w
            nh = bh / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not lines:
            continue

        # Speichern
        fname = f"{prefix}_{saved:06d}.jpg"
        cv2.imwrite(str(AUTO_IMG_DIR / fname), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        with open(AUTO_LBL_DIR / (Path(fname).stem + '.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        saved += 1

        if saved % 50 == 0:
            pct = idx / total * 100
            print(f"  {video_path.name}: {saved} gelabelt "
                  f"({pct:.0f}% durch)")

    cap.release()
    print(f"  → {saved} Bilder mit Survivors gelabelt  |  "
          f"{skipped_no_det} Frames ohne Detection übersprungen")
    return saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--videos',    required=True,
                   help='Glob-Pattern (z.B. "videos/*.mp4") oder Ordner')
    p.add_argument('--fps',       type=float, default=1.0,
                   help='Samples pro Sekunde aus Video (default: 1)')
    p.add_argument('--conf',      type=float, default=0.65,
                   help='Minimum Confidence für High-Quality Labels')
    p.add_argument('--max',       type=int,   default=2000,
                   help='Maximale Bilder pro Video')
    args = p.parse_args()

    # Videos auflösen
    pattern = args.videos
    if Path(pattern).is_dir():
        videos = sorted(Path(pattern).glob('*.mp4')) + \
                 sorted(Path(pattern).glob('*.webm')) + \
                 sorted(Path(pattern).glob('*.mkv'))
    else:
        videos = [Path(p) for p in glob.glob(pattern)]

    if not videos:
        print(f"Keine Videos gefunden: {pattern}")
        return

    print(f"[AutoLabel] {len(videos)} Videos zum Labeln")
    print(f"[AutoLabel] Sample-Rate: {args.fps} FPS  |  "
          f"Conf-Threshold: {args.conf}  |  Max/Video: {args.max}")

    model, classes = get_model()

    total_saved = 0
    for v in videos:
        total_saved += label_video(v, model, classes, args.fps, args.conf, args.max)

    print(f"\n[AutoLabel] Fertig! {total_saved} neue gelabelte Bilder.")
    print(f"            Gespeichert in: {AUTO_IMG_DIR} / {AUTO_LBL_DIR}")


if __name__ == '__main__':
    main()
