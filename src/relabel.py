"""
relabel.py — Labelt bestehende Bilder neu mit aktuellem HUD-Filter
Speichert Bilder die nach Filter keine Survivors mehr haben in 'empty/'
"""

from pathlib import Path
import shutil
import cv2
from ultralytics import YOLO

IMG_DIR   = Path('data/labeled/images')
LBL_DIR   = Path('data/labeled/labels')
EMPTY_DIR = Path('data/empty')
BASE      = 'yolov8l.pt'
FINETUNED = Path('models/best.pt')
CONF      = 0.45
IMGSZ     = 1280

HUD_REGIONS = [
    (0.00, 0.00, 0.18, 0.75),
    (0.85, 0.00, 1.00, 0.20),
    (0.30, 0.70, 0.70, 1.00),
    (0.00, 0.80, 0.25, 1.00),
    (0.82, 0.25, 1.00, 1.00),
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


def main():
    images = sorted(list(IMG_DIR.glob('*.jpg')) + list(IMG_DIR.glob('*.png')))
    print(f"Relabeln: {len(images)} Bilder")

    if FINETUNED.exists():
        model = YOLO(str(FINETUNED))
        classes = None
    else:
        model = YOLO(BASE)
        classes = [0]

    EMPTY_DIR.mkdir(exist_ok=True)

    moved     = 0
    relabeled = 0
    for i, img_path in enumerate(images):
        if i % 25 == 0:
            print(f"  {i}/{len(images)} — relabeled={relabeled}, leer={moved}")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        results = model(img, classes=classes, conf=CONF,
                        imgsz=IMGSZ, verbose=False)[0]

        lines = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            c = float(box.conf[0])
            if c < CONF:
                continue
            if (x2-x1) * (y2-y1) < 1500:
                continue
            if in_hud((x1, y1, x2, y2), w, h):
                continue
            cx = (x1+x2) / 2 / w
            cy = (y1+y2) / 2 / h
            bw = (x2-x1) / w
            bh = (y2-y1) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        lbl_path = LBL_DIR / (img_path.stem + '.txt')

        if lines:
            with open(lbl_path, 'w') as f:
                f.write('\n'.join(lines) + '\n')
            relabeled += 1
        else:
            # Keine validen Detections → beiseite schaffen, LABEL BEHALTEN als Backup!
            shutil.move(str(img_path), str(EMPTY_DIR / img_path.name))
            if lbl_path.exists():
                # Label als Backup ins empty-Verzeichnis kopieren, dann aus labels/ entfernen
                shutil.copy(str(lbl_path), str(EMPTY_DIR / lbl_path.name))
                lbl_path.unlink()
            moved += 1

    print(f"\nFertig!")
    print(f"  Relabeled:  {relabeled}")
    print(f"  Ohne Survivor nach Filter: {moved} → nach 'data/empty/' verschoben")


if __name__ == '__main__':
    main()
