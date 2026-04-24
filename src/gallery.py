"""
gallery.py — Zeigt alle gelabelten Bilder mit Boxes zum Durchklicken
Nutzung: python src/gallery.py

Steuerung:
  Pfeil LINKS/RECHTS  = Blättern
  D                   = Bild + Label LÖSCHEN (Qualitätskontrolle)
  G                   = Grid-Ansicht (mehrere Bilder gleichzeitig)
  ESC                 = Beenden
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# --raw Argument: zeigt data/raw/ statt data/labeled/images/
if '--raw' in sys.argv:
    IMG_DIR = Path('data/raw')
    LBL_DIR = Path('data/_nonexistent_labels')  # keine Labels
    MODE = 'raw'
else:
    IMG_DIR = Path('data/labeled/images')
    LBL_DIR = Path('data/labeled/labels')
    MODE = 'labeled'
VIEW_W  = 1600
VIEW_H  = 900


def load_boxes(label_path: Path):
    """Lädt YOLO-Labels → pixel boxes"""
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            boxes.append((cx, cy, bw, bh))
    return boxes


def draw_boxes(img: np.ndarray, boxes: list) -> np.ndarray:
    """Zeichnet YOLO-Boxes auf das Bild."""
    out = img.copy()
    h, w = out.shape[:2]
    for cx, cy, bw, bh in boxes:
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)

        label = "SURVIVOR"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1-th-10), (x1+tw+8, y1), (0, 255, 0), -1)
        cv2.putText(out, label, (x1+4, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out


def fit_to_view(img: np.ndarray) -> np.ndarray:
    """Skaliert auf VIEW_W x VIEW_H, zentriert mit schwarzem Rand."""
    h, w = img.shape[:2]
    scale = min(VIEW_W / w, VIEW_H / h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
    oy = (VIEW_H - nh) // 2
    ox = (VIEW_W - nw) // 2
    canvas[oy:oy+nh, ox:ox+nw] = resized
    return canvas


def add_info_bar(canvas: np.ndarray, idx: int, total: int,
                 fname: str, n_boxes: int) -> np.ndarray:
    """Fügt Info-Leiste oben hinzu."""
    cv2.rectangle(canvas, (0, 0), (VIEW_W, 50), (15, 15, 25), -1)
    info = f"[{idx+1} / {total}]   {fname}   |   Boxes: {n_boxes}"
    cv2.putText(canvas, info, (14, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

    cv2.rectangle(canvas, (0, VIEW_H-30), (VIEW_W, VIEW_H), (15, 15, 25), -1)
    cv2.putText(canvas,
                "PFEIL LINKS/RECHTS = Bild wechseln  |  "
                "D = Bild loeschen  |  G = Grid-Ansicht  |  ESC = Ende",
                (14, VIEW_H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return canvas


def make_grid(image_paths: list, start_idx: int, rows: int = 4, cols: int = 6) -> np.ndarray:
    """4x6 Grid mit Thumbnails + Boxes."""
    n = rows * cols
    cell_w = VIEW_W // cols
    cell_h = (VIEW_H - 50) // rows
    canvas = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)

    for i in range(n):
        idx = start_idx + i
        if idx >= len(image_paths):
            break
        img_path = image_paths[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        boxes = load_boxes(LBL_DIR / (img_path.stem + '.txt'))
        img = draw_boxes(img, boxes)

        h, w = img.shape[:2]
        scale = min(cell_w / w, cell_h / h) * 0.92
        nw, nh = int(w*scale), int(h*scale)
        thumb = cv2.resize(img, (nw, nh))

        r = i // cols
        c = i % cols
        oy = r * cell_h + (cell_h - nh) // 2
        ox = c * cell_w + (cell_w - nw) // 2
        canvas[oy:oy+nh, ox:ox+nw] = thumb

        # Kleine Nummer in der Ecke
        cv2.putText(canvas, f"{idx+1}", (ox+4, oy+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.rectangle(canvas, (0, VIEW_H-50), (VIEW_W, VIEW_H), (15, 15, 25), -1)
    cv2.putText(canvas,
                f"Grid {start_idx+1}-{min(start_idx+n, len(image_paths))} / {len(image_paths)}"
                f"   |   PFEILE = Bloecke  |  G = Zurueck zu Einzelansicht  |  ESC = Ende",
                (14, VIEW_H-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return canvas


def main():
    # --forced Filter: nur Force-Captures (die harten Fälle mit Busch/Gen)
    filter_prefix = None
    if '--forced' in sys.argv:
        filter_prefix = 'forced_'
    elif '--boosted' in sys.argv:
        filter_prefix = 'boosted_'

    # Neueste zuerst (reverse sort nach Timestamp im Namen)
    all_images = sorted(IMG_DIR.glob('*.jpg'), reverse=True)

    if filter_prefix:
        images = [p for p in all_images if p.name.startswith(filter_prefix)]
        print(f"Filter: {filter_prefix}*  |  {len(images)} Bilder gefunden")
    else:
        images = all_images

    if not images:
        print(f"Keine Bilder")
        return

    print(f"{len(images)} Bilder geladen (neueste zuerst).")
    cv2.namedWindow('Gallery', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gallery', VIEW_W, VIEW_H)

    idx = 0
    grid_mode = False

    while True:
        if grid_mode:
            view = make_grid(images, idx - (idx % 24))
        else:
            if idx >= len(images):
                idx = len(images) - 1
            img_path = images[idx]
            img = cv2.imread(str(img_path))
            if img is None:
                idx = (idx + 1) % len(images)
                continue
            boxes = load_boxes(LBL_DIR / (img_path.stem + '.txt'))
            img = draw_boxes(img, boxes)
            view = fit_to_view(img)
            view = add_info_bar(view, idx, len(images), img_path.name, len(boxes))

        cv2.imshow('Gallery', view)
        key = cv2.waitKeyEx(0)

        if key == 27:    # ESC
            break
        elif key in (2555904, ord('d')) and not grid_mode and key != ord('d'):
            # Arrow right (Windows)
            idx = (idx + 1) % len(images)
        elif key == 2555904:  # Arrow right
            if grid_mode:
                idx = min(len(images) - 1, idx + 24)
            else:
                idx = (idx + 1) % len(images)
        elif key == 2424832:  # Arrow left
            if grid_mode:
                idx = max(0, idx - 24)
            else:
                idx = (idx - 1) % len(images)
        elif key == ord('g') or key == ord('G'):
            grid_mode = not grid_mode
        elif key == ord('d') or key == ord('D'):
            if not grid_mode:
                img_path = images[idx]
                lbl_path = LBL_DIR / (img_path.stem + '.txt')
                img_path.unlink(missing_ok=True)
                lbl_path.unlink(missing_ok=True)
                print(f"Geloescht: {img_path.name}")
                images = sorted(IMG_DIR.glob('*.jpg'))
                if not images:
                    break
                idx = idx % len(images)

    cv2.destroyAllWindows()
    print(f"Fertig. {len(list(IMG_DIR.glob('*.jpg')))} Bilder uebrig.")


if __name__ == '__main__':
    main()
