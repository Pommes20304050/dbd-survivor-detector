"""
label_tool.py — Survivor in Screenshots einrahmen (YOLO-Format)
Starten: python src/label_tool.py

Bedienung:
  Linke Maustaste ziehen   → Box zeichnen
  Rechter Klick auf Box    → Box löschen
  S                        → Speichern + nächstes Bild
  N                        → Nächstes Bild (ohne Speichern)
  P                        → Vorheriges Bild
  D                        → Letzte Box rückgängig
  A                        → Auto-Vorschlag via YOLO (COCO 'person')
  ESC                      → Beenden
"""

from pathlib import Path

import cv2
import numpy as np

RAW_DIR    = Path('data/raw')
IMG_DIR    = Path('data/labeled/images')
LABEL_DIR  = Path('data/labeled/labels')

DISPLAY_W = 1280
DISPLAY_H = 720

# Globaler State (Mouse-Callback braucht das)
state = {
    'boxes':    [],   # Liste von [x1, y1, x2, y2] in Original-Koordinaten
    'drawing':  False,
    'start':    None,
    'scale':    1.0,
    'offset':   (0, 0),
    'img_h':    0,
    'img_w':    0,
}


def on_mouse(event, x, y, flags, _):
    ox, oy = state['offset']
    # Display-Koordinaten → Original-Koordinaten
    rx = int((x - ox) / state['scale'])
    ry = int((y - oy) / state['scale'])
    rx = max(0, min(state['img_w'] - 1, rx))
    ry = max(0, min(state['img_h'] - 1, ry))

    if event == cv2.EVENT_LBUTTONDOWN:
        state['drawing'] = True
        state['start']   = (rx, ry)
    elif event == cv2.EVENT_MOUSEMOVE and state['drawing']:
        state['current'] = (rx, ry)
    elif event == cv2.EVENT_LBUTTONUP and state['drawing']:
        state['drawing'] = False
        x1, y1 = state['start']
        x2, y2 = rx, ry
        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
            state['boxes'].append([min(x1, x2), min(y1, y2),
                                    max(x1, x2), max(y1, y2)])
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Box an der Klick-Position entfernen
        for i, (bx1, by1, bx2, by2) in enumerate(state['boxes']):
            if bx1 <= rx <= bx2 and by1 <= ry <= by2:
                state['boxes'].pop(i)
                break


def render(img: np.ndarray, img_idx: int, total: int, status: str = '') -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(DISPLAY_W / w, DISPLAY_H / h)
    dw, dh = int(w * scale), int(h * scale)

    display = cv2.resize(img, (dw, dh))
    canvas = np.full((DISPLAY_H, DISPLAY_W, 3), 30, dtype=np.uint8)
    ox = (DISPLAY_W - dw) // 2
    oy = (DISPLAY_H - dh) // 2
    canvas[oy:oy+dh, ox:ox+dw] = display

    state['scale']  = scale
    state['offset'] = (ox, oy)
    state['img_w']  = w
    state['img_h']  = h

    # Bestehende Boxes zeichnen
    for box in state['boxes']:
        x1, y1, x2, y2 = box
        px1 = int(x1 * scale + ox)
        py1 = int(y1 * scale + oy)
        px2 = int(x2 * scale + ox)
        py2 = int(y2 * scale + oy)
        cv2.rectangle(canvas, (px1, py1), (px2, py2), (0, 255, 0), 2)

    # Aktuell gezeichnete Box
    if state['drawing'] and state.get('current') and state.get('start'):
        x1, y1 = state['start']
        x2, y2 = state['current']
        px1 = int(x1 * scale + ox)
        py1 = int(y1 * scale + oy)
        px2 = int(x2 * scale + ox)
        py2 = int(y2 * scale + oy)
        cv2.rectangle(canvas, (px1, py1), (px2, py2), (0, 200, 255), 2)

    # Info-Leiste oben
    info = f"[{img_idx+1}/{total}]  Boxes: {len(state['boxes'])}  |  {status}"
    cv2.rectangle(canvas, (0, 0), (DISPLAY_W, 40), (10, 10, 10), -1)
    cv2.putText(canvas, info, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Box-Groessen Indikator (zeigt wie gross letzte Box relativ zum Bild ist)
    if state['boxes']:
        bx1, by1, bx2, by2 = state['boxes'][-1]
        bh_pct = (by2 - by1) / h * 100
        size_info = f"Letzte Box: {int(by2-by1)}x{int(bx2-bx1)}px  ({bh_pct:.0f}% Bildhoehe)"
        if bh_pct < 3:
            size_color = (100, 150, 255)
            size_note = " = weit entfernt"
        elif bh_pct < 15:
            size_color = (100, 255, 150)
            size_note = " = mittlere Distanz"
        else:
            size_color = (255, 220, 100)
            size_note = " = nah"
        cv2.putText(canvas, size_info + size_note,
                    (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, size_color, 1)

    # Controls unten
    help_text = "L-Klick=Box | R-Klick=BoxLoesch | S=Speichern | N=Naechste | D=Undo | X=BildLoesch | A=Auto | H=Hilfe | ESC=Ende"
    cv2.rectangle(canvas, (0, DISPLAY_H-30), (DISPLAY_W, DISPLAY_H), (10, 10, 10), -1)
    cv2.putText(canvas, help_text, (12, DISPLAY_H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return canvas


def show_help():
    """Zeigt eine visuelle Anleitung als Fenster."""
    h, w = 720, 1100
    guide = np.full((h, w, 3), 30, dtype=np.uint8)

    # Header
    cv2.rectangle(guide, (0, 0), (w, 60), (15, 50, 100), -1)
    cv2.putText(guide, "LABEL-GUIDE — So labelst du Survivors richtig",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Beispiel-Layout: links RICHTIG, rechts FALSCH
    # LINKS: Richtig (tight box)
    cv2.putText(guide, "RICHTIG", (50, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.rectangle(guide, (30, 110), (510, 520), (50, 50, 50), -1)
    # Mock-Survivor
    cv2.rectangle(guide, (240, 220), (310, 420), (100, 80, 60), -1)
    cv2.circle(guide, (275, 210), 20, (100, 80, 60), -1)
    # Tight box drumrum
    cv2.rectangle(guide, (235, 185), (315, 425), (0, 255, 0), 3)
    cv2.putText(guide, "Tight — Kopf bis Fuesse", (50, 550),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(guide, "Kein Abstand drumrum", (50, 575),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(guide, "Alle sichtbaren Teile", (50, 600),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    # RECHTS: Falsch (too loose)
    cv2.putText(guide, "FALSCH — zu gross", (580, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2)
    cv2.rectangle(guide, (560, 110), (1070, 520), (50, 50, 50), -1)
    cv2.rectangle(guide, (780, 220), (850, 420), (100, 80, 60), -1)
    cv2.circle(guide, (815, 210), 20, (100, 80, 60), -1)
    # Zu grosse Box
    cv2.rectangle(guide, (680, 140), (950, 490), (80, 80, 255), 3)
    cv2.putText(guide, "Zu viel Luft drumrum", (580, 550),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 1)
    cv2.putText(guide, "Verwirrt die KI", (580, 575),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 1)

    # Distanzen-Sektion unten
    cv2.line(guide, (0, 630), (w, 630), (100, 100, 100), 1)
    cv2.putText(guide, "DISTANZ-KATEGORIEN (relativ zur Bildhoehe):",
                (20, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(guide, "Nah: 15-40%   |   Mittel: 5-15%   |   Weit: 1-5%   |   Boxen anlegen wenn Survivor erkennbar",
                (20, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(guide, "ESC oder beliebige Taste = zurueck",
                (20, 712), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow('Label Guide', guide)
    cv2.waitKey(0)
    cv2.destroyWindow('Label Guide')


def save_label(img_path: Path, boxes: list, img_shape: tuple):
    """Speichert im YOLO-Format: class cx cy w h (normalisiert 0..1)"""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    h, w = img_shape[:2]
    dst_img   = IMG_DIR / img_path.name
    dst_label = LABEL_DIR / (img_path.stem + '.txt')

    # Bild kopieren
    if not dst_img.exists():
        import shutil
        shutil.copy(img_path, dst_img)

    with open(dst_label, 'w') as f:
        for x1, y1, x2, y2 in boxes:
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = abs(x2 - x1) / w
            bh = abs(y2 - y1) / h
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def load_existing_labels(img_path: Path, img_shape: tuple) -> list:
    """Lädt bestehende Labels wenn vorhanden."""
    label_file = LABEL_DIR / (img_path.stem + '.txt')
    if not label_file.exists():
        return []

    h, w = img_shape[:2]
    boxes = []
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            boxes.append([x1, y1, x2, y2])
    return boxes


def auto_suggest(img: np.ndarray) -> list:
    """Nutzt YOLO COCO Basis-Modell um Personen vorzuschlagen."""
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        results = model(img, classes=[0], conf=0.3, verbose=False)[0]
        return [list(map(int, box.xyxy[0])) for box in results.boxes]
    except Exception as e:
        print(f"Auto-Vorschlag Fehler: {e}")
        return []


def main():
    if not RAW_DIR.exists():
        print(f"Ordner fehlt: {RAW_DIR}")
        print("Zuerst Screenshots aufnehmen: python src/capture.py --screen --n 100")
        return

    images = sorted(RAW_DIR.glob('*.jpg')) + sorted(RAW_DIR.glob('*.png'))
    if not images:
        print(f"Keine Bilder in {RAW_DIR}")
        return

    print(f"{len(images)} Bilder gefunden. Los geht's!")

    cv2.namedWindow('Label Tool', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Label Tool', DISPLAY_W, DISPLAY_H)
    cv2.setMouseCallback('Label Tool', on_mouse)

    idx = 0
    status = ''
    while 0 <= idx < len(images):
        img_path = images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            idx += 1
            continue

        state['boxes']   = load_existing_labels(img_path, img.shape)
        state['drawing'] = False
        status = "geladen" if state['boxes'] else ""

        while True:
            canvas = render(img, idx, len(images), status)
            cv2.imshow('Label Tool', canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:   # ESC
                cv2.destroyAllWindows()
                return
            elif key == ord('s'):
                save_label(img_path, state['boxes'], img.shape)
                status = f"gespeichert ({len(state['boxes'])} Boxes)"
                idx += 1
                break
            elif key == ord('n'):
                idx += 1
                break
            elif key == ord('p'):
                idx = max(0, idx - 1)
                break
            elif key == ord('d'):
                if state['boxes']:
                    state['boxes'].pop()
            elif key == ord('x') or key == ord('X'):
                # Bild komplett loeschen (Original + bereits gelabelte Version)
                img_path.unlink(missing_ok=True)
                # Auch aus labeled/ entfernen falls schon gespeichert
                labeled_img = IMG_DIR / img_path.name
                labeled_lbl = LABEL_DIR / (img_path.stem + '.txt')
                labeled_img.unlink(missing_ok=True)
                labeled_lbl.unlink(missing_ok=True)
                print(f"Geloescht: {img_path.name}")
                # Aus der Liste entfernen und bei gleichem Index bleiben
                images.pop(idx)
                if idx >= len(images):
                    idx = len(images) - 1
                if not images:
                    cv2.destroyAllWindows()
                    return
                break
            elif key == ord('h'):
                show_help()
            elif key == ord('a'):
                status = "Auto-Vorschlag lädt..."
                canvas = render(img, idx, len(images), status)
                cv2.imshow('Label Tool', canvas)
                cv2.waitKey(1)
                state['boxes'].extend(auto_suggest(img))
                status = f"Auto-Vorschlag: {len(state['boxes'])} Boxes"

    cv2.destroyAllWindows()
    print(f"Fertig. {len(list(LABEL_DIR.glob('*.txt')))} Bilder gelabelt.")


if __name__ == '__main__':
    main()
