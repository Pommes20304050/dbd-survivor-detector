"""
capture.py — Screenshots für das Labeling aufnehmen
Aus Videos: python src/capture.py --video "pfad.mp4" --n 200
Aus Screen: python src/capture.py --screen --n 100
"""

import argparse
import random
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import mss

OUT_DIR = Path('data/raw')


def capture_from_video(video_path: str, n: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Gleichmäßig verteilte Frames + leichte Zufallsstreuung
    indices = sorted(set(
        int(i * total / n + random.randint(-10, 10))
        for i in range(n)
    ))
    indices = [max(0, min(total - 1, i)) for i in indices]

    prefix = Path(video_path).stem[:20].replace(' ', '_')
    saved = 0
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = f"{prefix}_{frame_idx:07d}.jpg"
        cv2.imwrite(str(OUT_DIR / fname), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved += 1
        if saved % 20 == 0:
            print(f"  {saved}/{n} gespeichert...")
    cap.release()
    print(f"Fertig: {saved} Screenshots in {OUT_DIR}")


def capture_from_screen(n: int, interval: float = 2.0):
    """
    Macht alle X Sekunden einen Screenshot.
    SPACE = sofort speichern, ESC = beenden.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    t_next = time.time()

    print(f"Auto-Screenshots alle {interval}s  |  SPACE = sofort  |  ESC = Ende")

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        while saved < n:
            frame = np.array(sct.grab(monitor))[:, :, :3]

            preview = cv2.resize(frame, (960, 540))
            cv2.putText(preview, f"{saved}/{n}  (SPACE=sofort, ESC=Ende)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Capture', preview)

            now = time.time()
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
            if key == ord(' ') or now >= t_next:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                cv2.imwrite(str(OUT_DIR / f"screen_{ts}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved += 1
                t_next = now + interval

    cv2.destroyAllWindows()
    print(f"Fertig: {saved} Screenshots")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--video', help='Screenshots aus Videodatei')
    g.add_argument('--screen', action='store_true', help='Screenshots live vom Screen')
    p.add_argument('--n', type=int, default=200, help='Anzahl Screenshots')
    p.add_argument('--interval', type=float, default=2.0,
                   help='Sekunden zwischen Auto-Screenshots (Screen-Modus)')
    args = p.parse_args()

    if args.video:
        capture_from_video(args.video, args.n)
    else:
        capture_from_screen(args.n, args.interval)
