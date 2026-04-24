"""
iterative_train.py — Self-Training Schleife für optimale Survivor-Erkennung

Strategie: Auto-Label → Train → besseres Auto-Label → Train → ...
Nach jeder Runde werden die Labels strenger gefiltert,
weil das Modell besser wird.

python src/iterative_train.py --videos "videos/" --rounds 5 --epochs 40
"""

import argparse
import glob
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

from auto_label import label_video

BASE_MODEL   = 'yolov8l.pt'   # Large: 43M params, beste Erkennung auch auf Distanz
IMGSZ        = 1280             # Hoehere Aufloesung = weit entfernte Survivors erkennbar
MODEL_OUT    = Path('models/best.pt')
IMG_DIR      = Path('data/labeled/images')
LBL_DIR      = Path('data/labeled/labels')
DATASET_ROOT = Path('data/dataset')
HISTORY      = Path('data/training_history.txt')


def prepare_dataset(val_ratio: float = 0.15):
    """Erstellt YOLO-Split aus allen gelabelten Bildern."""
    import random

    images = sorted(IMG_DIR.glob('*.jpg'))
    images = [img for img in images if (LBL_DIR / (img.stem + '.txt')).exists()]
    if len(images) < 10:
        raise RuntimeError(f"Nur {len(images)} Labels — brauche mindestens 10.")

    random.seed(42)
    random.shuffle(images)
    n_val   = max(5, int(len(images) * val_ratio))
    val_set = set(img.name for img in images[:n_val])

    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)
    for split in ('train', 'val'):
        (DATASET_ROOT / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATASET_ROOT / 'labels' / split).mkdir(parents=True, exist_ok=True)

    for img in images:
        split = 'val' if img.name in val_set else 'train'
        shutil.copy(img, DATASET_ROOT / 'images' / split / img.name)
        shutil.copy(LBL_DIR / (img.stem + '.txt'),
                    DATASET_ROOT / 'labels' / split / (img.stem + '.txt'))

    yaml_path = DATASET_ROOT / 'data.yaml'
    yaml_path.write_text(
        f"path: {DATASET_ROOT.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['survivor']\n"
    )
    print(f"[Dataset] {len(images)-n_val} train  |  {n_val} val")
    return yaml_path


def train_round(yaml_path: Path, epochs: int, batch: int, run_name: str) -> tuple[Path, float]:
    """Trainiert YOLO und gibt Pfad zum best.pt + mAP zurück."""
    # Start-Modell: entweder bestehend oder Basis
    start_model = str(MODEL_OUT) if MODEL_OUT.exists() else BASE_MODEL
    print(f"[Train] Start von: {start_model}")

    model = YOLO(start_model)
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=IMGSZ,               # 1280 = erkennt Survivors auch in Distanz
        batch=batch,
        device=0,
        project='runs',
        name=run_name,
        exist_ok=True,
        patience=15,
        optimizer='AdamW',
        lr0=5e-4,
        cos_lr=True,
        augment=True,

        # Starke Farb-Augmentation fuer viele Skin-Varianten:
        hsv_h=0.025,               # Farbton-Variation (war 0.015)
        hsv_s=0.8,                 # Saettigung stark variieren
        hsv_v=0.5,                 # Helligkeit stark variieren (Tag/Nacht Maps)

        # Geometrische Augmentation:
        fliplr=0.5,                # horizontal gespiegelt
        scale=0.6,                 # 0.4x bis 1.6x Zoom (kritisch fuer Distanz!)
        degrees=5.0,               # leichte Rotation
        translate=0.15,            # Position variieren
        shear=2.0,                 # leicht schief

        # Kontext-Augmentation:
        mosaic=1.0,                # 4 Bilder zu einem → viele Kontexte
        mixup=0.15,                # Bilder uebereinander mischen
        copy_paste=0.3,            # Survivors zwischen Bildern kopieren (massiv mehr Varianten!)

        verbose=True,
    )

    best = Path(f'runs/{run_name}/weights/best.pt')
    if not best.exists():
        raise RuntimeError("Training hat kein best.pt erzeugt")

    # mAP auslesen
    try:
        metrics = YOLO(str(best)).val(data=str(yaml_path), device=0, verbose=False)
        m_ap50 = float(metrics.results_dict.get('metrics/mAP50(B)', 0))
    except Exception as e:
        print(f"mAP-Eval Fehler: {e}")
        m_ap50 = 0.0

    # Als aktives Modell speichern
    MODEL_OUT.parent.mkdir(exist_ok=True)
    shutil.copy(best, MODEL_OUT)

    return best, m_ap50


def run_iterative(video_paths: list[Path], rounds: int, epochs: int,
                  initial_conf: float = 0.55, initial_fps: float = 0.5,
                  batch: int = 16, max_per_video: int = 1500):
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    history = []

    print(f"\n{'='*64}")
    print(f"  ITERATIVES SELF-TRAINING  —  {rounds} Runden, {epochs} Epochen")
    print(f"  Videos: {len(video_paths)}")
    print(f"{'='*64}\n")

    best_map = 0.0

    for round_i in range(1, rounds + 1):
        # Jede Runde: strengere Confidence + mehr Frames (weil Modell besser)
        conf = initial_conf + (round_i - 1) * 0.05
        conf = min(conf, 0.85)
        fps  = initial_fps * (1 + (round_i - 1) * 0.5)

        print(f"\n┏{'━'*62}┓")
        print(f"┃ RUNDE {round_i}/{rounds}  conf={conf:.2f}  fps={fps:.2f}  ┃")
        print(f"┗{'━'*62}┛")

        # 1. Auto-Labeling
        print(f"\n[Runde {round_i}] Phase 1: Auto-Labeling")
        from auto_label import get_model
        model, classes = get_model()

        new_labels = 0
        for video in video_paths:
            new_labels += label_video(
                video, model, classes,
                fps_sample=fps, conf_thresh=conf, max_per_video=max_per_video
            )
        print(f"[Runde {round_i}] {new_labels} neue Labels")

        # 2. Training
        print(f"\n[Runde {round_i}] Phase 2: Training ({epochs} Epochen)")
        yaml_path = prepare_dataset()
        t0 = time.time()
        best_path, map50 = train_round(
            yaml_path, epochs, batch, run_name=f"survivor_r{round_i}"
        )
        elapsed = time.time() - t0

        history.append({
            'round':   round_i,
            'conf':    conf,
            'labels':  new_labels,
            'map50':   map50,
            'time_s':  elapsed,
        })

        print(f"\n[Runde {round_i}] FERTIG — mAP@50 = {map50:.3f}  "
              f"(in {elapsed/60:.1f} Min)")

        # History speichern
        with open(HISTORY, 'w') as f:
            f.write("Runde | Conf | Labels | mAP@50 | Zeit (min)\n")
            f.write("-" * 52 + "\n")
            for h in history:
                f.write(f"  {h['round']:>2}  | {h['conf']:.2f} | "
                        f"{h['labels']:>6} | {h['map50']:.3f} | "
                        f"{h['time_s']/60:5.1f}\n")

        # Early Stopping wenn mAP sich kaum verbessert
        if round_i > 1 and map50 - best_map < 0.005:
            print(f"\nmAP verbessert sich nicht mehr ({map50:.3f} vs best {best_map:.3f}) — Stop.")
            break
        best_map = max(best_map, map50)

    print(f"\n{'='*64}")
    print(f"  FERTIG! Bestes mAP@50: {best_map:.3f}")
    print(f"  Modell: {MODEL_OUT}")
    print(f"  Historie: {HISTORY}")
    print(f"{'='*64}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--videos',  required=True,
                   help='Ordner mit Videos oder Glob-Pattern')
    p.add_argument('--rounds',  type=int,   default=5)
    p.add_argument('--epochs',  type=int,   default=40)
    p.add_argument('--batch',   type=int,   default=16)
    p.add_argument('--conf',    type=float, default=0.55,
                   help='Start-Confidence (steigt über Runden)')
    p.add_argument('--fps',     type=float, default=0.5,
                   help='Start-Sampling-Rate aus Videos')
    args = p.parse_args()

    if Path(args.videos).is_dir():
        vids = sorted(Path(args.videos).glob('*.mp4')) + \
               sorted(Path(args.videos).glob('*.webm')) + \
               sorted(Path(args.videos).glob('*.mkv'))
    else:
        vids = [Path(p) for p in glob.glob(args.videos)]

    if not vids:
        print(f"Keine Videos gefunden: {args.videos}")
        raise SystemExit(1)

    run_iterative(
        vids,
        rounds=args.rounds,
        epochs=args.epochs,
        initial_conf=args.conf,
        initial_fps=args.fps,
        batch=args.batch,
    )
