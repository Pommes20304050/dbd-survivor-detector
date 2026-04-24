"""
train.py — Fine-tuning von YOLOv8 auf gelabelten DBD Screenshots
Starten: python src/train.py
"""

import argparse
import random
import re
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

LABELED_IMG   = Path('data/labeled/images')
LABELED_LBL   = Path('data/labeled/labels')
DATASET_ROOT  = Path('data/dataset')      # YOLO-Split Struktur
MODEL_OUT     = Path('models')
BASE_MODEL    = 'yolov8l.pt'   # Large: massiv bessere Distanz-Erkennung


_BOOST_RE = re.compile(r'^boosted_\d+_(.+)$')

def _group_key(img_path):
    """Gruppiert Boost-Duplikate mit Original damit sie nicht zwischen train/val leaken."""
    m = _BOOST_RE.match(img_path.stem)
    return m.group(1) if m else img_path.stem


def split_dataset(val_ratio: float = 0.15, seed: int = 42):
    """Erstellt train/val Split im YOLO-Format — Boost-Duplikate bleiben zusammen."""
    images = sorted(list(LABELED_IMG.glob('*.jpg')) + list(LABELED_IMG.glob('*.png')))

    # Nur Bilder mit zugehörigem Label
    images = [img for img in images if (LABELED_LBL / (img.stem + '.txt')).exists()]

    if len(images) < 10:
        raise RuntimeError(f"Zu wenige gelabelte Bilder: {len(images)}. "
                           "Mindestens 10 nötig, besser 50-100.")

    # Gruppieren nach Original-Stem, damit Boost-Kopien in denselben Split gehen
    groups = {}
    for img in images:
        groups.setdefault(_group_key(img), []).append(img)

    group_keys = list(groups.keys())
    random.seed(seed)
    random.shuffle(group_keys)

    n_val_groups = max(1, int(len(group_keys) * val_ratio))
    val_keys = set(group_keys[:n_val_groups])
    val_set = {img.name for k in val_keys for img in groups[k]}

    # Alte Struktur entfernen
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)

    for split in ('train', 'val'):
        (DATASET_ROOT / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATASET_ROOT / 'labels' / split).mkdir(parents=True, exist_ok=True)

    for img in images:
        split = 'val' if img.name in val_set else 'train'
        shutil.copy(img, DATASET_ROOT / 'images' / split / img.name)
        lbl = LABELED_LBL / (img.stem + '.txt')
        shutil.copy(lbl, DATASET_ROOT / 'labels' / split / lbl.name)

    # data.yaml erzeugen
    yaml_path = DATASET_ROOT / 'data.yaml'
    yaml_path.write_text(
        f"path: {DATASET_ROOT.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['survivor']\n"
    )

    n_val = len([img for img in images if img.name in val_set])
    print(f"Split: {len(images)-n_val} train  |  {n_val} val  "
          f"(grupppiert über {len(group_keys)} Originale)")
    return yaml_path


def train(epochs: int = 50, imgsz: int = 1280, batch: int = 8):
    yaml_path = split_dataset()

    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project='runs',
        name='survivor',
        exist_ok=True,
        patience=15,
        pretrained=True,
        optimizer='AdamW',
        lr0=1e-3,
        cos_lr=True,
        augment=True,
        hsv_h=0.025, hsv_s=0.8, hsv_v=0.5,
        fliplr=0.5,
        scale=0.6,
        degrees=5.0,
        translate=0.15,
        shear=2.0,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
    )

    # Bestes Modell nach models/ kopieren
    MODEL_OUT.mkdir(exist_ok=True)
    candidates = [
        Path('runs/survivor/weights/best.pt'),
        Path('runs/detect/survivor/weights/best.pt'),
        Path('runs/detect/runs/survivor/weights/best.pt'),
    ]
    best = next((p for p in candidates if p.exists()), None)
    if best:
        shutil.copy(best, MODEL_OUT / 'best.pt')
        print(f"\nFertig! Modell gespeichert: {MODEL_OUT / 'best.pt'}")
    else:
        print("Warnung: Kein best.pt gefunden")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--imgsz',  type=int, default=1280)
    p.add_argument('--batch',  type=int, default=8)
    args = p.parse_args()
    train(args.epochs, args.imgsz, args.batch)
