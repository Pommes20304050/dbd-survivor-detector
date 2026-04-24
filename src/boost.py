"""
boost.py — Oversampling der schweren Fälle (Force-Captures)
Dupliziert 'forced_*' Bilder mehrfach ins Training-Dataset
damit YOLO sie häufiger während des Trainings sieht.

python src/boost.py --factor 3   # 3x Duplizierung
"""

import argparse
import shutil
from pathlib import Path

IMG_DIR = Path('data/labeled/images')
LBL_DIR = Path('data/labeled/labels')
BOOST_PREFIX = 'boosted_'


def clear_old_boosts():
    """Alte Boost-Dateien entfernen — alle Extensions."""
    removed = 0
    for pattern in (f'{BOOST_PREFIX}*.jpg', f'{BOOST_PREFIX}*.png', f'{BOOST_PREFIX}*.jpeg'):
        for img in IMG_DIR.glob(pattern):
            img.unlink()
            removed += 1
    for lbl in LBL_DIR.glob(f'{BOOST_PREFIX}*.txt'):
        lbl.unlink()
    if removed:
        print(f"[Boost] {removed} alte Boost-Kopien entfernt")


def boost_rare_samples(factor: int = 3):
    clear_old_boosts()

    # Finde alle schweren Fälle (Force-Captures = forced_*)
    rare_images = sorted(list(IMG_DIR.glob('forced_*.jpg')) + list(IMG_DIR.glob('forced_*.png')))
    if not rare_images:
        print("[Boost] Keine forced_* Bilder gefunden!")
        return

    print(f"[Boost] {len(rare_images)} Force-Captures gefunden")
    print(f"[Boost] Dupliziere jeweils {factor-1}x (Faktor {factor})")

    created = 0
    for img_path in rare_images:
        lbl_path = LBL_DIR / (img_path.stem + '.txt')
        if not lbl_path.exists():
            continue

        # 2..factor-mal kopieren (originale bleibt)
        for copy_idx in range(1, factor):
            new_stem = f"{BOOST_PREFIX}{copy_idx}_{img_path.stem}"
            new_img = IMG_DIR / (new_stem + '.jpg')
            new_lbl = LBL_DIR / (new_stem + '.txt')
            shutil.copy(img_path, new_img)
            shutil.copy(lbl_path, new_lbl)
            created += 1

    total_after = len(list(IMG_DIR.glob('*.jpg')))
    print(f"\n[Boost] {created} neue Kopien erstellt")
    print(f"[Boost] Datensatz jetzt: {total_after} Bilder")
    print(f"[Boost] Schwere Fälle sind nun {factor}x im Training")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--factor', type=int, default=3,
                   help='Wie oft schwere Fälle dupliziert werden (default: 3)')
    args = p.parse_args()
    boost_rare_samples(args.factor)
