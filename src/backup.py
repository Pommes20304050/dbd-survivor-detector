"""
backup.py — Erstellt automatisch ein Backup der nächsten Version
Findet die nächste freie Versionsnummer und kopiert:
  - alle gelabelten Bilder + Labels
  - das aktuelle Modell (best.pt)
  - die Trainings-Metriken

python src/backup.py                # Auto-Nummer (v3, v4, ...)
python src/backup.py --version v3   # Feste Version
"""

import argparse
import shutil
import time
from pathlib import Path

DATA_DIR        = Path('data')
MODELS_DIR      = Path('models')
LABELED_IMG     = DATA_DIR / 'labeled' / 'images'
LABELED_LBL     = DATA_DIR / 'labeled' / 'labels'
ACTIVE_MODEL    = MODELS_DIR / 'best.pt'
RUN_DIR         = Path('runs/detect/runs/survivor')


def next_version() -> str:
    """Findet nächste freie Versionsnummer (v1, v2, v3, ...)."""
    existing = [d.name for d in DATA_DIR.iterdir()
                if d.is_dir() and d.name.startswith('backup_v')]
    nums = []
    for name in existing:
        try:
            nums.append(int(name.replace('backup_v', '')))
        except ValueError:
            pass
    next_num = max(nums, default=0) + 1
    return f'v{next_num}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--version', default=None,
                   help='Feste Version (z.B. v3). Sonst automatisch.')
    args = p.parse_args()

    version = args.version or next_version()
    backup_dir = DATA_DIR / f'backup_{version}'

    if backup_dir.exists():
        print(f"[Backup] Fehler: {backup_dir} existiert bereits!")
        return

    print(f"[Backup] Erstelle: {backup_dir}")
    backup_dir.mkdir(parents=True)

    # 1. Bilder kopieren
    n_imgs = 0
    if LABELED_IMG.exists():
        target_img = backup_dir / 'images'
        shutil.copytree(LABELED_IMG, target_img)
        n_imgs = len(list(target_img.glob('*.jpg')))
        print(f"  Bilder:   {n_imgs}")

    # 2. Labels kopieren
    n_lbls = 0
    if LABELED_LBL.exists():
        target_lbl = backup_dir / 'labels'
        shutil.copytree(LABELED_LBL, target_lbl)
        n_lbls = len(list(target_lbl.glob('*.txt')))
        print(f"  Labels:   {n_lbls}")

    # 3. Modell kopieren (sowohl als best_vX.pt als auch best.pt im Backup)
    if ACTIVE_MODEL.exists():
        shutil.copy(ACTIVE_MODEL, backup_dir / f'best_{version}.pt')
        print(f"  Modell:   best_{version}.pt")
        # Zusätzlich in models/ als versionierte Kopie
        versioned_model = MODELS_DIR / f'best_{version}.pt'
        if not versioned_model.exists():
            shutil.copy(ACTIVE_MODEL, versioned_model)
            print(f"            + models/best_{version}.pt (für einfaches Zurück-Wechseln)")

    # 4. Trainingsrun kopieren
    n_run = 0
    if RUN_DIR.exists():
        target_run = backup_dir / 'training_run'
        shutil.copytree(RUN_DIR, target_run)
        n_run = len(list(target_run.iterdir()))
        print(f"  Training: {n_run} Dateien")

    # 5. Info-Datei
    info = backup_dir / 'info.txt'
    info.write_text(
        f"Backup: {version}\n"
        f"Datum: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Bilder: {n_imgs}\n"
        f"Labels: {n_lbls}\n"
        f"Trainings-Files: {n_run}\n"
    )

    # Gesamt-Größe
    def dir_size_mb(path: Path) -> float:
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / 1024 / 1024

    size = dir_size_mb(backup_dir)
    print(f"\n[Backup] Fertig!  {backup_dir}  ({size:.0f} MB)")
    print(f"\nVerfügbare Backups:")
    for bkp in sorted(DATA_DIR.glob('backup_v*')):
        if bkp.is_dir():
            info_file = bkp / 'info.txt'
            if info_file.exists():
                first_line = info_file.read_text().splitlines()[0]
                print(f"  {bkp.name:15s}  {first_line}")


if __name__ == '__main__':
    main()
