# DBD Survivor Detector

YOLOv8-basiertes Echtzeit-Erkennungssystem für Survivor-Charaktere in Dead by Daylight.

## Projekt-Status

| Modell | mAP@50 | Recall | Precision | mAP@50-95 |
|--------|--------|--------|-----------|-----------|
| v1     | 97.0%  | 86.6%  | 95.7%     | 68.7%     |
| v2     | 97.9%  | 96.3%  | 94.1%     | 72.0%     |
| **v3** | **96.8%** | **95.7%** | **95.8%** | **81.4%** |

v3 ist das finale Produktions-Modell mit 2210 Trainings-Bildern.

## Setup

```bash
pip install -r requirements.txt
```

## Workflow

### 1. Screenshots sammeln
```bash
python src/live_auto_label.py      # Live während DBD spielt
python src/capture.py --video X    # Aus Video extrahieren
```

### 2. Labeln
```bash
python src/label_tool.py           # Manuelles Labeling
python src/gallery.py              # Galerie zum Überprüfen
```

### 3. Trainieren
```bash
python src/boost.py                # Oversampling harter Fälle
python src/train.py --epochs 200   # Training mit Early Stopping
```

### 4. Nutzen
```bash
python src/monitor.py --screen     # Live-Monitor
python src/detector.py             # Einfache Detektion
python src/test_models.py          # Modelle vergleichen
```

## Struktur

```
src/
├── live_auto_label.py   # Live Screenshot + Auto-Label mit F-Hotkey
├── label_tool.py        # OpenCV-basiertes Label-Tool
├── gallery.py           # Bild-Browser mit Lösch-Funktion
├── capture.py           # Screenshots aus Video/Screen
├── auto_label.py        # Batch Auto-Labeling für Videos
├── train.py             # YOLOv8 Training
├── boost.py             # Oversampling seltener Fälle
├── monitor.py           # Live-Erkennung (Screen/Video)
├── detector.py          # Einfacher Detektor
├── test_models.py       # Side-by-Side Modellvergleich
├── relabel.py           # Labels mit aktuellem Modell neu erstellen
└── backup.py            # Versions-Backup System

data/                    # Training-Daten (gitignored)
models/                  # Trainierte Modelle (gitignored)
runs/                    # Training-Runs (gitignored)
```

## Technische Details

- **Modell:** YOLOv8-Large (43M Parameter)
- **Input:** 1280x1280 (konfigurierbar)
- **Augmentation:** HSV 0.8, Scale 0.6, Copy-Paste 0.3, Mosaic 1.0
- **HUD-Filter:** Ignoriert Killer-Hand, Survivor-Portraits, Perk-Icons
- **Self-Training Loop:** Iterative Verbesserung mit Force-Captures
- **Oversampling:** 3x Boost für manuell gelabelte schwere Fälle

## Hardware-Anforderungen

- GPU: NVIDIA mit CUDA-Support (12 GB VRAM empfohlen)
- Python 3.10+
- Windows 10/11 oder Linux

## Performance (RTX 4070 Super)

| Config              | FPS  |
|---------------------|------|
| 1280px + TTA        | 20   |
| 1280px ohne TTA     | 43   |
| 960px               | 75   |
| 640px               | 110  |

## License

**All Rights Reserved** — Copyright © 2026 Pommes20304050

Proprietary and confidential. See [LICENSE](LICENSE) for details.
No copying, distribution, or modification permitted.
