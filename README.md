# DBD Survivor Detector

Real-time YOLOv8-based detection system for Survivor characters in Dead by Daylight.
Transparent overlay + real-time dashboard with GPU/CPU monitoring, TensorRT acceleration, and live configuration.

![Dashboard](docs/dashboard.png)

> ## ⚠️ Project Status: Work in Progress
>
> **This project still needs significantly more training samples to reach production-level accuracy.**
>
> The current v3 model was trained on 2210 images from a limited set of matches. While it already achieves 96.8% mAP@50 on its test split, real-world performance varies based on:
>
> - **Maps not seen in training** — detection quality drops on unfamiliar environments
> - **Unusual skins / cosmetics** — rare outfits may not be recognized
> - **Extreme distances** — survivors far away are still inconsistently detected
> - **Heavy occlusion** — survivors mostly hidden behind objects need more samples
> - **Dark maps / night variants** — fewer low-light training samples exist
>
> **To improve the model, we need more data:**
> - 5,000-10,000+ diverse training images
> - Coverage of all 40+ DBD maps
> - Every survivor with multiple outfit variants
> - Edge cases: injured, crouched, locker-hidden, pallet-stunned, hooked
>
> Contributions welcome via Force-Capture (F-hotkey) during gameplay. Collected samples can be labeled via `label_tool.py` and fed back into training.

## Features

- **YOLOv8-Large Fine-tuned Model** — 96.8% mAP@50 on custom training data (2210 images)
- **TensorRT FP16 Engine** — 2-3x faster than PyTorch (84 FPS @ 1280px)
- **Transparent Overlay** on DBD with click-through
- **Live Dashboard** with sidebar navigation, FPS/GPU/VRAM/Temp chart
- **5 Quick Modes** (Maximum / Standard / Competitive / Minimal / Stream)
- **6 Performance Profiles** from 1920px+TTA down to 480px Extreme
- **HUD-Filter** ignores killer hand, survivor portraits, perk icons
- **Aura-Filter** distinguishes red killer aura from survivors wearing red
- **Monitor Selection** via dropdown (dxcam + mss fallback)
- **Global F-Hotkey** for force-capture during gameplay
- **Self-Training Loop** with auto-labeling + boost mechanism

### Preview

![Preview 1](docs/preview1.png)
![Preview 2](docs/preview2.png)

## Performance (RTX 4070 Super)

| Config             | FPS  | VRAM  |
|--------------------|------|-------|
| Ultra (TRT+TTA)    | 60   | 1.5GB |
| High (960px+TTA)   | 130  | 1GB   |
| Balanced (768px)   | 180  | 700MB |
| Fast (640px)       | 250  | 500MB |
| MAX GPU (1920+TTA) | 12   | 4GB   |

## Requirements

### Hardware
- **GPU:** NVIDIA with CUDA 12.x, min 6GB VRAM (12GB+ recommended)
- **RAM:** 16GB+
- **CPU:** Modern x64 processor (i5 / Ryzen 5 8th gen+)
- **OS:** Windows 10/11 (Linux works, overlay Windows-only)

### Python
- Python 3.10, 3.11 or 3.12
- NVIDIA driver 535+ with CUDA Toolkit 12.x

### Python Packages

See `requirements.txt`:

```
torch>=2.0.0              # PyTorch with CUDA support
torchvision>=0.15.0
ultralytics>=8.0.0        # YOLOv8
opencv-python>=4.8.0
mss>=9.0.0                # Screen capture (fallback)
dxcam>=0.3.0              # DirectX screen capture (faster)
numpy>=1.24.0
pandas>=2.0.0
pynput>=1.7.6             # Global hotkeys
PyQt5>=5.15.0             # Transparent overlay
flask>=2.0.0              # Web dashboard server
flask-cors>=6.0.0
tqdm>=4.60.0
pynvml>=13.0.0            # NVIDIA GPU monitoring
psutil>=5.9.0             # CPU monitoring
```

### Optional (for maximum performance)

```
tensorrt-cu12             # 2-3x faster inference
onnx>=1.21.0
onnxslim
onnxruntime-gpu           # Alternative to TensorRT
customtkinter>=5.2.0      # For ui.py (training GUI)
```

### Installation

```bash
# PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# All other dependencies
pip install -r requirements.txt

# Optional: TensorRT
pip install --extra-index-url https://pypi.nvidia.com tensorrt-cu12 onnx onnxslim
```

## Quick Start

```bash
# Start everything
start_v3.bat          # or: python overlay/overlay_server.py

# Stop everything
stop.bat
```

Browser opens automatically at `http://localhost:8765` with the dashboard.

## Project Structure

```
dbd-survivor-detector/
├── overlay/
│   ├── overlay_server.py     # Main system (Qt overlay + Flask + YOLO)
│   ├── templates/index.html  # Dashboard HTML
│   └── static/
│       ├── style.css         # Cyberpunk orange/black design
│       └── app.js            # Live updates, chart, API
├── src/
│   ├── live_auto_label.py    # Live screenshot + auto-label (F hotkey)
│   ├── label_tool.py         # OpenCV manual labeling
│   ├── gallery.py            # Image browser with delete
│   ├── capture.py            # Screenshots from video/screen
│   ├── auto_label.py         # Batch video labeling
│   ├── train.py              # YOLOv8 training
│   ├── boost.py              # Oversampling rare samples
│   ├── monitor.py            # Simple live monitor
│   ├── detector.py           # Standalone detector
│   ├── test_models.py        # Side-by-side model comparison
│   ├── relabel.py            # Relabel with current model
│   └── backup.py             # Versioning
├── models/                   # Trained models (gitignored)
├── data/                     # Training data (gitignored)
├── start_v3.bat              # Launcher
├── stop.bat                  # Shutdown script
└── DESIGN_PROMPT.md          # Prompt for AI design tools
```

## Training Workflow

1. **Collect screenshots** via `live_auto_label.py` (F hotkey during gameplay)
2. **Label** with `label_tool.py`
3. **Oversample hard cases** with `boost.py`
4. **Train** with `train.py`
5. **Test** with `test_models.py`

## Models

| Version | Images | mAP@50 | Recall | Precision | mAP@50-95 |
|---------|--------|--------|--------|-----------|-----------|
| v1      | 241    | 97.0%  | 86.6%  | 95.7%     | 68.7%     |
| v2      | 1011   | 97.9%  | 96.3%  | 94.1%     | 72.0%     |
| **v3**  | **2210** | **96.8%** | **95.7%** | **95.8%** | **81.4%** |

v3 is the final production model — significantly better box precision (mAP@50-95).

## Dashboard Features

- **Performance Monitor** — Switchable chart (FPS / GPU / VRAM / Temp)
- **Detection Quality Donut** — Confidence distribution
- **6 Mini-Stats** — FPS, GPU load, VRAM, Temp, Power, Session
- **Live Neural Feed** — MJPEG stream with AI boxes
- **Monitor Dropdown** — Switch between connected displays
- **AI Freedom Slider** — Live confidence adjustment (5-100%)
- **Quick Modes** — One-click setup
- **Advanced Settings** — Box thickness, color, glow, filters

## License

**All Rights Reserved** — Copyright © 2026 Pommes20304050

Proprietary and confidential. See [LICENSE](LICENSE) for details.
No copying, distribution, or modification permitted.
