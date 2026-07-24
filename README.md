<<<<<<< HEAD
# NeuraRoads ADAS

**Real-time Advanced Driver Assistance System.** Takes any dashcam driving video
and produces a fully annotated output showing object detection, tracking,
per-object distance & speed, time-to-collision, robust lane detection, a
bird's-eye-view map, and prioritised ADAS warnings — all at once.

> Built for a Hong Kong client. Detector trained **from scratch** on a 10-class
> dataset (no pretrained COCO weights). Runs on a GTX 1660 SUPER and degrades
> cleanly to a Jetson Nano.

---

## Features

| # | Capability | How |
|---|------------|-----|
| 1 | **Object detection** (10 classes) | YOLOv8m, trained from scratch |
| 2 | **Object tracking** (stable IDs) | ByteTrack |
| 3 | **Distance estimation** (m) | Monocular pinhole + known real heights, per-track smoothing |
| 4 | **Speed estimation** (km/h) | Distance-derivative + per-track Kalman; ego speed via optical flow |
| 5 | **Lane detection** | Hybrid: UFLDv2 deep (optional) + robust classical fallback + temporal smoothing |
| 6 | **Bird's-eye view** | Metric top-down mini-map from calibration |
| 7 | **ADAS warnings** | Collision / pedestrian / lane-departure / direction with icons, fade & pulse |
| + | **TTC** on every box, colour-coded danger, adaptive-quality FPS control | |

### The 10 classes
`0 Bicycle · 1 Bus · 2 Car · 3 Bike · 4 Pedestrian · 5 Ped on Bike · 6 Traffic Light · 7 Sign Board · 8 Train · 9 Truck`

Detections always display **readable names** (e.g. `Car`, `Truck`), never numeric ids.

---

## Quick start

The project ships with a ready `venv` (Python 3.12, CUDA 12.1 PyTorch). In VSCode
it auto-activates. From a terminal:

```powershell
# Windows (repo root)
venv\Scripts\Activate.ps1
```

### 1) Verify the dataset
```powershell
python src/scripts/prepare_dataset.py
```

### 2) Train the detector from scratch
```powershell
python src/training/train_yolo.py                 # full 150-epoch schedule
python src/training/train_yolo.py --epochs 50 --batch 8   # quicker / low-VRAM
```
Output: `models/trained/neuraroads_yolov8m/weights/best.pt`.

### 3) (Optional) Export for speed
```powershell
python src/training/export_model.py --format onnx --half
python src/scripts/convert_to_tensorrt.py --half          # TensorRT FP16
```

### 4) Run inference on a video
```powershell
python src/inference/inference_video.py --source data/raw/videos/dashcam_samples/drive.mp4
```
The annotated video lands in `src/results/videos/output_videos/`.

> **Before the model is trained** you can still see lanes + BEV + HUD + ego speed:
> add `--no-detector` to any inference command.

### Webcam / batch
```powershell
python src/inference/inference_webcam.py --camera 0
python src/inference/batch_inference.py --input data/raw/videos/test_videos
=======
# NeuraRoads — Advanced Driver Assistance System

![Status](https://img.shields.io/badge/Status-In%20Development-orange)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Jetson%20Nano-lightgrey)

A production-grade, monocular vision-based Advanced Driver Assistance System designed for real-time deployment on constrained hardware. Built for a Hong Kong client, NeuraRoads integrates object detection, multi-object tracking, distance estimation, lane detection, and intelligent safety alerting into a single unified pipeline.

---

## Overview

Most ADAS solutions require expensive LiDAR sensors or high-end GPUs. NeuraRoads achieves real-time performance using a single dashcam feed, optimized specifically for hardware with limited VRAM (GTX 970, 4GB). The system is designed for deployment on NVIDIA Jetson Nano for edge use cases.

---

## Core Features

| Feature | Description | Status |
|---|---|---|
| Object Detection | YOLOv8-based real-time detection of vehicles, pedestrians, and road objects | In Development |
| Multi-Object Tracking | DeepSORT integration for persistent object tracking across frames | In Development |
| Distance Estimation | Bird's Eye View (BEV) transformation for accurate depth estimation | In Development |
| Speed Estimation | Per-object speed calculation using frame-to-frame displacement | In Development |
| Lane Detection | Deep learning-based lane boundary detection | In Development |
| ADAS Warnings | Rule-based safety alert system for collision, lane departure, and proximity | In Development |

---

## Tech Stack

- **Detection:** YOLOv8 (Ultralytics)
- **Tracking:** DeepSORT / ByteTrack
- **Vision:** OpenCV
- **Deep Learning:** PyTorch
- **Language:** Python 3.10
- **Target Hardware:** GTX 970 (4GB VRAM) → NVIDIA Jetson Nano

---

## System Architecture

```
Camera Feed (Dashcam)
        │
        ▼
  Frame Preprocessing
  (Resize, Normalize)
        │
        ▼
  YOLOv8 Detection
  (Vehicles, Pedestrians, Objects)
        │
        ▼
  DeepSORT Tracking
  (Persistent Object IDs)
        │
        ├──────────────────────┐
        ▼                      ▼
BEV Distance Estimation   Lane Detection
(Depth from Monocular)    (DL-based)
        │                      │
        └──────────┬───────────┘
                   ▼
         ADAS Warning Engine
     (Collision / Lane / Proximity)
                   │
                   ▼
         Visualization Output
>>>>>>> 984cd830bde636833813bbd6d65d61f8886ea216
```

---

<<<<<<< HEAD
## Output layout

```
Top bar    : FPS | NEURAROADS ADAS | ego speed | time
Center     : colour-coded boxes (name, ID, distance, speed, TTC)
             lanes drawn + drivable area shaded (blue normal / red leaving)
Top-right  : bird's-eye-view mini-map
Center-top : active ADAS icon + warning text (fade + pulse)
Bottom     : closest-3 bar graph + active-warnings list
```

Icon priority: **Collision > Pedestrian > Lane Departure > Direction**.

---

## Project structure

```
configs/                YAML knobs (model, ADAS thresholds, camera, jetson)
data/
  annotations/          classes.txt, data.yaml
  datasets/             YOLO train/val/test (images + labels)
  icons/                9 ADAS icons
  raw/videos/           sample dashcam clips
models/                 trained weights + exported engines
src/
  utils/                config, logging, calibration, video I/O, metrics, HUD
  core/                 detector, tracker, distance, speed, lane, bev
  adas/                 collision, lane departure, warning system, alert manager
  pipeline/             single-frame engine + real-time runner
  training/             from-scratch training, validation, export
  inference/            video / webcam / batch entry points
  preprocessing/        frame extraction, annotation tools, augmentation
  scripts/              dataset prep, calibration, benchmark, TensorRT, setup
  deployment/           FastAPI service, Docker, Jetson Nano
  tests/                pytest suite
  notebooks/            data / training / performance / calibration
docs/                   installation, user manual, API, calibration, full docs
=======
## Performance Targets

- Real-time inference at **30+ FPS** on GTX 970 (4GB VRAM)
- Object detection accuracy: **95%+**
- Deployment-ready for **NVIDIA Jetson Nano** (edge hardware)

---

## Project Structure

```
NeuraRoads/
├── README.md
├── requirements.txt
├── setup.sh
├── LICENSE
├── .gitignore
├── .gitattributes                    # Git LFS (large files)
│
├── docs/
│   ├── PROJECT_DOCUMENTATION.md
│   ├── API_REFERENCE.md
│   ├── INSTALLATION_GUIDE.md
│   ├── USER_MANUAL.md
│   ├── CALIBRATION_GUIDE.md
│   └── images/
│       ├── architecture_diagram.png
│       ├── sample_outputs/
│       └── screenshots/
│
├── configs/
│   ├── model_config.yaml
│   ├── camera_calibration.yaml
│   ├── adas_thresholds.yaml
│   └── jetson_config.yaml
│
├── data/                             # Not tracked (dataset too large)
│   ├── raw/
│   ├── processed/
│   ├── datasets/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/
│
├── models/                           # Not tracked (binary weights)
│   ├── pretrained/
│   ├── trained/
│   └── weights/
│
├── src/
│   ├── core/
│   │   ├── detector.py              # YOLOv8 detection module
│   │   ├── tracker.py               # ByteTrack/DeepSORT tracking
│   │   ├── distance_estimator.py    # Distance calculation
│   │   ├── speed_estimator.py       # Speed calculation
│   │   ├── lane_detector.py         # Lane detection module
│   │   └── bev_transformer.py       # Bird's Eye View transformation
│   │
│   ├── adas/
│   │   ├── warning_system.py        # ADAS warning logic
│   │   ├── collision_detector.py    # FCW implementation
│   │   ├── lane_departure.py        # LDW implementation
│   │   └── alert_manager.py         # Alert prioritization
│   │
│   ├── utils/
│   │   ├── video_processor.py       # Video I/O operations
│   │   ├── calibration.py           # Camera calibration
│   │   ├── visualization.py         # Drawing, overlays
│   │   ├── config_loader.py         # YAML config parser
│   │   ├── logger.py                # Custom logging
│   │   └── metrics.py               # Performance metrics
│   │
│   ├── preprocessing/
│   │   ├── frame_extractor.py
│   │   ├── data_augmentation.py
│   │   └── annotation_converter.py
│   │
│   └── pipeline/
│       ├── inference_pipeline.py    # Main inference pipeline
│       └── realtime_pipeline.py     # Jetson real-time pipeline
│
├── training/
│   ├── train_yolo.py
│   ├── validate.py
│   ├── export_model.py
│   └── hyperparameters.yaml
│
├── scripts/
│   ├── setup_environment.sh
│   ├── download_pretrained.py
│   ├── prepare_dataset.py
│   ├── calibrate_camera.py
│   ├── convert_to_tensorrt.py
│   └── benchmark.py
│
├── inference/
│   ├── inference_video.py
│   ├── inference_webcam.py
│   └── batch_inference.py
│
├── tests/
│   ├── test_detector.py
│   ├── test_tracker.py
│   ├── test_distance_estimator.py
│   ├── test_speed_estimator.py
│   ├── test_lane_detector.py
│   └── test_pipeline.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_performance_analysis.ipynb
│   └── 04_calibration_demo.ipynb
│
├── assets/
│   ├── icons/
│   ├── fonts/
│   └── sounds/
│
├── results/
│   ├── videos/
│   ├── metrics/
│   └── visualizations/
│
└── deployment/
    ├── jetson/
    │   ├── install_jetson.sh
    │   ├── optimize_jetson.sh
    │   └── jetson_inference.py
    ├── docker/
    │   ├── Dockerfile
    │   └── docker-compose.yml
    └── api/
        ├── app.py
        └── endpoints.py
>>>>>>> 984cd830bde636833813bbd6d65d61f8886ea216
```

---

<<<<<<< HEAD
## Configuration

Everything is config-driven (nothing hardcoded). Key files:

- `configs/model_config.yaml` — detector, tracker, distance, speed, lane, BEV, pipeline, HUD.
- `configs/adas_thresholds.yaml` — danger bands, TTC levels, icon/message mapping, priorities.
- `configs/camera_calibration.yaml` — intrinsics, mounting, homography (run `calibrate_camera.py`).
- `configs/jetson_config.yaml` — Jetson Nano overlay (merged over the base; no code forks).

Device (CPU/GPU) is **auto-detected**; add `--jetson` to inference to apply the Nano overlay.

---

## Performance

- Target: 60 FPS; the client accepted **30–40 FPS** for maximum accuracy (YOLOv8m).
- FPS is protected by TensorRT/FP16, threaded I/O, lane/BEV cadence control and
  adaptive-quality fallback. Benchmark with:
  ```powershell
  python src/scripts/benchmark.py --source <clip> --frames 300
  ```

---

## Testing

```powershell
python -m pytest src/tests -q
=======
## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/SMAmjad/NeuraRoads.git
cd NeuraRoads

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the system
python main.py
>>>>>>> 984cd830bde636833813bbd6d65d61f8886ea216
```

---

<<<<<<< HEAD
## Deployment

- **API:** `uvicorn deployment.api.app:app` (from `src/`), or Docker:
  `docker compose -f src/deployment/docker/docker-compose.yml up --build`.
- **Jetson Nano:** `bash src/deployment/jetson/install_jetson.sh` then
  `bash src/deployment/jetson/optimize_jetson.sh`.

See `docs/` for the full guides.

---

## Notes on the dataset

The provided set is heavily imbalanced (Car ≈ 55%, **Train ≈ 0.01%**). Balance
rare classes before training:
```powershell
python src/preprocessing/data_augmentation.py --images data/datasets/images/train `
  --labels data/datasets/labels/train --out-images aug/images --out-labels aug/labels `
  --classes 8,3,5,0 --multiplier 4
```

---

© 2026 NeuraRoads. All rights reserved. See `LICENSE`.
=======
## Dataset

The training dataset is not included in this repository due to size constraints.
Available upon request — contact via [LinkedIn](https://www.linkedin.com/in/shkmamjad) or [Email](mailto:shkmamjad@gmail.com).

---

## Deployment Target

This system is being developed on Windows (Ryzen 5 3600, GTX 970, 16GB RAM) and will be deployed to **NVIDIA Jetson Nano** for edge inference in a production dashcam environment.

---

## Client

Built for a client based in **Hong Kong** as a production dashcam intelligence system.

---

## Author

**Sheikh Muhammad Amjad**  
AI/ML Engineer 
[LinkedIn](https://www.linkedin.com/in/shkmamjad) · [GitHub](https://github.com/SMAmjad) · [Email](mailto:shkmamjad@gmail.com)
>>>>>>> 984cd830bde636833813bbd6d65d61f8886ea216
