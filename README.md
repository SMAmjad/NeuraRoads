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
```

---

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
```

---

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
```

---

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
Software Engineer (AI/ML) at Cubix, Karachi  
[LinkedIn](https://www.linkedin.com/in/shkmamjad) · [GitHub](https://github.com/SMAmjad) · [Email](mailto:shkmamjad@gmail.com)
