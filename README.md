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
```

---

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
```

---

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
```

---

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
