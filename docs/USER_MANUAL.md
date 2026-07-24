# NeuraRoads ADAS - User Manual

This manual covers day-to-day use: preparing data, training, running inference
and reading the annotated output.

---

## 1. The workflow at a glance

```
prepare_dataset  ->  (optional) augment rare classes  ->  train_yolo
     ->  validate  ->  (optional) export to ONNX/TensorRT  ->  inference_video
```

Before the model is trained you can still run the pipeline with `--no-detector`
to preview lanes, BEV, ego speed and the HUD.

---

## 2. Preparing the dataset

```powershell
python src/scripts/prepare_dataset.py            # audit + class distribution
python src/scripts/prepare_dataset.py --write-yaml   # also (re)write data.yaml
```

Balance under-represented classes (e.g. Train, Bike) before training:

```powershell
python src/preprocessing/data_augmentation.py `
  --images data/datasets/images/train --labels data/datasets/labels/train `
  --out-images data/datasets/images/train_aug --out-labels data/datasets/labels/train_aug `
  --classes 8,3,5,0 --multiplier 4
```

Extract frames from raw videos for extra annotation:

```powershell
python src/preprocessing/frame_extractor.py --source clip.mp4 --fps 2
```

---

## 3. Training (from scratch)

```powershell
python src/training/train_yolo.py                       # full schedule (150 epochs)
python src/training/train_yolo.py --epochs 50 --batch 8 # faster / low VRAM
python src/training/train_yolo.py --device cpu          # CPU (slow)
python src/training/train_yolo.py --resume              # resume last run
```

- Weights: `models/trained/neuraroads_yolov8m/weights/best.pt`.
- On the 6 GB GPU, if you hit CUDA OOM, lower `--batch` (8 → 4).
- Tune everything in `src/training/hyperparameters.yaml`.

Validate and see per-class AP:

```powershell
python src/training/validate.py --split val
```

---

## 4. Exporting for speed (optional)

```powershell
python src/training/export_model.py --format onnx --half     # portable GPU
python src/scripts/convert_to_tensorrt.py --half             # TensorRT FP16
```

The detector auto-selects the fastest available backend (`engine > onnx > pt`).

---

## 5. Running inference

### Video file
```powershell
python src/inference/inference_video.py --source clip.mp4
python src/inference/inference_video.py --source clip.mp4 --no-preview --output out.mp4
```

### Webcam (live)
```powershell
python src/inference/inference_webcam.py --camera 0 --record
```

### Batch (a folder of videos)
```powershell
python src/inference/batch_inference.py --input data/raw/videos/test_videos
```

Controls in the preview window: **`q`** or **`Esc`** to stop.

Common flags: `--no-detector` (run before training), `--jetson` (Nano overlay),
`--max-frames N` (limit), `--no-preview` (headless).

---

## 6. Reading the annotated output

| Region | Shows |
|--------|-------|
| **Top bar** | FPS · `NEURAROADS ADAS` · ego speed (km/h) · clock |
| **Bounding boxes** | colour = danger (green→yellow→orange→red); label = name, ID, distance, speed, TTC |
| **Lanes** | blue lines + shaded drivable area; **red** when leaving lane |
| **Top-right** | bird's-eye-view map with range rings and object dots |
| **Center-top** | the single active ADAS icon + message (fades / pulses) |
| **Bottom bar** | closest-3 object bars + active-warnings list |

### ADAS warnings

| Warning | Trigger | Icon |
|---------|---------|------|
| Collision — warning | vehicle < 10 m or TTC < 3 s | `fcws-warning.png` (red, fast pulse, red tint) |
| Collision — caution | 10–20 m or TTC 3–5 s | `FCWS-prompt.png` (yellow, slow pulse) |
| Road clear | safe lead vehicle | `FCWS-normal.png` (green) |
| Pedestrian | ped within 15 m (red < 6 m) | `warn.png` |
| Lane departure | drifting out of lane | `LTA-left_lanes.png` / `LTA-right_lanes.png` |
| Direction | curve / roundabout / straight | `left_turn` / `right_turn` / `straight.png` |

Priority (only one main icon at a time): **Collision > Pedestrian > Lane Departure > Direction**.

---

## 7. Performance

```powershell
python src/scripts/benchmark.py --source clip.mp4 --frames 300
```

If FPS is low, the pipeline auto-reduces lane/BEV cadence (adaptive quality).
Tune targets in `model_config.yaml → pipeline`.

---

## 8. Tuning behaviour

- **Warning distances / TTC / messages / icons** → `configs/adas_thresholds.yaml`
- **Real object heights (distance accuracy)** → `model_config.yaml → distance_estimator.real_heights_m`
- **Lane robustness** → `model_config.yaml → lane_detector`
- **Camera geometry (distance/BEV/speed)** → `configs/camera_calibration.yaml` (see [CALIBRATION_GUIDE](CALIBRATION_GUIDE.md))

---

## 9. Troubleshooting

| Symptom | Fix |
|---------|-----|
| "No detector weights found" | Train first, or add `--no-detector` to preview. |
| CUDA out of memory during training | Lower `--batch` (8 → 4). |
| Distances look off | Calibrate the camera (HFOV, height, horizon). |
| Lane departure false alarms on curves | Handled by curve suppression; adjust `lane_departure.curve_curvature_min_m`. |
| No preview window | You may have installed headless OpenCV — see the install guide's reconcile step. |
