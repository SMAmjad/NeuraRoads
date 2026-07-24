# NeuraRoads ADAS - Project Documentation

Full technical documentation of the architecture, module responsibilities, data
contracts and the design decisions behind each subsystem.

---

## 1. System overview

NeuraRoads is a monocular, real-time ADAS. A dashcam frame flows through a
perception stack (detection → tracking → distance → speed → lane → BEV) and an
ADAS decision layer (collision, lane departure, pedestrian, direction) into a HUD
renderer. The whole thing is config-driven and device-agnostic.

```
                       ┌─────────────────────────── NeuraRoadsPipeline ───────────────────────────┐
 frame ─► fit/resize ─►│ Detector ─► Tracker ─► Distance ─► Speed(ego+obj) ─► Lane ─► BEV          │─► annotated frame
                       │                    │                                   │                   │
                       │             CollisionDetector ─► WarningSystem ─► AlertManager ─► Visualizer│
                       └───────────────────────────────────────────────────────────────────────────┘
```

Heavy stages (lane, BEV) run on a configurable cadence and are reused between
frames; an adaptive-quality controller lowers cadence if FPS drops.

---

## 2. Design decisions (client-approved)

| Area | Choice | Why |
|------|--------|-----|
| Detector | **YOLOv8m from scratch** | Highest accuracy of the realtime options; client accepted 30–40 FPS. Trained with no COCO weights per requirement. |
| Tracker | **ByteTrack** | Near-zero overhead (no appearance net), strong ID stability in traffic — best fit for the FPS budget. |
| Distance | **Pinhole + known heights** | Robust monocular metric distance; blended with a flat-road ground model and per-track smoothing. |
| Speed | **Distance-derivative + Kalman** | Smooth per-object range rate → absolute speed; ego speed from road optical flow. No extra sensor. |
| Lane | **Hybrid (UFLDv2 + classical + smoothing)** | Deep net for curves/night when weights present; robust classical fallback (colour + edges + road-edge) always available; temporal Kalman prevents flicker/false warnings. |
| BEV | **Metric projection from calibration** | Object placement from distance + lateral `(x-cx)·d/fx`; more robust than a fragile homography. |

---

## 3. Module reference

### utils/
- **logger.py** — loguru sinks (console + rotating file). No `print` anywhere.
- **config_loader.py** — YAML loading, deep-merge overlays (Jetson), path
  resolution, `resolve_device` auto-detect. `ConfigLoader` convenience object.
- **calibration.py** — `CameraCalibration`: intrinsics (auto-rescaled to frame
  size), distortion, pinhole distance, flat-road ground distance, horizon.
- **video_processor.py** — threaded `VideoReader`/`VideoWriter`, `letterbox`, `FPSMeter`.
- **metrics.py** — `PerformanceTracker`/`StageTimer`, box geometry (IoU, centres),
  `system_stats`.
- **visualization.py** — `Visualizer`: boxes, labels, lanes, BEV composite,
  top/bottom bars, ADAS icon (fade+pulse), screen tint.

### core/
- **detector.py** — `Detector` (Ultralytics YOLO), backend auto-select
  (engine>onnx>pt), returns `Detection` with readable names.
- **tracker.py** — `ObjectTracker` (ByteTrack/BoT-SORT) → `TrackedObject`
  (the central per-object record enriched downstream).
- **distance_estimator.py** — `DistanceEstimator`: pinhole + ground blend +
  per-track EMA.
- **speed_estimator.py** — `SpeedEstimator` (per-track `_KalmanCV1D`) +
  `EgoSpeedEstimator` (sparse optical flow of the road).
- **lane_detector.py** — `ClassicalLaneDetector`, `DeepLaneDetector` (UFLDv2,
  auto-disables without weights), `LaneSmoother`, `LaneDetector` (hybrid).
- **bev_transformer.py** — `BEVTransformer`: metric top-down mini-map + optional
  homography warp.

### adas/
- **collision_detector.py** — pure kinematics: fills `ttc_s`/`color_key`,
  finds forward + pedestrian threats (`CollisionAssessment`).
- **lane_departure.py** — `LaneDepartureWarning`: hysteresis + debounce + curve
  suppression → `LaneDepartureState`.
- **warning_system.py** — `WarningSystem`: builds prioritised `WarningEvent`s
  (icons/messages/colours from config).
- **alert_manager.py** — `AlertManager`: min-display time, fade, pulse; optional audio.

### pipeline/
- **inference_pipeline.py** — `NeuraRoadsPipeline.process_frame` (the engine).
- **realtime_pipeline.py** — `RealtimePipeline.run` (threaded I/O, preview,
  adaptive quality, output writing).

### training / inference / preprocessing / scripts / deployment / tests / notebooks
See the [USER_MANUAL](USER_MANUAL.md) and [API_REFERENCE](API_REFERENCE.md).

---

## 4. Data contracts

**Detection** — `box[x1,y1,x2,y2]`, `confidence`, `class_id`, `class_name`.

**TrackedObject** — `track_id`, `box`, `class_id`, `class_name`, `confidence`,
and the enriched fields `distance_m`, `speed_kmh`, `ttc_s`, `color_key`, plus
`extra` (`range_rate_mps`, `closing_speed_kmh`).

**LaneResult** — `lines[]`, `fill`, `left_fit`, `right_fit`, `ego_offset`
(−1 left..+1 right), `curvature_m`, `direction`, `source`, `confidence`.

**Pipeline state (to the HUD)** — `objects[]`, `lane`, `bev`, `alert`,
`closest[]`, `warnings[]`, `ego_speed_kmh`, `fps`.

**WarningEvent** — `kind`, `level`, `message`, `icon`, `color_key`, `priority`,
`pulse`, `is_alert`, `screen_tint`.

---

## 5. Configuration surface

| File | Governs |
|------|---------|
| `model_config.yaml` | detector, tracker, distance, speed, lane, BEV, pipeline, video, visualization |
| `adas_thresholds.yaml` | colour bands, FCWS/pedestrian/lane/direction thresholds, icon+message map, priorities, icon behaviour |
| `camera_calibration.yaml` | intrinsics, distortion, mounting, homography, speed scale |
| `jetson_config.yaml` | Nano overlay (deep-merged over `model_config`) |

Class ids are grouped semantically under `detector.groups` (vehicles,
vulnerable, pedestrians, collision_relevant, …) so no class id is hardcoded in code.

---

## 6. Performance strategy (60 FPS target)

1. **FP16 / TensorRT** detector inference.
2. **Threaded** frame decode + encode (I/O off the critical path).
3. **Cadence control** — lane/BEV every N frames, interpolated between.
4. **Adaptive quality** — raise cadence automatically when FPS < floor.
5. **Processing resolution cap** — configurable `process_width/height`.

Observed on the GTX 1660 SUPER (classical lanes, detection off, real footage):
26–49 FPS depending on resolution; the detector adds the main GPU cost once trained.

---

## 7. Jetson Nano

The Nano runs the **same code** with `jetson_config.yaml` merged on top:
YOLOv8n + INT8 TensorRT, `imgsz 416`, classical lanes, reduced cadence,
`target_fps 20`. Deployment scripts set MAXN power, lock clocks, add swap and
build the INT8 engine.

---

## 8. Safety notes

All safety-critical numbers (distance, speed, TTC) are estimates from a single
camera and degrade under rain/night/occlusion. The system is an assistance aid,
not an autonomous controller. Warnings use debounce + hysteresis + curve
suppression to avoid nuisance alerts, and temporal smoothing to avoid flicker.
