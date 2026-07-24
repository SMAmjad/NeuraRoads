# NeuraRoads ADAS - API Reference

Two APIs are documented here: the **Python API** (use the pipeline in your own
code) and the **HTTP API** (the FastAPI inference service).

---

## Python API

All modules live under `src/`; add it to `sys.path` (entry scripts do this
automatically) or run from within `src`.

### The pipeline (recommended entry point)

```python
from pipeline.inference_pipeline import NeuraRoadsPipeline
import cv2

pipe = NeuraRoadsPipeline(frame_rate=30)          # allow_no_detector=True to skip detection
frame = cv2.imread("frame.jpg")
annotated, state = pipe.process_frame(frame, dt=1/30)

# state = {objects[], lane, bev, alert, closest[], warnings[], ego_speed_kmh, fps}
cv2.imwrite("out.jpg", annotated)
```

```python
from pipeline.realtime_pipeline import RealtimePipeline
rt = RealtimePipeline()                            # overlay="jetson_config" for the Nano
summary = rt.run("clip.mp4", output_path="out.mp4", show_preview=False)
```

### Individual modules

```python
from utils.config_loader import load_config, resolve_device
from utils.calibration import CameraCalibration
from core.detector import Detector
from core.tracker import ObjectTracker
from core.distance_estimator import DistanceEstimator
from core.speed_estimator import SpeedEstimator
from core.lane_detector import LaneDetector
from core.bev_transformer import BEVTransformer

cfg = load_config("model_config")
device = resolve_device(cfg["device"])
calib = CameraCalibration.from_config(frame_size=(1280, 720))

det = Detector(cfg, device)                        # needs trained weights
tracks = ObjectTracker(cfg["tracker"], frame_rate=30).update(det.detect(frame), frame)
DistanceEstimator(cfg["distance_estimator"], calib).estimate(tracks)
SpeedEstimator(cfg["speed_estimator"]).estimate(tracks, dt=1/30)
lane = LaneDetector(cfg, device).detect(frame)
```

### Key signatures

| Call | Returns |
|------|---------|
| `Detector(cfg, device).detect(frame)` | `list[Detection]` |
| `ObjectTracker(cfg, fps).update(dets, frame)` | `list[TrackedObject]` |
| `DistanceEstimator(cfg, calib).estimate(objs)` | same list, `distance_m` filled |
| `SpeedEstimator(cfg).update_ego(frame, dt)` | `float` ego km/h |
| `SpeedEstimator(cfg).estimate(objs, dt)` | same list, `speed_kmh` filled |
| `LaneDetector(cfg, device).detect(frame)` | `LaneResult` |
| `BEVTransformer(cfg, calib, colors).render(objs)` | BGR canvas |
| `CollisionDetector(mcfg, acfg).assess(objs, size)` | `CollisionAssessment` |
| `WarningSystem(acfg).evaluate(assess, lane_state, lane)` | `(events, primary)` |
| `AlertManager(acfg).update(primary)` | `alert` dict |
| `Visualizer(vcfg, bcfg).render(frame, state)` | annotated frame |

---

## HTTP API (FastAPI)

Start the service (from `src/`, or via Docker):

```powershell
uvicorn deployment.api.app:app --host 0.0.0.0 --port 8000
# before training:  NR_ALLOW_NO_DETECTOR=1 uvicorn deployment.api.app:app
```

Interactive docs: `http://localhost:8000/docs`.

### Endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET | `/health` | — | `{status, device, detector_loaded, lane_mode}` |
| GET | `/config` | — | active class map + key settings |
| POST | `/infer/image` | multipart `file` (image) | annotated PNG |
| POST | `/infer/video` | multipart `file` (video) | annotated MP4 |

### Examples

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/infer/image \
     -F "file=@frame.jpg" --output annotated.png

curl -X POST http://localhost:8000/infer/video \
     -F "file=@clip.mp4" --output annotated.mp4
```

```python
import requests
r = requests.post("http://localhost:8000/infer/image",
                  files={"file": open("frame.jpg", "rb")})
open("annotated.png", "wb").write(r.content)
```

### Environment variables

| Var | Meaning |
|-----|---------|
| `NR_ALLOW_NO_DETECTOR` | `"1"` → run lane/ego/HUD only (no trained model needed) |
| `NR_OVERLAY` | config overlay name, e.g. `jetson_config` |

> Video inference is synchronous — fine for short clips. For long videos use the
> CLI (`inference_video.py`) or add a background-job queue.
