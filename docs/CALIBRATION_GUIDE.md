# NeuraRoads ADAS - Camera Calibration Guide

Distance, speed and the bird's-eye view all depend on the camera model in
`configs/camera_calibration.yaml`. Good calibration is the single biggest lever
on distance/speed accuracy — the client's key judging criteria.

---

## 1. What the model contains

| Field | Meaning | Affects |
|-------|---------|---------|
| `calibrated_resolution` | resolution the numbers were measured at (auto-rescaled) | everything |
| `intrinsics.fx/fy` | focal length (px) | pinhole distance |
| `intrinsics.cx/cy` | principal point (px) | lateral offset / BEV |
| `intrinsics.horizontal_fov_deg` | lens HFOV | focal length fallback |
| `distortion.*` | lens distortion | rectification (optional) |
| `mounting.height_m` | camera height above road | ground-plane distance, BEV |
| `mounting.pitch_deg` | downward tilt | ground-plane distance |
| `mounting.horizon_y_frac` | image row of the horizon | ground projection cutoff |
| `speed_calibration.flow_to_kmh_scale` | optical-flow → km/h | ego speed |

The defaults assume a ~70° HFOV 1080p dashcam mounted ~1.25 m high, looking level.

---

## 2. Quick calibration from lens spec (FOV mode)

If you only know the lens field of view and mounting, this is enough for good
results:

```powershell
python src/scripts/calibrate_camera.py fov `
  --hfov 70 --width 1920 --height 1080 `
  --cam-height 1.25 --pitch 0 --horizon 0.52
```

This computes `fx = (width/2) / tan(HFOV/2)` and writes the config.

**Finding the horizon fraction:** open a frame, note the pixel row where the road
meets the sky, divide by image height. Straight, level roads → ~0.5.

---

## 3. Full calibration from a chessboard (most accurate)

1. Print a chessboard (e.g. 9×6 inner corners), measure a square (e.g. 25 mm).
2. Take 15–30 photos of it with the **same camera/lens**, from varied angles.
3. Put them in a folder and run:

```powershell
python src/scripts/calibrate_camera.py chessboard `
  --images calib_photos/ --cols 9 --rows 6 --square 0.025 `
  --cam-height 1.25 --horizon 0.52
```

The RMS reprojection error is printed — aim for **< 1.0 px**. The tool writes
`fx, fy, cx, cy` and distortion coefficients.

---

## 4. Validating distance accuracy

1. Park a car a **known distance** ahead (e.g. 20 m) and capture a frame.
2. Run inference (`--no-detector` off) and read the box distance, or use the
   calibration notebook `src/notebooks/04_calibration_demo.ipynb`.
3. If the reading is consistently off by a factor, adjust `fx/fy` (distance scales
   linearly with focal length) or verify `real_heights_m` for that class in
   `model_config.yaml`.

Rule of thumb: `distance = fy × real_height_m / bbox_height_px`.

---

## 5. Tuning ego speed

Ego speed comes from optical flow of the road, scaled by `flow_to_kmh_scale`.
To calibrate:

1. Record a clip at a **known constant speed** (e.g. from GPS/speedometer).
2. Run inference and compare the displayed ego speed.
3. Scale `speed_calibration.flow_to_kmh_scale` by `(known / displayed)`.
   (Setting it here overrides the value in `model_config.yaml`.)

---

## 6. Bird's-eye view

The BEV places objects using calibration (`fx`, `cx`) plus each object's
distance, so once distance is accurate the BEV is too. The metric window is set
in `model_config.yaml → bev.range_m` (`forward`, `lateral`). For an explicit
ground homography, set `homography.computed: true` and provide the 3×3 matrix.

---

## 7. Common pitfalls

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| All distances too small/large | wrong `fx/fy` (FOV guess) | chessboard calibrate, or adjust HFOV |
| Distances drift near/far incorrectly | wrong `mounting.height_m` / `pitch` | measure and set them |
| Objects above road counted | wrong `horizon_y_frac` | set to the true horizon row |
| Ego speed way off | wrong `flow_to_kmh_scale` | calibrate against a known-speed clip |
| Different video resolution | none — intrinsics auto-rescale to the frame | ensure `calibrated_resolution` is correct |
