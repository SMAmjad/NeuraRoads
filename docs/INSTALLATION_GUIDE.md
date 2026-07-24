# NeuraRoads ADAS - Installation Guide

## Reference environment

The system was built and verified on:

| Component | Value |
|-----------|-------|
| OS | Windows 10 Pro x64 |
| CPU | AMD Ryzen 5 5600 |
| GPU | MSI GTX 1660 SUPER 6 GB (Turing, FP16) |
| RAM | 16 GB DDR4 3200 |
| Python | 3.12.10 |
| CUDA toolkit | 12.1 |
| PyTorch | 2.5.1+cu121 |

The repository already contains a working `venv/` created for this machine.
If you are on that machine, you do not need to reinstall anything — just activate.

---

## Windows (the delivery machine)

```powershell
# From the project root
venv\Scripts\Activate.ps1

# Sanity check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# -> 2.5.1+cu121 True
```

In VSCode the interpreter is pre-selected (`.vscode/settings.json`) and the venv
auto-activates in every new terminal. If VSCode shows "package not installed"
hints, reload the window or run **Python: Select Interpreter → ./venv**.

---

## Recreating the environment from scratch

If you ever need to rebuild (new machine, corrupted venv):

```powershell
# 1) Create a fresh venv (Python 3.12)
python -m venv venv
venv\Scripts\Activate.ps1

# 2) Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# 3) Install CUDA 12.1 PyTorch FIRST (so pip doesn't grab CPU wheels)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 `
    --index-url https://download.pytorch.org/whl/cu121

# 4) Install the rest
pip install -r requirements.txt

# 5) Reconcile OpenCV (albumentations pulls the headless build, which lacks
#    cv2.imshow used by the preview window)
pip uninstall -y opencv-python opencv-python-headless
pip install --no-cache-dir opencv-python==4.10.0.84
```

An exact-version lock is provided in `requirements.lock.txt`.

### Linux / WSL

```bash
bash setup.sh            # CUDA 12.1 GPU build
CUDA=cpu bash setup.sh   # CPU-only
```

---

## GPU / CUDA notes

- The GTX 1660 SUPER has **no INT8 Tensor Cores**, so on the desktop use **FP16**
  (TensorRT `--half`). INT8 is reserved for the Jetson Nano.
- Device selection is automatic (`device.mode: auto` in `model_config.yaml`).
  Force with `--device cpu` on training/validation scripts if needed.

---

## Optional: TensorRT

ONNX Runtime (GPU) is installed and gives a reliable speed-up out of the box.
For maximum speed, install the NVIDIA `tensorrt` wheel matching your CUDA, then:

```powershell
python src/scripts/convert_to_tensorrt.py --half
```

## Optional: UFLDv2 deep lane weights

The object detector is trained from scratch, so nothing is downloaded for it.
The deep lane branch is optional; without weights the robust classical detector
is used automatically. To enable the deep branch, place a UFLDv2 CULane ResNet-34
checkpoint at `models/weights/ufldv2_culane_res34.pth` (see
`src/scripts/download_pretrained.py`).

---

## Verifying the install

```powershell
python -m pytest src/tests -q                     # 24 tests should pass (1 skips w/o weights)
python src/scripts/prepare_dataset.py             # dataset audit
python src/inference/inference_video.py --source data/raw/videos/dashcam_samples/lane.mp4 --no-detector --max-frames 60 --no-preview
```

See [INSTALLATION_GUIDE](INSTALLATION_GUIDE.md) issues? Check
[CALIBRATION_GUIDE](CALIBRATION_GUIDE.md) and [USER_MANUAL](USER_MANUAL.md).
