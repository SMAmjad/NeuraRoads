"""YOLOv8 object detector wrapper (Ultralytics backend).

Loads the from-scratch-trained NeuraRoads model and runs detection, returning a
list of :class:`Detection` objects with human-readable class names (never raw
indices). Automatically selects the fastest available backend file
(TensorRT ``.engine`` > ONNX ``.onnx`` > PyTorch ``.pt``) and honours the
device / half-precision settings resolved from config.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils.config_loader import resolve_device, resolve_path
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Detection:
    """A single detection in the original frame's pixel coordinates.

    Attributes:
        box: ``[x1, y1, x2, y2]`` in pixels.
        confidence: Detection confidence in ``[0, 1]``.
        class_id: Integer class index (0-9 for NeuraRoads).
        class_name: Human-readable class name (e.g. ``"Car"``).
    """

    box: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def xyxy(self) -> np.ndarray:
        """The box as an ``[x1, y1, x2, y2]`` float array."""
        return self.box

    @property
    def width(self) -> float:
        return float(self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])

    @property
    def center(self) -> tuple[float, float]:
        return (float(self.box[0] + self.box[2]) / 2.0, float(self.box[1] + self.box[3]) / 2.0)


class Detector:
    """Ultralytics YOLOv8 detector configured from ``model_config.detector``."""

    def __init__(self, cfg: Dict[str, Any], device: Optional[str] = None) -> None:
        """Build the detector.

        Args:
            cfg: The full ``model_config`` dict (needs ``detector`` and ``device``).
            device: Explicit torch device; if None it is auto-resolved.

        Raises:
            FileNotFoundError: If no usable weights/engine/onnx file exists.
            RuntimeError: If the Ultralytics model fails to load.
        """
        self.det_cfg: Dict[str, Any] = cfg.get("detector", {})
        self.device = device or resolve_device(cfg.get("device"))
        self.half = bool(cfg.get("device", {}).get("half_precision", True)) and "cuda" in self.device

        self.imgsz = int(self.det_cfg.get("imgsz", 640))
        self.conf = float(self.det_cfg.get("conf_threshold", 0.25))
        self.iou = float(self.det_cfg.get("iou_threshold", 0.45))
        self.max_det = int(self.det_cfg.get("max_detections", 300))
        self.agnostic = bool(self.det_cfg.get("agnostic_nms", False))
        self.keep_classes: Optional[Sequence[int]] = self.det_cfg.get("keep_classes")

        # Config class names are authoritative for readable labels.
        raw_names = self.det_cfg.get("class_names", {})
        self.class_names: Dict[int, str] = {int(k): str(v) for k, v in raw_names.items()}

        self.weights_path = self._select_backend_file()
        self.backend = self.weights_path.suffix.lower().lstrip(".")
        # Prefer a lean direct-onnxruntime path for .onnx (strips the ultralytics
        # per-call overhead => the max-FPS inference path; also Jetson-friendly).
        self.use_ort = False
        self._ort_sess = None
        self._model = None
        if self.backend == "onnx" and bool(self.det_cfg.get("ort_direct", True)):
            self._try_setup_ort()
        if not self.use_ort:
            self._model = self._load_model()
        log.info("Detector ready: backend={} ort={} device={} half={} imgsz={} ({} classes).",
                 self.backend, self.use_ort, self.device, self.half, self.imgsz, len(self.class_names))

    def _try_setup_ort(self) -> None:
        """Set up a direct onnxruntime session for the ONNX weights (fast path)."""
        try:
            import onnxruntime as ort

            avail = ort.get_available_providers()
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if "cuda" in self.device and "CUDAExecutionProvider" in avail
                         else ["CPUExecutionProvider"])
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._ort_sess = ort.InferenceSession(str(self.weights_path), sess_options=so,
                                                  providers=providers)
            self._ort_in = self._ort_sess.get_inputs()[0].name
            self._ort_fp16 = "float16" in str(self._ort_sess.get_inputs()[0].type)
            self.use_ort = True
            log.info("Detector direct-ORT active ({}).", self._ort_sess.get_providers()[0])
        except Exception as exc:
            log.warning("Direct ORT setup failed ({}); using ultralytics backend.", exc)
            self.use_ort = False
            self._ort_sess = None

    # -- backend / model ----------------------------------------------------
    def _select_backend_file(self) -> Path:
        """Pick the fastest existing model file per the ``backend`` preference."""
        pref = str(self.det_cfg.get("backend", "auto")).lower()
        engine = resolve_path(self.det_cfg.get("tensorrt_engine", "")) if self.det_cfg.get("tensorrt_engine") else None
        onnx = resolve_path(self.det_cfg.get("onnx_model", "")) if self.det_cfg.get("onnx_model") else None
        pt = resolve_path(self.det_cfg.get("weights", "")) if self.det_cfg.get("weights") else None

        order = {
            "auto": [engine, onnx, pt],
            "tensorrt": [engine, pt, onnx],
            "onnx": [onnx, pt, engine],
            "pytorch": [pt, onnx, engine],
        }.get(pref, [engine, onnx, pt])

        for cand in order:
            if cand is not None and cand.is_file():
                return cand
        # Nothing found -> actionable error.
        tried = [str(p) for p in (pt, onnx, engine) if p is not None]
        raise FileNotFoundError(
            "No detector weights found. Train the model first with "
            "`python src/training/train_yolo.py` (writes best.pt), or export an "
            f"engine/onnx. Looked for: {tried}"
        )

    def _load_model(self):
        """Instantiate the Ultralytics model, falling back to CPU on GPU errors."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("ultralytics is not installed") from exc
        try:
            model = YOLO(str(self.weights_path), task="detect")
            # Sync readable names from the model if config omitted any.
            if hasattr(model, "names") and model.names:
                for k, v in model.names.items():
                    self.class_names.setdefault(int(k), str(v))
            return model
        except Exception as exc:
            raise RuntimeError(f"Failed to load detector model {self.weights_path}: {exc}") from exc

    def name_of(self, class_id: int) -> str:
        """Return the readable class name for ``class_id`` (fallback to id)."""
        return self.class_names.get(int(class_id), f"cls{int(class_id)}")

    # -- inference ----------------------------------------------------------
    def warmup(self, imgsz: Optional[int] = None) -> None:
        """Run a dummy forward pass so the first real frame isn't slow."""
        size = imgsz or self.imgsz
        dummy = np.zeros((size, size, 3), dtype=np.uint8)
        try:
            self.detect(dummy)
            log.debug("Detector warmup complete.")
        except Exception as exc:
            log.warning("Detector warmup failed: {}", exc)

    # -- direct onnxruntime path (lean, no ultralytics overhead) ------------
    @staticmethod
    def _letterbox(img: np.ndarray, new: int) -> Tuple[np.ndarray, float, int, int]:
        """Aspect-preserving resize + pad to a square, matching YOLOv8 preprocess."""
        h, w = img.shape[:2]
        r = min(new / h, new / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dw, dh = new - nw, new - nh
        top, left = dh // 2, dw // 2
        out = cv2.copyMakeBorder(resized, top, dh - top, left, dw - left,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return out, r, left, top

    def _ort_detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect via the direct ORT session: letterbox -> infer -> decode + NMS."""
        img, r, left, top = self._letterbox(frame, self.imgsz)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        if self._ort_fp16:
            x = x.astype(np.float16)
        out = self._ort_sess.run(None, {self._ort_in: x})[0]  # (1, 4+nc, N)
        p = np.squeeze(out, 0).T.astype(np.float32)            # (N, 4+nc)
        if p.shape[1] < 5:
            return []
        boxes, scores = p[:, :4], p[:, 4:]
        cls = scores.argmax(1)
        conf = scores.max(1)
        # Drop any non-finite predictions defensively (FP16 on some GPUs can emit
        # NaN); keeps a bad value from ever reaching integer pixel conversions.
        keep = (conf >= self.conf) & np.isfinite(conf) & np.isfinite(boxes).all(axis=1)
        if self.keep_classes is not None:
            keep &= np.isin(cls, list(self.keep_classes))
        boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
        if len(boxes) == 0:
            return []
        # xywh (letterbox space) -> xyxy in original-frame pixels.
        xy, wh = boxes[:, :2], boxes[:, 2:4]
        xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)
        xyxy[:, [0, 2]] -= left
        xyxy[:, [1, 3]] -= top
        xyxy /= r
        H, W = frame.shape[:2]
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W - 1)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H - 1)
        # Class-aware NMS via a per-class coordinate offset (vectorised numpy;
        # avoids the slow cv2.dnn.NMSBoxes Python-list conversion).
        off = cls.astype(np.float32) * (max(H, W) + 1.0)
        nms_boxes = xyxy + off[:, None]
        idxs = self._nms(nms_boxes, conf, self.iou)[: self.max_det]
        return [Detection(box=xyxy[i].astype(np.float32), confidence=float(conf[i]),
                          class_id=int(cls[i]), class_name=self.name_of(int(cls[i])))
                for i in idxs]

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
        """Greedy IoU NMS in pure numpy. ``boxes`` are ``[x1,y1,x2,y2]``."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[iou <= iou_thr]
        return keep

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame.

        Args:
            frame: BGR image ``(H, W, 3)``.

        Returns:
            List of :class:`Detection` in the frame's pixel coordinates. Returns
            an empty list on inference failure (never raises during streaming).
        """
        if self.use_ort:
            try:
                return self._ort_detect(frame)
            except Exception as exc:
                log.error("Detection (ORT) failed: {}", exc)
                return []
        try:
            results = self._model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                agnostic_nms=self.agnostic,
                classes=list(self.keep_classes) if self.keep_classes else None,
                half=self.half,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            log.error("Detection failed: {}", exc)
            return []

        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        out: List[Detection] = []
        for i in range(len(xyxy)):
            cid = int(cls[i])
            out.append(Detection(
                box=xyxy[i].astype(np.float32),
                confidence=float(conf[i]),
                class_id=cid,
                class_name=self.name_of(cid),
            ))
        return out

    def __call__(self, frame: np.ndarray) -> List[Detection]:
        return self.detect(frame)
