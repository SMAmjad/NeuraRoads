"""Video I/O helpers: threaded reader, threaded writer and letterbox utilities.

Decoding and encoding are moved onto background threads so the GPU pipeline is
not stalled by disk / codec latency - important for the 60 FPS target. Also
provides :func:`letterbox` for aspect-preserving resize to the model input size.
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


def letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int] = (720, 1280),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize ``image`` to ``new_shape`` preserving aspect ratio with padding.

    Args:
        image: Input BGR image ``(H, W, 3)``.
        new_shape: Target ``(height, width)``.
        color: Padding colour (BGR).

    Returns:
        Tuple ``(padded_image, ratio, (pad_w, pad_h))`` where ``ratio`` is the
        scale applied and ``pad_w/pad_h`` are the one-sided paddings in pixels.
        These let callers map coordinates back to the original image.
    """
    h0, w0 = image.shape[:2]
    new_h, new_w = new_shape
    ratio = min(new_h / h0, new_w / w0)
    unpad_w, unpad_h = int(round(w0 * ratio)), int(round(h0 * ratio))
    resized = cv2.resize(image, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (new_w - unpad_w) / 2.0
    pad_h = (new_h - unpad_h) / 2.0
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return out, ratio, (pad_w, pad_h)


class VideoReader:
    """Threaded frame reader for a video file or webcam index.

    Frames are decoded on a background thread into a bounded queue so the main
    processing loop never blocks on I/O. Use as an iterator or call :meth:`read`.
    """

    def __init__(
        self,
        source: Union[str, Path, int],
        queue_size: int = 8,
        threaded: bool = True,
    ) -> None:
        """Open a video source.

        Args:
            source: Path to a video file, or an integer webcam index.
            queue_size: Max buffered frames when threaded.
            threaded: Decode on a background thread when True.

        Raises:
            IOError: If the source cannot be opened.
        """
        self.source = source if isinstance(source, int) else str(source)
        self.threaded = threaded
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise IOError(f"Could not open video source: {source!r}")

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.is_webcam = isinstance(source, int)

        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=queue_size)
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if self.threaded:
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()

        log.info(
            "Opened {} ({}x{} @ {:.1f} FPS, {} frames).",
            "webcam" if self.is_webcam else Path(str(source)).name,
            self.width, self.height, self.fps,
            self.frame_count if self.frame_count > 0 else "stream",
        )

    def _reader_loop(self) -> None:
        """Background decode loop (threaded mode)."""
        while not self._stopped.is_set():
            ok, frame = self._cap.read()
            if not ok:
                self._queue.put(None)  # sentinel: end of stream
                break
            # Block briefly if consumer is behind; drop nothing for files.
            self._queue.put(frame)
        self._stopped.set()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return ``(ok, frame)``; ``ok`` is False at end of stream."""
        if self.threaded:
            if self._stopped.is_set() and self._queue.empty():
                return False, None
            try:
                frame = self._queue.get(timeout=5.0)
            except queue.Empty:
                return False, None
            if frame is None:
                return False, None
            return True, frame
        ok, frame = self._cap.read()
        return ok, (frame if ok else None)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate frames until the stream ends."""
        while True:
            ok, frame = self.read()
            if not ok:
                break
            yield frame

    def release(self) -> None:
        """Stop the reader thread and release the capture."""
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self._cap.release()
        except Exception:  # pragma: no cover
            pass

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class VideoWriter:
    """Threaded video writer that encodes frames off the main loop."""

    def __init__(
        self,
        output_path: Union[str, Path],
        fps: float,
        frame_size: Tuple[int, int],
        codec: str = "mp4v",
        threaded: bool = True,
        queue_size: int = 16,
    ) -> None:
        """Open a video writer.

        Args:
            output_path: Destination file path (parent dirs are created).
            fps: Output frames per second.
            frame_size: ``(width, height)`` of frames that will be written.
            codec: FourCC codec string (e.g. ``mp4v``, ``avc1``, ``XVID``).
            threaded: Encode on a background thread when True.
            queue_size: Max buffered frames when threaded.
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.threaded = threaded

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(self.output_path), fourcc, float(fps), self.frame_size)
        if not self._writer.isOpened():
            # Fallback to a widely-available codec.
            log.warning("Codec '{}' failed; falling back to 'mp4v'.", codec)
            self._writer = cv2.VideoWriter(
                str(self.output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                float(fps), self.frame_size,
            )
        if not self._writer.isOpened():
            raise IOError(f"Could not open VideoWriter for {self.output_path}")

        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=queue_size)
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if self.threaded:
            self._thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._thread.start()
        log.info("Writing output to {} ({}x{} @ {:.0f} FPS, codec={}).",
                 self.output_path.name, self.frame_size[0], self.frame_size[1], fps, codec)

    def _writer_loop(self) -> None:
        """Background encode loop (threaded mode)."""
        while True:
            frame = self._queue.get()
            if frame is None:
                break
            self._writer.write(frame)

    def write(self, frame: np.ndarray) -> None:
        """Queue (threaded) or write (sync) a single BGR frame.

        Frames whose size differs from ``frame_size`` are resized to fit.
        """
        if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
            frame = cv2.resize(frame, self.frame_size)
        if self.threaded:
            self._queue.put(frame)
        else:
            self._writer.write(frame)

    def release(self) -> None:
        """Flush pending frames and close the writer."""
        if self.threaded and self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=5.0)
        try:
            self._writer.release()
        except Exception:  # pragma: no cover
            pass

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class FPSMeter:
    """Lightweight rolling FPS meter (exponential moving average)."""

    def __init__(self, alpha: float = 0.1) -> None:
        """Args: alpha: EMA smoothing factor (higher = more responsive)."""
        self.alpha = alpha
        self._last: Optional[float] = None
        self.fps: float = 0.0

    def tick(self) -> float:
        """Record a frame boundary and return the current smoothed FPS."""
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0 else (1 - self.alpha) * self.fps + self.alpha * inst
        self._last = now
        return self.fps
