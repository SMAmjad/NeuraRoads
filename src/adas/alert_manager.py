"""Temporal display manager for the single center-top ADAS icon.

Turns the stream of per-frame "primary" :class:`~adas.warning_system.WarningEvent`
decisions into a smooth on-screen presentation:

* **Anti-flicker** - an icon stays for at least ``min_display_time_s`` unless a
  higher-priority event escalates (e.g. prompt -> collision warning).
* **Fade in/out** - alpha ramps over ``fade_in_s`` / ``fade_out_s``.
* **Pulse** - danger icons gently zoom at a configured frequency.

The output is a plain ``alert`` dict consumed by
:meth:`utils.visualization.Visualizer.draw_alert`. Optional audio cues are
supported (disabled by default) via ``winsound`` on Windows.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from adas.warning_system import WarningEvent
from utils.config_loader import resolve_path
from utils.logger import get_logger

log = get_logger(__name__)


class AlertManager:
    """Manages icon display timing, fade and pulse for the primary warning."""

    def __init__(self, adas_cfg: Dict[str, Any]) -> None:
        """Args: adas_cfg: full ``adas_thresholds`` config."""
        beh = adas_cfg.get("icon_behaviour", {})
        self.min_display = float(beh.get("min_display_time_s", 0.8))
        self.fade_in = float(beh.get("fade_in_s", 0.2))
        self.fade_out = float(beh.get("fade_out_s", 0.3))
        self.pulse_slow_hz = float(beh.get("pulse_slow_hz", 1.5))
        self.pulse_fast_hz = float(beh.get("pulse_fast_hz", 4.0))
        self.pulse_amp = 0.12
        self.max_active = int(beh.get("max_active_warnings", 4))

        self._current: Optional[WarningEvent] = None
        self._shown_since: float = 0.0

        # Audio (optional, best-effort, Windows winsound).
        audio = adas_cfg.get("audio", {})
        self.audio_enabled = bool(audio.get("enabled", False))
        self.audio_cooldown = float(audio.get("cooldown_s", 2.0))
        self._audio_files = {
            "collision": audio.get("collision_wav"),
            "pedestrian": audio.get("pedestrian_wav"),
            "lane_departure": audio.get("lane_wav"),
        }
        self._last_audio: Dict[str, float] = {}

    # -- switching logic ----------------------------------------------------
    def _should_switch(self, new: Optional[WarningEvent], now: float) -> bool:
        """Decide whether to replace the current icon with ``new``."""
        if new is None:
            return False
        if self._current is None:
            return True
        if new.kind == self._current.kind and new.level == self._current.level:
            return False  # same thing, keep timing
        # Escalation to higher priority wins immediately (safety first).
        if new.priority > self._current.priority:
            return True
        # Otherwise respect the minimum display time (anti-flicker).
        return (now - self._shown_since) >= self.min_display

    def _pulse_scale(self, event: WarningEvent, now: float) -> float:
        """Compute the pulse zoom factor for the current event."""
        if event.pulse == "fast":
            hz = self.pulse_fast_hz
        elif event.pulse == "slow":
            hz = self.pulse_slow_hz
        else:
            return 1.0
        return 1.0 + self.pulse_amp * (0.5 + 0.5 * math.sin(2 * math.pi * hz * now))

    def _fade_alpha(self, now: float) -> float:
        """Compute the fade-in alpha for the current event."""
        elapsed = now - self._shown_since
        if elapsed >= self.fade_in:
            return 1.0
        return max(0.0, min(1.0, elapsed / max(self.fade_in, 1e-3)))

    # -- main update --------------------------------------------------------
    def update(self, primary: Optional[WarningEvent],
               now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Update display state and return the ``alert`` dict for rendering.

        Args:
            primary: The highest-priority event this frame (may be None).
            now: Monotonic timestamp (defaults to ``time.perf_counter()``).

        Returns:
            An ``alert`` dict for the visualizer, or None if nothing to show.
        """
        now = time.perf_counter() if now is None else now
        if self._should_switch(primary, now):
            self._current = primary
            self._shown_since = now
            if primary is not None and primary.is_alert:
                self._maybe_play_audio(primary, now)

        if self._current is None:
            return None

        alpha = self._fade_alpha(now)
        pulse = self._pulse_scale(self._current, now)
        alert = {
            "icon": self._current.icon,
            "message": self._current.message,
            "color_key": self._current.color_key,
            "alpha": alpha,
            "pulse_scale": pulse,
            "kind": self._current.kind,
            "level": self._current.level,
        }
        if self._current.screen_tint is not None:
            alert["screen_tint"] = self._current.screen_tint
            alert["screen_tint_alpha"] = self._current.screen_tint_alpha
        return alert

    # -- audio (optional) ---------------------------------------------------
    def _maybe_play_audio(self, event: WarningEvent, now: float) -> None:
        """Play an alert sound if enabled, present and past the cooldown."""
        if not self.audio_enabled:
            return
        wav = self._audio_files.get(event.kind)
        if not wav:
            return
        last = self._last_audio.get(event.kind, 0.0)
        if now - last < self.audio_cooldown:
            return
        path = resolve_path(wav)
        if not path.is_file():
            return
        try:
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            self._last_audio[event.kind] = now
        except Exception as exc:  # never break rendering for audio
            log.debug("Audio playback failed: {}", exc)

    def reset(self) -> None:
        """Clear display state (e.g. between videos)."""
        self._current = None
        self._shown_since = 0.0
        self._last_audio.clear()
