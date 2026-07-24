"""Configuration loading, deep-merge overlays, path resolution and device auto-detect.

All YAML configs live in ``<project_root>/configs``. Modules never read YAML
directly; they call :func:`load_config` (or use :class:`ConfigLoader`) and get a
plain ``dict`` back, plus helpers to resolve project-relative paths to absolute
:class:`pathlib.Path` objects and to auto-select the compute device.

Example::

    from utils.config_loader import load_config, resolve_device

    cfg = load_config("model_config")                 # base config
    jet = load_config("model_config", overlay="jetson_config")  # deep-merged
    device = resolve_device(cfg["device"])            # 'cuda:0' or 'cpu'
"""
from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from utils.logger import get_logger

log = get_logger(__name__)

# <project_root>/src/utils/config_loader.py  ->  parents[2] == <project_root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"

# Cache raw parsed YAML by absolute path to avoid re-reading from disk.
_RAW_CACHE: Dict[str, Dict[str, Any]] = {}


def _config_path(name: str) -> Path:
    """Resolve a config *name* (with or without .yaml) to an absolute path.

    Searches ``configs/`` first, then ``data/annotations/`` (for data.yaml).

    Args:
        name: Config stem or filename, e.g. ``"model_config"`` or ``"data.yaml"``.

    Returns:
        Absolute path to the YAML file.

    Raises:
        FileNotFoundError: If no matching file exists.
    """
    stem = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    candidates = [
        CONFIGS_DIR / stem,
        PROJECT_ROOT / "data" / "annotations" / stem,
        PROJECT_ROOT / stem,
        Path(name),  # allow an explicit absolute/relative path
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"Config '{name}' not found. Looked in: {[str(c) for c in candidates]}"
    )


def _read_yaml(path: Path) -> Dict[str, Any]:
    """Read and cache a YAML file into a dict (empty dict if the file is empty)."""
    key = str(path)
    if key in _RAW_CACHE:
        return copy.deepcopy(_RAW_CACHE[key])
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        log.error("Failed to read config {}: {}", path, exc)
        raise
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} did not parse to a mapping (got {type(data)}).")
    _RAW_CACHE[key] = copy.deepcopy(data)
    return data


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base`` (overlay wins), returning a new dict.

    Nested dicts merge key-by-key; any non-dict value (including lists) is
    replaced wholesale. Neither input is mutated.

    Args:
        base: The base mapping.
        overlay: The mapping whose values take precedence.

    Returns:
        A new merged dict.
    """
    result = copy.deepcopy(base)
    for key, val in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_config(name: str, overlay: Optional[str] = None) -> Dict[str, Any]:
    """Load a config by name, optionally deep-merging an overlay config on top.

    Args:
        name: Base config name (e.g. ``"model_config"``).
        overlay: Optional overlay config name (e.g. ``"jetson_config"``) merged
            over the base. Overlay values win.

    Returns:
        The (merged) configuration as a plain dict.
    """
    base = _read_yaml(_config_path(name))
    if overlay:
        over = _read_yaml(_config_path(overlay))
        merged = deep_merge(base, over)
        log.debug("Loaded config '{}' with overlay '{}'.", name, overlay)
        return merged
    log.debug("Loaded config '{}'.", name)
    return base


def get_in(cfg: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Fetch a nested value using a dotted path, returning *default* if absent.

    Example::

        get_in(cfg, "detector.imgsz", 640)

    Args:
        cfg: The config dict.
        dotted_key: Dot-separated key path.
        default: Value returned when any segment is missing.

    Returns:
        The nested value or ``default``.
    """
    node: Any = cfg
    for part in dotted_key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node


def resolve_path(path_like: Union[str, Path]) -> Path:
    """Resolve a possibly project-relative path to an absolute :class:`Path`.

    Absolute paths are returned as-is (normalised). Relative paths are anchored
    at :data:`PROJECT_ROOT` so configs stay machine-independent.

    Args:
        path_like: A path string or Path, absolute or relative to the project.

    Returns:
        An absolute, normalised path (not required to exist).
    """
    p = Path(path_like)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def resolve_device(device_cfg: Optional[Dict[str, Any]] = None) -> str:
    """Auto-detect the compute device from a ``device`` config block.

    Honors ``mode`` (``auto``/``cuda``/``cpu``) and ``gpu_index``. In ``auto``
    mode, returns ``cuda:<idx>`` when a GPU is available, else ``cpu``. Torch is
    imported lazily so this module is usable without torch installed.

    Args:
        device_cfg: The ``device`` sub-dict of a config. Defaults to auto.

    Returns:
        A torch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    """
    device_cfg = device_cfg or {}
    mode = str(device_cfg.get("mode", "auto")).lower()
    idx = int(device_cfg.get("gpu_index", 0))

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
    except Exception:  # torch missing or broken -> CPU
        cuda_ok = False

    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        if not cuda_ok:
            log.warning("device.mode='cuda' but no GPU is available; falling back to CPU.")
            return "cpu"
        return f"cuda:{idx}"
    # auto
    resolved = f"cuda:{idx}" if cuda_ok else "cpu"
    log.info("Auto-selected device: {}", resolved)
    return resolved


def apply_torch_runtime(device_cfg: Dict[str, Any]) -> None:
    """Apply global torch runtime flags (cudnn benchmark, seed) from config.

    Safe no-op if torch is unavailable.

    Args:
        device_cfg: The ``device`` sub-dict of a config.
    """
    try:
        import torch

        torch.backends.cudnn.benchmark = bool(device_cfg.get("cudnn_benchmark", True))
    except Exception as exc:
        log.debug("Could not apply torch runtime flags: {}", exc)


class ConfigLoader:
    """Convenience object bundling a base config, its overlay and path helpers.

    Attributes:
        cfg: The merged configuration dict.
        device: The resolved torch device string.
    """

    def __init__(self, name: str = "model_config", overlay: Optional[str] = None) -> None:
        """Load ``name`` (optionally merged with ``overlay``) and resolve device."""
        self.name = name
        self.overlay = overlay
        self.cfg: Dict[str, Any] = load_config(name, overlay)
        self.device: str = resolve_device(self.cfg.get("device"))
        apply_torch_runtime(self.cfg.get("device", {}))

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Dotted-path accessor into the loaded config (see :func:`get_in`)."""
        return get_in(self.cfg, dotted_key, default)

    def path(self, dotted_key: str, default: Any = None) -> Optional[Path]:
        """Fetch a config value by dotted key and resolve it as a project path."""
        raw = self.get(dotted_key, default)
        return resolve_path(raw) if raw is not None else None

    def __getitem__(self, key: str) -> Any:
        return self.cfg[key]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        ov = f", overlay='{self.overlay}'" if self.overlay else ""
        return f"ConfigLoader(name='{self.name}'{ov}, device='{self.device}')"


# Clear caches (useful in tests / notebooks after editing YAML on disk).
def clear_cache() -> None:
    """Drop cached parsed YAML and memoised loads."""
    _RAW_CACHE.clear()
