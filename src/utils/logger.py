"""Project-wide logging built on `loguru`.

Every NeuraRoads module logs through :func:`get_logger`. No module should ever
use ``print``. The first call configures a shared sink set (colored console +
rotating file); subsequent calls are cheap and return the same bound logger.

Typical use::

    from utils.logger import get_logger

    log = get_logger(__name__)
    log.info("Detector ready on {}", device)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

from loguru import logger

# Module-level guard so we configure sinks exactly once per process.
_CONFIGURED: bool = False

# Default location for rotating log files (created lazily).
_DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "src" / "results" / "metrics" / "performance_logs"

_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[name]}</cyan> - "
    "<level>{message}</level>"
)
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{extra[name]}:{function}:{line} - {message}"
)


def configure_logging(
    level: str = "INFO",
    log_dir: Optional[Union[str, Path]] = None,
    log_file: str = "neuraroads.log",
    to_file: bool = True,
    rotation: str = "20 MB",
    retention: str = "10 days",
) -> None:
    """Configure the global loguru sinks. Safe to call multiple times.

    Args:
        level: Minimum level for the console sink (``DEBUG``/``INFO``/...).
        log_dir: Directory for the rotating log file. Defaults to the project
            performance-logs directory.
        log_file: File name for the rotating log file.
        to_file: Whether to also write logs to a file sink.
        rotation: loguru rotation policy (size or time).
        retention: loguru retention policy.
    """
    global _CONFIGURED

    logger.remove()  # drop loguru's default stderr handler

    # Bind a default ``name`` so the format string never KeyErrors when a raw
    # ``logger`` (unbound) is used somewhere.
    logger.configure(extra={"name": "neuraroads"})

    logger.add(
        sys.stderr,
        level=level.upper(),
        format=_CONSOLE_FORMAT,
        colorize=True,
        backtrace=False,
        diagnose=False,
        enqueue=True,  # thread/process safe (pipeline uses worker threads)
    )

    if to_file:
        target_dir = Path(log_dir) if log_dir is not None else _DEFAULT_LOG_DIR
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            logger.add(
                target_dir / log_file,
                level="DEBUG",
                format=_FILE_FORMAT,
                rotation=rotation,
                retention=retention,
                encoding="utf-8",
                enqueue=True,
                backtrace=True,
                diagnose=False,
            )
        except OSError as exc:  # never let logging setup crash the app
            logger.warning("Could not create file log sink at {}: {}", target_dir, exc)

    _CONFIGURED = True


def get_logger(name: str = "neuraroads", level: str = "INFO"):
    """Return a loguru logger bound to ``name``.

    Configures the global sinks on first use. Pass ``__name__`` from the calling
    module for readable, source-attributed logs.

    Args:
        name: Logical name shown in each log line (usually ``__name__``).
        level: Console level to use if logging is not yet configured.

    Returns:
        A loguru logger with ``extra['name']`` bound to ``name``.
    """
    if not _CONFIGURED:
        configure_logging(level=level)
    return logger.bind(name=name)
