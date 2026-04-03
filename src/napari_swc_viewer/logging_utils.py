"""Runtime logging helpers for napari_swc_viewer."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_DEBUG_ENV_VAR = "NAPARI_SWC_VIEWER_DEBUG"
_LOG_FILE_ENV_VAR = "NAPARI_SWC_VIEWER_LOG_FILE"
_LOGGER_NAME = "napari_swc_viewer"
_DEFAULT_LOG_FILE = Path.home() / ".napari-swc-viewer" / "debug.log"
_LOG_FORMAT = (
    "%(asctime)s %(process)d %(threadName)s "
    "%(levelname)s %(name)s: %(message)s"
)


def _env_enabled(value: str | None) -> bool:
    """Return whether an environment variable value should enable debug mode."""
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def debug_logging_enabled() -> bool:
    """Return whether debug logging should be enabled for the plugin."""
    return _env_enabled(os.environ.get(_DEBUG_ENV_VAR))


def resolve_log_path() -> Path:
    """Return the configured log file path."""
    override = os.environ.get(_LOG_FILE_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return _DEFAULT_LOG_FILE


def configure_debug_logging() -> Path | None:
    """Install opt-in debug logging for the plugin and its child loggers.

    Returns the configured log-file path when debug logging is enabled,
    otherwise ``None``.
    """
    if not debug_logging_enabled():
        return None

    logger = logging.getLogger(_LOGGER_NAME)
    if getattr(logger, "_napari_swc_viewer_debug_configured", False):
        existing_path = getattr(logger, "_napari_swc_viewer_log_path", None)
        return Path(existing_path) if existing_path is not None else resolve_log_path()

    log_path = resolve_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.set_name("napari_swc_viewer_debug_file")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    stream_handler.set_name("napari_swc_viewer_debug_stream")

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger._napari_swc_viewer_debug_configured = True
    logger._napari_swc_viewer_log_path = str(log_path)
    logger.debug("Configured debug logging at %s", log_path)
    return log_path
