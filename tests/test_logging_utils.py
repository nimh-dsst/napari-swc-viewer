"""Tests for opt-in plugin debug logging."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from napari_swc_viewer.logging_utils import configure_debug_logging


def _reset_plugin_logger() -> None:
    """Restore the plugin logger to an unconfigured state."""
    logger = logging.getLogger("napari_swc_viewer")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    for attr in (
        "_napari_swc_viewer_debug_configured",
        "_napari_swc_viewer_log_path",
    ):
        if hasattr(logger, attr):
            delattr(logger, attr)


@pytest.fixture(autouse=True)
def _cleanup_plugin_logger(monkeypatch):
    """Ensure debug logger state does not leak across tests."""
    _reset_plugin_logger()
    monkeypatch.delenv("NAPARI_SWC_VIEWER_DEBUG", raising=False)
    monkeypatch.delenv("NAPARI_SWC_VIEWER_LOG_FILE", raising=False)
    yield
    _reset_plugin_logger()


def test_configure_debug_logging_is_noop_when_disabled() -> None:
    """Disabled debug mode should not change plugin logger state."""
    logger = logging.getLogger("napari_swc_viewer")

    log_path = configure_debug_logging()

    assert log_path is None
    assert logger.handlers == []


def test_configure_debug_logging_adds_file_and_stream_handlers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Enabled debug mode should install both file and stream handlers."""
    log_file = tmp_path / "swc-viewer.log"
    monkeypatch.setenv("NAPARI_SWC_VIEWER_DEBUG", "1")
    monkeypatch.setenv("NAPARI_SWC_VIEWER_LOG_FILE", str(log_file))

    log_path = configure_debug_logging()
    logger = logging.getLogger("napari_swc_viewer")

    assert log_path == log_file
    assert logger.level == logging.DEBUG
    assert logger.propagate is False
    assert [handler.name for handler in logger.handlers] == [
        "napari_swc_viewer_debug_file",
        "napari_swc_viewer_debug_stream",
    ]
    logger.debug("hello from test")
    for handler in logger.handlers:
        handler.flush()
    assert log_file.exists()
    assert "hello from test" in log_file.read_text()


def test_configure_debug_logging_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    """Repeated setup should reuse the same handlers without duplication."""
    log_file = tmp_path / "debug.log"
    monkeypatch.setenv("NAPARI_SWC_VIEWER_DEBUG", "1")
    monkeypatch.setenv("NAPARI_SWC_VIEWER_LOG_FILE", str(log_file))

    first = configure_debug_logging()
    second = configure_debug_logging()
    logger = logging.getLogger("napari_swc_viewer")

    assert first == log_file
    assert second == log_file
    assert len(logger.handlers) == 2


def test_configure_debug_logging_honors_custom_log_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Custom log-file env var should control the output path."""
    nested = tmp_path / "logs" / "custom.log"
    monkeypatch.setenv("NAPARI_SWC_VIEWER_DEBUG", "true")
    monkeypatch.setenv("NAPARI_SWC_VIEWER_LOG_FILE", str(nested))

    log_path = configure_debug_logging()

    assert log_path == nested
    assert nested.parent.exists()
