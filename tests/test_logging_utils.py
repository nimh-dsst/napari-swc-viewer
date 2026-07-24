"""Tests for opt-in plugin debug logging."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from napari_swc_viewer.logging_utils import (
    configure_debug_logging,
    startup_timing,
)

_NAPARI_SLICER_LOGGER_NAME = "napari.components._layer_slicer"


def _reset_plugin_logger() -> None:
    """Restore the plugin logger to an unconfigured state."""
    logger = logging.getLogger("napari_swc_viewer")
    napari_logger = logging.getLogger(_NAPARI_SLICER_LOGGER_NAME)
    handlers = list(logger.handlers)
    for handler in list(napari_logger.handlers):
        if handler.name == "napari_swc_viewer_debug_file":
            napari_logger.removeHandler(handler)
            if handler not in handlers:
                handlers.append(handler)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    for handler in handlers:
        handler.close()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    napari_logger.setLevel(logging.NOTSET)
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
    napari_logger = logging.getLogger(_NAPARI_SLICER_LOGGER_NAME)
    assert all(
        handler.name != "napari_swc_viewer_debug_file"
        for handler in napari_logger.handlers
    )


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
    napari_logger = logging.getLogger(_NAPARI_SLICER_LOGGER_NAME)
    assert napari_logger.level == logging.DEBUG
    assert [
        handler.name
        for handler in napari_logger.handlers
        if handler.name == "napari_swc_viewer_debug_file"
    ] == ["napari_swc_viewer_debug_file"]
    assert all(
        handler.name != "napari_swc_viewer_debug_stream"
        for handler in napari_logger.handlers
    )
    logger.debug("hello from test")
    napari_logger.debug("_LayerSlicer.shutdown test")
    for handler in logger.handlers:
        handler.flush()
    assert log_file.exists()
    log_text = log_file.read_text()
    assert "hello from test" in log_text
    assert log_text.count("_LayerSlicer.shutdown test") == 1
    assert _NAPARI_SLICER_LOGGER_NAME in log_text


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
    napari_logger = logging.getLogger(_NAPARI_SLICER_LOGGER_NAME)
    assert sum(
        handler.name == "napari_swc_viewer_debug_file"
        for handler in napari_logger.handlers
    ) == 1


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


def test_startup_timing_logs_start_and_ok_records(caplog) -> None:
    """Timing spans should emit stable startup_timing records."""
    logger = logging.getLogger("napari_swc_viewer.tests.startup_timing")

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        with startup_timing(logger, "unit_event", phase="setup") as timing:
            timing.set(count=3)

    messages = [record.getMessage() for record in caplog.records]

    assert any(
        "startup_timing event=unit_event status=start elapsed_s=0.000000"
        in message
        and "phase=setup" in message
        for message in messages
    )
    assert any(
        "startup_timing event=unit_event status=ok elapsed_s=" in message
        and "phase=setup" in message
        and "count=3" in message
        for message in messages
    )


def test_startup_timing_logs_exception_with_exc_info(caplog) -> None:
    """Error spans should log the exception without swallowing it."""
    logger = logging.getLogger("napari_swc_viewer.tests.startup_timing_error")

    with pytest.raises(RuntimeError, match="boom"):
        with caplog.at_level(logging.DEBUG, logger=logger.name):
            with startup_timing(logger, "unit_error", log_start=False):
                raise RuntimeError("boom")

    error_records = [
        record
        for record in caplog.records
        if "startup_timing event=unit_error status=error" in record.getMessage()
    ]
    assert len(error_records) == 1
    assert error_records[0].exc_info is not None
    assert error_records[0].exc_info[0] is RuntimeError


def test_startup_timing_is_quiet_when_debug_disabled(caplog) -> None:
    """Timing spans should avoid DEBUG records when the logger is not enabled."""
    logger = logging.getLogger("napari_swc_viewer.tests.startup_timing_disabled")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        with startup_timing(logger, "disabled_event") as timing:
            timing.set(count=1)

    assert not any("startup_timing" in record.getMessage() for record in caplog.records)
