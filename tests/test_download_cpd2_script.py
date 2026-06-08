"""Tests for the CPD2 BIL downloader script."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    """Load the CPD2 downloader CLI script as a module."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "download_cpd2.py"
    spec = importlib.util.spec_from_file_location("download_cpd2_script", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corrected_registered_filename_rewrites_cpd2_suffix():
    """CPD2 parquet filenames should map back to BIL registered filenames."""
    module = _load_script_module()

    assert (
        module.corrected_registered_filename(
            "1119749665_17545_3134-X21894-Y19320_reg_right.swc"
        )
        == "1119749665_17545_3134-X21894-Y19320_reg.swc"
    )
    assert (
        module.corrected_registered_filename(
            "1119749665_17545_3134-X21894-Y19320_reg.swc"
        )
        == "1119749665_17545_3134-X21894-Y19320_reg.swc"
    )


def test_builtin_target_list_has_77_unique_registered_filenames():
    """The embedded target list should describe the full CPD2 source set."""
    module = _load_script_module()

    filenames = module.CPD2_REGISTERED_FILENAMES

    assert len(filenames) == 77
    assert len(set(filenames)) == 77
    assert all(name.endswith("_reg.swc") for name in filenames)
    assert not any(name.endswith("_reg_right.swc") for name in filenames)


def test_extract_swc_urls_deduplicates_doi_page_links():
    """The DOI page contains duplicate link text and href URLs."""
    module = _load_script_module()
    markup = """
    <a href="https://download.brainimagelibrary.org/0f/cd/id/a_reg.swc">
      https://download.brainimagelibrary.org/0f/cd/id/a_reg.swc
    </a>
    <a href="https://download.brainimagelibrary.org/0f/cd/id/b.swc">b</a>
    """

    assert module.extract_swc_urls(markup) == [
        "https://download.brainimagelibrary.org/0f/cd/id/a_reg.swc",
        "https://download.brainimagelibrary.org/0f/cd/id/b.swc",
    ]


def test_match_target_urls_uses_filename_component():
    """Matching should not assume BIL's non-obvious directory structure."""
    module = _load_script_module()
    urls = [
        "https://download.brainimagelibrary.org/0f/cd/id/mouseID_1/a_reg.swc",
        "https://download.brainimagelibrary.org/0f/cd/id/mouseID_2/b_reg.swc",
    ]

    matched, missing = module.match_target_urls(("b_reg.swc", "missing_reg.swc"), urls)

    assert matched == [
        (
            "b_reg.swc",
            "https://download.brainimagelibrary.org/0f/cd/id/mouseID_2/b_reg.swc",
        )
    ]
    assert missing == ["missing_reg.swc"]
