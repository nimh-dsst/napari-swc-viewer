from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_script_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "add_flatmap_columns_to_parquet.py"
    )
    spec = importlib.util.spec_from_file_location(
        "add_flatmap_columns_to_parquet_script",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_accepts_lookup_directory_for_v3_whole_parquet() -> None:
    script = _load_script_module()

    parsed = script.parse_args(
        [
            "source.parquet",
            "output.parquet",
            "--lookup-dir",
            "lookups",
            "--lookup-resolution",
            "10",
        ]
    )

    assert parsed.lookup_dir == Path("lookups")
    assert parsed.lookup_resolution == 10.0
    assert parsed.flatmap is None
    assert parsed.depth is None


def test_cli_retains_legacy_single_style_arguments() -> None:
    script = _load_script_module()

    parsed = script.parse_args(
        [
            "source.parquet",
            "output.parquet",
            "--flatmap",
            "flatmap.nrrd",
            "--depth",
            "depth.nrrd",
            "--file-id",
            "one.swc",
        ]
    )

    assert parsed.lookup_dir is None
    assert parsed.flatmap == Path("flatmap.nrrd")
    assert parsed.depth == Path("depth.nrrd")
    assert parsed.file_ids == ["one.swc"]


def test_cli_rejects_row_filter_for_v3_whole_parquet() -> None:
    script = _load_script_module()

    with pytest.raises(SystemExit):
        script.parse_args(
            [
                "source.parquet",
                "output.parquet",
                "--lookup-dir",
                "lookups",
                "--file-id",
                "one.swc",
            ]
        )


def test_cli_lookup_directory_dispatches_to_dual_augmentation(
    monkeypatch,
    capsys,
) -> None:
    script = _load_script_module()
    calls: dict[str, object] = {}
    lookup_set = object()

    def discover(path, **kwargs):
        calls["lookup_dir"] = path
        calls["discover_kwargs"] = kwargs
        return lookup_set

    def augment(source, output, selected_lookup_set, **kwargs):
        calls["source"] = source
        calls["output"] = output
        calls["lookup_set"] = selected_lookup_set
        calls["augment_kwargs"] = kwargs
        return SimpleNamespace(
            rows=2,
            direct_rows=1,
            mirrored_depth_rows=1,
            mirrored_rows=0,
            unmapped_rows=0,
            output_parquet=output,
            lookup_set_id="fls1-test",
        )

    monkeypatch.setattr(script, "discover_flatmap_lookup_set", discover)
    monkeypatch.setattr(script, "augment_neuron_parquet_with_flatmaps", augment)

    result = script.main(
        [
            "source.parquet",
            "output.parquet",
            "--lookup-dir",
            "lookups",
            "--lookup-resolution",
            "10",
        ]
    )

    assert result == 0
    assert calls["lookup_dir"] == Path("lookups")
    assert calls["lookup_set"] is lookup_set
    assert calls["source"] == Path("source.parquet")
    assert calls["output"] == Path("output.parquet")
    assert calls["discover_kwargs"]["lookup_resolution_um"] == 10.0
    assert "Lookup set ID: fls1-test" in capsys.readouterr().out

