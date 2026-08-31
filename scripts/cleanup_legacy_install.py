"""Remove editable-install metadata left by the former distribution name."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
import shutil
import subprocess
import sys


LEGACY_DISTRIBUTION = "napari-swc-viewer"


def main() -> int:
    """Uninstall the legacy distribution and remove its generated egg-info."""
    repository_root = Path(__file__).resolve().parents[1]
    legacy_egg_info = repository_root / "src" / "napari_swc_viewer.egg-info"

    try:
        distribution(LEGACY_DISTRIBUTION)
    except PackageNotFoundError:
        shutil.rmtree(legacy_egg_info, ignore_errors=True)
        return 0

    completed = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", LEGACY_DISTRIBUTION],
        check=False,
    )
    shutil.rmtree(legacy_egg_info, ignore_errors=True)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
