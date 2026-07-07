from __future__ import annotations

from pathlib import Path
import sys


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


DEFAULT_OUTPUT_DIR = Path.home() / "Documents" / "mid"
