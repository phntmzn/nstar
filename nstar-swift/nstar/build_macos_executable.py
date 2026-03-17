from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ENTRYPOINT = PROJECT_ROOT / "generation.py"
DEFAULT_DIST_DIR = PROJECT_ROOT / "dist-macos"
DEFAULT_WORK_DIR = PROJECT_ROOT / "build" / "pyinstaller" / "work"
DEFAULT_SPEC_DIR = PROJECT_ROOT / "build" / "pyinstaller" / "spec"
DEFAULT_NAME = "nstar"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the nstar CLI into a standalone macOS Unix executable."
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help="Executable name to create inside the dist directory.",
    )
    parser.add_argument(
        "--entrypoint",
        default=str(DEFAULT_ENTRYPOINT),
        help="Python entrypoint to package.",
    )
    parser.add_argument(
        "--dist-dir",
        default=str(DEFAULT_DIST_DIR),
        help="Directory where the built executable will be written.",
    )
    parser.add_argument(
        "--work-dir",
        default=str(DEFAULT_WORK_DIR),
        help="PyInstaller work directory.",
    )
    parser.add_argument(
        "--spec-dir",
        default=str(DEFAULT_SPEC_DIR),
        help="Directory where the PyInstaller spec file will be written.",
    )
    parser.add_argument(
        "--skip-model-copy",
        action="store_true",
        help="Do not copy progression_model.pt next to the built executable.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previous build outputs before building.",
    )
    parser.add_argument(
        "--onedir",
        action="store_true",
        help="Build an onedir layout instead of a onefile executable.",
    )
    return parser.parse_args()


def ensure_macos() -> None:
    if sys.platform != "darwin":
        raise SystemExit("This build script is intended to run on macOS.")


def ensure_pyinstaller() -> None:
    if importlib.util.find_spec("PyInstaller.__main__") is None:
        raise SystemExit(
            "PyInstaller is not installed in this Python environment. "
            "Install it first with: python3 -m pip install pyinstaller"
        )


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def build_executable(
    *,
    name: str,
    entrypoint: Path,
    dist_dir: Path,
    work_dir: Path,
    spec_dir: Path,
    skip_model_copy: bool,
    clean: bool,
    onedir: bool = False,
) -> Path:
    if not entrypoint.exists():
        raise SystemExit(f"Entrypoint does not exist: {entrypoint}")

    if clean:
        remove_path(dist_dir)
        remove_path(work_dir)
        spec_file = spec_dir / f"{name}.spec"
        remove_path(spec_file)

    dist_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)
    pyinstaller_config_dir = work_dir.parent / "config"
    pyinstaller_config_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir" if onedir else "--onefile",
        "--clean",
        "--name",
        name,
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(spec_dir),
        "--paths",
        str(PROJECT_ROOT),
        "--collect-all",
        "torch",
        str(entrypoint),
    ]

    print("[build] running:", " ".join(command))
    environment = dict(os.environ)
    environment.setdefault("PYINSTALLER_CONFIG_DIR", str(pyinstaller_config_dir))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=environment)

    executable_path = dist_dir / name / name if onedir else dist_dir / name
    if not executable_path.exists():
        raise SystemExit(f"Build completed but executable was not found: {executable_path}")

    if not skip_model_copy:
        runtime_dir = executable_path.parent
        if DEFAULT_MODEL_DIR.exists():
            destination_dir = runtime_dir / DEFAULT_MODEL_DIR.name
            if destination_dir.exists():
                shutil.rmtree(destination_dir)
            shutil.copytree(DEFAULT_MODEL_DIR, destination_dir)
            print(f"[build] copied model checkpoints to {destination_dir}")

        legacy_model_path = PROJECT_ROOT / "progression_model.pt"
        if legacy_model_path.exists():
            destination = runtime_dir / legacy_model_path.name
            shutil.copy2(legacy_model_path, destination)
            print(f"[build] copied legacy model checkpoint to {destination}")

    print(f"[build] executable ready at {executable_path}")
    return executable_path


def main() -> None:
    args = parse_args()
    ensure_macos()
    ensure_pyinstaller()

    build_executable(
        name=args.name,
        entrypoint=Path(args.entrypoint).expanduser().resolve(),
        dist_dir=Path(args.dist_dir).expanduser().resolve(),
        work_dir=Path(args.work_dir).expanduser().resolve(),
        spec_dir=Path(args.spec_dir).expanduser().resolve(),
        skip_model_copy=args.skip_model_copy,
        clean=args.clean,
        onedir=args.onedir,
    )


if __name__ == "__main__":
    main()
