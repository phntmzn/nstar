from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final
import sys


# --- Named model defaults ---

@dataclass(frozen=True)
class ModelProfile:
    name: str
    seed: int
    hidden_size: int = 32
    embedding_size: int = 16
    epochs: int = 600
    learning_rate: float = 0.01


MODEL_PROFILES: Final[dict[str, ModelProfile]] = {
    "phntmzn": ModelProfile(name="phntmzn", seed=11),
    "slaymor": ModelProfile(name="slaymor", seed=23),
    "phntm12a": ModelProfile(name="phntm12a", seed=37),
    "phntm38a": ModelProfile(name="phntm38a", seed=47),
    "zonedtodeath": ModelProfile(name="zonedtodeath", seed=59),
    "zonedout4evr": ModelProfile(name="zonedout4evr", seed=71),
}

DEFAULT_MODEL_NAME: Final[str] = "phntmzn"


# --- Runtime path helpers ---

def runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def model_names() -> tuple[str, ...]:
    return tuple(MODEL_PROFILES)


def get_model_profile(model_name: str) -> ModelProfile:
    # Keep model-name validation in one place so both CLIs fail with the same
    # error message when an unknown profile is requested.
    try:
        return MODEL_PROFILES[model_name]
    except KeyError as exc:
        known_names = ", ".join(model_names())
        raise ValueError(f"Unknown model name '{model_name}'. Expected one of: {known_names}") from exc


def model_checkpoint_dir(base_dir: Path | None = None) -> Path:
    # Frozen macOS builds look for checkpoints next to the executable, while
    # source runs keep them next to the project files.
    return (base_dir or runtime_base_dir()) / "models"


def model_checkpoint_path(model_name: str, base_dir: Path | None = None) -> Path:
    # Validate the model name before building the checkpoint path so callers do
    # not silently write unexpected files.
    get_model_profile(model_name)
    return model_checkpoint_dir(base_dir) / f"{model_name}.pt"
