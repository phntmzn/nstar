from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random

from modelprofiles import (
    DEFAULT_MODEL_NAME,
    get_model_profile,
    model_checkpoint_path,
    model_names,
)
from musicdata import (
    CHORD_DEGREES,
    F_SHARP_MINOR_SCALE,
    KEY_NAME,
    NOTE_TO_SEMITONE,
    PROGRESSION_LENGTH,
    PROGRESSION_TEMPLATES,
    RHYTHM_WEIGHTS,
    TRANSFORMATIONS,
)

import torch
import torch.nn as nn

# --- Shared constants ---

START_TOKEN = "<START>"
CHECKPOINT_VERSION = 1


# --- Music context and training configuration ---

@dataclass(frozen=True)
class MusicContext:
    key_name: str
    scale_notes: tuple[str, ...]
    chord_degrees: dict[str, tuple[int, int, int]]
    note_to_semitone: dict[str, int]
    rhythm_weights: dict[float, int]
    transformations: tuple[str, ...]
    progression_length: int

    @property
    def root_note(self) -> str:
        return self.scale_notes[0]

    @property
    def root_semitone(self) -> int:
        return self.note_to_semitone[self.root_note]


MUSIC_CONTEXT = MusicContext(
    key_name=KEY_NAME,
    scale_notes=F_SHARP_MINOR_SCALE,
    chord_degrees=CHORD_DEGREES,
    note_to_semitone=NOTE_TO_SEMITONE,
    rhythm_weights=RHYTHM_WEIGHTS,
    transformations=TRANSFORMATIONS,
    progression_length=PROGRESSION_LENGTH,
)


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = DEFAULT_MODEL_NAME
    hidden_size: int = 32
    embedding_size: int = 16
    epochs: int = 600
    learning_rate: float = 0.01
    seed: int = get_model_profile(DEFAULT_MODEL_NAME).seed
    log_every: int = 0
    checkpoint_path: str | None = None
    force_retrain: bool = False


def chord_vocabulary() -> list[str]:
    ordered_vocab: list[str] = []

    for template in PROGRESSION_TEMPLATES:
        for chord_name in template:
            if chord_name not in ordered_vocab:
                ordered_vocab.append(chord_name)

    return ordered_vocab


def chord_degree_center(chord_name: str) -> float:
    degrees = MUSIC_CONTEXT.chord_degrees[chord_name]
    return sum(degrees) / len(degrees)


def weighted_average_rhythm() -> float:
    weighted_total = sum(duration * weight for duration, weight in MUSIC_CONTEXT.rhythm_weights.items())
    total_weight = sum(MUSIC_CONTEXT.rhythm_weights.values())
    return weighted_total / total_weight


def template_context_vector(template: tuple[str, ...]) -> list[float]:
    # Reduce a full progression template to a small fixed feature vector that
    # conditions the GRU on key, cadence shape, and chord distribution.
    template_centers = [chord_degree_center(chord_name) for chord_name in template]
    unique_ratio = len(set(template)) / max(len(template), 1)
    cadence_center = chord_degree_center(template[-1])
    return [
        1.0 if "minor" in MUSIC_CONTEXT.key_name.lower() else 0.0,
        len(MUSIC_CONTEXT.scale_notes) / 12.0,
        MUSIC_CONTEXT.root_semitone / 11.0,
        len(MUSIC_CONTEXT.chord_degrees) / 16.0,
        MUSIC_CONTEXT.progression_length / 16.0,
        len(MUSIC_CONTEXT.transformations) / 8.0,
        weighted_average_rhythm() / 4.0,
        sum(template_centers) / max(len(template_centers), 1) / max(len(MUSIC_CONTEXT.scale_notes) - 1, 1),
        cadence_center / max(len(MUSIC_CONTEXT.scale_notes) - 1, 1),
        unique_ratio,
    ]


class ProgressionRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        embedding_size: int = 16,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(
            input_size=embedding_size + context_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, vocab_size - 1)

    def forward(self, token_ids: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.embedding(token_ids)
        # Repeat the same progression-level context at every time step so each
        # next-chord prediction sees both token history and global template cues.
        repeated_context = context.unsqueeze(1).expand(-1, token_embeddings.size(1), -1)
        rnn_input = torch.cat([token_embeddings, repeated_context], dim=-1)
        rnn_output, _ = self.rnn(rnn_input)
        return self.output(rnn_output)


SimpleNeuralNet = ProgressionRNN


# --- Inference wrapper ---

@dataclass(frozen=True)
class ProgressionModel:
    model_name: str
    network: ProgressionRNN
    chords: list[str]
    chord_to_index: dict[str, int]
    context_size: int
    start_index: int
    hidden_size: int
    embedding_size: int
    epochs: int
    learning_rate: float
    seed: int
    final_loss: float | None

    def history_to_token_ids(self, history: list[str]) -> list[int]:
        token_ids = [self.start_index]
        token_ids.extend(self.chord_to_index[chord_name] for chord_name in history)
        return token_ids

    def predict_probabilities(
        self,
        history: list[str],
        context_vector: list[float] | None = None,
    ) -> list[float]:
        self.network.eval()
        token_ids = torch.tensor([self.history_to_token_ids(history)], dtype=torch.long)
        context = torch.tensor([context_vector or template_context_vector(tuple(history) or random.choice(PROGRESSION_TEMPLATES))], dtype=torch.float32)
        with torch.no_grad():
            logits = self.network(token_ids, context)[0, -1]
            probabilities = torch.softmax(logits, dim=0)
        return probabilities.tolist()

    def sample_next_chord(
        self,
        history: list[str],
        temperature: float = 0.9,
        context_vector: list[float] | None = None,
    ) -> str:
        probabilities = self.predict_probabilities(history, context_vector=context_vector)
        adjusted = [
            max(probability, 1e-9) ** (1.0 / max(temperature, 1e-3))
            for probability in probabilities
        ]
        return random.choices(self.chords, weights=adjusted, k=1)[0]

    def _nudge_off_training_template(
        self,
        progression: list[str],
        context_vector: list[float],
    ) -> list[str]:
        # If repeated resampling still lands on a memorized template, walk
        # backward and swap in the next most likely chord that breaks the exact
        # training-sequence match.
        for position in range(len(progression) - 1, -1, -1):
            history = progression[:position]
            probabilities = self.predict_probabilities(history, context_vector=context_vector)
            ranked_indices = sorted(
                range(len(self.chords)),
                key=lambda index: probabilities[index],
                reverse=True,
            )
            for candidate_index in ranked_indices:
                candidate_chord = self.chords[candidate_index]
                if candidate_chord == progression[position]:
                    continue
                candidate_progression = progression.copy()
                candidate_progression[position] = candidate_chord
                if tuple(candidate_progression) not in PROGRESSION_TEMPLATES:
                    return candidate_progression
        return progression

    def generate_progression(
        self,
        length: int | None = None,
        temperature: float = 0.9,
        seed_template: tuple[str, ...] | None = None,
        allow_training_template: bool = False,
        max_resamples: int = 12,
    ) -> list[str]:
        target_length = length or MUSIC_CONTEXT.progression_length
        seed = seed_template or random.choice(PROGRESSION_TEMPLATES)
        context_vector = template_context_vector(seed)
        fallback_progression: list[str] = []

        for attempt in range(max(1, max_resamples)):
            progression: list[str] = []
            # Raise temperature slightly on each retry so the model has a better
            # chance of escaping a memorized training row.
            attempt_temperature = max(temperature, 0.85) * (1.0 + 0.08 * attempt)
            for _ in range(target_length):
                progression.append(
                    self.sample_next_chord(
                        progression,
                        temperature=attempt_temperature,
                        context_vector=context_vector,
                    )
                )

            fallback_progression = progression
            if allow_training_template or tuple(progression) not in PROGRESSION_TEMPLATES:
                return progression

        if allow_training_template:
            return fallback_progression

        return self._nudge_off_training_template(fallback_progression, context_vector)


# --- Dataset construction and gradient training ---

def _build_training_dataset(
    chords: list[str],
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    start_index = len(chords)
    chord_to_index = {chord_name: index for index, chord_name in enumerate(chords)}
    token_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    context_rows: list[list[float]] = []

    for template in PROGRESSION_TEMPLATES:
        token_sequence = [start_index]
        token_sequence.extend(chord_to_index[chord_name] for chord_name in template[:-1])
        target_sequence = [chord_to_index[chord_name] for chord_name in template]
        token_rows.append(token_sequence)
        target_rows.append(target_sequence)
        context_rows.append(template_context_vector(template))

    return (
        torch.tensor(token_rows, dtype=torch.long),
        torch.tensor(target_rows, dtype=torch.long),
        torch.tensor(context_rows, dtype=torch.float32),
    )


def train_progression_model(
    model_name: str = DEFAULT_MODEL_NAME,
    hidden_size: int = 32,
    embedding_size: int = 16,
    epochs: int = 600,
    learning_rate: float = 0.01,
    seed: int = 0,
    log_every: int = 0,
    ) -> ProgressionModel:
    torch.manual_seed(seed)
    chords = chord_vocabulary()
    start_index = len(chords)
    chord_to_index = {chord_name: index for index, chord_name in enumerate(chords)}
    token_tensor, target_tensor, context_tensor = _build_training_dataset(chords)
    network = ProgressionRNN(
        vocab_size=len(chords) + 1,
        context_size=context_tensor.shape[1],
        embedding_size=embedding_size,
        hidden_size=hidden_size,
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    final_loss: float | None = None

    network.train()
    # Train against every position in every hardcoded eight-bar template.
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = network(token_tensor, context_tensor)
        loss = loss_fn(logits.reshape(-1, len(chords)), target_tensor.reshape(-1))
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        if log_every and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            print(f"[neuralnetwork] epoch {epoch}/{epochs} loss={final_loss:.6f}")

    network.eval()
    return ProgressionModel(
        model_name=model_name,
        network=network,
        chords=chords,
        chord_to_index=chord_to_index,
        context_size=context_tensor.shape[1],
        start_index=start_index,
        hidden_size=hidden_size,
        embedding_size=embedding_size,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        final_loss=final_loss,
    )


# --- Runtime cache and config resolution ---

_ACTIVE_TRAINING_CONFIG = TrainingConfig()
_CACHED_MODEL: ProgressionModel | None = None
_CACHED_MODEL_KEY: tuple[str, int, int, int, float, int, str, bool] | None = None


def clear_progression_model_cache() -> None:
    global _CACHED_MODEL
    global _CACHED_MODEL_KEY
    _CACHED_MODEL = None
    _CACHED_MODEL_KEY = None


def resolve_training_config(
    model_name: str = DEFAULT_MODEL_NAME,
    hidden_size: int | None = None,
    embedding_size: int | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
    seed: int | None = None,
    log_every: int = 0,
    checkpoint_path: str | None = None,
    force_retrain: bool = False,
) -> TrainingConfig:
    profile = get_model_profile(model_name)
    return TrainingConfig(
        model_name=model_name,
        hidden_size=hidden_size if hidden_size is not None else profile.hidden_size,
        embedding_size=embedding_size if embedding_size is not None else profile.embedding_size,
        epochs=epochs if epochs is not None else profile.epochs,
        learning_rate=learning_rate if learning_rate is not None else profile.learning_rate,
        seed=seed if seed is not None else profile.seed,
        log_every=log_every,
        checkpoint_path=checkpoint_path,
        force_retrain=force_retrain,
    )


def configure_progression_model(
    model_name: str = DEFAULT_MODEL_NAME,
    hidden_size: int | None = None,
    embedding_size: int | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
    seed: int | None = None,
    log_every: int = 0,
    checkpoint_path: str | None = None,
    force_retrain: bool = False,
) -> TrainingConfig:
    global _ACTIVE_TRAINING_CONFIG
    _ACTIVE_TRAINING_CONFIG = resolve_training_config(
        model_name=model_name,
        hidden_size=hidden_size,
        embedding_size=embedding_size,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        log_every=log_every,
        checkpoint_path=checkpoint_path,
        force_retrain=force_retrain,
    )
    clear_progression_model_cache()
    return _ACTIVE_TRAINING_CONFIG


def current_training_config() -> TrainingConfig:
    return _ACTIVE_TRAINING_CONFIG


def resolved_checkpoint_path(config: TrainingConfig) -> Path:
    # Named models default to models/<model-name>.pt, but callers can override
    # the location explicitly for experiments or manual exports.
    if config.checkpoint_path:
        return Path(config.checkpoint_path).expanduser().resolve()
    return model_checkpoint_path(config.model_name)


def _training_config_key(config: TrainingConfig) -> tuple[str, int, int, int, float, int, str, bool]:
    checkpoint_path = str(resolved_checkpoint_path(config))
    return (
        config.model_name,
        config.hidden_size,
        config.embedding_size,
        config.epochs,
        config.learning_rate,
        config.seed,
        checkpoint_path,
        config.force_retrain,
    )


def save_progression_model(model: ProgressionModel, checkpoint_path: str | Path) -> None:
    path = Path(checkpoint_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "model_name": model.model_name,
            "state_dict": model.network.state_dict(),
            "chords": model.chords,
            "context_size": model.context_size,
            "start_index": model.start_index,
            "hidden_size": model.hidden_size,
            "embedding_size": model.embedding_size,
            "epochs": model.epochs,
            "learning_rate": model.learning_rate,
            "seed": model.seed,
            "final_loss": model.final_loss,
        },
        path,
    )


# --- Checkpoint loading and lazy train/load entrypoint ---

def load_progression_model(checkpoint_path: str | Path) -> ProgressionModel:
    payload = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location="cpu")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported progression model checkpoint version: {payload.get('version')}")

    chords = list(payload["chords"])
    chord_to_index = {chord_name: index for index, chord_name in enumerate(chords)}
    network = ProgressionRNN(
        vocab_size=len(chords) + 1,
        context_size=int(payload["context_size"]),
        embedding_size=int(payload["embedding_size"]),
        hidden_size=int(payload["hidden_size"]),
    )
    network.load_state_dict(payload["state_dict"])
    network.eval()

    model_name = payload.get("model_name", DEFAULT_MODEL_NAME)
    return ProgressionModel(
        model_name=model_name,
        network=network,
        chords=chords,
        chord_to_index=chord_to_index,
        context_size=int(payload["context_size"]),
        start_index=int(payload["start_index"]),
        hidden_size=int(payload["hidden_size"]),
        embedding_size=int(payload["embedding_size"]),
        epochs=int(payload["epochs"]),
        learning_rate=float(payload["learning_rate"]),
        seed=int(payload["seed"]),
        final_loss=float(payload["final_loss"]) if payload["final_loss"] is not None else None,
    )


def _load_or_train_progression_model(config: TrainingConfig) -> ProgressionModel:
    checkpoint_path = resolved_checkpoint_path(config)

    if checkpoint_path.exists() and not config.force_retrain:
        model = load_progression_model(checkpoint_path)
        if config.log_every:
            loss_text = "unknown" if model.final_loss is None else f"{model.final_loss:.6f}"
            print(
                f"[neuralnetwork:{model.model_name}] loaded checkpoint {checkpoint_path} "
                f"(epochs={model.epochs}, loss={loss_text})"
            )
        return model

    if config.log_every:
        print(
            f"[neuralnetwork:{config.model_name}] training GRU "
            f"(epochs={config.epochs}, hidden={config.hidden_size}, "
            f"embedding={config.embedding_size}, lr={config.learning_rate}, seed={config.seed})"
        )

    model = train_progression_model(
        model_name=config.model_name,
        hidden_size=config.hidden_size,
        embedding_size=config.embedding_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        seed=config.seed,
        log_every=config.log_every,
    )

    save_progression_model(model, checkpoint_path)
    if config.log_every:
        print(f"[neuralnetwork:{config.model_name}] saved checkpoint {checkpoint_path}")

    return model


# --- Public generation helpers ---

def get_progression_model(config: TrainingConfig | None = None) -> ProgressionModel:
    global _CACHED_MODEL
    global _CACHED_MODEL_KEY

    active_config = config or _ACTIVE_TRAINING_CONFIG
    key = _training_config_key(active_config)
    if _CACHED_MODEL is None or _CACHED_MODEL_KEY != key:
        _CACHED_MODEL = _load_or_train_progression_model(active_config)
        _CACHED_MODEL_KEY = key
    return _CACHED_MODEL


def generate_progression_template(
    length: int | None = None,
    temperature: float = 0.9,
    allow_training_template: bool = False,
) -> list[str]:
    return get_progression_model().generate_progression(
        length=length,
        temperature=temperature,
        allow_training_template=allow_training_template,
    )


# --- Command-line interface ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the chord progression GRU and print sampled templates.")
    parser.add_argument("--model-name", choices=model_names(), default=DEFAULT_MODEL_NAME, help="Named model to train or load.")
    parser.add_argument("--epochs", type=int, default=600, help="Epochs used to train the GRU.")
    parser.add_argument("--hidden-size", type=int, default=32, help="Hidden size for the GRU.")
    parser.add_argument("--embedding-size", type=int, default=16, help="Embedding size for chord tokens.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Optimizer learning rate.")
    parser.add_argument("--seed", type=int, default=None, help="Torch seed for model initialization. Defaults to the selected model profile.")
    parser.add_argument("--log-every", type=int, default=100, help="Print loss every N epochs; 0 disables logs.")
    parser.add_argument("--checkpoint-path", default="", help="Optional checkpoint path for the trained model. Defaults to models/<model-name>.pt.")
    parser.add_argument("--retrain", action="store_true", help="Ignore any saved checkpoint and retrain.")
    parser.add_argument("--samples", type=int, default=1, help="How many progression templates to print.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature for generated templates.")
    parser.add_argument("--train-all-models", action="store_true", help="Train or load every named model profile.")
    parser.add_argument("--list-models", action="store_true", help="Print the supported named models and exit.")
    parser.add_argument(
        "--allow-training-template",
        action="store_true",
        help="Allow samples to exactly match one of the training templates.",
    )
    return parser.parse_args()


def _run_single_model(args: argparse.Namespace, model_name: str) -> None:
    config = configure_progression_model(
        model_name=model_name,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        log_every=args.log_every,
        checkpoint_path=args.checkpoint_path or None,
        force_retrain=args.retrain,
    )
    model = get_progression_model(config)
    checkpoint_path = resolved_checkpoint_path(config)
    loss_text = "unknown" if model.final_loss is None else f"{model.final_loss:.6f}"
    print(
        f"[neuralnetwork:{model.model_name}] ready checkpoint={checkpoint_path} "
        f"epochs={model.epochs} loss={loss_text}"
    )
    for sample_index in range(1, args.samples + 1):
        progression = generate_progression_template(
            temperature=args.temperature,
            allow_training_template=args.allow_training_template,
        )
        print(f"[{model.model_name} sample {sample_index}] {progression}")


def main() -> None:
    args = parse_args()
    if args.list_models:
        for model_name in model_names():
            profile = get_model_profile(model_name)
            print(f"{model_name} seed={profile.seed}")
        return

    if args.train_all_models:
        if args.checkpoint_path:
            raise ValueError("--checkpoint-path cannot be combined with --train-all-models.")
        for model_name in model_names():
            _run_single_model(args, model_name)
        return

    _run_single_model(args, args.model_name)


if __name__ == "__main__":
    main()
