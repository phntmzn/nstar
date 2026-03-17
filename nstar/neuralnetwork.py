from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import random

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

START_TOKEN = "<START>"


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
        repeated_context = context.unsqueeze(1).expand(-1, token_embeddings.size(1), -1)
        rnn_input = torch.cat([token_embeddings, repeated_context], dim=-1)
        rnn_output, _ = self.rnn(rnn_input)
        return self.output(rnn_output)


SimpleNeuralNet = ProgressionRNN


@dataclass(frozen=True)
class ProgressionModel:
    network: ProgressionRNN
    chords: list[str]
    chord_to_index: dict[str, int]
    context_size: int
    start_index: int

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

    def generate_progression(
        self,
        length: int | None = None,
        temperature: float = 0.9,
        seed_template: tuple[str, ...] | None = None,
    ) -> list[str]:
        target_length = length or MUSIC_CONTEXT.progression_length
        seed = seed_template or random.choice(PROGRESSION_TEMPLATES)
        context_vector = template_context_vector(seed)
        progression: list[str] = []

        for _ in range(target_length):
            progression.append(
                self.sample_next_chord(
                    progression,
                    temperature=temperature,
                    context_vector=context_vector,
                )
            )

        return progression


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
    hidden_size: int = 32,
    embedding_size: int = 16,
    epochs: int = 600,
    learning_rate: float = 0.01,
    seed: int = 0,
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

    network.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = network(token_tensor, context_tensor)
        loss = loss_fn(logits.reshape(-1, len(chords)), target_tensor.reshape(-1))
        loss.backward()
        optimizer.step()

    network.eval()
    return ProgressionModel(
        network=network,
        chords=chords,
        chord_to_index=chord_to_index,
        context_size=context_tensor.shape[1],
        start_index=start_index,
    )


@lru_cache(maxsize=1)
def get_progression_model() -> ProgressionModel:
    return train_progression_model()


def generate_progression_template(
    length: int | None = None,
    temperature: float = 0.9,
) -> list[str]:
    return get_progression_model().generate_progression(length=length, temperature=temperature)


if __name__ == "__main__":
    print(generate_progression_template())
