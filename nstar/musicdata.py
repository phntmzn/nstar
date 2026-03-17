from __future__ import annotations

from typing import Final

KEY_NAME: Final[str] = "F# minor"

F_SHARP_MINOR_SCALE: Final[tuple[str, ...]] = ("F#", "G#", "A", "B", "C#", "D", "E")

NOTE_TO_SEMITONE: Final[dict[str, int]] = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}

# Scale-degree triads/suspensions used by the generator. Keep these to three
# notes because generation.py voices harmony as a three-note chord.
CHORD_DEGREES: Final[dict[str, tuple[int, int, int]]] = {
    "F#m": (0, 2, 4),
    "F#m7": (0, 2, 4),
    "G#dim": (1, 3, 5),
    "A": (2, 4, 6),
    "Asus": (2, 3, 6),
    "Bm": (3, 5, 0),
    "Bm7": (3, 5, 0),
    "C#m": (4, 6, 1),
    "D": (5, 0, 2),
    "D6": (5, 0, 2),
    "D7": (5, 0, 2),
    "E": (6, 1, 3),
}

# Eight-bar templates are the training set for neuralnetwork.py and the motif
# seeds for generation.py. Variants deliberately reuse the same harmonic center
# while changing dominant motion and cadential shape.
PROGRESSION_TEMPLATES: Final[tuple[tuple[str, ...], ...]] = (
    ("F#m", "D", "A", "E", "Bm", "C#m", "D", "C#m"),
    ("F#m", "Bm", "D", "C#m", "F#m", "E", "D", "C#m"),
    ("F#m", "D", "Bm", "C#m", "F#m", "A", "E", "C#m"),
    ("F#m", "E", "D", "Bm", "F#m", "D", "C#m", "C#m"),
    ("Bm", "C#m", "A", "C#m", "C#m", "D", "F#m", "G#dim"),
    ("Bm", "C#m", "A", "C#m", "C#m", "D", "F#m", "C#m"),
    ("F#m7", "D", "A", "E", "Bm7", "C#m", "D", "C#m"),
    ("F#m", "D6", "A", "E", "Bm", "C#m", "D", "F#m"),
    ("F#m", "Bm", "A", "E", "F#m", "D", "C#m", "F#m"),
    ("F#m", "D", "A", "C#m", "Bm", "D", "E", "F#m"),
    ("F#m", "Asus", "D", "C#m", "Bm", "E", "D", "F#m"),
    ("F#m", "E", "Bm", "C#m", "F#m", "D", "E", "F#m"),

    ("F#m", "D", "E", "C#m", "Bm", "D", "A", "C#m"),
    ("F#m", "A", "E", "D", "Bm", "C#m", "F#m", "E"),
    ("F#m", "D", "A", "E", "F#m", "Bm", "C#m", "F#m"),
    ("F#m", "Bm", "C#m", "D", "A", "E", "C#m", "F#m"),
    ("F#m", "E", "D", "A", "Bm", "C#m", "D", "F#m"),
    ("F#m", "C#m", "D", "A", "Bm", "E", "C#m", "F#m"),
    ("Bm", "D", "A", "E", "F#m", "C#m", "D", "F#m"),
    ("F#m", "D", "Bm", "E", "A", "C#m", "D", "F#m"),
    ("F#m", "E", "C#m", "D", "Bm", "A", "E", "F#m"),
    ("F#m", "A", "Bm", "C#m", "D", "A", "E", "F#m"),
    ("F#m", "D", "A", "Bm", "C#m", "D", "E", "F#m"),
    ("Bm", "F#m", "D", "A", "E", "C#m", "Bm", "F#m"),
    ("F#m", "Dmaj7", "A", "E", "Bm7", "C#m7", "D", "F#m"),
    ("F#m", "Bm7", "E", "A", "D", "C#m", "Bm", "F#m"),
    ("F#m", "E", "D", "C#m", "Bm", "A", "D", "F#m"),
    ("F#m", "A", "D", "E", "Bm", "C#m", "A", "F#m"),
    ("F#m", "C#m", "Bm", "D", "A", "E", "D", "F#m"),
    ("F#m", "D", "E", "A", "Bm", "C#m", "D", "F#m"),
    ("F#m", "Bm", "D", "A", "E", "D", "C#m", "F#m"),
    ("F#m", "E", "A", "D", "Bm", "C#m", "E", "F#m"),
    ("Bm", "D", "F#m", "E", "A", "C#m", "D", "F#m"),
    ("F#m", "D", "A", "E", "G#dim", "C#m", "D", "F#m"),
    ("F#m", "Bm", "E", "C#m", "D", "A", "Bm", "F#m"),
    ("F#m", "D6", "Bm", "C#m", "A", "E", "D", "F#m"),
    ("F#m7", "Bm7", "E", "A", "D", "C#m", "Bm7", "F#m7"),
    ("F#m", "A", "C#m", "D", "Bm", "E", "A", "F#m"),
    ("F#m", "E", "D", "Bm", "A", "C#m", "D", "F#m"),
    ("F#m", "D", "Bm", "A", "E", "C#m", "D", "F#m"),
    ("F#m", "C#m", "A", "E", "Bm", "D", "C#m", "F#m"),
    ("F#m", "Bm", "C#m", "F#m", "D", "A", "E", "F#m"),
)

# Optional slot-level choices for future logic-based progression selection.
PROGRESSION_SLOT_OPTIONS: Final[dict[str, tuple[str, ...]]] = {
    "P1": ("F#m", "Bm"),
    "P2": ("D", "E", "C#m"),
    "P3": ("A", "Bm", "D"),
    "P4": ("E", "C#m"),
    "P5": ("F#m", "Bm", "C#m"),
    "P6": ("D", "E"),
    "P7": ("C#m", "D", "E"),
    "P8": ("F#m", "C#m", "G#dim"),
}

RHYTHM_WEIGHTS: Final[dict[float, int]] = {
    0.5: 5,
    1.0: 6,
    1.5: 3,
    2.0: 3,
    4.0: 1,
}

TRANSFORMATIONS: Final[tuple[str, ...]] = (
    "identity",
    "transpose_up",
    "transpose_down",
    "invert",
    "retrograde",
)

PROGRESSION_LENGTH: Final[int] = len(PROGRESSION_TEMPLATES[0])
CHORD_NAMES: Final[tuple[str, ...]] = tuple(CHORD_DEGREES)


def chord_notes(chord_name: str) -> tuple[str, str, str]:
    return tuple(F_SHARP_MINOR_SCALE[degree] for degree in CHORD_DEGREES[chord_name])


def validate_music_data() -> None:
    if not F_SHARP_MINOR_SCALE:
        raise ValueError("F_SHARP_MINOR_SCALE must not be empty.")

    if len(set(F_SHARP_MINOR_SCALE)) != len(F_SHARP_MINOR_SCALE):
        raise ValueError("F_SHARP_MINOR_SCALE contains duplicate note names.")

    if PROGRESSION_LENGTH == 0:
        raise ValueError("PROGRESSION_TEMPLATES must not be empty.")

    for chord_name, degrees in CHORD_DEGREES.items():
        if len(degrees) != 3:
            raise ValueError(f"{chord_name} must contain exactly three degrees.")

        for degree in degrees:
            if degree < 0 or degree >= len(F_SHARP_MINOR_SCALE):
                raise ValueError(f"{chord_name} contains an out-of-range degree: {degree}")

    for template in PROGRESSION_TEMPLATES:
        if len(template) != PROGRESSION_LENGTH:
            raise ValueError("All progression templates must be the same length.")

        for chord_name in template:
            if chord_name not in CHORD_DEGREES:
                raise ValueError(f"Unknown chord in progression template: {chord_name}")

    if not RHYTHM_WEIGHTS:
        raise ValueError("RHYTHM_WEIGHTS must not be empty.")

    for duration, weight in RHYTHM_WEIGHTS.items():
        if duration <= 0:
            raise ValueError(f"Invalid rhythm duration: {duration}")
        if weight <= 0:
            raise ValueError(f"Invalid rhythm weight for duration {duration}: {weight}")

    if len(set(TRANSFORMATIONS)) != len(TRANSFORMATIONS):
        raise ValueError("TRANSFORMATIONS contains duplicates.")

    for slot_name, slot_options in PROGRESSION_SLOT_OPTIONS.items():
        if not slot_options:
            raise ValueError(f"{slot_name} must define at least one chord option.")
        for chord_name in slot_options:
            if chord_name not in CHORD_DEGREES:
                raise ValueError(f"Unknown chord in {slot_name}: {chord_name}")


validate_music_data()

