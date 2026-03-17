from __future__ import annotations

from musicdata import PROGRESSION_SLOT_OPTIONS

PROGRESSION_TEMPLATE = PROGRESSION_SLOT_OPTIONS


def choose_chord(slot: str, preferred: str | None = None) -> str:
    options = PROGRESSION_TEMPLATE[slot]
    if preferred in options:
        return preferred
    return options[0]
