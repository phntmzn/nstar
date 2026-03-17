from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
import random
import sys
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from pathlib import Path

from midiutil import MIDIFile
from midiutil.MidiFile import MINOR, SHARPS
from modelprofiles import DEFAULT_MODEL_NAME, model_checkpoint_path, model_names
from musicdata import (
    CHORD_DEGREES,
    F_SHARP_MINOR_SCALE,
    KEY_NAME,
    NOTE_TO_SEMITONE,
    PROGRESSION_LENGTH,
    RHYTHM_WEIGHTS,
    TRANSFORMATIONS,
)
from neuralnetwork import (
    configure_progression_model,
    generate_progression_template,
    get_progression_model,
)






# Constants
def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


DEFAULT_TARGET = 20_000_000_000
DEFAULT_BARS = 64
DEFAULT_BPM = 157
BEATS_PER_BAR = 4
DEFAULT_OUTPUT_DIR = Path("/Users/macbookair/Documents/mid")
DEFAULT_QUEUE_SIZE = 256
DEFAULT_NAME_PREFIX = DEFAULT_MODEL_NAME

MELODY_TRACK = 0
CHORD_TRACK = 1
MELODY_CHANNEL = 0
CHORD_CHANNEL = 1
MELODY_PROGRAM = 0
CHORD_PROGRAM = 48


@dataclass(frozen=True)
class SectionSpec:
    name: str
    family: str
    transformation: str = "identity"
    refresh: bool = False

    def __post_init__(self) -> None:
        if self.transformation not in TRANSFORMATIONS:
            raise ValueError(f"Unsupported transformation: {self.transformation}")


def build_section_plan() -> list[SectionSpec]:
    return [
        SectionSpec("intro", "A", "identity", refresh=True),
        SectionSpec("verse", "A", "transpose_up"),
        SectionSpec("pre_hook", "B", "identity", refresh=True),
        SectionSpec("hook", "B", "retrograde"),
        SectionSpec("verse_b", "A", "transpose_down"),
        SectionSpec("bridge", "C", "invert", refresh=True),
        SectionSpec("hook_b", "B", "identity"),
        SectionSpec("outro", "A", "identity"),
    ]


def choose_progression_template() -> list[str]:
    return generate_progression_template(length=PROGRESSION_LENGTH, temperature=0.85)


def build_model_config(args: argparse.Namespace) -> dict[str, object]:
    checkpoint_path = args.model_path.strip() or str(model_checkpoint_path(args.model_name, app_base_dir()))
    return {
        "model_name": args.model_name,
        "hidden_size": args.hidden_size,
        "embedding_size": args.embedding_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.model_seed,
        "log_every": args.train_log_every,
        "checkpoint_path": checkpoint_path,
        "force_retrain": args.retrain_model,
    }


def prepare_progression_model(model_config: dict[str, object]) -> None:
    configure_progression_model(**model_config)
    model = get_progression_model()
    loss_text = "unknown" if model.final_loss is None else f"{model.final_loss:.6f}"
    print(f"[neuralnetwork:{model.model_name}] ready epochs={model.epochs} loss={loss_text}")

# -- CLI TOOLS
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate unique 64-bar MIDI songs in F# minor."
    )
    parser.add_argument(
        "--model-name",
        choices=model_names(),
        default=DEFAULT_MODEL_NAME,
        help="Named progression model to use for training and generation.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help="Number of unique MIDI files to save.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where MIDI files will be written.",
    )
    parser.add_argument(
        "--producers",
        type=int,
        default=max(multiprocessing.cpu_count() - 1, 1),
        help="Number of producer processes.",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=DEFAULT_QUEUE_SIZE,
        help="Max buffered songs between producers and the writer.",
    )
    parser.add_argument(
        "--name-prefix",
        default="",
        help="Filename prefix used for saved MIDI files. Defaults to the selected model name.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=600,
        help="Epochs used to train the progression model before generating songs.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="Hidden size for the progression GRU.",
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=16,
        help="Embedding size for progression chord tokens.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Optimizer learning rate for the progression model.",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="Torch seed used when training the progression model. Defaults to the selected model profile.",
    )
    parser.add_argument(
        "--train-log-every",
        type=int,
        default=100,
        help="Print progression-model loss every N epochs; 0 disables training logs.",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Checkpoint path for the trained progression model. Defaults to models/<model-name>.pt.",
    )
    parser.add_argument(
        "--retrain-model",
        action="store_true",
        help="Ignore any saved progression-model checkpoint and retrain before generating songs.",
    )
    return parser.parse_args()



# -- music
def degree_to_note(degree: int) -> str:
    return F_SHARP_MINOR_SCALE[degree % len(F_SHARP_MINOR_SCALE)]


def shift_degree(degree: int, steps: int) -> int:
    return (degree + steps) % len(F_SHARP_MINOR_SCALE)


def invert_degree(degree: int) -> int:
    return (-degree) % len(F_SHARP_MINOR_SCALE)


def note_to_midi(note: str, octave: int) -> int:
    return (octave + 1) * 12 + NOTE_TO_SEMITONE[note]


def generate_bar_rhythm() -> list[float]:
    remaining = float(BEATS_PER_BAR)
    durations: list[float] = []

    while remaining > 0:
        options = [value for value in RHYTHM_WEIGHTS if value <= remaining + 1e-9]
        weights = [RHYTHM_WEIGHTS[value] for value in options]
        duration = random.choices(options, weights=weights, k=1)[0]
        durations.append(duration)
        remaining = round(remaining - duration, 3)

    return durations


def choose_degree(chord_name: str, previous_degree: int | None) -> int:
    chord_tones = CHORD_DEGREES[chord_name]
    if previous_degree is None:
        return random.choice(chord_tones)

    if random.random() < 0.65:
        return random.choice(chord_tones)

    return shift_degree(previous_degree, random.choice([-2, -1, 1, 2]))


def choose_octave(note: str, previous_pitch: int | None) -> int:
    candidates = [(octave, note_to_midi(note, octave)) for octave in (4, 5)]
    if previous_pitch is None:
        return 5 if random.random() < 0.6 else 4
    return min(candidates, key=lambda item: abs(item[1] - previous_pitch))[0]


def generate_bar(chord_name: str, previous_pitch: int | None) -> tuple[dict, int]:
    melody_events = []

    for duration in generate_bar_rhythm():
        previous_degree = melody_events[-1]["degree"] if melody_events else None
        degree = choose_degree(chord_name, previous_degree)
        note = degree_to_note(degree)
        octave = choose_octave(note, previous_pitch)
        pitch = note_to_midi(note, octave)
        melody_events.append(
            {
                "degree": degree,
                "duration": duration,
                "velocity": random.randint(78, 108),
                "octave": octave,
            }
        )
        previous_pitch = pitch

    return {"chord": chord_name, "melody": melody_events}, previous_pitch


def generate_phrase(template: list[str], previous_pitch: int | None) -> tuple[list[dict], int | None]:
    phrase = []

    for chord_name in template:
        bar, previous_pitch = generate_bar(chord_name, previous_pitch)
        phrase.append(bar)

    return phrase, previous_pitch


def transform_phrase(phrase: list[dict], transformation: str) -> list[dict]:
    source_bars = list(reversed(phrase)) if transformation == "retrograde" else phrase
    transformed: list[dict] = []

    for bar in source_bars:
        melody_source = list(reversed(bar["melody"])) if transformation == "retrograde" else bar["melody"]
        new_bar = {"chord": bar["chord"], "melody": []}

        for event in melody_source:
            new_event = event.copy()
            if transformation == "transpose_up":
                new_event["degree"] = shift_degree(new_event["degree"], 1)
            elif transformation == "transpose_down":
                new_event["degree"] = shift_degree(new_event["degree"], -1)
            elif transformation == "invert":
                new_event["degree"] = invert_degree(new_event["degree"])
            new_bar["melody"].append(new_event)

        transformed.append(new_bar)

    return transformed


def phrase_last_pitch(phrase: list[dict]) -> int | None:
    if not phrase or not phrase[-1]["melody"]:
        return None

    last_event = phrase[-1]["melody"][-1]
    return note_to_midi(degree_to_note(last_event["degree"]), last_event["octave"])


def generate_section_phrase(
    section: SectionSpec,
    motif_library: dict[str, list[dict]],
    previous_pitch: int | None,
) -> tuple[list[dict], int | None]:
    if section.refresh or section.family not in motif_library:
        template = choose_progression_template()
        motif_library[section.family], previous_pitch = generate_phrase(template, previous_pitch)

    section_phrase = transform_phrase(copy.deepcopy(motif_library[section.family]), section.transformation)
    return section_phrase, phrase_last_pitch(section_phrase)


def generate_song() -> dict:
    bars: list[dict] = []
    previous_pitch: int | None = None
    motif_library: dict[str, list[dict]] = {}
    section_plan = build_section_plan()

    for section in section_plan:
        section_phrase, previous_pitch = generate_section_phrase(
            section,
            motif_library,
            previous_pitch,
        )
        bars.extend(section_phrase)

    bars = bars[:DEFAULT_BARS]
    bars[-2]["chord"] = "C#m"
    bars[-1]["chord"] = "F#m"
    bars[-1]["melody"][-1]["degree"] = 0
    bars[-1]["melody"][-1]["octave"] = 4
    bars[-1]["melody"][-1]["duration"] = max(1.0, bars[-1]["melody"][-1]["duration"])

    return {
        "tempo_bpm": random.randint(88, 132),
        "key": KEY_NAME,
        "bars": bars,
        "structure": [section.name for section in section_plan],
    }


def hash_song(song: dict) -> str:
    payload = json.dumps(song, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def next_output_number(output_dir: Path, name_prefix: str) -> int:
    max_number = 0

    for midi_path in output_dir.glob("*.mid"):
        stem = midi_path.stem
        if not stem.startswith(name_prefix):
            continue

        suffix = stem[len(name_prefix):]
        if not suffix:
            continue

        if not suffix.isdigit():
            continue
        max_number = max(max_number, int(suffix))

    return max_number + 1


def output_path_for_number(output_dir: Path, name_prefix: str, song_number: int) -> Path:
    return output_dir / f"{name_prefix}{song_number:02d}.mid"


def chord_pitches(chord_name: str) -> list[int]:
    degrees = CHORD_DEGREES[chord_name]
    octaves = [3, 3, 4]
    pitches = []

    for degree, octave in zip(degrees, octaves):
        note = degree_to_note(degree)
        pitches.append(note_to_midi(note, octave))

    return pitches




# --- Write to midi ---
def write_song_to_midi(song: dict, midi_path: Path) -> None:
    midi = MIDIFile(
        numTracks=2,
        removeDuplicates=False,
        deinterleave=True,
        adjust_origin=False,
        file_format=1,
    )
    tempo = song["tempo_bpm"]

    midi.addTrackName(MELODY_TRACK, 0, "Melody")
    midi.addTrackName(CHORD_TRACK, 0, "Harmony")
    midi.addTempo(MELODY_TRACK, 0, tempo)
    midi.addTempo(CHORD_TRACK, 0, tempo)
    midi.addTimeSignature(MELODY_TRACK, 0, 4, 2, 24)
    midi.addTimeSignature(CHORD_TRACK, 0, 4, 2, 24)
    midi.addKeySignature(MELODY_TRACK, 0, 3, SHARPS, MINOR)
    midi.addKeySignature(CHORD_TRACK, 0, 3, SHARPS, MINOR)
    midi.addProgramChange(MELODY_TRACK, MELODY_CHANNEL, 0, MELODY_PROGRAM)
    midi.addProgramChange(CHORD_TRACK, CHORD_CHANNEL, 0, CHORD_PROGRAM)

    for bar_index, bar in enumerate(song["bars"]):
        bar_start = bar_index * BEATS_PER_BAR
        time_offset = 0.0

        for event in bar["melody"]:
            note = degree_to_note(event["degree"])
            pitch = note_to_midi(note, event["octave"])
            midi.addNote(
                MELODY_TRACK,
                MELODY_CHANNEL,
                pitch,
                bar_start + time_offset,
                event["duration"],
                event["velocity"],
            )
            time_offset += event["duration"]

        for pitch in chord_pitches(bar["chord"]):
            midi.addNote(CHORD_TRACK, CHORD_CHANNEL, pitch, bar_start, BEATS_PER_BAR, 72)

    midi_path.parent.mkdir(parents=True, exist_ok=True)
    with midi_path.open("wb") as handle:
        midi.writeFile(handle)


def producer(
    song_queue: Queue,
    stop_event: Event,
    seed_value: int,
    model_config: dict[str, object],
) -> None:
    random.seed(seed_value)
    configure_progression_model(**model_config)
    get_progression_model()

    while not stop_event.is_set():
        song = generate_song()
        song_hash = hash_song(song)
        song_queue.put(("song", song, song_hash))


def consume_and_save(
    song_queue: Queue,
    stop_event: Event,
    output_dir: Path,
    target: int,
    name_prefix: str,
) -> None:
    saved = 0
    seen_hashes: set[str] = set()
    output_dir.mkdir(parents=True, exist_ok=True)
    next_number = next_output_number(output_dir, name_prefix)

    while saved < target:
        item_type, song, song_hash = song_queue.get()
        if item_type != "song":
            continue

        if song_hash in seen_hashes:
            continue

        midi_path = output_path_for_number(output_dir, name_prefix, next_number)
        if midi_path.exists():
            next_number += 1
            continue

        write_song_to_midi(song, midi_path)
        seen_hashes.add(song_hash)
        saved += 1
        next_number += 1

        if saved == 1 or saved % 100 == 0 or saved == target:
            print(f"Saved {saved} MIDI files to {output_dir}")

    stop_event.set()
    print(f"Finished saving {target} unique MIDI files.")


def start_producers(
    song_queue: Queue,
    stop_event: Event,
    producer_count: int,
    model_config: dict[str, object],
) -> list[multiprocessing.Process]:
    producers = []

    for seed in range(producer_count):
        process = multiprocessing.Process(
            target=producer,
            args=(song_queue, stop_event, seed, model_config),
        )
        process.start()
        producers.append(process)

    return producers


def stop_producers(producers: list[multiprocessing.Process], stop_event: Event) -> None:
    stop_event.set()

    for process in producers:
        process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    producer_count = max(1, args.producers)
    name_prefix = args.name_prefix.strip() or args.model_name
    model_config = build_model_config(args)
    prepare_progression_model(model_config)
    producer_model_config = dict(model_config)
    producer_model_config["log_every"] = 0
    producer_model_config["force_retrain"] = False
    song_queue: Queue = multiprocessing.Queue(maxsize=max(1, args.queue_size))
    stop_event = multiprocessing.Event()
    producers = start_producers(song_queue, stop_event, producer_count, producer_model_config)

    try:
        consume_and_save(song_queue, stop_event, output_dir, args.target, name_prefix)
    except KeyboardInterrupt:
        print("Stopping song generation.")
        stop_event.set()
    finally:
        stop_producers(producers, stop_event)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
