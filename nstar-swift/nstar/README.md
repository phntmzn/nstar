# nstar

Generate 64-bar MIDI songs in F# minor.

The program combines:

- a PyTorch GRU in `neuralnetwork.py` for chord progression generation
- rule-based melody and arrangement logic in `generation.py`
- named model profiles in `modelprofiles.py`
- fixed harmony/rhythm source data in `musicdata.py`

New MIDI files are saved by default to:

`~/Documents/mid`

## What the program does

Each generated song:

- is 64 bars long
- stays in F# minor
- writes a melody track and a harmony track
- uses a trained chord progression model plus procedural melody generation

Important: the neural net only handles chord progression sampling. Melody, rhythm, section structure, and transformations are still procedural.

## Project files

- `generation.py`: main CLI for generating MIDI songs
- `neuralnetwork.py`: train, save, load, and sample the chord progression model
- `modelprofiles.py`: named model registry and default checkpoint locations
- `musicdata.py`: chord vocabulary, progression templates, rhythm weights, and key data
- `requirements.txt`: Python dependencies
- `models/`: saved per-model checkpoints after training

## Setup

Create a local virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

If you want to build a standalone macOS executable, also install PyInstaller in the same environment:

```bash
.venv/bin/python -m pip install pyinstaller
```

## Quick start

Generate one MIDI file:

```bash
.venv/bin/python generation.py --model-name phntmzn --target 1 --producers 1
```

This will:

1. load or train the chord progression model
2. generate one song
3. save a MIDI file into `~/Documents/mid`

Use `--target` explicitly. The code default is very large (`20_000_000_000`), which is not a practical first run.

## Main usage

Generate a small batch:

```bash
.venv/bin/python generation.py \
  --model-name phntmzn \
  --target 10 \
  --producers 1 \
  --name-prefix phntmzn
```

Retrain the progression model and show training loss while generating:

```bash
.venv/bin/python generation.py \
  --model-name slaymor \
  --target 1 \
  --producers 1 \
  --epochs 120 \
  --train-log-every 40 \
  --retrain-model
```

Write MIDI files somewhere else:

```bash
.venv/bin/python generation.py \
  --target 5 \
  --output-dir ~/Documents/mid \
  --producers 1
```

## Swift GUI

The repo also includes a native Swift/AppKit GUI that calls the existing Python and PyTorch backend through `gui_bridge.py`.

Run the GUI from source:

```bash
swift run nstar-gui
```

Build a `.app` bundle:

```bash
bash build_swift_app.sh
```

This creates:

- app bundle: `dist-swift/NStar.app`
- bundled backend: `dist-swift/NStar.app/Contents/Resources/backend/nstar`

Notes:

- the GUI is native Swift, but checkpoint loading and generation still happen in Python
- the app needs a Python interpreter with `torch` and `midiutil` installed
- the GUI lets you choose that interpreter at runtime
- the Swift target is set to macOS 10.13, but the usable macOS floor still depends on the Python/PyTorch build you provide

## Neural network usage

If you want to train or inspect the chord progression model directly:

```bash
.venv/bin/python neuralnetwork.py \
  --model-name phntmzn \
  --epochs 120 \
  --log-every 40 \
  --samples 3 \
  --retrain
```

This command:

- trains the GRU
- prints loss during training
- saves the checkpoint to `models/phntmzn.pt`
- prints sampled chord progressions

To allow exact training-template outputs:

```bash
.venv/bin/python neuralnetwork.py --allow-training-template
```

## Named models

Supported model names:

- `phntmzn`
- `slaymor`
- `phntm12a`
- `phntm38a`
- `zonedtodeath`
- `zonedout4evr`

List the supported model profiles:

```bash
.venv/bin/python neuralnetwork.py --list-models
```

Train all named models into `models/`:

```bash
.venv/bin/python neuralnetwork.py --train-all-models --epochs 600 --log-every 200
```

Each named model gets its own checkpoint:

- `models/phntmzn.pt`
- `models/slaymor.pt`
- `models/phntm12a.pt`
- `models/phntm38a.pt`
- `models/zonedtodeath.pt`
- `models/zonedout4evr.pt`

## Build a macOS executable

For the older Python-only CLI packaging flow:

Build a standalone Unix executable for macOS:

```bash
.venv/bin/python build_macos_executable.py --clean
```

By default this creates:

- executable: `dist-macos/nstar`
- copied checkpoints: `dist-macos/models/` if the project already has named model checkpoints

The frozen executable uses `models/<model-name>.pt` next to the executable as its default model path, so copying the checkpoint directory beside the binary lets it start without retraining.

Useful build options:

- `--name`: output executable name
- `--dist-dir`: output directory for the built binary
- `--work-dir`: PyInstaller work directory
- `--spec-dir`: where the `.spec` file is written
- `--skip-model-copy`: build the binary without copying the `models/` checkpoint directory
- `--clean`: remove previous build outputs first

## Useful CLI options

### `generation.py`

- `--model-name`: named progression model to use
- `--target`: number of MIDI files to save
- `--output-dir`: destination directory for MIDI files
- `--producers`: number of worker processes
- `--queue-size`: queue size between producers and writer
- `--name-prefix`: filename prefix; defaults to the selected model name
- `--epochs`: progression-model training epochs
- `--hidden-size`: GRU hidden size
- `--embedding-size`: chord embedding size
- `--learning-rate`: optimizer learning rate
- `--model-seed`: torch seed for model training
- `--train-log-every`: print loss every N epochs; `0` disables logging
- `--model-path`: checkpoint path for the trained progression model
- `--retrain-model`: ignore saved checkpoint and retrain

### `neuralnetwork.py`

- `--model-name`: named model to train or load
- `--epochs`: training epochs
- `--hidden-size`: GRU hidden size
- `--embedding-size`: embedding size
- `--learning-rate`: optimizer learning rate
- `--seed`: torch seed
- `--log-every`: print loss every N epochs
- `--checkpoint-path`: where to save or load the model checkpoint
- `--retrain`: force retraining even if a checkpoint exists
- `--samples`: number of progression samples to print
- `--temperature`: sampling temperature
- `--train-all-models`: train or load all named models
- `--list-models`: print supported named models
- `--allow-training-template`: allow output to exactly match a training template

## Output and checkpoints

- MIDI output folder default: `~/Documents/mid`
- Model checkpoint default: `models/<model-name>.pt` next to the running script or executable

The MIDI output directory is created automatically if it does not exist.

## Notes and limitations

- The progression model trains from a small hardcoded set of progression templates in `musicdata.py`.
- The neural net improves progression selection, but it does not learn the full song end to end.
- Melody generation is still rule-based.
- With more training data, the model would have a wider range of harmonic output.

## Typical workflow

1. Install dependencies.
2. Train one named model or all named models with `neuralnetwork.py`.
3. Run `generation.py` with `--model-name` and a small `--target`, such as `1` or `10`.
4. If you want fresh model weights, add `--retrain-model`.
5. Check the generated MIDI files in `~/Documents/mid`.
