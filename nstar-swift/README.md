# nstar

Generate 64-bar MIDI songs in F# minor.

The program combines:

- a PyTorch GRU in `neuralnetwork.py` for chord progression generation
- rule-based melody and arrangement logic in `generation.py`
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
- `musicdata.py`: chord vocabulary, progression templates, rhythm weights, and key data
- `requirements.txt`: Python dependencies
- `progression_model.pt`: saved progression-model checkpoint after training

## Setup

Create a local virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Quick start

Generate one MIDI file:

```bash
.venv/bin/python generation.py --target 1 --producers 1
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
  --target 10 \
  --producers 1 \
  --name-prefix phntmzn
```

Retrain the progression model and show training loss while generating:

```bash
.venv/bin/python generation.py \
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

The repo now also includes a native Swift/AppKit GUI that keeps the existing Python and PyTorch backend.

Run the GUI from source:

```bash
swift run nstar-gui
```

Build a `.app` bundle:

```bash
bash build_swift_app.sh
```

By default this creates:

- app bundle: `dist-swift/NStar.app`
- bundled backend: `dist-swift/NStar.app/Contents/Resources/backend/nstar`

Important:

- the GUI is native Swift, but generation still runs through Python and PyTorch
- the app needs a Python interpreter with `torch` and `midiutil` installed
- the first field in the GUI lets you point the app at that interpreter
- the AppKit target is set to macOS 10.13, but the practical minimum macOS version is still limited by the Python/PyTorch runtime you use

## Neural network usage

If you want to train or inspect the chord progression model directly:

```bash
.venv/bin/python neuralnetwork.py \
  --epochs 120 \
  --log-every 40 \
  --samples 3 \
  --checkpoint-path progression_model.pt \
  --retrain
```

This command:

- trains the GRU
- prints loss during training
- saves the checkpoint to `progression_model.pt`
- prints sampled chord progressions

To allow exact training-template outputs:

```bash
.venv/bin/python neuralnetwork.py --allow-training-template
```

## Useful CLI options

### `generation.py`

- `--target`: number of MIDI files to save
- `--output-dir`: destination directory for MIDI files
- `--producers`: number of worker processes
- `--queue-size`: queue size between producers and writer
- `--name-prefix`: filename prefix
- `--epochs`: progression-model training epochs
- `--hidden-size`: GRU hidden size
- `--embedding-size`: chord embedding size
- `--learning-rate`: optimizer learning rate
- `--model-seed`: torch seed for model training
- `--train-log-every`: print loss every N epochs; `0` disables logging
- `--model-path`: checkpoint path for the trained progression model
- `--retrain-model`: ignore saved checkpoint and retrain

### `neuralnetwork.py`

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
- `--allow-training-template`: allow output to exactly match a training template

## Output and checkpoints

- MIDI output folder default: `~/Documents/mid`
- Model checkpoint default: `progression_model.pt` in the project directory

The MIDI output directory is created automatically if it does not exist.

## Notes and limitations

- The progression model trains from a small hardcoded set of progression templates in `musicdata.py`.
- The neural net improves progression selection, but it does not learn the full song end to end.
- Melody generation is still rule-based.
- With more training data, the model would have a wider range of harmonic output.

## Typical workflow

1. Install dependencies.
2. Run `generation.py` with a small `--target`, such as `1` or `10`.
3. If you want fresh model weights, add `--retrain-model`.
4. If you want to inspect training behavior directly, run `neuralnetwork.py`.
5. Check the generated MIDI files in `~/Documents/mid`.
