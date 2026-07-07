from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import json
import platform
import sys
import traceback
from pathlib import Path

from appconfig import DEFAULT_OUTPUT_DIR
from modelprofiles import DEFAULT_MODEL_NAME, get_model_profile, model_checkpoint_path, model_names


BRIDGE_VERSION = 1


def _module_status(module_name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "error": str(exc),
        }

    return {
        "available": True,
        "version": getattr(module, "__version__", None),
        "error": None,
    }


def _status_payload() -> dict[str, object]:
    torch_status = _module_status("torch")
    midiutil_status = _module_status("midiutil")
    models: list[dict[str, object]] = []

    for model_name in model_names():
        checkpoint_path = model_checkpoint_path(model_name)
        profile = get_model_profile(model_name)
        models.append(
            {
                "name": model_name,
                "seed": profile.seed,
                "hidden_size": profile.hidden_size,
                "embedding_size": profile.embedding_size,
                "epochs": profile.epochs,
                "learning_rate": profile.learning_rate,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_exists": checkpoint_path.exists(),
            }
        )

    return {
        "ok": True,
        "bridge_version": BRIDGE_VERSION,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "default_model_name": DEFAULT_MODEL_NAME,
        "default_output_dir": str(DEFAULT_OUTPUT_DIR),
        "generation_available": torch_status["available"] and midiutil_status["available"],
        "dependencies": {
            "torch": torch_status,
            "midiutil": midiutil_status,
        },
        "models": models,
    }


def _generate_payload(args: argparse.Namespace) -> dict[str, object]:
    try:
        from generation import run_generation
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Unable to load generation backend: {exc}",
        }

    try:
        with contextlib.redirect_stdout(sys.stderr):
            generation_kwargs = {
                "model_name": args.model_name,
                "target": args.target,
                "output_dir": Path(args.output_dir).expanduser().resolve(),
                "producers": args.producers,
                "queue_size": args.queue_size,
                "name_prefix": args.name_prefix,
                "epochs": args.epochs,
                "hidden_size": args.hidden_size,
                "embedding_size": args.embedding_size,
                "learning_rate": args.learning_rate,
                "model_seed": args.model_seed,
                "train_log_every": args.train_log_every,
                "model_path": args.model_path,
            }

            if "retrain_model" in inspect.signature(run_generation).parameters:
                generation_kwargs["retrain_model"] = True

            saved_paths = run_generation(**generation_kwargs)
    except KeyboardInterrupt:
        return {
            "ok": False,
            "error": "Generation cancelled.",
        }
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return {
            "ok": False,
            "error": str(exc),
        }

    return {
        "ok": True,
        "saved_count": len(saved_paths),
        "saved_files": [str(path) for path in saved_paths],
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "model_name": args.model_name,
        "retrained": True,
    }


def _train_payload(args: argparse.Namespace) -> dict[str, object]:
    try:
        from neuralnetwork import (
            configure_progression_model,
            get_progression_model,
            resolved_checkpoint_path,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Unable to load training backend: {exc}",
        }

    target_models = list(model_names()) if args.train_all_models else [args.model_name]
    results: list[dict[str, object]] = []

    for model_name in target_models:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                # Retraining is the whole point of this command, so always force
                # a fresh training run regardless of any existing checkpoint.
                config = configure_progression_model(
                    model_name=model_name,
                    hidden_size=args.hidden_size,
                    embedding_size=args.embedding_size,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    seed=args.model_seed,
                    log_every=args.train_log_every,
                    checkpoint_path=(args.model_path or None) if not args.train_all_models else None,
                    force_retrain=True,
                )
                model = get_progression_model(config)
        except KeyboardInterrupt:
            results.append({"model_name": model_name, "ok": False, "error": "Training cancelled."})
            break
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            results.append({"model_name": model_name, "ok": False, "error": str(exc)})
            continue

        checkpoint_path = resolved_checkpoint_path(config)
        results.append(
            {
                "model_name": model.model_name,
                "ok": True,
                "epochs": model.epochs,
                "final_loss": model.final_loss,
                "checkpoint_path": str(checkpoint_path),
            }
        )

    overall_ok = bool(results) and all(result["ok"] for result in results)
    return {
        "ok": overall_ok,
        "models": results,
    }


def _print_payload(payload: dict[str, object]) -> None:
    json.dump(payload, sys.stdout, indent=None, separators=(",", ":"))
    sys.stdout.write("\n")
    sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JSON bridge for the NStar Swift GUI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Print backend runtime details as JSON.")

    generate_parser = subparsers.add_parser("generate", help="Run song generation and print the result as JSON.")
    generate_parser.add_argument("--model-name", choices=model_names(), default=DEFAULT_MODEL_NAME)
    generate_parser.add_argument("--target", type=int, default=1)
    generate_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    generate_parser.add_argument("--producers", type=int, default=1)
    generate_parser.add_argument("--queue-size", type=int, default=256)
    generate_parser.add_argument("--name-prefix", default="")
    generate_parser.add_argument("--epochs", type=int, default=600)
    generate_parser.add_argument("--hidden-size", type=int, default=32)
    generate_parser.add_argument("--embedding-size", type=int, default=16)
    generate_parser.add_argument("--learning-rate", type=float, default=0.01)
    generate_parser.add_argument("--model-seed", type=int, default=None)
    generate_parser.add_argument("--train-log-every", type=int, default=100)
    generate_parser.add_argument("--model-path", default="")
    generate_parser.add_argument(
        "--retrain-model",
        action="store_true",
        help="Compatibility flag. Generate always retrains the model before output.",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train (or retrain) one or all progression model checkpoints and print the result as JSON.",
    )
    train_parser.add_argument("--model-name", choices=model_names(), default=DEFAULT_MODEL_NAME)
    train_parser.add_argument("--epochs", type=int, default=600)
    train_parser.add_argument("--hidden-size", type=int, default=32)
    train_parser.add_argument("--embedding-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=0.01)
    train_parser.add_argument("--model-seed", type=int, default=None)
    train_parser.add_argument("--train-log-every", type=int, default=100)
    train_parser.add_argument("--model-path", default="")
    train_parser.add_argument(
        "--train-all-models",
        action="store_true",
        help="Retrain every named model profile instead of just --model-name.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "status":
        _print_payload(_status_payload())
        return

    if args.command == "generate":
        _print_payload(_generate_payload(args))
        return

    if args.command == "train":
        _print_payload(_train_payload(args))
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
