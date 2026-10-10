from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data import category_labels, read_jsonl, write_jsonl
from .metrics import exact_quad_f1_by_instance
from .timing import format_duration


def _parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="asqp-s2it",
        description="S2IT aspect sentiment quadruple extraction",
    )
    commands = root.add_subparsers(dest="command", required=True)

    setup = commands.add_parser("prepare-parser", help="download Stanza dependency models")
    setup.add_argument("--languages", default="en", help="comma-separated Stanza language codes")

    language_setup = commands.add_parser(
        "prepare-language",
        help="detect and store language in JSONL before uploading data to Kaggle",
    )
    language_setup.add_argument("--input", required=True)
    language_setup.add_argument("--output", required=True)
    language_setup.add_argument("--seed", type=int, default=0)

    fit = commands.add_parser("train", help="train the three S2IT stages")
    fit.add_argument("--train", dest="train_file", default="asqp_data/train.jsonl")
    fit.add_argument("--dev", dest="dev_file", default="asqp_data/dev.jsonl")
    fit.add_argument("--output-dir", default="runs/s2it")
    fit.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    fit.add_argument("--language", default="auto", help="Stanza language code or auto")
    fit.add_argument("--epochs", type=int, default=5)
    fit.add_argument("--batch-size", type=int, default=4)
    fit.add_argument("--gradient-accumulation-steps", type=int, default=2)
    fit.add_argument("--learning-rate", type=float, default=5e-5)
    fit.add_argument("--lora-rank", type=int, default=32)
    fit.add_argument("--max-length", type=int, default=2048)
    fit.add_argument("--parser-gpu", action="store_true")

    predict = commands.add_parser("predict", help="predict quadruples and report exact-tuple F1")
    predict.add_argument("--input", default="asqp_data/test.jsonl")
    predict.add_argument("--model-dir", default="runs/s2it/final")
    predict.add_argument("--output", default="runs/s2it/predictions.jsonl")
    predict.add_argument("--language", default="auto")
    predict.add_argument("--parser-gpu", action="store_true")
    predict.add_argument("--max-new-tokens", type=int, default=512)
    predict.add_argument("--categories-file", help="JSON list; defaults to model's categories.json")

    evaluate = commands.add_parser("evaluate", help="evaluate JSONL predictions against gold")
    evaluate.add_argument("--gold", default="asqp_data/test.jsonl")
    evaluate.add_argument("--predictions", default="runs/s2it/predictions.jsonl")
    evaluate.add_argument(
        "--training-metrics",
        default="runs/s2it/final/training_metrics.json",
        help="training_metrics.json from the model run",
    )
    return root


def _prediction_quads(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = str(row.get("instance_id", ""))
            quads = row.get("quadruples")
            if not instance_id or not isinstance(quads, list):
                raise ValueError(f"{path}:{line_number}: expected instance_id and quadruples list")
            if instance_id in result:
                raise ValueError(f"{path}:{line_number}: duplicate instance_id {instance_id!r}")
            result[instance_id] = quads
    return result


def _evaluate(gold_path: str | Path, prediction_path: str | Path) -> dict[str, float | int]:
    gold = read_jsonl(gold_path)
    predicted = _prediction_quads(prediction_path)
    return exact_quad_f1_by_instance(
        {instance.instance_id: instance.quads for instance in gold},
        predicted,
    )


def _load_training_metrics(path: str | Path) -> dict[str, Any] | None:
    metrics_path = Path(path)
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict) or not isinstance(metrics.get("training_seconds"), (int, float)):
        raise ValueError(f"Invalid training metrics file: {metrics_path}")
    return metrics


def _print_scores(scores: dict[str, float | int], training_metrics: dict[str, Any] | None) -> None:
    result: dict[str, Any] = {"evaluation": scores}
    if training_metrics is not None:
        seconds = float(training_metrics["training_seconds"])
        result["training_time"] = {
            "formatted": training_metrics.get("training_time", format_duration(seconds)),
            "seconds": seconds,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _predict(args: argparse.Namespace) -> None:
    from .model import S2ITPredictor
    from .syntax import DependencyParser

    model_dir = Path(args.model_dir)
    categories_path = Path(args.categories_file) if args.categories_file else model_dir / "categories.json"
    if categories_path.exists():
        categories = json.loads(categories_path.read_text(encoding="utf-8"))
    else:
        categories = category_labels(read_jsonl("asqp_data/train.jsonl"))
    if not isinstance(categories, list) or not all(isinstance(item, str) for item in categories):
        raise ValueError(f"Category file must contain a JSON list of strings: {categories_path}")
    predictor = S2ITPredictor(
        model_dir,
        DependencyParser(args.language, args.parser_gpu),
    )
    instances = read_jsonl(args.input, require_gold=False)
    rows = []
    for index, instance in enumerate(instances, start=1):
        quadruples = predictor.predict(instance, categories, args.max_new_tokens)
        rows.append(
            {
                "instance_id": instance.instance_id,
                "text": instance.text,
                "language": instance.language,
                "quadruples": quadruples,
            }
        )
        if index % 100 == 0:
            print(f"Predicted {index}/{len(instances)}")
    write_jsonl(args.output, rows)
    if any(instance.quads for instance in instances):
        scores = _evaluate(args.input, args.output)
        _print_scores(scores, _load_training_metrics(model_dir / "training_metrics.json"))
    else:
        training_metrics = _load_training_metrics(model_dir / "training_metrics.json")
        if training_metrics is not None:
            _print_scores({}, training_metrics)
    print(f"Saved predictions to {args.output}")


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare-parser":
        from .syntax import download_parser_models

        download_parser_models([value.strip() for value in args.languages.split(",") if value.strip()])
    elif args.command == "prepare-language":
        from .language_prep import annotate_languages

        counts = annotate_languages(args.input, args.output, seed=args.seed)
        print(json.dumps({"language_counts": counts, "output": args.output}, ensure_ascii=False, indent=2))
    elif args.command == "train":
        from .syntax import DependencyParser
        from .training import train

        train(
            read_jsonl(args.train_file),
            read_jsonl(args.dev_file),
            args.output_dir,
            args.model,
            DependencyParser(args.language, args.parser_gpu),
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            max_length=args.max_length,
        )
    elif args.command == "predict":
        _predict(args)
    elif args.command == "evaluate":
        _print_scores(
            _evaluate(args.gold, args.predictions),
            _load_training_metrics(args.training_metrics),
        )


if __name__ == "__main__":
    main()
