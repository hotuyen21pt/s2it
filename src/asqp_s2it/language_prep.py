from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _detect_language(text: str, detect: Any) -> str | None:
    if not text.strip():
        return None
    from langdetect import LangDetectException

    try:
        return detect(text)
    except LangDetectException:
        return None


def annotate_languages(
    source: str | Path,
    destination: str | Path,
    seed: int = 0,
) -> dict[str, int]:
    try:
        from langdetect import DetectorFactory, detect
    except ImportError as error:
        raise RuntimeError(
            "Language annotation requires the optional `langdetect` package. "
            "Install it on the data-preparation machine with "
            "`pip install -e '.[language-detection]'`. Kaggle training/inference "
            "does not require langdetect once JSONL has been annotated."
        ) from error

    DetectorFactory.seed = seed
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[int, dict[str, Any], str | None]] = []
    detected_counts: Counter[str] = Counter()
    with source_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source_path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"{source_path}:{line_number}: expected a JSON object with text")
            language = _detect_language(row["text"], detect)
            if language is not None:
                detected_counts[language] += 1
            rows.append((line_number, row, language))

    if rows and not detected_counts:
        raise ValueError(
            f"{source_path}: langdetect could not detect a language from any record; "
            "provide at least one informative text record."
        )
    fallback_language = (
        min(detected_counts, key=lambda language: (-detected_counts[language], language))
        if detected_counts
        else None
    )
    counts: Counter[str] = Counter()
    fallback_lines: list[int] = []
    with destination_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for line_number, row, language in rows:
            if language is None:
                language = fallback_language
                fallback_lines.append(line_number)
            row["language"] = language
            counts[language] += 1
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    if fallback_lines:
        print(
            f"Warning: {len(fallback_lines)} records in {source_path} had empty or "
            f"undetectable text; assigned the most common language {fallback_language!r}. "
            f"First affected lines: {fallback_lines[:10]}"
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect sentence languages once and store them in JSONL records."
    )
    parser.add_argument("--input", required=True, help="source JSONL file")
    parser.add_argument("--output", required=True, help="annotated output JSONL file")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    counts = annotate_languages(args.input, args.output, seed=args.seed)
    print(json.dumps({"language_counts": counts, "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
