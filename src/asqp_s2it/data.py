from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Quad:
    aspect_term: str | None
    aspect_category: str
    opinion_term: str | None
    sentiment: str

    def as_dict(self) -> dict[str, str | None]:
        return {
            "aspect_term": self.aspect_term,
            "aspect_category": self.aspect_category,
            "opinion_term": self.opinion_term,
            "sentiment": self.sentiment,
        }


@dataclass(frozen=True)
class Instance:
    instance_id: str
    text: str
    quads: tuple[Quad, ...]
    language: str | None = None


def _parse_quad(raw: dict[str, Any], source: Path, line_number: int) -> Quad:
    required = ("aspect_term", "aspect_category", "opinion_term", "sentiment")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"{source}:{line_number}: quad missing fields: {', '.join(missing)}")

    aspect = raw["aspect_term"]
    opinion = raw["opinion_term"]
    category = raw["aspect_category"]
    sentiment = raw["sentiment"]
    if aspect is not None and not isinstance(aspect, str):
        raise ValueError(f"{source}:{line_number}: aspect_term must be a string or null")
    if opinion is not None and not isinstance(opinion, str):
        raise ValueError(f"{source}:{line_number}: opinion_term must be a string or null")
    if not isinstance(category, str) or not category:
        raise ValueError(f"{source}:{line_number}: aspect_category must be a non-empty string")
    if not isinstance(sentiment, str) or not sentiment:
        raise ValueError(f"{source}:{line_number}: sentiment must be a non-empty string")
    return Quad(aspect, category, opinion, sentiment)


def read_jsonl(path: str | Path, *, require_gold: bool = True) -> list[Instance]:
    source = Path(path)
    records: list[Instance] = []
    seen_ids: set[str] = set()
    with source.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"{source}:{line_number}: each JSONL record must be an object")
            text = raw.get("text")
            instance_id = raw.get("instance_id", f"{source.stem}-{line_number}")
            language = raw.get("language")
            if not isinstance(text, str):
                raise ValueError(f"{source}:{line_number}: text must be a string")
            if language is not None and (not isinstance(language, str) or not language):
                raise ValueError(f"{source}:{line_number}: language must be a non-empty string")
            if str(instance_id) in seen_ids:
                raise ValueError(f"{source}:{line_number}: duplicate instance_id {instance_id!r}")
            seen_ids.add(str(instance_id))
            if require_gold and "quads" not in raw:
                raise ValueError(f"{source}:{line_number}: missing gold 'quads' field")
            raw_quads = raw.get("quads", [])
            if require_gold and not isinstance(raw_quads, list):
                raise ValueError(f"{source}:{line_number}: quads must be a list")
            if not isinstance(raw_quads, list):
                raw_quads = []
            quads = tuple(
                _parse_quad(item, source, line_number)
                for item in raw_quads
                if isinstance(item, dict)
            )
            if require_gold and len(quads) != len(raw_quads):
                raise ValueError(f"{source}:{line_number}: every quad must be an object")
            records.append(Instance(str(instance_id), text, quads, language))
    return records


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def category_labels(instances: Iterable[Instance]) -> list[str]:
    return sorted({quad.aspect_category for item in instances for quad in item.quads})
