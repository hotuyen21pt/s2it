from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
import unicodedata

from .data import Quad

LABEL_FIELDS = ("aspect_term", "aspect_category", "opinion_term", "sentiment")


def _normalize(value: object) -> str | None:
    if value is None:
        return None
    text = unicodedata.normalize("NFC", str(value))
    return re.sub(r"\s+", " ", text).strip().casefold()


def quad_key(quad: Quad | Mapping[str, object]) -> tuple[str | None, ...]:
    values = (
        quad.aspect_term,
        quad.aspect_category,
        quad.opinion_term,
        quad.sentiment,
    ) if isinstance(quad, Quad) else tuple(quad.get(field) for field in LABEL_FIELDS)
    return tuple(_normalize(value) for value in values)


def exact_quad_f1(
    gold: Iterable[Quad | Mapping[str, object]],
    predicted: Iterable[Quad | Mapping[str, object]],
) -> dict[str, float | int]:
    gold_set = {quad_key(quad) for quad in gold}
    predicted_set = {quad_key(quad) for quad in predicted}
    true_positives = len(gold_set & predicted_set)
    precision = true_positives / len(predicted_set) if predicted_set else 0.0
    recall = true_positives / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "predicted": len(predicted_set),
        "gold": len(gold_set),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def exact_quad_f1_by_instance(
    gold: Mapping[str, Iterable[Quad | Mapping[str, object]]],
    predicted: Mapping[str, Iterable[Quad | Mapping[str, object]]],
) -> dict[str, float | int]:
    if set(gold) != set(predicted):
        missing = sorted(set(gold) - set(predicted))
        extra = sorted(set(predicted) - set(gold))
        raise ValueError(
            f"Gold/prediction instance IDs differ: {len(missing)} missing, {len(extra)} extra"
        )
    totals = {"true_positives": 0, "predicted": 0, "gold": 0}
    for instance_id, gold_quads in gold.items():
        scores = exact_quad_f1(gold_quads, predicted[instance_id])
        for key in totals:
            totals[key] += int(scores[key])
    precision = totals["true_positives"] / totals["predicted"] if totals["predicted"] else 0.0
    recall = totals["true_positives"] / totals["gold"] if totals["gold"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**totals, "precision": precision, "recall": recall, "f1": f1}
