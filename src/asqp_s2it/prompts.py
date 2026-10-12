from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Iterable, Sequence

from .data import Instance

if TYPE_CHECKING:  # `syntax` pulls in Stanza; prompts only need the type.
    from .syntax import ParsedText

SENTIMENTS = ("negative", "neutral", "positive")

# S2IT (Table 4) shows that symbol/bracket templates degrade performance and
# that the dependency tree, the candidates and the labels must all be rendered
# as natural language so they match the pretraining objective of the LLM.
# Every prompt and every target below follows the templates in Appendix A.2.
IMPLICIT_TOKEN = "NULL"
_IMPLICIT_VALUES = {"", "null", "none", "nil", "implicit", "n/a"}

ASPECT_TERM = "aspect_term"
OPINION_TERM = "opinion_term"
ASPECT_CATEGORY = "aspect_category"
SENTIMENT = "sentiment"

PAIR_FIELDS = (ASPECT_TERM, OPINION_TERM)
REVERSE_PAIR_FIELDS = (OPINION_TERM, ASPECT_TERM)
QUAD_FIELDS = (ASPECT_TERM, OPINION_TERM, ASPECT_CATEGORY, SENTIMENT)

_LABEL_TO_FIELD = {
    "aspect": ASPECT_TERM,
    "opinion": OPINION_TERM,
    "category": ASPECT_CATEGORY,
    "sentiment": SENTIMENT,
}
_FIELD_TO_LABEL = {field: label for label, field in _LABEL_TO_FIELD.items()}
_LABEL_PATTERN = re.compile(
    r"(?:^|[,;.\s])(aspect|opinion|category|sentiment)\s*:\s*",
    re.IGNORECASE,
)
_EMPTY_RECORD_SET = "none"


# --------------------------------------------------------------------------
# Natural-language serialization of labels ("aspect: service, opinion: ok | ...")
# --------------------------------------------------------------------------


def _render_value(value: object) -> str:
    if value is None:
        return IMPLICIT_TOKEN
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or IMPLICIT_TOKEN


def render_records(
    records: Iterable[dict[str, object]],
    fields: Sequence[str],
) -> str:
    """Serialize labelled records the way S2IT Appendix A.2 writes its outputs."""
    rendered = [
        ", ".join(f"{_FIELD_TO_LABEL[field]}: {_render_value(row.get(field))}" for field in fields)
        for row in records
    ]
    return " | ".join(rendered) if rendered else _EMPTY_RECORD_SET


def _clean_value(raw: str) -> str | None:
    text = raw.strip().strip("\"'").strip()
    text = re.sub(r"[,;.]+$", "", text).strip()
    return None if text.casefold() in _IMPLICIT_VALUES else text


def _parse_segment(segment: str) -> dict[str, str | None]:
    matches = list(_LABEL_PATTERN.finditer(segment))
    record: dict[str, str | None] = {}
    for position, match in enumerate(matches):
        field = _LABEL_TO_FIELD[match.group(1).casefold()]
        start = match.end()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(segment)
        record[field] = _clean_value(segment[start:end])
    return record


def _keep(record: dict[str, str | None], fields: Sequence[str]) -> bool:
    if any(field not in record for field in fields):
        return False
    return not any(
        record[field] is None for field in fields if field in (ASPECT_CATEGORY, SENTIMENT)
    )


_JSON_OBJECT = re.compile(r"\{[^{}]*\}")


def _json_field_pattern(field: str) -> re.Pattern[str]:
    return re.compile(
        rf'"{field}"\s*:\s*(null|"(?:[^"\\]|\\.)*")',
        re.IGNORECASE,
    )


_JSON_FIELD_PATTERNS = {
    field: _json_field_pattern(field)
    for field in (ASPECT_TERM, OPINION_TERM, ASPECT_CATEGORY, SENTIMENT)
}


def _parse_json_records(text: str, fields: Sequence[str]) -> list[dict[str, str | None]]:
    """Fallback for adapters trained on the earlier JSON target format.

    Scanned with a regex rather than `json.loads` so a single unescaped quote or
    a completion truncated by `max_new_tokens` costs one record, not the run.
    """
    records: list[dict[str, str | None]] = []
    for block in _JSON_OBJECT.findall(text):
        record: dict[str, str | None] = {}
        for field in fields:
            match = _JSON_FIELD_PATTERNS[field].search(block)
            if match is None:
                continue
            raw = match.group(1)
            if raw.casefold() == "null":
                record[field] = None
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw.strip('"')
            record[field] = _clean_value(str(value))
        if _keep(record, fields):
            records.append({field: record[field] for field in fields})
    return records


def parse_records(text: str, fields: Sequence[str]) -> list[dict[str, str | None]]:
    """Parse a natural-language label string, skipping malformed segments.

    Generation is never guaranteed to be well formed, so a segment the model
    mangled is dropped instead of aborting the whole prediction run.
    """
    cleaned = re.sub(r"```(?:\w+)?", " ", text)
    cleaned = re.sub(r"^\s*output\s*:", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or cleaned.casefold().rstrip(".") == _EMPTY_RECORD_SET:
        return []
    if "{" in cleaned and '"' in cleaned:
        json_records = _parse_json_records(cleaned, fields)
        if json_records:
            return json_records
    records: list[dict[str, str | None]] = []
    for segment in cleaned.split("|"):
        record = _parse_segment(segment)
        if _keep(record, fields):
            records.append({field: record[field] for field in fields})
    return records


def render_pairs(pairs: Iterable[dict[str, object]], *, reverse: bool = False) -> str:
    return render_records(pairs, REVERSE_PAIR_FIELDS if reverse else PAIR_FIELDS)


def parse_pairs(text: str, *, reverse: bool = False) -> list[dict[str, str | None]]:
    parsed = parse_records(text, REVERSE_PAIR_FIELDS if reverse else PAIR_FIELDS)
    return [{ASPECT_TERM: row[ASPECT_TERM], OPINION_TERM: row[OPINION_TERM]} for row in parsed]


def render_quads(quads: Iterable[dict[str, object]]) -> str:
    return render_records(quads, QUAD_FIELDS)


def parse_quads(text: str) -> list[dict[str, str | None]]:
    return parse_records(text, QUAD_FIELDS)


# --------------------------------------------------------------------------
# Stage prompts
# --------------------------------------------------------------------------


def _category_line(categories: Sequence[str]) -> str:
    return "category list: " + " | ".join(categories)


def _candidate_values(candidates: Sequence[dict[str, object]], field: str) -> str:
    """Bare candidate list of the node classification tasks: `service | bathroom`."""
    return " | ".join(_render_value(row.get(field)) for row in candidates) or _EMPTY_RECORD_SET


def _subgraph_description(
    parsed: "ParsedText",
    candidate: dict[str, object],
    fields: Sequence[str],
) -> str:
    """One-hop subgraph wording from S2IT Eq. 4: `x, which is connected to (...)`."""
    return " ".join(
        f"{_FIELD_TO_LABEL[field]}: {_render_value(candidate.get(field))}, "
        f"{parsed.neighbors(candidate.get(field))}."
        for field in fields
    )


def extraction_prompt(text: str, relations: str, reverse: bool = False) -> str:
    """Step 1, Global Syntax-guided Extraction (S2IT Sec. 2.2)."""
    first, second = ("opinion", "aspect") if reverse else ("aspect", "opinion")
    return (
        f"task: (extract {first}, {second})\n"
        f"Instruction:\nGiven a sentence and related dependency relations, extract "
        f"{first} and {second} (both implicit and explicit) from the sentence and return "
        f"pair({first}, {second}). Pay attention to the one or multi hop dependency "
        f"relationships between aspect and opinion. Write {IMPLICIT_TOKEN} for an implicit "
        f"element and keep the exact span for an explicit one. Separate pairs with ' | ' "
        f"and answer with the pairs only.\n"
        f"Input:\nsentence: {text}\ndependency relation: {relations}\n"
        f"Output:"
    )


def classification_prompt(
    text: str,
    pairs: list[dict[str, str | None]],
    parsed: "ParsedText",
    categories: list[str],
) -> str:
    """Step 2, Local Syntax-guided Classification (S2IT Sec. 2.3)."""
    subgraph = " | ".join(
        _subgraph_description(parsed, pair, PAIR_FIELDS) for pair in pairs
    )
    return (
        "task: (classification (aspect, opinion) to (category, sentiment))\n"
        "Instruction:\nGiven a sentence, related dependency relations (will be presented "
        "in the form of subgraph) and (aspect, opinion) candidates, determine the category "
        "of the aspect and the sentiment (positive, neutral, negative) of the opinion and "
        "return the quadruple(aspect, opinion, category, sentiment). Keep each candidate "
        "aspect and opinion unchanged, choose a category from the category list, and "
        "separate quadruples with ' | '. Answer with the quadruples only.\n"
        f"Input:\nsentence: {text}\nsubgraph: {subgraph}\n"
        f"candidate: {render_pairs(pairs)}\n{_category_line(categories)}\n"
        "Output:"
    )


def auxiliary_examples(
    instance: Instance,
    parsed: "ParsedText",
    categories: list[str],
) -> list[tuple[str, str]]:
    """Step 3, Fine-grained Structure Instruction Tuning (S2IT Sec. 2.4).

    Element link prediction in both directions plus node classification of each
    sentiment element, exactly the six auxiliary tasks of Appendix A.2.
    """
    examples: list[tuple[str, str]] = []
    relations = parsed.global_relations()
    text = instance.text
    pairs = [
        {ASPECT_TERM: quad.aspect_term, OPINION_TERM: quad.opinion_term}
        for quad in instance.quads
    ]

    def add_link(source: str, target: str) -> None:
        source_label, target_label = _FIELD_TO_LABEL[source], _FIELD_TO_LABEL[target]
        candidates = [{source: getattr(quad, source)} for quad in instance.quads]
        outputs = [
            {source: getattr(quad, source), target: getattr(quad, target)}
            for quad in instance.quads
        ]
        prompt = (
            f"task: (linking {source_label} to {target_label})\n"
            f"Instruction:\nGiven a sentence, related dependency relations and known "
            f"{source_label}s, determine the {target_label} (both implicit and explicit) "
            f"related to the each {source_label} from dependency relation and return the "
            f"pair({source_label}, {target_label}). Separate pairs with ' | ' and answer "
            f"with the pairs only.\n"
            f"Input:\nsentence: {text}\ndependency relation: {relations}\n"
            f"candidates: {render_records(candidates, (source,))}\n"
            f"Output:"
        )
        examples.append((prompt, render_records(outputs, (source, target))))

    def add_node_classification(role: str, field: str) -> None:
        role_label, field_label = _FIELD_TO_LABEL[role], _FIELD_TO_LABEL[field]
        # Node classification only ever sees the element being classified; handing
        # the model the paired element too would leak the link it must not use.
        candidates = [{role: getattr(quad, role)} for quad in instance.quads]
        outputs = [
            {role: getattr(quad, role), field: getattr(quad, field)}
            for quad in instance.quads
        ]
        subgraph = " | ".join(
            _subgraph_description(parsed, candidate, (role,)) for candidate in candidates
        )
        allowed = (
            _category_line(categories)
            if field == ASPECT_CATEGORY
            else "sentiment list: " + " | ".join(SENTIMENTS)
        )
        prompt = (
            f"task: (classification {role_label} to {field_label})\n"
            f"Instruction:\nGiven a sentence, related dependency relations (will be "
            f"presented in the form of subgraph) and known {role_label}s (both implicit "
            f"and explicit), determine the {field_label} related to the each {role_label}s "
            f"from dependency relation and return pair ({role_label}, {field_label}). "
            f"Separate pairs with ' | ' and answer with the pairs only.\n"
            f"Input:\nsentence: {text}\nsubgraph: {subgraph}\n"
            f"candidate {role_label}: {_candidate_values(candidates, role)}\n{allowed}\n"
            f"Output:"
        )
        examples.append((prompt, render_records(outputs, (role, field))))

    add_link(ASPECT_TERM, OPINION_TERM)
    add_link(OPINION_TERM, ASPECT_TERM)
    for field in (ASPECT_CATEGORY, SENTIMENT):
        for role in (ASPECT_TERM, OPINION_TERM):
            add_node_classification(role, field)
    # Replaying the Step 2 task keeps the quadruple objective alive while the
    # adapter is tuned on the auxiliary structure tasks.
    examples.append(
        (
            classification_prompt(text, pairs, parsed, categories),
            render_quads([quad.as_dict() for quad in instance.quads]),
        )
    )
    return examples
