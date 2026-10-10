from __future__ import annotations

import json

from .data import Instance
from .syntax import ParsedText

SENTIMENTS = ("negative", "neutral", "positive")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def extraction_prompt(text: str, relations: str, reverse: bool = False) -> str:
    task = "opinion-aspect" if reverse else "aspect-opinion"
    fields = (
        '{"pairs":[{"opinion_term":string|null,"aspect_term":string|null}]}'
        if reverse
        else '{"pairs":[{"aspect_term":string|null,"opinion_term":string|null}]}'
    )
    return (
        f"Task: extract {task} pairs.\n"
        "Given a sentence and its dependency relations, extract every explicit or implicit "
        "aspect and opinion pair. Use null for an implicit element. Preserve the exact text "
        "span and do not infer category or sentiment. Return only JSON with this schema: "
        f"{fields}.\n"
        f"Sentence: {text}\nDependency relations: {relations}"
    )


def classification_prompt(
    text: str,
    pairs: list[dict[str, str | None]],
    parsed: ParsedText,
    categories: list[str],
) -> str:
    local = []
    for pair in pairs:
        aspect = pair.get("aspect_term")
        opinion = pair.get("opinion_term")
        local.append(
            f"aspect: {aspect if aspect is not None else 'implicit'}, "
            f"{parsed.neighbors(aspect)}; "
            f"opinion: {opinion if opinion is not None else 'implicit'}, "
            f"{parsed.neighbors(opinion)}"
        )
    return (
        "Task: classify extracted aspect-opinion pairs.\n"
        "Given a sentence, one-hop dependency subgraphs, and candidate pairs, assign the "
        "aspect category and opinion sentiment. Keep the candidate aspect and opinion "
        "unchanged. Choose an exact category from the supplied category list and sentiment "
        "from negative, neutral, positive. Return only JSON with this schema: "
        '{"quadruples":[{"aspect_term":string|null,"aspect_category":string,'
        '"opinion_term":string|null,"sentiment":string}]}.\n'
        f"Sentence: {text}\nLocal subgraphs: {' | '.join(local)}\n"
        f"Candidates: {_json(pairs)}\nCategories: {_json(categories)}"
    )


def auxiliary_examples(
    instance: Instance,
    parsed: ParsedText,
    categories: list[str],
) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    pairs = [
        {"aspect_term": quad.aspect_term, "opinion_term": quad.opinion_term}
        for quad in instance.quads
    ]
    relation_text = parsed.global_relations()
    text = instance.text

    def add(
        task: str,
        candidates: list[dict[str, str | None]],
        output: list[dict[str, object]],
        *,
        use_local_subgraphs: bool = False,
    ) -> None:
        if use_local_subgraphs:
            subgraphs = [
                "; ".join(
                    f"{field.removesuffix('_term')}: "
                    f"{candidate.get(field) if candidate.get(field) is not None else 'implicit'}, "
                    f"{parsed.neighbors(candidate.get(field))}"
                    for field in ("aspect_term", "opinion_term")
                    if field in candidate
                )
                for candidate in candidates
            ]
            structure_label = "One-hop subgraphs"
            structure = " | ".join(subgraphs)
        else:
            structure_label = "Dependency relations"
            structure = relation_text
        prompt = (
            f"Task: {task}.\nSentence: {text}\n"
            f"{structure_label}: {structure}\n"
            f"Candidates: {_json(candidates)}\nCategories: {_json(categories)}\n"
            f"Sentiments: {_json(SENTIMENTS)}\n"
            "Return only JSON with a top-level 'labels' list using the candidate fields."
        )
        examples.append((prompt, _json({"labels": output})))

    aspect_candidates = [{"aspect_term": q.aspect_term} for q in instance.quads]
    opinion_candidates = [{"opinion_term": q.opinion_term} for q in instance.quads]
    add(
        "link aspect to opinion",
        aspect_candidates,
        [{"aspect_term": q.aspect_term, "opinion_term": q.opinion_term} for q in instance.quads],
    )
    add(
        "link opinion to aspect",
        opinion_candidates,
        [{"opinion_term": q.opinion_term, "aspect_term": q.aspect_term} for q in instance.quads],
    )
    for field in ("aspect_category", "sentiment"):
        for role in ("aspect_term", "opinion_term"):
            inputs = [
                {
                    "aspect_term": q.aspect_term,
                    "opinion_term": q.opinion_term,
                    role: getattr(q, role),
                }
                for q in instance.quads
            ]
            outputs = [
                {role: getattr(q, role), field: getattr(q, field)}
                for q in instance.quads
            ]
            add(
                f"classify {role.removesuffix('_term')} to {field}",
                inputs,
                outputs,
                use_local_subgraphs=True,
            )
    add(
        "classify pairs to category and sentiment",
        pairs,
        [quad.as_dict() for quad in instance.quads],
        use_local_subgraphs=True,
    )
    return examples
