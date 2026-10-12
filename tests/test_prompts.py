from asqp_s2it.data import Instance, Quad
from asqp_s2it.prompts import (
    auxiliary_examples,
    classification_prompt,
    extraction_prompt,
    parse_pairs,
    parse_quads,
    render_pairs,
    render_quads,
)


class StubParse:
    """Duck-typed `ParsedText` so prompt tests do not need a Stanza install."""

    def global_relations(self) -> str:
        return "root depend service | service modify ok | bathroom modify filthy"

    def neighbors(self, term: str | None) -> str:
        if term is None:
            return "implicit (no token span)"
        return "which is connected to (bathroom, ok) within one hop"


QUADS = [
    {
        "aspect_term": "service",
        "opinion_term": "ok",
        "aspect_category": "service general",
        "sentiment": "negative",
    },
    {
        "aspect_term": None,
        "opinion_term": "filthy",
        "aspect_category": "ambience general",
        "sentiment": "negative",
    },
]


def make_instance() -> Instance:
    return Instance(
        "s-1",
        "service ok but unfriendly , filthy bathroom .",
        tuple(
            Quad(q["aspect_term"], q["aspect_category"], q["opinion_term"], q["sentiment"])
            for q in QUADS
        ),
        "en",
    )


def test_targets_use_the_papers_natural_language_template():
    rendered = render_quads(QUADS)
    assert rendered == (
        "aspect: service, opinion: ok, category: service general, sentiment: negative | "
        "aspect: NULL, opinion: filthy, category: ambience general, sentiment: negative"
    )
    assert parse_quads(rendered) == QUADS


def test_pairs_round_trip_in_both_generation_directions():
    forward = render_pairs(QUADS)
    assert forward == "aspect: service, opinion: ok | aspect: NULL, opinion: filthy"
    assert parse_pairs(forward) == [
        {"aspect_term": "service", "opinion_term": "ok"},
        {"aspect_term": None, "opinion_term": "filthy"},
    ]
    reverse = render_pairs(QUADS, reverse=True)
    assert reverse.startswith("opinion: ok, aspect: service")
    assert parse_pairs(reverse, reverse=True) == parse_pairs(forward)


def test_malformed_output_is_skipped_instead_of_raising():
    # The Kaggle run died here: an unescaped quote inside a term plus a
    # completion cut short by max_new_tokens.
    truncated = (
        '{"quadruples":[{"aspect_term":"the 15" screen","opinion_term":null,'
        '"aspect_category":"display","sentiment":"negative"},'
        '{"aspect_term":"x","opinion_term":"y","aspect_category":"c","sentiment":"positive"}'
    )
    assert len(parse_quads(truncated)) == 2
    assert parse_quads("I am sorry, I cannot help with that.") == []
    assert parse_quads("none") == []
    assert parse_quads("aspect: wifi, opinion: NULL, category: tech, sentiment: negative") == [
        {
            "aspect_term": "wifi",
            "opinion_term": None,
            "aspect_category": "tech",
            "sentiment": "negative",
        }
    ]


def test_quads_missing_a_label_are_dropped_not_half_parsed():
    assert parse_quads("aspect: wifi, opinion: slow") == []
    assert parse_quads("aspect: wifi, opinion: slow, category: NULL, sentiment: negative") == []


def test_prompts_carry_the_syntax_view_each_stage_needs():
    parsed = StubParse()
    extraction = extraction_prompt("service ok .", parsed.global_relations())
    assert "dependency relation: root depend service" in extraction
    assert "subgraph" not in extraction

    classification = classification_prompt("service ok .", QUADS, parsed, ["service general"])
    assert "which is connected to (bathroom, ok) within one hop" in classification
    assert "candidate: aspect: service, opinion: ok" in classification
    assert "category list: service general" in classification


def test_node_classification_never_shows_the_element_it_must_not_use():
    examples = auxiliary_examples(make_instance(), StubParse(), ["service general"])
    tasks = [prompt.splitlines()[0] for prompt, _ in examples]
    assert tasks[:6] == [
        "task: (linking aspect to opinion)",
        "task: (linking opinion to aspect)",
        "task: (classification aspect to category)",
        "task: (classification opinion to category)",
        "task: (classification aspect to sentiment)",
        "task: (classification opinion to sentiment)",
    ]
    aspect_to_category = next(
        prompt for prompt, _ in examples if "classification aspect to category" in prompt
    )
    assert "candidate aspect: service | NULL" in aspect_to_category
    assert "opinion:" not in aspect_to_category.split("Input:")[1]

    link_prompt, link_target = examples[0]
    assert "dependency relation:" in link_prompt
    assert link_target == "aspect: service, opinion: ok | aspect: NULL, opinion: filthy"
