from asqp_s2it.metrics import exact_quad_f1, exact_quad_f1_by_instance


def test_exact_quad_f1_requires_all_four_fields():
    gold = [{
        "aspect_term": "hotel",
        "aspect_category": "FACILITY",
        "opinion_term": "clean",
        "sentiment": "positive",
    }]
    changed_category = [{**gold[0], "aspect_category": "EXPERIENCE"}]
    assert exact_quad_f1(gold, changed_category)["f1"] == 0.0
    assert exact_quad_f1(gold, gold)["f1"] == 1.0


def test_exact_quad_f1_uses_instance_level_quad_sets_and_normalizes_case():
    gold = [{
        "aspect_term": "Hotel",
        "aspect_category": "FACILITY",
        "opinion_term": "very clean",
        "sentiment": "positive",
    }]
    predicted = [{
        "aspect_term": "hotel",
        "aspect_category": "facility",
        "opinion_term": "very   clean",
        "sentiment": "POSITIVE",
    }]
    assert exact_quad_f1(gold, predicted)["f1"] == 1.0


def test_implicit_terms_are_compared_as_null():
    quad = {
        "aspect_term": None,
        "aspect_category": "EXPERIENCE",
        "opinion_term": "peaceful",
        "sentiment": "positive",
    }
    assert exact_quad_f1([quad], [quad])["true_positives"] == 1


def test_quadruple_in_another_instance_is_not_a_match():
    quad = {
        "aspect_term": "hotel",
        "aspect_category": "FACILITY",
        "opinion_term": "clean",
        "sentiment": "positive",
    }
    result = exact_quad_f1_by_instance(
        {"sentence-1": [quad], "sentence-2": []},
        {"sentence-1": [], "sentence-2": [quad]},
    )
    assert result["true_positives"] == 0
    assert result["f1"] == 0.0
