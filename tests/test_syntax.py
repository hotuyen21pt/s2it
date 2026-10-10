from types import SimpleNamespace

import pytest

from asqp_s2it import syntax
from asqp_s2it.syntax import DependencyParser


def test_custom_stanza_models_are_loaded_from_direct_paths(tmp_path, monkeypatch):
    model_paths = {}
    for setting in ("tokenize_model_path", "pos_model_path", "lemma_model_path", "depparse_model_path"):
        path = tmp_path / f"{setting}.pt"
        path.touch()
        model_paths[setting] = str(path)
    resources_file = tmp_path / "resources.json"
    resources_file.write_text("{}", encoding="utf-8")
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return lambda text: SimpleNamespace(
            sentences=[SimpleNamespace(words=[SimpleNamespace(text=text, head=0)])]
        )

    monkeypatch.setattr(syntax.stanza, "Pipeline", fake_pipeline)
    DependencyParser._pipeline.cache_clear()
    try:
        parser = DependencyParser(
            model_paths=model_paths,
            model_language="en",
            resources_filepath=str(resources_file),
        )
        parsed = parser.parse("hello", language="en")
    finally:
        DependencyParser._pipeline.cache_clear()

    assert parsed.words == ("hello",)
    assert captured["processors"] == {
        "tokenize": "default",
        "pos": "default",
        "depparse": "default",
    }
    assert captured["package"] is None
    assert captured["allow_unknown_language"] is True
    assert captured["resources_filepath"] == str(resources_file)
    assert captured["tokenize_model_path"] == model_paths["tokenize_model_path"]


def test_custom_stanza_model_rejects_mismatched_language(tmp_path):
    model_paths = {}
    for setting in ("tokenize_model_path", "pos_model_path", "lemma_model_path", "depparse_model_path"):
        path = tmp_path / f"{setting}.pt"
        path.touch()
        model_paths[setting] = str(path)
    resources_file = tmp_path / "resources.json"
    resources_file.write_text("{}", encoding="utf-8")

    parser = DependencyParser(
        model_paths=model_paths,
        model_language="en",
        resources_filepath=str(resources_file),
    )

    with pytest.raises(ValueError, match="supports 'en'.*tagged 'es'"):
        parser.parse("hola", language="es")


def test_resources_directory_is_passed_for_each_record_language(tmp_path, monkeypatch):
    captured = []

    def fake_pipeline(**kwargs):
        captured.append(kwargs)
        return lambda text: SimpleNamespace(
            sentences=[SimpleNamespace(words=[SimpleNamespace(text=text, head=0)])]
        )

    monkeypatch.setattr(syntax.stanza, "Pipeline", fake_pipeline)
    DependencyParser._pipeline.cache_clear()
    try:
        parser = DependencyParser(resources_dir=str(tmp_path))
        parser.parse("hello", language="en")
        parser.parse("bonjour", language="fr")
        parser.parse("你好", language="zh")
    finally:
        DependencyParser._pipeline.cache_clear()

    assert [pipeline["lang"] for pipeline in captured] == ["en", "fr", "zh-hans"]
    assert all(pipeline["dir"] == str(tmp_path) for pipeline in captured)
    assert all(pipeline["download_method"] is None for pipeline in captured)
    assert all(
        pipeline["processors"] == {
            "tokenize": "default",
            "pos": "default",
            "lemma": "identity",
            "depparse": "default",
        }
        for pipeline in captured
    )
