import json
import sys
from types import ModuleType

from asqp_s2it.data import read_jsonl
from asqp_s2it.language_prep import annotate_languages


def test_read_jsonl_preserves_precomputed_language(tmp_path):
    source = tmp_path / "sample.jsonl"
    source.write_text(
        json.dumps(
            {
                "instance_id": "s1",
                "text": "Buenos días",
                "language": "es",
                "quads": [
                    {
                        "aspect_term": "hotel",
                        "aspect_category": "FACILITY",
                        "opinion_term": "bueno",
                        "sentiment": "positive",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    instance = read_jsonl(source)[0]
    assert instance.language == "es"


def test_missing_language_remains_none_for_explicit_parser_configuration(tmp_path):
    source = tmp_path / "sample.jsonl"
    source.write_text(
        json.dumps({"instance_id": "s1", "text": "hello", "quads": []}) + "\n",
        encoding="utf-8",
    )
    assert read_jsonl(source)[0].language is None


def test_language_prep_ignores_existing_language_and_uses_langdetect(tmp_path, monkeypatch):
    class LangDetectException(Exception):
        pass

    fake_langdetect = ModuleType("langdetect")
    fake_langdetect.LangDetectException = LangDetectException

    class DetectorFactory:
        seed = None

    fake_langdetect.DetectorFactory = DetectorFactory

    def detect(text):
        if not any(character.isalpha() for character in text):
            raise LangDetectException("No features in text")
        return "es"

    fake_langdetect.detect = detect
    monkeypatch.setitem(sys.modules, "langdetect", fake_langdetect)

    source = tmp_path / "source.jsonl"
    destination = tmp_path / "tagged.jsonl"
    source.write_text(
        "\n".join(
            (
                json.dumps({"text": "Texto de prueba", "language": "en", "quads": []}),
                json.dumps({"text": "!!!", "language": "fr", "quads": []}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    counts = annotate_languages(source, destination)
    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]

    assert counts == {"es": 2}
    assert [row["language"] for row in rows] == ["es", "es"]
