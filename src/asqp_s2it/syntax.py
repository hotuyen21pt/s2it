from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import stanza

_LANGUAGE_ALIASES = {"zh": "zh-hans", "zh-cn": "zh-hans", "zh-tw": "zh-hant", "no": "nb"}
_REQUIRED_MODEL_PATHS = {
    "tokenize_model_path",
    "pos_model_path",
    "lemma_model_path",
    "depparse_model_path",
}


@dataclass(frozen=True)
class ParsedText:
    words: tuple[str, ...]
    heads: tuple[int, ...]

    def global_relations(self) -> str:
        relations: list[str] = []
        for index, head in enumerate(self.heads):
            if head == -1:
                relations.append(f"root depend {self.words[index]}")
            else:
                relations.append(f"{self.words[head]} modify {self.words[index]}")
        return " | ".join(relations) if relations else "no dependency relations"

    def neighbors(self, term: str | None) -> str:
        if term is None:
            return "implicit (no token span)"
        target = _find_term(self.words, term)
        if target is None:
            return "no aligned token span"
        target_indices = set(target)
        adjacent: set[int] = set()
        for index, head in enumerate(self.heads):
            if index in target_indices and head != -1:
                adjacent.add(head)
            if head in target_indices:
                adjacent.add(index)
        adjacent.difference_update(target_indices)
        neighbors = ", ".join(self.words[index] for index in sorted(adjacent))
        return f"which is connected to ({neighbors}) within one hop"


def _find_term(words: tuple[str, ...], term: str) -> tuple[int, ...] | None:
    wanted = tuple(token.casefold() for token in term.split())
    if not wanted:
        return None
    normalized = tuple(token.casefold() for token in words)
    for start in range(len(normalized) - len(wanted) + 1):
        if normalized[start : start + len(wanted)] == wanted:
            return tuple(range(start, start + len(wanted)))
    return None


class DependencyParser:
    """Lazy Stanza parser; language='auto' detects and loads models per language."""

    def __init__(
        self,
        language: str = "auto",
        use_gpu: bool = False,
        model_paths: Mapping[str, str] | None = None,
        model_language: str | None = None,
        resources_filepath: str | None = None,
        resources_dir: str | None = None,
    ) -> None:
        self.language = language
        self.use_gpu = use_gpu
        self.model_paths = tuple(sorted((key, str(Path(value).expanduser())) for key, value in (model_paths or {}).items()))
        self.model_language = (
            _LANGUAGE_ALIASES.get(model_language, model_language) if model_language else None
        )
        self.resources_filepath = (
            str(Path(resources_filepath).expanduser()) if resources_filepath else None
        )
        self.resources_dir = str(Path(resources_dir).expanduser()) if resources_dir else None
        if model_paths is not None:
            missing = _REQUIRED_MODEL_PATHS.difference(model_paths)
            if missing:
                raise ValueError(f"Missing Stanza model path settings: {', '.join(sorted(missing))}")
            missing_files = [path for _, path in self.model_paths if not Path(path).is_file()]
            if missing_files:
                raise FileNotFoundError(f"Stanza model files do not exist: {', '.join(missing_files)}")
            if not self.model_language:
                raise ValueError("Set model_language when supplying custom Stanza model paths.")
            if not self.resources_filepath or not Path(self.resources_filepath).is_file():
                raise FileNotFoundError(
                    "Custom Stanza model files still require a local resources.json. "
                    "Set resources_filepath to a valid local file."
                )

    @staticmethod
    @lru_cache(maxsize=8)
    def _pipeline(
        language: str,
        use_gpu: bool,
        model_paths: tuple[tuple[str, str], ...] = (),
        model_language: str | None = None,
        resources_filepath: str | None = None,
        resources_dir: str | None = None,
    ):
        try:
            custom_models = bool(model_paths)
            pipeline_options = dict(model_paths)
            processors = {
                "tokenize": "default",
                "pos": "default",
                "lemma": "identity",
                "depparse": "default",
            }
            if custom_models:
                pipeline_options.update(
                    package=None,
                    allow_unknown_language=True,
                    resources_filepath=resources_filepath,
                )
                processors = {
                    "tokenize": "default",
                    "pos": "default",
                    "lemma": "default",
                    "depparse": "default",
                }
            else:
                # A package="default" can add every processor in that package,
                # including constituency models we do not download or use.
                pipeline_options["package"] = None
            return stanza.Pipeline(
                lang=language,
                processors=processors,
                use_gpu=use_gpu,
                download_method=None,
                verbose=False,
                **({"dir": resources_dir} if resources_dir else {}),
                **pipeline_options,
            )
        except Exception as error:
            raise RuntimeError(
                f"Cannot load Stanza '{language}' dependency model. "
                f"Run `asqp-s2it prepare-parser --languages {language}` first, or check "
                "the configured local Stanza model files and resources.json."
            ) from error

    def parse(self, text: str, language: str | None = None) -> ParsedText:
        language = language or self.language
        if language == "auto":
            raise ValueError(
                "This sentence has no precomputed language. Annotate the JSONL with "
                "`asqp-s2it prepare-language` or set an explicit parser language."
            )
        language = _LANGUAGE_ALIASES.get(language, language)
        if self.model_paths and language != self.model_language:
            raise ValueError(
                f"Configured Stanza model supports '{self.model_language}', but this record "
                f"is tagged '{language}'. Attach matching models or correct the language tag."
            )
        document = self._pipeline(
            language,
            self.use_gpu,
            self.model_paths,
            self.model_language,
            self.resources_filepath,
            self.resources_dir,
        )(text)
        words: list[str] = []
        heads: list[int] = []
        for sentence in document.sentences:
            offset = len(words)
            words.extend(word.text for word in sentence.words)
            for word in sentence.words:
                heads.append(-1 if word.head == 0 else offset + word.head - 1)
        return ParsedText(tuple(words), tuple(heads))


def download_parser_models(languages: list[str]) -> None:
    for language in languages:
        stanza.download(
            _LANGUAGE_ALIASES.get(language, language),
            package="default",
        )
