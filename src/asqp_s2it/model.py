from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import Instance, Quad
from .prompts import classification_prompt, extraction_prompt
from .syntax import DependencyParser


def _chat_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Model output does not contain a JSON object: {text!r}")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model output JSON must be an object")
    return value


class S2ITPredictor:
    def __init__(
        self,
        model_path: str | Path,
        parser: DependencyParser,
        device: str | None = None,
    ) -> None:
        self.parser = parser
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        adapter = Path(model_path) / "adapter_config.json"
        if adapter.exists():
            adapter_config = PeftConfig.from_pretrained(model_path)
            dtype = (
                torch.bfloat16
                if self.device == "cuda" and torch.cuda.is_bf16_supported()
                else torch.float16
                if self.device == "cuda"
                else torch.float32
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                adapter_config.base_model_name_or_path,
                torch_dtype=dtype,
            )
            self.model = PeftModel.from_pretrained(self.model, model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
            )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _generate(self, prompt: str, max_new_tokens: int, num_beams: int = 1) -> str:
        encoded = self.tokenizer(
            _chat_prompt(self.tokenizer, prompt),
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)
        output = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = output[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def predict(
        self,
        instance: Instance,
        categories: list[str],
        max_new_tokens: int = 512,
    ) -> list[dict[str, str | None]]:
        parsed = self.parser.parse(instance.text, instance.language)
        pairs_by_key: dict[tuple[str | None, str | None], dict[str, str | None]] = {}
        for reverse in (False, True):
            raw_pairs = _extract_json(
                self._generate(
                    extraction_prompt(instance.text, parsed.global_relations(), reverse=reverse),
                    max_new_tokens,
                    num_beams=4,
                )
            ).get("pairs")
            if not isinstance(raw_pairs, list):
                raise ValueError(
                    f"Expected a 'pairs' list for instance {instance.instance_id} "
                    f"(reverse={reverse})"
                )
            for pair in raw_pairs:
                if not isinstance(pair, dict):
                    raise ValueError(f"Invalid pair generated for instance {instance.instance_id}")
                aspect, opinion = pair.get("aspect_term"), pair.get("opinion_term")
                if (aspect is not None and not isinstance(aspect, str)) or (
                    opinion is not None and not isinstance(opinion, str)
                ):
                    raise ValueError(
                        f"Invalid pair labels generated for instance {instance.instance_id}"
                    )
                key = (
                    aspect.strip().casefold() if aspect is not None else None,
                    opinion.strip().casefold() if opinion is not None else None,
                )
                pairs_by_key.setdefault(
                    key,
                    {"aspect_term": aspect, "opinion_term": opinion},
                )
        pairs = list(pairs_by_key.values())
        if not pairs:
            return []
        raw_quads = _extract_json(
            self._generate(
                classification_prompt(instance.text, pairs, parsed, categories),
                max_new_tokens,
            )
        ).get("quadruples")
        if not isinstance(raw_quads, list):
            raise ValueError(f"Expected a 'quadruples' list for instance {instance.instance_id}")
        results: list[dict[str, str | None]] = []
        for raw in raw_quads:
            if not isinstance(raw, dict):
                raise ValueError(f"Invalid quadruple generated for instance {instance.instance_id}")
            quad = Quad(
                aspect_term=raw.get("aspect_term"),
                aspect_category=raw.get("aspect_category"),
                opinion_term=raw.get("opinion_term"),
                sentiment=raw.get("sentiment"),
            )
            if (
                (quad.aspect_term is not None and not isinstance(quad.aspect_term, str))
                or not isinstance(quad.aspect_category, str)
                or (quad.opinion_term is not None and not isinstance(quad.opinion_term, str))
                or not isinstance(quad.sentiment, str)
            ):
                raise ValueError(f"Incomplete quadruple generated for instance {instance.instance_id}")
            results.append(quad.as_dict())
        return results
