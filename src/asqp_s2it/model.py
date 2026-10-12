from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import Instance
from .prompts import classification_prompt, extraction_prompt, parse_pairs, parse_quads
from .syntax import DependencyParser

# S2IT Appendix B: beam search of size 4 for both generation steps.
BEAM_SIZE = 4


def _dtype_kwargs(dtype) -> dict[str, Any]:
    """`torch_dtype` is deprecated in transformers 5, which renamed it `dtype`.

    Kaggle's image moves between major versions, so pick the name the
    installed library actually accepts instead of pinning to one of them.
    """
    import transformers

    try:
        major = int(transformers.__version__.split(".")[0])
    except (AttributeError, ValueError):
        major = 4
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}


def _chat_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


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
        self.skipped: list[dict[str, str]] = []
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
                **_dtype_kwargs(dtype),
            )
            self.model = PeftModel.from_pretrained(self.model, model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **_dtype_kwargs(torch.float32),
            )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _generate(self, prompt: str, max_new_tokens: int, num_beams: int = BEAM_SIZE) -> str:
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
            # Greedy decoding ignores these, and leaving them set only produces
            # the `temperature`/`top_p`/`top_k` warnings seen in the Kaggle log.
            temperature=None,
            top_p=None,
            top_k=None,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = output[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _record_skip(self, instance: Instance, stage: str, raw: str) -> None:
        self.skipped.append(
            {"instance_id": instance.instance_id, "stage": stage, "output": raw[:500]}
        )

    def predict(
        self,
        instance: Instance,
        categories: list[str],
        max_new_tokens: int = 512,
    ) -> list[dict[str, str | None]]:
        parsed = self.parser.parse(instance.text, instance.language)
        relations = parsed.global_relations()

        # Step 1: Global Syntax-guided Extraction, generated in both directions
        # and merged, following the bidirectional decoding of Appendix B.
        pairs_by_key: dict[tuple[str | None, str | None], dict[str, str | None]] = {}
        for reverse in (False, True):
            raw = self._generate(
                extraction_prompt(instance.text, relations, reverse=reverse),
                max_new_tokens,
            )
            extracted = parse_pairs(raw, reverse=reverse)
            if not extracted:
                self._record_skip(instance, f"extraction(reverse={reverse})", raw)
                continue
            for pair in extracted:
                aspect, opinion = pair["aspect_term"], pair["opinion_term"]
                key = (
                    aspect.strip().casefold() if aspect is not None else None,
                    opinion.strip().casefold() if opinion is not None else None,
                )
                pairs_by_key.setdefault(
                    key, {"aspect_term": aspect, "opinion_term": opinion}
                )
        pairs = list(pairs_by_key.values())
        if not pairs:
            return []

        # Step 2: Local Syntax-guided Classification over the extracted pairs.
        raw = self._generate(
            classification_prompt(instance.text, pairs, parsed, categories),
            max_new_tokens,
        )
        quadruples = parse_quads(raw)
        if not quadruples:
            # One unparsable completion costs this sentence, not the whole run.
            self._record_skip(instance, "classification", raw)
        return quadruples
