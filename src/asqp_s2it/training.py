from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .budget import Budget, Deadline
from .data import Instance, category_labels
from .prompts import (
    auxiliary_examples,
    classification_prompt,
    extraction_prompt,
    render_pairs,
    render_quads,
)
from .syntax import DependencyParser
from .timing import format_duration


def _training_examples(
    instances: list[Instance],
    parser: DependencyParser,
    categories: list[str],
    split_name: str = "data",
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    extraction: list[tuple[str, str]] = []
    classification: list[tuple[str, str]] = []
    structural: list[tuple[str, str]] = []
    preparation_started = time.perf_counter()
    progress_interval = max(1, (len(instances) + 19) // 20)
    print(f"Preparing {split_name} syntax examples: {len(instances)} records", flush=True)
    for index, instance in enumerate(instances, start=1):
        parsed = parser.parse(instance.text, instance.language)
        pairs = [
            {"aspect_term": quad.aspect_term, "opinion_term": quad.opinion_term}
            for quad in instance.quads
        ]
        relations = parsed.global_relations()
        # Bidirectional generation of the (a, o) pair, as in S2IT Appendix B.
        for reverse in (False, True):
            extraction.append(
                (
                    extraction_prompt(instance.text, relations, reverse=reverse),
                    render_pairs(pairs, reverse=reverse),
                )
            )

        class_prompt = classification_prompt(instance.text, pairs, parsed, categories)
        class_output = render_quads([quad.as_dict() for quad in instance.quads])
        classification.append((class_prompt, class_output))
        structural.extend(auxiliary_examples(instance, parsed, categories))
        if index % progress_interval == 0 or index == len(instances):
            elapsed = time.perf_counter() - preparation_started
            eta = elapsed * (len(instances) - index) / index
            print(
                f"Preparing {split_name}: {index}/{len(instances)} records "
                f"({index / len(instances) * 100:.1f}%) | "
                f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}",
                flush=True,
            )
    return extraction, classification, structural


class TimeBudgetCallback(TrainerCallback):
    """Ends a stage at its deadline so the adapter is still saved.

    Overrunning a Kaggle session kills the notebook before `save_pretrained`,
    losing the whole run, so a stage that cannot finish stops early instead.
    """

    def __init__(self, deadline: Deadline) -> None:
        self.deadline = deadline

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.deadline.expired():
            control.should_training_stop = True
            control.should_evaluate = False
            control.should_save = False
        return control


class TrainingProgressCallback(TrainerCallback):
    def __init__(self, stage_name: str) -> None:
        self.stage_name = stage_name
        self.started_at: float | None = None

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.started_at = time.perf_counter()
        print(
            f"[{self.stage_name}] Training started: {state.max_steps} optimizer steps, "
            f"{args.num_train_epochs:g} epochs.",
            flush=True,
        )

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if self.started_at is None:
            return
        total_steps = max(1, state.max_steps)
        current_step = min(state.global_step, total_steps)
        elapsed = time.perf_counter() - self.started_at
        eta = elapsed * (total_steps - current_step) / current_step if current_step else 0.0
        epoch = state.epoch or 0.0
        loss = (logs or {}).get("loss")
        loss_text = f" | loss {loss:.4f}" if isinstance(loss, (int, float)) else ""
        print(
            f"[{self.stage_name}] step {current_step}/{total_steps} "
            f"({current_step / total_steps * 100:.1f}%) | "
            f"epoch {epoch:.2f}/{args.num_train_epochs:g} | "
            f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}"
            f"{loss_text}",
            flush=True,
        )


@dataclass
class CausalDataCollator:
    tokenizer: Any
    max_length: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(row["input_ids"]) for row in features)
        max_length = min(max_length, self.max_length)
        input_ids, attention_masks, labels = [], [], []
        pad_id = self.tokenizer.pad_token_id
        for row in features:
            ids = row["input_ids"][-max_length:]
            row_labels = row["labels"][-max_length:]
            padding = max_length - len(ids)
            input_ids.append(ids + [pad_id] * padding)
            attention_masks.append([1] * len(ids) + [0] * padding)
            labels.append(row_labels + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class PromptDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: list[tuple[str, str]],
        tokenizer: Any,
        max_length: int,
        split_name: str = "dataset",
    ) -> None:
        self.rows: list[dict[str, list[int]]] = []
        started_at = time.perf_counter()
        progress_interval = max(1, (len(examples) + 19) // 20)
        print(f"Tokenizing {split_name}: {len(examples)} examples", flush=True)
        for index, (prompt, completion) in enumerate(examples, start=1):
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(
                completion + tokenizer.eos_token,
                add_special_tokens=False,
            )["input_ids"]
            if len(completion_ids) >= max_length:
                completion_ids = completion_ids[-max_length:]
                prompt_ids = []
            else:
                prompt_ids = prompt_ids[-(max_length - len(completion_ids)) :]
            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids
            self.rows.append({"input_ids": input_ids, "labels": labels})
            if index % progress_interval == 0 or index == len(examples):
                elapsed = time.perf_counter() - started_at
                eta = elapsed * (len(examples) - index) / index
                print(
                    f"Tokenizing {split_name}: {index}/{len(examples)} examples "
                    f"({index / len(examples) * 100:.1f}%) | "
                    f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}",
                    flush=True,
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


def train(
    train_instances: list[Instance],
    dev_instances: list[Instance],
    output_dir: str | Path,
    model_name: str,
    parser: DependencyParser,
    epochs: int = 5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 5e-5,
    lora_rank: int = 32,
    max_length: int = 2048,
    time_budget_seconds: float | None = None,
    stage_fractions: tuple[float, float, float] = (0.25, 0.25, 0.50),
) -> dict[str, Any]:
    """Run the three S2IT stages, optionally inside a wall-clock budget.

    `time_budget_seconds` is the time this call may take end to end. It is split
    across the stages by `stage_fractions`; fine-grained structural tuning gets
    the largest share because it has the most examples per sentence.
    """
    started_at = time.perf_counter()
    budget = Budget(time_budget_seconds)
    if not budget.unlimited:
        print(
            f"Session budget: {format_duration(time_budget_seconds)} for "
            f"preparation, three tuning stages and saving.",
            flush=True,
        )
    categories = category_labels(train_instances)
    if not categories:
        raise ValueError("Training set contains no aspect categories")
    print("Preparing syntax-guided training examples...", flush=True)
    preparation_started = time.perf_counter()
    train_stages = _training_examples(train_instances, parser, categories, "train")
    dev_stages = _training_examples(dev_instances, parser, categories, "dev")
    preparation_seconds = time.perf_counter() - preparation_started
    if not dev_instances:
        raise ValueError("Development set is empty")

    model_load_started = time.perf_counter()
    print("Loading base model and tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype)
    model.config.use_cache = False
    # Reentrant gradient checkpointing needs grad-enabled inputs even when the
    # base model is frozen and only LoRA adapters are trainable.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model_load_seconds = time.perf_counter() - model_load_started
    print(
        f"Base model ready in {format_duration(model_load_seconds)} "
        f"({model_load_seconds:.2f} seconds).",
        flush=True,
    )
    collator = CausalDataCollator(tokenizer, max_length)
    stage_names = ("global_extraction", "local_classification", "fine_grained_structure")
    stage_times: dict[str, float] = {}
    # Reserve time to save the adapter once the last stage ends.
    save_reserve = 0.0 if budget.unlimited else min(300.0, max(60.0, budget.remaining() * 0.02))
    remaining_fractions = list(stage_fractions)
    for stage_index, (stage_name, examples, _dev_examples) in enumerate(
        zip(stage_names, train_stages, dev_stages)
    ):
        if not examples:
            raise ValueError(f"Training stage {stage_name} has no examples")
        print(f"Fine-tuning stage {stage_name}: {len(examples)} examples", flush=True)
        dataset_started = time.perf_counter()
        print(f"Tokenizing {stage_name} train/dev examples...", flush=True)
        train_dataset = PromptDataset(examples, tokenizer, max_length, f"{stage_name} train")
        dev_dataset = PromptDataset(_dev_examples, tokenizer, max_length, f"{stage_name} dev")
        print(
            f"Tokenized {len(train_dataset)} train and {len(dev_dataset)} dev examples in "
            f"{format_duration(time.perf_counter() - dataset_started)}.",
            flush=True,
        )
        if budget.unlimited:
            stage_deadline = Deadline(budget, None, stage_name)
        else:
            share = remaining_fractions[stage_index] / sum(remaining_fractions[stage_index:])
            stage_deadline = Deadline(
                budget,
                max(0.0, budget.remaining() - save_reserve) * share,
                stage_name,
            )
            print(f"[{stage_name}] {stage_deadline.describe()}", flush=True)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(Path(output_dir) / stage_name),
                learning_rate=learning_rate,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                lr_scheduler_type="cosine",
                warmup_ratio=0.03,
                logging_steps=10,
                logging_first_step=True,
                disable_tqdm=False,
                save_strategy="no",
                eval_strategy="epoch",
                report_to="none",
                remove_unused_columns=False,
                gradient_checkpointing=True,
                bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            ),
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=collator,
            callbacks=[
                TrainingProgressCallback(stage_name),
                TimeBudgetCallback(stage_deadline),
            ],
        )
        stage_started = time.perf_counter()
        print(f"[{stage_name}] Starting training at {time.strftime('%H:%M:%S')}.", flush=True)
        trainer.train()
        stage_times[stage_name] = time.perf_counter() - stage_started
        print(
            f"Completed {stage_name} in "
            f"{format_duration(stage_times[stage_name])} "
            f"({stage_times[stage_name]:.2f} seconds)"
        )
    training_seconds = sum(stage_times.values())

    saving_started = time.perf_counter()
    destination = Path(output_dir) / "final"
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    (destination / "categories.json").write_text(
        json.dumps(categories, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    saving_seconds = time.perf_counter() - saving_started
    total_elapsed_seconds = time.perf_counter() - started_at
    timing = {
        "model": model_name,
        "train_records": len(train_instances),
        "dev_records": len(dev_instances),
        "training_seconds": training_seconds,
        "training_time": format_duration(training_seconds),
        "stage_training_seconds": stage_times,
        "preparation_seconds": preparation_seconds,
        "model_load_seconds": model_load_seconds,
        "saving_seconds": saving_seconds,
        "total_elapsed_seconds": total_elapsed_seconds,
        "total_elapsed_time": format_duration(total_elapsed_seconds),
        "budget": budget.report(),
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "max_length": max_length,
        },
    }
    timing_json = json.dumps(timing, ensure_ascii=False, indent=2)
    (destination / "training_metrics.json").write_text(timing_json, encoding="utf-8")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "training_metrics.json").write_text(timing_json, encoding="utf-8")
    print(f"Saved final LoRA adapter and tokenizer to {destination}")
    print(f"Total training time: {timing['training_time']} ({training_seconds:.2f} seconds)")
    print(f"Total elapsed time: {timing['total_elapsed_time']} ({total_elapsed_seconds:.2f} seconds)")
    print(f"Training metrics saved to {destination / 'training_metrics.json'}")
    return timing
