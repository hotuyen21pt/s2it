from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
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
from .resume import (
    PROGRESS_FILENAME,
    STAGE_NAMES,
    assert_same_categories,
    latest_checkpoint,
    read_progress,
    save_adapter,
    write_progress,
)
from .syntax import DependencyParser
from .timing import format_duration


#: TrainingArguments fields this project cannot silently lose. Transformers
#: renames and removes arguments across majors - `evaluation_strategy` became
#: `eval_strategy` in 4.41, `warmup_ratio` is gone in 5.x - and Kaggle's image
#: moves between majors, so everything else is filtered against the installed
#: signature and reported rather than assumed.
_ESSENTIAL_TRAINING_ARGS = (
    "output_dir",
    "learning_rate",
    "num_train_epochs",
    "max_steps",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "lr_scheduler_type",
    # Silently losing this would leave no checkpoint to resume from, and the
    # full training split needs more than one Kaggle session.
    "save_strategy",
)


def _accepted_training_args() -> set[str] | None:
    """Field names the installed TrainingArguments takes, or None if unknown.

    Tried most reliable first. A signature that collapsed to bare `*args,
    **kwargs` tells us nothing and is skipped; if every source does, None is
    returned and nothing is filtered, so a genuine mistake still raises.
    """
    import dataclasses

    sources = []
    for probe in (
        lambda: set(inspect.signature(TrainingArguments).parameters),
        lambda: {field.name for field in dataclasses.fields(TrainingArguments)},
        lambda: set(inspect.signature(TrainingArguments.__init__).parameters),
    ):
        try:
            sources.append(probe())
        except (TypeError, ValueError):
            continue
    for names in sources:
        names -= {"self", "args", "kwargs"}
        if names:
            return names
    return None


def _training_arguments(**desired: Any) -> TrainingArguments:
    """Build TrainingArguments, dropping fields this transformers cannot take."""
    import transformers

    accepted = _accepted_training_args()
    if accepted is None:
        # Could not introspect; pass everything through rather than guess.
        return TrainingArguments(**desired)
    lost_essential = [
        name for name in _ESSENTIAL_TRAINING_ARGS
        if name in desired and name not in accepted
    ]
    if lost_essential:
        raise RuntimeError(
            f"transformers {transformers.__version__} does not accept "
            f"{lost_essential} on TrainingArguments. Dropping these would change "
            "the training configuration silently, so the run stops instead. Pin "
            "the bundled transformers version, or map them to the new names."
        )
    dropped = sorted(name for name in desired if name not in accepted)
    if dropped:
        print(
            f"[transformers {transformers.__version__}] TrainingArguments does not "
            f"accept {dropped}; continuing without them.",
            flush=True,
        )
        # Losing these does not fail here, it fails later as an out-of-memory
        # error that looks unrelated, so say so now.
        memory_critical = sorted(
            {"gradient_checkpointing", "fp16", "bf16"}.intersection(dropped)
        )
        if memory_critical:
            print(
                f"  WARNING: {memory_critical} control precision and activation "
                "memory. Without them this run is likely to exhaust GPU memory. "
                "Set PIN_BUNDLED_VERSIONS = True to use the bundled transformers.",
                flush=True,
            )
    return TrainingArguments(**{k: v for k, v in desired.items() if k in accepted})


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


def _auto_device_map() -> str | None:
    """Shard across GPUs only when there is more than one.

    With a single device `device_map="auto"` offloads whatever does not fit to
    CPU instead of failing. That is roughly fifty times slower and presents as
    a run that simply never finishes, so one GPU loads normally and lets
    `check_gpu_memory` decide whether the configuration is possible at all.
    """
    if not torch.cuda.is_available():
        return None
    return "auto" if torch.cuda.device_count() > 1 else None


def assert_not_offloaded(model: Any) -> None:
    """Stop if accelerate placed any layer on CPU or disk.

    An offloaded layer copies its weights over PCIe on every forward and every
    backward pass. A stage that should take half an hour then reports an ETA of
    twenty-two hours, which is how this was found.
    """
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return
    placements: dict[str, int] = {}
    for device in device_map.values():
        placements[str(device)] = placements.get(str(device), 0) + 1
    print(f"Model placement: {placements}", flush=True)
    spilled = sorted(
        name for name, device in device_map.items() if str(device) in {"cpu", "disk"}
    )
    if spilled:
        raise RuntimeError(
            f"{len(spilled)} module(s) were offloaded to CPU or disk because they "
            f"did not fit in GPU memory, for example {spilled[:3]}.\n"
            "Training would still run, roughly fifty times slower, so it stops "
            "here instead.\n"
            "  - Switch the Kaggle accelerator to 'GPU T4 x2', or\n"
            "  - Lower LORA_RANK and MAX_LENGTH and record the deviation."
        )


def _model_parameter_count(model_name: str) -> int | None:
    """Parameter count from the config alone, without loading any weights."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        layers = config.num_hidden_layers
        hidden = config.hidden_size
        inter = config.intermediate_size
        heads = config.num_attention_heads
        kv_heads = getattr(config, "num_key_value_heads", heads)
        head_dim = getattr(config, "head_dim", hidden // heads)
    except Exception:
        return None
    kv = kv_heads * head_dim
    per_layer = 2 * hidden * hidden + 2 * hidden * kv + 3 * hidden * inter
    embeddings = config.vocab_size * hidden
    if not getattr(config, "tie_word_embeddings", False):
        embeddings *= 2
    return layers * per_layer + embeddings


def _lora_parameter_count(model_name: str, rank: int) -> int | None:
    """Trainable parameters LoRA adds to the seven targeted projections."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        hidden = config.hidden_size
        inter = config.intermediate_size
        heads = config.num_attention_heads
        kv_heads = getattr(config, "num_key_value_heads", heads)
        head_dim = getattr(config, "head_dim", hidden // heads)
    except Exception:
        return None
    kv = kv_heads * head_dim
    widths = [
        (hidden, hidden), (hidden, kv), (hidden, kv), (hidden, hidden),
        (hidden, inter), (hidden, inter), (inter, hidden),
    ]
    return config.num_hidden_layers * sum(rank * (a + b) for a, b in widths)


def check_gpu_memory(model_name: str, lora_rank: int, dtype_bytes: int = 2) -> None:
    """Fail before loading weights if this model and rank cannot possibly fit.

    The base model dominates: a 7B in fp16 is 14.2 GiB of a 16 GB card's
    14.6 GiB, so the LoRA weights and their two AdamW states decide whether a
    run is possible at all. Discovering that during `Trainer.__init__` wastes
    several minutes and produces a traceback that does not name the cause.
    """
    if not torch.cuda.is_available():
        return
    params = _model_parameter_count(model_name)
    lora = _lora_parameter_count(model_name, lora_rank)
    if params is None or lora is None:
        return
    gib = float(2**30)
    weights = params * dtype_bytes / gib
    # LoRA weights stay fp32, and AdamW keeps two states of the same size.
    adapter = lora * 4 * 3 / gib
    devices = torch.cuda.device_count()
    capacity = sum(
        torch.cuda.get_device_properties(i).total_memory for i in range(devices)
    ) / gib
    needed = weights + adapter
    print(
        f"GPU budget: {capacity:.2f} GiB across {devices} device(s) | "
        f"weights {weights:.2f} + LoRA r{lora_rank} {adapter:.2f} = {needed:.2f} GiB",
        flush=True,
    )
    headroom = capacity - needed
    if headroom <= 0:
        per_rank = lora / lora_rank
        affordable = max(
            (r for r in (64, 32, 16, 8, 4)
             if weights + per_rank * r * 4 * 3 / gib < capacity),
            default=None,
        )
        advice = (
            f"Lower LORA_RANK to {affordable}, which departs from the paper and "
            "should be recorded"
            if affordable
            else "Use a smaller backbone"
        )
        raise RuntimeError(
            f"This cannot fit: {needed:.2f} GiB of weights and optimiser state "
            f"against {capacity:.2f} GiB of GPU memory, before any activations.\n"
            f"  - Switch the Kaggle accelerator to 'GPU T4 x2' "
            f"({2 * capacity / max(devices, 1):.0f} GiB total), which shards the "
            f"model across both and keeps the paper's rank, or\n"
            f"  - {advice}."
        )
    # A 7B in fp16 leaves very little room on a 16 GB card. The run that
    # trained for eight hours had 0.15 GiB spare at rank 8 and sequence length
    # 768, so a thin margin is survivable but worth saying out loud.
    if headroom < 0.5:
        print(
            f"  WARNING: only {headroom * 1024:.0f} MiB spare for activations. "
            "Keep MAX_LENGTH and BATCH_SIZE small, or use 'GPU T4 x2'.",
            flush=True,
        )


def _set_seed(seed: int) -> None:
    """Seed every RNG that affects the adapter, before it is created."""
    try:
        from transformers import set_seed

        set_seed(seed)
    except ImportError:  # pragma: no cover - transformers always provides it
        import random

        random.seed(seed)
        torch.manual_seed(seed)


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
    stage_fractions: tuple[float, float, float] | None = None,
    seed: int = 444,
    evaluate_dev: bool = False,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Run the three S2IT stages, optionally inside a wall-clock budget.

    `time_budget_seconds` is the time this call may take end to end. It is
    split across the stages in proportion to how many examples each one
    actually builds - 2, 1 and 7 per sentence, so 20/10/70 - rather than by a
    fixed guess. Pass `stage_fractions` to override.

    `evaluate_dev` is off by default. Nothing here selects a checkpoint on the
    development loss - there is no early stopping and no load_best_model - so
    evaluating every epoch spends budget without changing the result. On this
    data it is 92,300 forward passes, which can exceed the time left for
    training itself and leave the adapter undertrained for no reason.

    `resume_from` is the output directory of an earlier call. The full training
    split at the paper's five epochs is longer than one Kaggle session, so a
    session that runs out of time saves what it has and a later one continues:
    stages already marked complete in `training_progress.json` are skipped, and
    the first unfinished stage restarts from its last `checkpoint-N`, which
    carries the optimiser state and the position in the cosine schedule. The
    five epochs therefore stay one run rather than becoming several short ones.
    A stage stopped part-way through an epoch repeats that epoch, since
    checkpoints are written at epoch boundaries.
    """
    started_at = time.perf_counter()
    # LoRA matrices are initialised in get_peft_model below, before any Trainer
    # exists to seed the RNG, so without this two runs produce different
    # adapters and neither can be reproduced.
    _set_seed(seed)
    budget = Budget(time_budget_seconds)
    resume_dir = Path(resume_from) if resume_from is not None else None
    progress: dict[str, dict[str, Any]] = {}
    if resume_dir is not None:
        if not (resume_dir / "final" / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"resume_from={resume_dir} holds no final/adapter_config.json. "
                "Point it at the output directory of the earlier run - the one "
                f"containing final/ and {PROGRESS_FILENAME} - not at the adapter."
            )
        progress = read_progress(resume_dir)
        finished = [n for n in STAGE_NAMES if progress.get(n, {}).get("complete")]
        print(
            f"Resuming from {resume_dir}: "
            + (
                f"{', '.join(finished)} already complete."
                if finished
                else "no stage finished in the earlier session."
            ),
            flush=True,
        )
    if not budget.unlimited:
        print(
            f"Session budget: {format_duration(time_budget_seconds)} for "
            f"preparation, three tuning stages and saving.",
            flush=True,
        )
    categories = category_labels(train_instances)
    if not categories:
        raise ValueError("Training set contains no aspect categories")
    if resume_dir is not None:
        assert_same_categories(resume_dir / "final", categories)
    print("Preparing syntax-guided training examples...", flush=True)
    preparation_started = time.perf_counter()
    train_stages = _training_examples(train_instances, parser, categories, "train")
    if evaluate_dev:
        dev_stages = _training_examples(dev_instances, parser, categories, "dev")
    else:
        dev_stages = ([], [], [])
        print(
            "Development evaluation is off: no checkpoint is selected on it, so "
            "it would only consume the budget. Pass evaluate_dev=True to log it.",
            flush=True,
        )
    preparation_seconds = time.perf_counter() - preparation_started
    if not dev_instances:
        raise ValueError("Development set is empty")

    check_gpu_memory(model_name, lora_rank)
    model_load_started = time.perf_counter()
    print(f"Loading {model_name}", flush=True)
    print(
        "  Reading ~15 GB of weights from the Kaggle input mount; this is the "
        "longest step with no output, often several minutes.",
        flush=True,
    )
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=_auto_device_map(),
        **_dtype_kwargs(model_dtype),
    )
    assert_not_offloaded(model)
    model.config.use_cache = False
    # Reentrant gradient checkpointing needs grad-enabled inputs even when the
    # base model is frozen and only LoRA adapters are trainable.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if resume_dir is None:
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
    else:
        # Continue the earlier session's adapter rather than creating a new one.
        # is_trainable keeps the LoRA matrices requiring grad; without it peft
        # loads them for inference and the run reports zero trainable weights.
        adapter_dir = resume_dir / "final"
        print(f"Loading the adapter to continue from {adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    model_load_seconds = time.perf_counter() - model_load_started
    print(
        f"Base model ready in {format_duration(model_load_seconds)} "
        f"({model_load_seconds:.2f} seconds).",
        flush=True,
    )
    collator = CausalDataCollator(tokenizer, max_length)
    stage_names = STAGE_NAMES
    stage_times: dict[str, float] = {}
    trained_here: list[str] = []
    inherited: list[str] = []
    resumed_a_stage = False
    # Reserve time to save the adapter once the last stage ends.
    save_reserve = 0.0 if budget.unlimited else min(300.0, max(60.0, budget.remaining() * 0.02))
    if stage_fractions is None:
        # Each stage costs in proportion to its example count: a fixed 25/25/50
        # starved fine-grained tuning, which builds seven examples per sentence,
        # while over-funding classification, which builds one.
        sizes = [len(stage) for stage in train_stages]
        total_examples = sum(sizes) or 1
        remaining_fractions = [size / total_examples for size in sizes]
        print(
            "Budget split by example count: "
            + ", ".join(
                f"{name} {share:.0%} ({size} examples)"
                for name, share, size in zip(stage_names, remaining_fractions, sizes)
            ),
            flush=True,
        )
    else:
        remaining_fractions = list(stage_fractions)
    for stage_index, (stage_name, examples, _dev_examples) in enumerate(
        zip(stage_names, train_stages, dev_stages)
    ):
        if not examples:
            raise ValueError(f"Training stage {stage_name} has no examples")
        if progress.get(stage_name, {}).get("complete"):
            inherited.append(stage_name)
            print(
                f"[{stage_name}] finished in an earlier session "
                f"({progress[stage_name].get('epochs_done', epochs):g} epochs); "
                "skipping. Its budget share goes to the stages that are left.",
                flush=True,
            )
            continue
        print(f"Fine-tuning stage {stage_name}: {len(examples)} examples", flush=True)
        dataset_started = time.perf_counter()
        print(f"Tokenizing {stage_name} train/dev examples...", flush=True)
        train_dataset = PromptDataset(examples, tokenizer, max_length, f"{stage_name} train")
        dev_dataset = (
            PromptDataset(_dev_examples, tokenizer, max_length, f"{stage_name} dev")
            if _dev_examples
            else None
        )
        print(
            f"Tokenized {len(train_dataset)} train and "
            f"{len(dev_dataset) if dev_dataset is not None else 0} dev examples in "
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
            args=_training_arguments(
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
                # One checkpoint per epoch, one kept. The full training split
                # does not fit a single Kaggle session, and this is what
                # resume_from picks up; the checkpoint carries the optimiser
                # state and the cosine schedule, which a bare adapter does not.
                save_strategy="epoch",
                save_total_limit=1,
                eval_strategy="epoch" if dev_dataset is not None else "no",
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
        # Only the first unfinished stage has a checkpoint to pick up; any
        # stage after it never started, so it runs from the beginning.
        stage_checkpoint: Path | None = None
        if resume_dir is not None and not resumed_a_stage:
            resumed_a_stage = True
            stage_checkpoint = latest_checkpoint(resume_dir / stage_name)
            if stage_checkpoint is not None:
                print(
                    f"[{stage_name}] continuing from {stage_checkpoint.name}; the "
                    "optimiser state and the position in the cosine schedule come "
                    f"with it, so this stays one {epochs}-epoch run.",
                    flush=True,
                )
            else:
                print(
                    f"[{stage_name}] the earlier session stopped before its first "
                    "epoch ended, so there is no checkpoint; restarting the stage.",
                    flush=True,
                )
        stage_started = time.perf_counter()
        print(f"[{stage_name}] Starting training at {time.strftime('%H:%M:%S')}.", flush=True)
        trainer.train(
            resume_from_checkpoint=str(stage_checkpoint)
            if stage_checkpoint is not None
            else None
        )
        stage_times[stage_name] = time.perf_counter() - stage_started
        epochs_done = float(getattr(trainer.state, "epoch", 0.0) or 0.0)
        progress[stage_name] = {
            "epochs_done": round(epochs_done, 4),
            "epochs_target": float(epochs),
            "global_step": int(getattr(trainer.state, "global_step", 0) or 0),
            "complete": epochs_done >= epochs - 1e-6,
        }
        trained_here.append(stage_name)
        # Save now rather than only after the last stage: this is what the next
        # session loads, and a session killed between stages would lose the lot.
        save_adapter(model, tokenizer, output_dir, categories)
        write_progress(output_dir, progress)
        print(
            f"Completed {stage_name} in "
            f"{format_duration(stage_times[stage_name])} "
            f"({stage_times[stage_name]:.2f} seconds)"
        )
        if not progress[stage_name]["complete"]:
            print(
                f"[{stage_name}] stopped at its deadline after {epochs_done:.2f} of "
                f"{epochs:g} epochs. Save Version, then set RESUME_FROM to this "
                "run's output directory to carry on from here.",
                flush=True,
            )
    training_seconds = sum(stage_times.values())

    saving_started = time.perf_counter()
    destination = save_adapter(model, tokenizer, output_dir, categories)
    write_progress(output_dir, progress)
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
        "resume": {
            "resumed_from": str(resume_dir) if resume_dir is not None else None,
            # training_seconds below covers this session only; stages listed as
            # inherited were paid for by an earlier one.
            "stages_trained_this_session": trained_here,
            "stages_inherited": inherited,
            "stages": progress,
            "complete": all(
                progress.get(name, {}).get("complete") for name in stage_names
            ),
        },
        "hyperparameters": {
            "seed": seed,
            "evaluate_dev": evaluate_dev,
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
    if timing["resume"]["complete"]:
        print("All three stages reached their full epoch count.")
    else:
        unfinished = [
            name for name in stage_names if not progress.get(name, {}).get("complete")
        ]
        print(
            f"Unfinished: {', '.join(unfinished)}. Save Version, attach this "
            "output to the next session and set RESUME_FROM to it."
        )
    print(f"Total training time: {timing['training_time']} ({training_seconds:.2f} seconds)")
    print(f"Total elapsed time: {timing['total_elapsed_time']} ({total_elapsed_seconds:.2f} seconds)")
    print(f"Training metrics saved to {destination / 'training_metrics.json'}")
    return timing
