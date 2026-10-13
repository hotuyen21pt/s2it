"""Bookkeeping that lets a second session continue the first one's training.

The full training split at the paper's five epochs is longer than one Kaggle
session, so `training.train()` saves what it reached and a later call picks it
up. Which stages are done, which checkpoint a stage stopped at and whether the
label space still matches are all plain filesystem questions, so they live
here, away from the torch import, and are tested without a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: The three stages, in the order they must run. Resume skips a prefix of
#: completed stages, which is only correct while that order holds.
STAGE_NAMES = ("global_extraction", "local_classification", "fine_grained_structure")

#: Written beside the adapter so a later session knows what is already done.
PROGRESS_FILENAME = "training_progress.json"


def latest_checkpoint(stage_dir: Path) -> Path | None:
    """The furthest `checkpoint-N` Trainer wrote for one stage, if any.

    Compared as integers: `checkpoint-900` sorts after `checkpoint-1000` as a
    string, which would resume from the older one and repeat hours of training.
    """
    stage_dir = Path(stage_dir)
    if not stage_dir.is_dir():
        return None
    checkpoints = []
    for candidate in stage_dir.glob("checkpoint-*"):
        if not candidate.is_dir():
            continue
        try:
            checkpoints.append((int(candidate.name.split("-")[-1]), candidate))
        except ValueError:
            continue
    if not checkpoints:
        return None
    return max(checkpoints)[1]


def read_progress(output_dir: str | Path) -> dict[str, dict[str, Any]]:
    """What an earlier run reached, or `{}` for a first session."""
    path = Path(output_dir) / PROGRESS_FILENAME
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    stages = payload.get("stages", payload)
    if not isinstance(stages, dict):
        raise ValueError(f"{path} is not a progress file written by train().")
    return {name: dict(state) for name, state in stages.items()}


def write_progress(output_dir: str | Path, progress: dict[str, dict[str, Any]]) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"stages": progress}, ensure_ascii=False, indent=2)
    (Path(output_dir) / PROGRESS_FILENAME).write_text(payload + "\n", encoding="utf-8")


def save_adapter(
    model: Any, tokenizer: Any, output_dir: str | Path, categories: list[str]
) -> Path:
    """Write the adapter that a later session, or inference, loads.

    Called after every stage rather than only at the end: a session killed
    between stages would otherwise lose everything it had just trained.
    """
    destination = Path(output_dir) / "final"
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    (destination / "categories.json").write_text(
        json.dumps(categories, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination


def assert_same_categories(adapter_dir: str | Path, categories: list[str]) -> None:
    """Refuse to resume onto a different label space.

    The category list goes into every stage-2 prompt. Continuing an adapter
    that was trained against a different one trains it on prompts it has never
    seen, and the damage only surfaces as a low F1 hours later.
    """
    path = Path(adapter_dir) / "categories.json"
    if not path.is_file():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous != categories:
        missing = sorted(set(previous) - set(categories))
        added = sorted(set(categories) - set(previous))
        raise ValueError(
            f"The adapter in {adapter_dir} was trained on a different category "
            f"set: {len(previous)} categories then, {len(categories)} now "
            f"(missing {missing[:3]}, added {added[:3]}). Resuming would change "
            "the prompts under it. Train from scratch on this data instead."
        )
