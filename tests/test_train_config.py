"""Two settings that decide whether a run is reproducible and affordable."""
import sys

import pytest

from tests.test_training_args import _stub_heavy_dependencies  # noqa: F401

_stub_heavy_dependencies()

from asqp_s2it import training  # noqa: E402


def test_dev_evaluation_is_off_by_default():
    """Nothing selects a checkpoint on the dev loss, so evaluating it is waste.

    On this data it is 92,300 forward passes across the three stages, which can
    outlast the training budget itself and leave the adapter undertrained for a
    reason unrelated to the method.
    """
    import inspect

    parameters = inspect.signature(training.train).parameters
    assert parameters["evaluate_dev"].default is False


def test_a_seed_is_part_of_the_signature():
    import inspect

    assert inspect.signature(training.train).parameters["seed"].default == 444


def test_set_seed_makes_torch_deterministic(monkeypatch):
    """The LoRA matrices are created before any Trainer exists to seed the RNG."""
    calls = []

    stub = sys.modules["transformers"]
    monkeypatch.setattr(stub, "set_seed", calls.append, raising=False)
    training._set_seed(444)
    assert calls == [444]


def test_seeding_falls_back_when_transformers_lacks_set_seed(monkeypatch):
    stub = sys.modules["transformers"]
    if hasattr(stub, "set_seed"):
        monkeypatch.delattr(stub, "set_seed")

    seeded = []
    monkeypatch.setattr(training.torch, "manual_seed", seeded.append, raising=False)
    # ImportError is what a missing name raises for `from x import y`.
    training._set_seed(7)
    assert seeded == [7]


def test_resuming_is_off_by_default():
    import inspect

    assert inspect.signature(training.train).parameters["resume_from"].default is None


def test_checkpointing_cannot_be_dropped_silently():
    """Without checkpoints there is nothing for a second session to resume.

    `_training_arguments` drops any field the installed transformers does not
    accept and merely prints what it lost. Losing this one would leave a run
    that looks fine, finishes at its deadline and cannot be continued, so it
    belongs in the set that raises instead.
    """
    assert "save_strategy" in training._ESSENTIAL_TRAINING_ARGS
