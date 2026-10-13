"""Bookkeeping that lets a second Kaggle session continue the first one.

The full training split at the paper's five epochs does not fit one session,
so `train()` records what each stage reached and a later call skips what is
done. None of that needs a GPU, so it is tested here directly.
"""

import json

import pytest

from asqp_s2it.resume import (
    PROGRESS_FILENAME,
    STAGE_NAMES,
    assert_same_categories,
    latest_checkpoint,
    read_progress,
    write_progress,
)


def test_stage_order_is_the_curriculum():
    # Resume skips a prefix of completed stages, which is only correct while
    # the stages run in this order.
    assert STAGE_NAMES == (
        "global_extraction",
        "local_classification",
        "fine_grained_structure",
    )


def test_progress_round_trips(tmp_path):
    progress = {
        "global_extraction": {"epochs_done": 5.0, "epochs_target": 5.0, "complete": True},
        "local_classification": {"epochs_done": 2.4, "epochs_target": 5.0, "complete": False},
    }
    write_progress(tmp_path, progress)
    assert read_progress(tmp_path) == progress


def test_missing_progress_file_reads_as_nothing_done(tmp_path):
    # A first session has no progress file; that must not look like an error.
    assert read_progress(tmp_path) == {}


def test_progress_file_rejects_foreign_json(tmp_path):
    (tmp_path / PROGRESS_FILENAME).write_text('{"stages": 3}', encoding="utf-8")
    with pytest.raises(ValueError, match="progress file"):
        read_progress(tmp_path)


def test_latest_checkpoint_orders_numerically_not_lexically(tmp_path):
    # checkpoint-900 sorts after checkpoint-1000 as a string, which would
    # resume from the older one and silently repeat hours of training.
    for step in (900, 1000, 150):
        (tmp_path / f"checkpoint-{step}").mkdir()
    assert latest_checkpoint(tmp_path).name == "checkpoint-1000"


def test_latest_checkpoint_ignores_unrelated_entries(tmp_path):
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-notanumber").mkdir()
    (tmp_path / "checkpoint-20.txt").write_text("", encoding="utf-8")
    assert latest_checkpoint(tmp_path).name == "checkpoint-10"


def test_no_checkpoint_when_the_stage_never_reached_an_epoch_end(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    assert latest_checkpoint(tmp_path / "never_created") is None


def test_resuming_onto_a_different_category_set_is_refused(tmp_path):
    # The category list goes into every stage-2 prompt. Continuing an adapter
    # trained against a different one trains it on prompts it has never seen,
    # and that only shows up as a low F1 hours later.
    (tmp_path / "categories.json").write_text(
        json.dumps(["room", "service"]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="different category"):
        assert_same_categories(tmp_path, ["room", "service", "food"])


def test_resuming_onto_the_same_category_set_is_allowed(tmp_path):
    (tmp_path / "categories.json").write_text(
        json.dumps(["room", "service"]), encoding="utf-8"
    )
    assert_same_categories(tmp_path, ["room", "service"])


def test_category_check_is_skipped_when_the_adapter_predates_the_file(tmp_path):
    assert_same_categories(tmp_path, ["room"])
