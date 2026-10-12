"""Rebuild s2it_kaggle.ipynb with zip-based inputs and a 9h session budget."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


md(r"""
# S2IT — Aspect Sentiment Quad Prediction (offline Kaggle)

Trains and evaluates S2IT with **no Internet**, inside a **9-hour** Kaggle session.

**Attach two inputs, then Run All:**

| Input | What |
|---|---|
| Dataset | `kaggle_offline_upload.zip`, built locally by `scripts/prepare_offline_bundle.py` |
| Model | `Qwen2.5-7B-Instruct` (already on Kaggle — *not* in the archive) |

Settings: **Accelerator = GPU**, **Internet = Off**.

The archive carries the project source, the language-tagged JSONL splits, the
precomputed Stanza parse cache and a pip wheelhouse. Stanza itself is never
installed here: the notebook reads `stanza_parses.jsonl`.
""")

code(r'''
# ============================== SESSION CONFIG ==============================
import sys
import time

# Kaggle runs this notebook under papermill, which redirects stdout to a file.
# Python then block-buffers it, so nothing reaches the log until 8 KB has
# accumulated - a run can look dead for a quarter of an hour while it is
# unpacking, installing and loading a 7B model. Line buffering makes progress
# visible as it happens.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

SESSION_START = time.perf_counter()

# --- Wall-clock budget -------------------------------------------------------
# Kaggle kills the session at the limit, which would lose the LoRA adapter.
# Training stops at its own deadline instead, so the adapter is always saved.
SESSION_BUDGET_HOURS     = 9.0   # Kaggle GPU session limit
SETUP_RESERVE_MINUTES    = 20    # wheels, model load, tokenisation
INFERENCE_RESERVE_MINUTES = 180  # full test split with beam-4 decoding
SAFETY_MARGIN_MINUTES    = 15    # never finish exactly on the limit

# --- Data --------------------------------------------------------------------
# The bundle already carries the agreed 1,530-sentence training subset and
# the full 1,831-sentence test split, shared byte-for-byte with the
# Tree-CoT-RT bundle. Leave this at 100 to use exactly what it is given.
DATA_PERCENT     = 100
DATA_SAMPLE_SEED = 444
MAX_EVAL_RECORDS = None  # None = the whole test split

# --- Fine-tuning -------------------------------------------------------------
# Table 5 of the paper, reproduced: lr 5e-5, 5 epochs, LoRA rank 32 /
# alpha 32 / dropout 0.1, cosine schedule.
# The paper's "batch size 4, gradient accumulation 2" is an effective batch of
# 8, and 1 x 8 below is the same optimiser trajectory while fitting 16 GB.
EPOCHS                      = 5     # = paper
BATCH_SIZE                  = 1     # paper 4, but 1 x 8 == 4 x 2 effective
GRADIENT_ACCUMULATION_STEPS = 8     # effective batch 8 = paper
LEARNING_RATE               = 5e-5  # = paper
# Needs 'GPU T4 x2'. A 7B in fp16 is 14.2 GiB of one card's 14.6, leaving
# 0.37 GiB, and rank 32 wants 0.92 GiB of weights plus AdamW state. On a
# single card use 8, and record it as a deviation. train() checks this
# before loading any weights and says which applies.
LORA_RANK                   = 32    # = paper (alpha 32, dropout 0.1 in training.py)
# Measured on this data: the longest stage-3 example is 786 tokens, so the
# previous 768 silently truncated one example in 3,600. 1024 truncates none.
MAX_LENGTH                  = 1024
# How the training budget is split across the three stages. None derives it
# from the example count each stage actually builds (2, 1 and 7 per sentence,
# so 20/10/70). A fixed 25/25/50 starved fine-grained tuning.
STAGE_FRACTIONS = None

# Nothing selects a checkpoint on the development loss - no early stopping, no
# load_best_model - so evaluating it every epoch is 92,300 forward passes that
# change nothing and can outlast the training budget itself. On only for
# diagnostics.
EVALUATE_DEV = False
# LoRA is initialised before any Trainer exists to seed the RNG, so without
# this two runs produce different adapters.
SEED = 444

# --- Inference ---------------------------------------------------------------
# Measured on this data: the label string is ~22 tokens, longest 136.
MAX_NEW_TOKENS = 256
# Install the bundled transformers/peft set instead of Kaggle's newer one.
# Kaggle's stack broke two runs at the Trainer API; the bundle is the
# combination this code was written against. False uses Kaggle's.
PIN_BUNDLED_VERSIONS = True

RUN_TRAINING   = True
RUN_INFERENCE  = True

# --- Paths -------------------------------------------------------------------
OUTPUT_DIR = "/kaggle/working/s2it_outputs"
# Point this at an attached previous run's `s2it_outputs/final` to evaluate an
# adapter trained in an earlier session (set RUN_TRAINING = False as well).
# Leave None to use the adapter this session trains.
ADAPTER_DIR = None
MODEL_DIR  = ADAPTER_DIR or f"{OUTPUT_DIR}/final"
BASE_MODEL = None       # auto-discovered; set to override
PROJECT_DIR = None      # auto-discovered from the attached zip
DATA_DIR = None

SESSION_BUDGET_SECONDS = SESSION_BUDGET_HOURS * 3600
INFERENCE_RESERVE_SECONDS = INFERENCE_RESERVE_MINUTES * 60
SAFETY_MARGIN_SECONDS = SAFETY_MARGIN_MINUTES * 60


def session_remaining():
    """Seconds left before the session budget runs out."""
    return SESSION_BUDGET_SECONDS - SAFETY_MARGIN_SECONDS - (time.perf_counter() - SESSION_START)


print(f"Session budget: {SESSION_BUDGET_HOURS:.1f} h "
      f"(reserving {INFERENCE_RESERVE_MINUTES} min for inference, "
      f"{SAFETY_MARGIN_MINUTES} min safety margin)")
print(f"Training gets about "
      f"{(SESSION_BUDGET_SECONDS - INFERENCE_RESERVE_SECONDS - SAFETY_MARGIN_SECONDS - SETUP_RESERVE_MINUTES * 60) / 3600:.1f} h")
''')

code(r'''
# ========================= LOCATE AND UNPACK INPUTS =========================
import os, sys
from pathlib import Path
from zipfile import ZipFile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"

input_root = Path("/kaggle/input")
if not input_root.is_dir():
    raise FileNotFoundError("/kaggle/input not found — run this notebook on Kaggle.")

# The bundle is uploaded as a single zip; unpack it into /kaggle/working once.
archive = next(iter(sorted(input_root.rglob("kaggle_offline_upload.zip"))), None)
if archive is not None:
    unpacked = Path("/kaggle/working/s2it_bundle")
    if not (unpacked / "project").is_dir():
        unpacked.mkdir(parents=True, exist_ok=True)
        with ZipFile(archive) as zf:
            zf.extractall(unpacked)
    PROJECT_DIR = str(unpacked / "project")
    DATA_DIR = str(unpacked / "offline_bundle")
    print("Unpacked archive:", archive)
else:
    # Fall back to a Dataset uploaded already-extracted.
    bundles = [p.parent for p in input_root.rglob("stanza_parses.jsonl")
               if (p.parent.parent / "project" / "src" / "asqp_s2it").is_dir()]
    if not bundles:
        raise FileNotFoundError(
            "Attach the Dataset built by scripts/prepare_offline_bundle.py "
            "(kaggle_offline_upload.zip, or its extracted project/ + offline_bundle/)."
        )
    DATA_DIR = str(bundles[0])
    PROJECT_DIR = str(Path(DATA_DIR).parent / "project")

if BASE_MODEL is None:
    configs = [p.parent for p in input_root.rglob("config.json") if "qwen" in str(p.parent).lower()]
    if not configs:
        raise FileNotFoundError("Attach Qwen2.5-7B-Instruct under Models, or set BASE_MODEL.")
    BASE_MODEL = str(sorted(configs, key=lambda p: ("7b" not in str(p).lower(), len(str(p))))[0])

WHEELHOUSE = str(Path(DATA_DIR) / "wheelhouse")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

for required in (
    Path(PROJECT_DIR) / "src" / "asqp_s2it" / "training.py",
    Path(DATA_DIR) / "train.jsonl",
    Path(DATA_DIR) / "dev.jsonl",
    Path(DATA_DIR) / "test.jsonl",
    Path(DATA_DIR) / "stanza_parses.jsonl",
    Path(BASE_MODEL) / "config.json",
):
    if not required.exists():
        raise FileNotFoundError(f"Missing offline input: {required}")

if ADAPTER_DIR is not None and not (Path(MODEL_DIR) / "adapter_config.json").is_file():
    raise FileNotFoundError(
        f"ADAPTER_DIR is set but {MODEL_DIR}/adapter_config.json is missing. "
        "Attach the previous run's output and point ADAPTER_DIR at its "
        "s2it_outputs/final directory."
    )

sys.path.insert(0, str(Path(PROJECT_DIR) / "src"))
print("Project :", PROJECT_DIR)
if ADAPTER_DIR is not None:
    print("Adapter :", MODEL_DIR, "(reusing a previous session)")
print("Data    :", DATA_DIR)
print("Model   :", BASE_MODEL)
print("Output  :", OUTPUT_DIR)
''')

code(r'''
# ===================== INSTALL WHEELS AND IMPORT PROJECT ====================
import subprocess

import importlib

# Kaggle ships a complete, CUDA-matched stack and keeps bumping it. Passing
# bare package names to pip here did nothing at all - pip answered
# "Requirement already satisfied" for every one and the bundled wheels were
# never installed. Rather than force a downgrade across a major transformers
# release onto Kaggle's much newer torch, the wheelhouse is now a fallback
# for packages that are genuinely absent, and the versions actually in use
# are printed below so a run can be reproduced.
REQUIRED = ("transformers", "tokenizers", "accelerate", "peft",
            "huggingface_hub", "safetensors")

# peft >= 0.19 *raises* instead of returning False when torchao is installed
# but older than it wants, and the Kaggle image pairs a new peft with a stale
# torchao. Nothing here uses torchao quantisation, so dropping it is the fix.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
               check=False)

# The bundled stack (transformers 4.48.3, peft 0.14.0, huggingface-hub 0.28.1
# and their matching tokenizers/accelerate/safetensors) is what this code was
# written against. Kaggle's own stack is newer and has already broken two runs
# at the Trainer API, so the pinned set is the default. Set this to False to
# run on whatever Kaggle ships; the code adapts, but reports what it lost.
if PIN_BUNDLED_VERSIONS:
    bundled = sorted(Path(WHEELHOUSE).glob("*.whl"))
    if bundled:
        print("Pinning the bundled stack:",
              ", ".join(w.name.split("-")[0] + " " + w.name.split("-")[1] for w in bundled))
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "--no-index", "--no-deps", "--force-reinstall", *[str(w) for w in bundled],
        ])
        importlib.invalidate_caches()
    else:
        print("PIN_BUNDLED_VERSIONS is set but the wheelhouse is empty.")

missing = []
for name in REQUIRED:
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)

if missing:
    print("Installing from the offline wheelhouse:", ", ".join(missing))
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--no-index", "--no-deps", "--find-links", WHEELHOUSE, *missing,
    ])
    for name in missing:
        importlib.invalidate_caches()
        importlib.import_module(name)
else:
    print("Every required package is already present; wheelhouse not needed.")

# Belt and braces: if torchao could not be removed, neutralise peft's probe.
try:
    import peft.tuners.lora.torchao as _lora_torchao
    _lora_torchao.is_torchao_available = lambda: False
except Exception:
    pass

import json, random
import pandas as pd
from IPython.display import display
import torch

from asqp_s2it.data import read_jsonl, category_labels, write_jsonl
from asqp_s2it.metrics import exact_quad_f1_by_instance
from asqp_s2it.model import S2ITPredictor
from asqp_s2it.syntax import CachedDependencyParser
from asqp_s2it.timing import format_duration
from asqp_s2it.training import train

import platform, transformers, peft, accelerate
print("python      :", platform.python_version())
print("torch       :", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("peft        :", peft.__version__, "| accelerate:", accelerate.__version__)
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU accelerator before running this notebook.")
''')

md(r"""
## 1. Data and dependency parses

Splits were language-tagged with `langdetect` and parsed with Stanza on the
preparation machine, so this session does neither.
""")

code(r'''
splits = {}
for name in ("train", "dev", "test"):
    records = read_jsonl(Path(DATA_DIR) / f"{name}.jsonl")
    if DATA_PERCENT < 100:
        rng = random.Random(DATA_SAMPLE_SEED + ("train", "dev", "test").index(name))
        keep = max(1, round(len(records) * DATA_PERCENT / 100))
        records = [records[i] for i in sorted(rng.sample(range(len(records)), keep))]
    splits[name] = records
train_records, dev_records, test_records = splits["train"], splits["dev"], splits["test"]
categories = category_labels(train_records)

display(pd.DataFrame([
    {"Split": k,
     "Sentences": len(v),
     "Quadruples": sum(len(r.quads) for r in v),
     "Languages": ", ".join(sorted({r.language for r in v if r.language}))}
    for k, v in splits.items()
]))

parser_started = time.perf_counter()
dependency_parser = CachedDependencyParser(Path(DATA_DIR) / "stanza_parses.jsonl")
parser_setup_seconds = time.perf_counter() - parser_started
print(f"Loaded the Stanza parse cache in {format_duration(parser_setup_seconds)}")
print(f"{len(categories)} aspect categories")
''')

md(r"""
## 2. Three-stage S2IT tuning

One LoRA adapter is tuned through the paper's three steps in order:

1. **Global Syntax-guided Extraction** — aspect→opinion and opinion→aspect pairs
   from the serialized dependency relations.
2. **Local Syntax-guided Classification** — category and sentiment from the
   one-hop subgraphs around each pair.
3. **Fine-grained Structural Tuning** — element link prediction plus node
   classification.

Each stage gets a slice of the remaining budget and **stops at its deadline**,
so the adapter is written even if the data is larger than the session allows.
`training_metrics.json` records which stages were truncated.

**Measured on this dataset**: roughly 35 min + 20 min + 1 h 55 min per epoch
for stages 1-3 at 8,557 sentences, so about 2 h 42 m per epoch. On the
1,530-sentence subset that is ~29 min per epoch, and the paper's 5 epochs
cost about 2 h 25 m. Stage 3 is the long one because it builds seven
auxiliary examples per sentence.
""")

code(r'''
if RUN_TRAINING:
    train_budget = session_remaining() - INFERENCE_RESERVE_SECONDS
    if train_budget <= 0:
        raise RuntimeError("No time left for training; raise SESSION_BUDGET_HOURS or cut the reserves.")
    print(f"Training budget: {format_duration(train_budget)}")

    notebook_training_started = time.perf_counter()
    training_metrics = train(
        train_records,
        dev_records,
        OUTPUT_DIR,
        BASE_MODEL,
        dependency_parser,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lora_rank=LORA_RANK,
        max_length=MAX_LENGTH,
        time_budget_seconds=train_budget,
        stage_fractions=STAGE_FRACTIONS,
        seed=SEED,
        evaluate_dev=EVALUATE_DEV,
    )
    training_metrics["notebook_training_wall_seconds"] = time.perf_counter() - notebook_training_started
    training_metrics["parser_setup_seconds"] = parser_setup_seconds
    (Path(MODEL_DIR) / "training_metrics.json").write_text(
        json.dumps(training_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
else:
    metrics_path = Path(MODEL_DIR) / "training_metrics.json"
    training_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
    print("Training skipped; using the existing adapter at", MODEL_DIR)

if training_metrics:
    stage_seconds = training_metrics.get("stage_training_seconds", {})
    rows = [
        {"Phase": "1. Global syntax-guided extraction", "Seconds": stage_seconds.get("global_extraction", 0.0)},
        {"Phase": "2. Local syntax-guided classification", "Seconds": stage_seconds.get("local_classification", 0.0)},
        {"Phase": "3. Fine-grained structural tuning", "Seconds": stage_seconds.get("fine_grained_structure", 0.0)},
        {"Phase": "Total tuning", "Seconds": training_metrics.get("training_seconds", 0.0)},
        {"Phase": "End to end (prep + load + save)", "Seconds": training_metrics.get("total_elapsed_seconds", 0.0)},
    ]
    timing_table = pd.DataFrame(rows)
    timing_table["Time"] = timing_table["Seconds"].map(format_duration)
    timing_table["Seconds"] = timing_table["Seconds"].round(2)
    display(timing_table)
    timing_table.to_csv(Path(OUTPUT_DIR) / "training_time.csv", index=False)

    truncated = training_metrics.get("budget", {}).get("truncated_phases", [])
    if truncated:
        print("Stopped early to stay inside the budget:", ", ".join(truncated))
        print("Raise SESSION_BUDGET_HOURS, lower DATA_PERCENT, or lower EPOCHS for full passes.")
    print("Session time left:", format_duration(max(0.0, session_remaining())))
''')

md(r"""
## 3. Prediction and exact-match F1

Inference follows the paper: the pair is decoded **bidirectionally** with beam
size 4, the two directions are merged, then the pairs are classified against
their local one-hop subgraphs.

A sentence whose output cannot be parsed contributes no quadruple instead of
aborting the run, and the loop stops at the inference deadline — any sentence
left unscored counts as a recall miss rather than silently shrinking the test
set, so the reported F1 is always over the full evaluation split.

### Accelerator: **GPU T4 x2**

Two reasons, and the second is a hard requirement:

- S2IT time goes into fine-tuning, which is compute-bound, and a T4 has
  65 TFLOPS of fp16 against the P100's 19.
- **Qwen2.5-7B in fp16 is 14.2 GiB and a single T4 has 14.6 GiB.** That
  leaves 0.37 GiB, while the paper's LoRA rank 32 needs 0.92 GiB once AdamW
  keeps its two states. One card can only run rank 8. Two cards shard the
  model through `device_map="auto"` and keep the paper's rank.

`train()` checks this before loading any weights and names the fix rather
than failing minutes later inside `Trainer.__init__` with a CUDA
out-of-memory traceback that never mentions LoRA.

### Budget

| | time |
|---|---|
| setup: wheels, model load, tokenisation | ~0h20 |
| training, 5 epochs over 1,530 sentences | ~2h25 |
| full 1,831-sentence test split, 3 generations/sentence, beam 4 | ~2h50 |
| safety margin | 0h15 |
| **total** | **~5h50 of 9h** |

Comfortable, which is the point: every hyperparameter is the paper's, and
the headroom absorbs the uncertainty in the inference estimate.

If the session still runs short, inference is what gets cut, and the metrics
table reports how many sentences were left unscored — F1 stays measured over
the full split either way.

### Why 1,530 training sentences

| sentences | epochs | train | + full eval | total |
|---|---|---|---|---|
| 8,557 | 2 | 5h24 | 2h50 | 8h47 |
| 8,557 | **5 = paper** | 13h30 | 2h50 | **16h53 — does not fit** |
| **1,530** | **5 = paper** | **2h25** | **2h50** | **5h50** |

1,530 is not an arbitrary cut: it is the size of the **Restaurant-ACOS
training split**, one of the two datasets this method was published on
(the other, Laptop-ACOS, has 2,934). Training at that scale with the
paper's 5 epochs is a faithful reproduction. Keeping all 8,557 sentences
but stopping at 2 of 5 epochs is not.

The same 1,530 sentences are in the Tree-CoT-RT bundle, and both evaluate
on the same 1,831 test sentences, so the two baselines are comparable to
each other and to anything else trained on that subset.

**Two sessions**, if you want all 8,557 sentences at 5 epochs anyway: run
this one with `RUN_INFERENCE = False`, *Save Version*, then next session
attach the output, set `RUN_TRAINING = False` and point `ADAPTER_DIR` at
the attached `s2it_outputs/final`.
""")

code(r'''
if RUN_INFERENCE:
    evaluation_records = test_records if MAX_EVAL_RECORDS is None else test_records[:MAX_EVAL_RECORDS]
    inference_budget = max(60.0, session_remaining())
    print(f"Inference budget: {format_duration(inference_budget)} for {len(evaluation_records)} sentences")

    predictor = S2ITPredictor(MODEL_DIR, dependency_parser)
    prediction_rows, unscored = [], 0
    inference_started = time.perf_counter()
    for index, instance in enumerate(evaluation_records, start=1):
        if time.perf_counter() - inference_started > inference_budget:
            unscored = len(evaluation_records) - index + 1
            prediction_rows.extend(
                {"instance_id": r.instance_id, "text": r.text, "language": r.language, "quadruples": []}
                for r in evaluation_records[index - 1:]
            )
            print(f"Inference budget reached; {unscored} sentences left unscored.")
            break
        prediction_rows.append({
            "instance_id": instance.instance_id,
            "text": instance.text,
            "language": instance.language,
            "quadruples": predictor.predict(instance, categories, max_new_tokens=MAX_NEW_TOKENS),
        })
        if index % 100 == 0 or index == len(evaluation_records):
            print(f"Predicted {index}/{len(evaluation_records)}")
    inference_seconds = time.perf_counter() - inference_started

    prediction_path = Path(OUTPUT_DIR) / "test_predictions.jsonl"
    write_jsonl(prediction_path, prediction_rows)
    evaluation = exact_quad_f1_by_instance(
        {r.instance_id: r.quads for r in evaluation_records},
        {r["instance_id"]: r["quadruples"] for r in prediction_rows},
    )

    result_rows = [
        {"Metric": "Precision", "Value": round(evaluation["precision"], 4)},
        {"Metric": "Recall", "Value": round(evaluation["recall"], 4)},
        {"Metric": "Exact quadruple micro-F1", "Value": round(evaluation["f1"], 4)},
        {"Metric": "True positive quadruples", "Value": evaluation["true_positives"]},
        {"Metric": "Predicted quadruples", "Value": evaluation["predicted"]},
        {"Metric": "Gold quadruples", "Value": evaluation["gold"]},
        {"Metric": "Sentences evaluated", "Value": len(evaluation_records)},
        {"Metric": "Sentences left unscored", "Value": unscored},
        {"Metric": "Unparsable model outputs", "Value": len(predictor.skipped)},
        {"Metric": "Inference time", "Value": format_duration(inference_seconds)},
    ]
    if training_metrics:
        result_rows.append({"Metric": "Training time",
                            "Value": training_metrics.get("training_time", "n/a")})
    results_table = pd.DataFrame(result_rows)
    display(results_table)
    results_table.to_csv(Path(OUTPUT_DIR) / "evaluation_metrics.csv", index=False)

    if predictor.skipped:
        (Path(OUTPUT_DIR) / "unparsable_outputs.json").write_text(
            json.dumps(predictor.skipped, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{len(predictor.skipped)} outputs could not be parsed; "
              "see unparsable_outputs.json")
    print("Session time left:", format_duration(max(0.0, session_remaining())))
else:
    print("Inference skipped. Set RUN_INFERENCE = True.")
''')

md(r"""
## Outputs

Download these from the notebook's **Output** tab:

- `s2it_outputs/final/` — LoRA adapter, tokenizer, `categories.json`, `training_metrics.json`
- `s2it_outputs/training_time.csv` — per-stage and total tuning time
- `s2it_outputs/test_predictions.jsonl` — predicted quadruples per sentence
- `s2it_outputs/evaluation_metrics.csv` — precision, recall, exact-quadruple F1
- `s2it_outputs/unparsable_outputs.json` — only written if some output failed to parse

`training_metrics.json` carries a `budget` block naming any stage that stopped
at its deadline; that tells you whether the run was time-limited or data-limited.
""")

notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = REPO / "s2it_kaggle.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {target} ({len(CELLS)} cells)")
