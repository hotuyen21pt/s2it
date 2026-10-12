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
import time
SESSION_START = time.perf_counter()

# --- Wall-clock budget -------------------------------------------------------
# Kaggle kills the session at the limit, which would lose the LoRA adapter.
# Training stops at its own deadline instead, so the adapter is always saved.
SESSION_BUDGET_HOURS     = 9.0   # Kaggle GPU session limit
SETUP_RESERVE_MINUTES    = 20    # wheels, model load, tokenisation
INFERENCE_RESERVE_MINUTES = 100  # beam-4 decoding is slow; see note below
SAFETY_MARGIN_MINUTES    = 15    # never finish exactly on the limit

# --- Data --------------------------------------------------------------------
DATA_PERCENT     = 100   # sample a slice of every split for a quick trial
DATA_SAMPLE_SEED = 444
MAX_EVAL_RECORDS = 600   # see 'Evaluating the full test split' below; None = all

# --- Fine-tuning -------------------------------------------------------------
# These are the values that fit a 16 GB Kaggle GPU with a 7B backbone.
# The S2IT paper uses rank 32 / max_length 2048 / 5 epochs on an A100; raise
# them only if you move to a bigger accelerator.
EPOCHS                      = 2
BATCH_SIZE                  = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE               = 5e-5
LORA_RANK                   = 8
MAX_LENGTH                  = 768
# Share of the training budget per stage. Fine-grained structural tuning gets
# the most because it builds seven auxiliary examples per sentence.
STAGE_FRACTIONS = (0.25, 0.25, 0.50)

# --- Inference ---------------------------------------------------------------
MAX_NEW_TOKENS = 256   # the paper's label format is short; 256 is ample
RUN_TRAINING   = True
RUN_INFERENCE  = True

# --- Paths -------------------------------------------------------------------
OUTPUT_DIR = "/kaggle/working/s2it_outputs"
MODEL_DIR  = f"{OUTPUT_DIR}/final"
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

sys.path.insert(0, str(Path(PROJECT_DIR) / "src"))
print("Project :", PROJECT_DIR)
print("Data    :", DATA_DIR)
print("Model   :", BASE_MODEL)
print("Output  :", OUTPUT_DIR)
''')

code(r'''
# ===================== INSTALL WHEELS AND IMPORT PROJECT ====================
import subprocess

# Only pinned local wheels; --no-deps avoids the resolver reaching the network.
# Kaggle already ships torch, numpy and pandas. Stanza is deliberately absent:
# asqp_s2it.syntax imports it lazily and this run uses the parse cache.
wheels = sorted(Path(WHEELHOUSE).glob("*.whl"))
if wheels:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--no-index", "--no-deps", "--find-links", WHEELHOUSE,
        *[w.name.split("-")[0].replace("_", "-") for w in wheels],
    ])
else:
    print("No wheelhouse found; relying on the preinstalled Kaggle packages.")

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

print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
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

**Measured on this dataset** (8,815 train sentences, 16 GB GPU, the settings
above): roughly 35 min + 20 min + 1 h 55 min per epoch for stages 1-3, so
`EPOCHS = 2` lands near 5 h 30 m and fits the training budget. Stage 3 is the
long one because it builds seven auxiliary examples per sentence.
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

### Evaluating the full test split

Beam-4 decoding runs three generations per sentence, so all 1,890 test
sentences need roughly 3 hours — more than one session has left after
training. To report F1 over the whole split, use two sessions:

1. **This session**: `RUN_INFERENCE = False`. Train, then *Save Version* so
   `s2it_outputs/final/` becomes a downloadable output.
2. **Next session**: attach this notebook's output as an input, set
   `RUN_TRAINING = False`, `MAX_EVAL_RECORDS = None`, and point `MODEL_DIR`
   at the attached `s2it_outputs/final`. The full 9 hours then goes to inference.

With the default `MAX_EVAL_RECORDS = 600` both fit in one session, and the
metrics table reports exactly how many sentences were scored.
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
