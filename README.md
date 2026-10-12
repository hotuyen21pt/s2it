# S2IT — Stepwise Syntax Integration Tuning (ASQP)

Reproduction of **S2IT** (Chen et al., *Findings of NAACL 2025*; the PDF is in
this repo as `2025.findings-naacl.379.pdf`) for Aspect Sentiment Quad
Prediction, run on a private hotel-review dataset.

**Purpose: this is a baseline.** Its scores will be published next to another
method, so faithfulness to the paper matters more than peak F1. Several
things here look like they could be "improved" and must not be — see
[Decisions that must not be reverted](#decisions-that-must-not-be-reverted).

This repo is coupled to a sibling repo, **tree** (Tree-CoT-RT), which is the
second baseline. The two must train and evaluate on byte-identical splits;
see [Shared data subset](#shared-data-subset).

---

## Current state

| | |
|---|---|
| Status | Code aligned with the paper; **not yet run end to end since the rewrite** |
| Tests | 22 pass. `tests/test_syntax.py` imports Stanza, so without it pytest fails at collection — run with `--ignore=tests/test_syntax.py` |
| Kaggle bundle | Built and ready at `kaggle_offline_upload.zip` (not in git) |
| Last real training run | Before the rewrite, on the old JSON format. **That adapter is obsolete** — the output format changed, so it must be retrained. |

The previous Kaggle run trained for 8h07 and then **crashed at inference** with
`JSONDecodeError`. That is fixed; see the parsing decision below.

---

## What the method does

Three fine-tuning stages run in order on one LoRA adapter:

1. **Global Syntax-guided Extraction** (§2.2) — given the sentence plus its
   dependency relations serialised to natural language, emit aspect–opinion
   pairs. Generated in both directions (aspect→opinion and opinion→aspect).
2. **Local Syntax-guided Classification** (§2.3) — given one-hop dependency
   subgraphs around each pair, assign category and sentiment.
3. **Fine-grained Structural Tuning** (§2.4) — six auxiliary tasks: element
   link prediction in both directions, and node classification of each
   element to category and to sentiment.

Inference: bidirectional pair extraction with beam size 4, merge, then
classify (Appendix B).

---

## Decisions that must not be reverted

These were deliberate corrections against the paper. Each has a reason a
future agent is likely to miss.

| Decision | Why |
|---|---|
| **Labels are natural language**, `aspect: service, opinion: ok \| ...`, not JSON | Table 4 of the paper: symbol/bracket templates cost 3.4 F1 on Restaurant and 2.0 on Laptop because they fight the LLM's pretraining objective. Appendix A.2 shows this exact format. JSON is the same kind of mismatch, and it is what caused the `JSONDecodeError` crash. |
| **The output parser never raises** | A malformed segment is dropped; the run continues. One unescaped quote used to abort evaluation after eight hours of training. `S2ITPredictor.skipped` records what was dropped. |
| The parser also reads the **old JSON format** | So an adapter trained before the rewrite can still be evaluated without retraining. |
| **Node classification sees only the element it classifies** | Passing the paired element leaked the very link the auxiliary task exists to learn. |
| **Beam 4 on both generation steps** | Appendix B specifies beam search for inference; the classification step was previously greedy. |
| **Stanza is imported lazily** (`syntax.py`) | A Kaggle run reads the precomputed parse cache and needs neither the package nor its models. This cut the offline wheelhouse from 17 packages to 6. Do not move the import back to module level. |
| `BATCH_SIZE=1`, `GRADIENT_ACCUMULATION_STEPS=8` | The paper's "batch 4, accumulation 2" is an **effective batch of 8**. 1×8 is the same optimiser trajectory and fits 16 GB. This is not a deviation. |
| `MAX_LENGTH = 1024` | Measured on this data: the longest stage-3 example is 786 tokens. The earlier 768 silently truncated one example in 3,600. |
| `MAX_NEW_TOKENS = 256` | Measured: the label string is ~22 tokens, longest 136. |

### Known deviations from the paper

- **Training set is 1,530 sentences, not all 8,557.** See
  [Why 1,530](#why-1530-sentences). Hyperparameters are the paper's.
- The dataset is this project's own hotel reviews, not Restaurant/Laptop
  ACOS, so the paper's published numbers are not comparable to these.
- **Single run.** The paper does not state a seed policy, but one run is one
  run; report it as such.
- **fp16, not bf16.** The paper does not state a precision; Kaggle's T4 and
  P100 have no bf16, so `training.py` falls back to fp16.

---

## Layout

```
src/asqp_s2it/
  prompts.py        prompt templates + natural-language label (de)serialisation
  training.py       the three stages, with a wall-clock deadline each
  model.py          S2ITPredictor: bidirectional extraction then classification
  syntax.py         Stanza wrapper (lazy import) + CachedDependencyParser
  data.py           JSONL reader, Quad/Instance types
  metrics.py        exact-quadruple micro-F1
  language_prep.py  langdetect tagging, run once at bundle-build time
  budget.py         Budget/Deadline: stop a phase before Kaggle kills the session
  cli.py            prepare-parser | prepare-language | train | predict | evaluate
scripts/
  prepare_offline_bundle.py   build the Kaggle upload (langdetect, Stanza, wheels, zip)
  build_notebook.py           regenerate s2it_kaggle.ipynb — EDIT THIS, not the .ipynb
tests/                        22 tests; test_syntax.py needs Stanza
```

> `s2it_kaggle.ipynb` is **generated**. Edit `scripts/build_notebook.py` and
> re-run it, or your change is lost on the next rebuild.

---

## Data

Not in git (too large, and distribution-restricted). Expected locally:

```
converted_apc/{train,dev,test}.jsonl     8,815 / 1,895 / 1,890 sentences
stanza_parses.jsonl                      12,234 cached Stanza parses
```

`asqp_data/` is also accepted as the directory name; the script checks both.

The data is **multilingual**. `langdetect` tags every record, then records
outside `{en, vi, zh-hans, ja, de, th, fr}` are dropped:

| | en | vi | de | fr | th | zh-hans | dropped |
|---|---|---|---|---|---|---|---|
| train | 6,918 | 1,346 | 131 | 111 | 50 | 1 | 258 |
| dev | 1,485 | 289 | 30 | 34 | 7 | 1 | 49 |
| test | 1,485 | 286 | 14 | 37 | 8 | 1 | 59 |

Leaving **8,557 / 1,846 / 1,831**.

### Building the Kaggle bundle

```bash
pip install -e ".[prepare]"        # stanza + langdetect, build machine only
python scripts/prepare_offline_bundle.py --sample-train 1530
```

Steps: langdetect tagging → download Stanza models for the languages present
→ cache every parse → download Linux/py3.12 wheels → zip.

`stanza_parses.jsonl` already covers all 12,030 unique sentences, so Stanza
need not run at all:

```bash
python scripts/prepare_offline_bundle.py \
    --reuse-parses stanza_parses.jsonl --sample-train 1530
```

`--reuse-parses` verifies the cache covers every sentence before accepting it.

Output: `kaggle_offline_upload.zip` (~15 MB) containing `project/`,
`offline_bundle/`, `UPLOAD_README.txt` and the notebook. Model weights are
**not** in it.

### Shared data subset

`--sample-train 1530` writes `offline_bundle/subset_ids.json` with the chosen
train ids **and the full dev/test id lists**. The tree repo consumes it:

```bash
cd ../tree && python scripts/prepare_offline_bundle.py \
    --subset-ids ../s2it/kaggle_offline_upload/offline_bundle/subset_ids.json
```

Both bundles then hold identical splits (verified: 1530 / 1846 / 1831, same
ids, same texts). Dev and test are recorded in full because this repo's
language filter removes records the tree repo still has — without alignment
the two F1 scores would be measured over different test sets.

**Any third method compared against these baselines must use the same
`subset_ids.json`.**

### The evaluation set

**The test split is never sampled.** Training was cut to 1,530 sentences on
purpose; evaluation keeps every sentence it can score.

| | sentences |
|---|---|
| `converted_apc/test.jsonl` as delivered | 1,890 |
| removed by the language filter | 59 |
| **evaluated, by both baselines** | **1,831** |

Those 59 are spread over 17 languages with no parser in the supported set
(ro, es, af, ca, no, da, it, sv, ko, cy, et, ru, sw, tl, nl, pt, hr). S2IT
structurally cannot score them: it needs a dependency parse, and Stanza has
no model for them here. 1,831 is therefore the largest test set the two
baselines can share.

So:

- **s2it vs tree** — valid, identical 1,831 sentences, verified id by id.
- **vs a third method** — valid only if it is scored on the same 1,831.
  Use `offline_bundle/subset_ids.json`, which lists them.
- **vs the numbers printed in the papers** — not comparable at all. Those
  are Restaurant/Laptop ACOS; this is a private hotel-review dataset.

A method that needs no parser could be scored on all 1,890, but then it is
no longer comparable to S2IT. Report 1,831 for everything.

### Why 1,530 sentences

1,530 is the size of the **Restaurant-ACOS training split**, one of the two
datasets S2IT was published on (Laptop-ACOS has 2,934). Reducing data while
keeping the paper's hyperparameters is a faithful reproduction at a published
scale. The alternative — all 8,557 sentences but 2 of the paper's 5 epochs —
is a cut-down method, which is worse for a baseline.

| sentences | epochs | train | + full eval | total |
|---|---|---|---|---|
| 8,557 | 2 | 5h24 | 2h50 | 8h47 |
| 8,557 | 5 = paper | 13h30 | 2h50 | **16h53 — does not fit** |
| **1,530** | **5 = paper** | **2h25** | **2h50** | **5h50** |

---

## Running on Kaggle

Upload `kaggle_offline_upload.zip` as a Dataset, import `s2it_kaggle.ipynb`,
attach the Dataset plus the **Qwen2.5-7B-Instruct** Kaggle Model, then Run All.

- **Accelerator: T4.** S2IT is dominated by fine-tuning, which is
  compute-bound, and the T4 has 65 TFLOPS fp16 against the P100's 19.
  (Tree-CoT-RT is the opposite case and wants the P100.)
- **Internet: Off.**

The notebook finds the zip under `/kaggle/input`, unpacks it, locates Qwen,
and installs the six wheels from the bundle.

### The Kaggle environment is not pinned
> **Two runs were lost to the Kaggle image's transformers.** `PIN_BUNDLED_VERSIONS`
> defaults to `True`, which force-installs the bundled transformers 4.48.3 /
> peft 0.14.0 set this code was written against. Set it to `False` to run on
> whatever Kaggle ships; `_training_arguments()` then filters `TrainingArguments`
> against the installed signature and prints what it had to drop.

The bundled wheelhouse is a **fallback, not a pin**. The notebook installs
from it only what is genuinely missing, and prints the versions actually in
use. Two findings forced this:

- Passing bare package names to `pip install --no-index --find-links` did
  nothing: pip answered *"Requirement already satisfied"* for all six and the
  bundled wheels were never installed. The run silently used Kaggle's stack.
- Kaggle moves fast. One observed image: Python 3.13, torch 2.11,
  transformers 5.16, peft 0.20. Forcing a downgrade to transformers 4.48 on
  that torch build is riskier than using what Kaggle tests together.

Consequences to be aware of:

- `peft >= 0.19` **raises** instead of returning `False` when `torchao` is
  installed but older than it wants, and the image pairs a new peft with a
  stale torchao. The notebook uninstalls `torchao` (nothing here uses torchao
  quantisation) and, as a fallback, neutralises peft's probe directly.
- `torch_dtype` was renamed `dtype` in transformers 5. `_dtype_kwargs()`
  picks the name the installed version accepts, so both majors work.
- **Record the printed versions with any result.** The environment is not
  reproducible from this repo alone.

### Session budget

Kaggle kills a session at the limit, which would lose the adapter. Each of the
three stages gets a deadline from `budget.py` and stops cleanly instead;
`training_metrics.json` records a `budget` block naming any stage that was
truncated. Inference likewise stops at its deadline and reports how many
sentences were left unscored, so F1 is always measured over the full split.

`ADAPTER_DIR` lets a second session evaluate an adapter trained in a first one
(`RUN_TRAINING = False`).

---

## Local development

```bash
pip install -e ".[dev]"
python -m pytest -q --ignore=tests/test_syntax.py   # 22 pass
```

`tests/test_syntax.py` imports Stanza at collection time, so it is ignored
above. With Stanza installed, drop the flag.

CLI:

```bash
asqp-s2it prepare-language --input raw.jsonl --output tagged.jsonl
asqp-s2it prepare-parser --languages en,vi,de,fr,th,zh-hans
asqp-s2it train   --train tagged_train.jsonl --dev tagged_dev.jsonl --output-dir runs/s2it
asqp-s2it predict --input tagged_test.jsonl --model-dir runs/s2it/final --output preds.jsonl
asqp-s2it evaluate --gold tagged_test.jsonl --predictions preds.jsonl
```

`train(..., time_budget_seconds=None)` disables the deadline, which is what
you want off Kaggle.

---

## Not in git

`converted_apc/`, `stanza_parses.jsonl`, `kaggle_offline_upload/`,
`kaggle_offline_upload.zip`, `runs/`, caches. All are reproducible from the
data plus `scripts/prepare_offline_bundle.py`. A fresh clone has no data.

---

## Open items

- Not yet run end to end since the rewrite. The first Kaggle run is the real
  test of the natural-language format.
- Runtime figures above are extrapolated from one logged run (2h42/epoch at
  8,557 sentences) and a memory-bandwidth model for inference. Expect ±40%.
- `tests/test_syntax.py` cannot run without Stanza, which needs Python ≤3.13
  (torch has no 3.14 wheels).
