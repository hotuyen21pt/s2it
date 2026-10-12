#!/usr/bin/env python
"""Build the offline Kaggle upload for S2IT, end to end, on your local machine.

Kaggle runs with no Internet, so everything the notebook needs has to be in one
uploaded archive. This script does all of it in one command:

  1. run langdetect once and write the language into every JSONL record,
  2. download the Stanza models for the languages actually present,
  3. parse every sentence with Stanza and cache words/heads to JSONL,
  4. download the Linux wheels Kaggle is missing,
  5. zip project source + data + parse cache + wheelhouse.

The Qwen model is NOT packaged: attach it separately as a Kaggle Model input.

    pip install -e ".[prepare]"
    python scripts/prepare_offline_bundle.py --data-dir asqp_data

Produces `kaggle_offline_upload.zip`; upload it as a Kaggle Dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from asqp_s2it.data import read_jsonl  # noqa: E402
from asqp_s2it.language_prep import annotate_languages  # noqa: E402

SPLITS = ("train", "dev", "test")

#: Stanza language codes this project supports; records in other languages are
#: dropped so no unused parser model is downloaded or packaged.
LANGUAGES = {"en", "vi", "zh-hans", "ja", "de", "th", "fr"}
ALIASES = {"zh": "zh-hans", "zh-cn": "zh-hans", "zh-tw": "zh-hans", "zh-hant": "zh-hans"}

#: Kaggle images ship torch, numpy and pandas but lag on these.
#: `syntax.py` imports Stanza lazily, so the Kaggle side never needs it.
WHEELS = (
    "transformers==4.48.3",
    "tokenizers==0.21.0",
    "accelerate==1.15.0",
    "peft==0.14.0",
    "huggingface-hub==0.28.1",
    "safetensors==0.6.2",
)
KAGGLE_PYTHON = "3.13"  # Kaggle image Python; check a run log if it changes
KAGGLE_PLATFORM = "manylinux2014_x86_64"

PROJECT_FILES = ("pyproject.toml", "README.md")


#: Directory names checked for train/dev/test.jsonl when --data-dir is omitted.
DATA_DIR_CANDIDATES = ("asqp_data", "converted_apc")


def default_data_dir() -> Path:
    for name in DATA_DIR_CANDIDATES:
        candidate = REPO_ROOT / name
        if all((candidate / f"{split}.jsonl").is_file() for split in SPLITS):
            return candidate
    return REPO_ROOT / DATA_DIR_CANDIDATES[0]


def log(message: str) -> None:
    print(f"==> {message}", flush=True)


def tag_languages(data_dir: Path, bundle: Path) -> None:
    log("Step 1/5: detecting languages with langdetect")
    for split in SPLITS:
        source = data_dir / f"{split}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(f"Missing split: {source}")
        tagged = bundle / f"{split}.jsonl"
        counts = annotate_languages(source, tagged)
        records = read_jsonl(tagged)
        retained = [
            row
            for row in records
            if (row.language or "") in LANGUAGES or (row.language or "") in ALIASES
        ]
        dropped = len(records) - len(retained)
        if not retained:
            raise ValueError(
                f"{split}: no record is in a supported language {sorted(LANGUAGES)}. "
                f"Detected: {dict(counts)}"
            )
        with tagged.open("w", encoding="utf-8", newline="\n") as out:
            for record in retained:
                out.write(
                    json.dumps(
                        {
                            "instance_id": record.instance_id,
                            "text": record.text,
                            "language": ALIASES.get(record.language or "", record.language),
                            "quads": [quad.as_dict() for quad in record.quads],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        note = f", dropped {dropped} unsupported" if dropped else ""
        print(f"    {split}: {len(retained)} records{note}")


def select_subset(
    bundle: Path,
    sample_train: int | None,
    seed: int,
    subset_ids: Path | None,
) -> dict[str, list[str]]:
    """Trim the training split to a fixed, reproducible set of sentences.

    Both baselines must train on exactly the same sentences or their scores
    cannot be compared, so the chosen ids are written to subset_ids.json and
    the other repository reads them back with --subset-ids.
    """
    chosen: dict[str, list[str]] = {}
    if subset_ids is not None:
        payload = json.loads(subset_ids.read_text(encoding="utf-8"))
        wanted = {split: set(ids) for split, ids in payload.items()}
        log(f"Applying the shared subset from {subset_ids}")
    elif sample_train is not None:
        import random

        rows = read_jsonl(bundle / "train.jsonl")
        if sample_train >= len(rows):
            raise SystemExit(
                f"--sample-train {sample_train} is not smaller than the {len(rows)} "
                "available training sentences; omit it to keep them all."
            )
        ids = sorted(row.instance_id for row in rows)   # sort first: stable across runs
        picked = random.Random(seed).sample(ids, sample_train)
        # dev and test are recorded in full, not sampled: the other repository
        # must evaluate on exactly these sentences or the F1 scores are not
        # comparable. This repo's language filter has already removed records
        # the other one still has.
        wanted = {
            "train": set(picked),
            "dev": {row.instance_id for row in read_jsonl(bundle / "dev.jsonl")},
            "test": {row.instance_id for row in read_jsonl(bundle / "test.jsonl")},
        }
        log(f"Sampling {sample_train}/{len(rows)} training sentences (seed {seed}); "
            "dev/test recorded in full for cross-repo alignment")
    else:
        return chosen

    for split in SPLITS:
        if split not in wanted:
            continue
        path = bundle / f"{split}.jsonl"
        kept = []
        with path.open(encoding="utf-8") as stream:
            lines = [line for line in stream if line.strip()]
        with path.open("w", encoding="utf-8", newline="\n") as out:
            for line in lines:
                row = json.loads(line)
                if str(row.get("instance_id")) in wanted[split]:
                    out.write(line if line.endswith("\n") else line + "\n")
                    kept.append(str(row.get("instance_id")))
        missing = wanted[split] - set(kept)
        if missing:
            raise SystemExit(
                f"{split}: {len(missing)} ids from the shared subset are not in this "
                f"bundle, for example {sorted(missing)[:3]}. The two repositories "
                "started from different data."
            )
        chosen[split] = kept
        print(f"    {split}: kept {len(kept)} sentences")
    (bundle / "subset_ids.json").write_text(
        json.dumps(chosen, indent=2) + "\n", encoding="utf-8"
    )
    return chosen


def present_languages(bundle: Path) -> list[str]:
    languages: set[str] = set()
    for split in SPLITS:
        languages.update(
            row.language for row in read_jsonl(bundle / f"{split}.jsonl") if row.language
        )
    return sorted(languages)


def download_stanza(languages: list[str], stanza_dir: Path) -> None:
    log(f"Step 2/5: downloading Stanza models for {', '.join(languages)}")
    try:
        import stanza
    except ImportError as error:
        raise SystemExit(
            "Stanza is required to build the bundle (but NOT to run on Kaggle).\n"
            "Install it on this machine:  pip install stanza"
        ) from error

    stanza_dir.mkdir(parents=True, exist_ok=True)
    os.environ["STANZA_RESOURCES_DIR"] = str(stanza_dir)
    for language in languages:
        target = stanza_dir / language
        if target.is_dir() and any(target.rglob("*.pt")):
            print(f"    {language}: already downloaded")
            continue
        # `processors=` keeps this to the depparse chain instead of the full
        # default package, which also pulls constituency and NER models.
        stanza.download(
            language,
            model_dir=str(stanza_dir),
            processors="tokenize,pos,lemma,depparse",
            package="default",
            verbose=False,
        )
        print(f"    {language}: downloaded")


def cache_parses(bundle: Path, stanza_dir: Path) -> int:
    log("Step 3/5: caching Stanza dependency parses")
    os.environ["STANZA_RESOURCES_DIR"] = str(stanza_dir)
    from asqp_s2it.syntax import DependencyParser  # imports Stanza lazily

    parser = DependencyParser("auto", resources_dir=str(stanza_dir))
    cache_path = bundle / "stanza_parses.jsonl"
    seen: set[tuple[str, str]] = set()
    count = 0
    with cache_path.open("w", encoding="utf-8", newline="\n") as out:
        for split in SPLITS:
            for record in read_jsonl(bundle / f"{split}.jsonl"):
                key = (record.text, record.language or "")
                if key in seen:
                    continue
                seen.add(key)
                parsed = parser.parse(record.text, record.language)
                out.write(
                    json.dumps(
                        {
                            "text": record.text,
                            "language": record.language,
                            "words": parsed.words,
                            "heads": parsed.heads,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                count += 1
                if count % 500 == 0:
                    print(f"    parsed {count} unique sentences", flush=True)
    print(f"    cached {count} unique parses")
    return count


def reuse_parses(bundle: Path, existing: Path) -> int:
    """Adopt a parse cache built elsewhere, after checking it covers the data.

    The cache format is (text, language) -> words/heads and does not depend on
    the model code, so a cache from an earlier bundle stays valid as long as
    the splits are unchanged. This is the escape hatch when Stanza cannot run
    on the local interpreter.
    """
    log("Step 3/5: reusing an existing Stanza parse cache")
    cached: set[tuple[str, str]] = set()
    rows = 0
    with existing.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{existing}:{line_number}: invalid JSON: {error}") from error
            for field in ("text", "language", "words", "heads"):
                if field not in row:
                    raise SystemExit(f"{existing}:{line_number}: missing field {field!r}")
            if len(row["words"]) != len(row["heads"]):
                raise SystemExit(f"{existing}:{line_number}: words/heads length mismatch")
            cached.add((row["text"], row["language"] or ""))
            rows += 1

    needed: set[tuple[str, str]] = set()
    for split in SPLITS:
        for record in read_jsonl(bundle / f"{split}.jsonl"):
            needed.add((record.text, record.language or ""))
    missing = needed - cached
    if missing:
        sample = list(missing)[:3]
        raise SystemExit(
            f"The reused cache covers {len(needed) - len(missing)}/{len(needed)} sentences.\n"
            f"{len(missing)} are missing, for example: {sample}\n"
            "The cache was built from different splits or a different language tagging. "
            "Rebuild it with Stanza instead of reusing."
        )
    shutil.copy2(existing, bundle / "stanza_parses.jsonl")
    print(f"    reused {rows} parses; all {len(needed)} sentences covered")
    return rows


def download_wheels(wheelhouse: Path) -> None:
    log(f"Step 4/5: downloading Linux py{KAGGLE_PYTHON} wheels")
    wheelhouse.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable, "-m", "pip", "download",
            "--dest", str(wheelhouse),
            "--only-binary=:all:",
            "--no-deps",
            "--python-version", KAGGLE_PYTHON,
            "--platform", KAGGLE_PLATFORM,
            *WHEELS,
        ]
    )
    print(f"    {len(list(wheelhouse.glob('*.whl')))} wheels in {wheelhouse}")


def copy_project(staging: Path) -> None:
    project = staging / "project"
    if project.exists():
        shutil.rmtree(project)
    (project / "src").mkdir(parents=True)
    shutil.copytree(
        REPO_ROOT / "src" / "asqp_s2it",
        project / "src" / "asqp_s2it",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in PROJECT_FILES:
        if (REPO_ROOT / name).is_file():
            shutil.copy2(REPO_ROOT / name, project / name)


def write_upload_readme(staging: Path, parse_count: int, languages: list[str]) -> None:
    (staging / "UPLOAD_README.txt").write_text(
        "S2IT offline Kaggle bundle\n"
        "==========================\n\n"
        "Contents\n"
        "  project/          S2IT source; the notebook imports from project/src\n"
        "  offline_bundle/   language-tagged JSONL splits, Stanza parse cache, wheelhouse\n\n"
        f"Cached dependency parses: {parse_count}\n"
        f"Languages: {', '.join(languages)}\n\n"
        "How to use on Kaggle\n"
        "  1. Upload kaggle_offline_upload.zip as a Kaggle Dataset.\n"
        "  2. Attach that Dataset to the notebook.\n"
        "  3. Attach Qwen2.5-7B-Instruct under Models (NOT in this archive).\n"
        "  4. Accelerator: GPU. Internet: OFF.\n"
        "  5. Run all cells; the config cell finds everything under /kaggle/input.\n\n"
        "The archive deliberately excludes model weights and training outputs.\n"
        "Stanza is NOT needed on Kaggle: the notebook reads stanza_parses.jsonl.\n",
        encoding="utf-8",
    )


def build_zip(staging: Path, archive_path: Path, notebook: Path | None) -> None:
    log("Step 5/5: writing the upload archive")
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(staging.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            # Forward slashes so the archive unpacks correctly on Kaggle/Linux.
            archive.write(path, path.relative_to(staging).as_posix())
        if notebook is not None and notebook.is_file():
            archive.write(notebook, notebook.name)
    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"    {archive_path} ({size_mb:.1f} MB)")


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help=f"train/dev/test.jsonl location (auto: {' or '.join(DATA_DIR_CANDIDATES)})",
    )
    cli.add_argument("--staging-dir", type=Path, default=REPO_ROOT / "kaggle_offline_upload")
    cli.add_argument("--archive", type=Path, default=REPO_ROOT / "kaggle_offline_upload.zip")
    cli.add_argument(
        "--stanza-dir",
        type=Path,
        default=Path(os.environ.get("STANZA_RESOURCES_DIR", Path.home() / "stanza_resources")),
        help="where Stanza models are downloaded to (not packaged into the zip)",
    )
    cli.add_argument(
        "--sample-train",
        type=int,
        help="keep only this many training sentences (writes subset_ids.json)",
    )
    cli.add_argument("--sample-seed", type=int, default=444)
    cli.add_argument(
        "--subset-ids",
        type=Path,
        help="reuse the subset_ids.json written by the other repository, so both "
             "baselines train on identical sentences",
    )
    cli.add_argument("--skip-wheels", action="store_true", help="reuse an existing wheelhouse")
    cli.add_argument(
        "--reuse-parses",
        type=Path,
        help="adopt an existing stanza_parses.jsonl instead of running Stanza; "
             "coverage against the splits is verified first",
    )
    args = cli.parse_args()

    staging = args.staging_dir
    bundle = staging / "offline_bundle"
    bundle.mkdir(parents=True, exist_ok=True)

    tag_languages(args.data_dir, bundle)
    select_subset(bundle, args.sample_train, args.sample_seed, args.subset_ids)
    languages = present_languages(bundle)
    if args.reuse_parses is not None:
        parse_count = reuse_parses(bundle, args.reuse_parses)
    else:
        download_stanza(languages, args.stanza_dir)
        parse_count = cache_parses(bundle, args.stanza_dir)
    if not args.skip_wheels:
        download_wheels(bundle / "wheelhouse")
    copy_project(staging)
    write_upload_readme(staging, parse_count, languages)
    build_zip(staging, args.archive, REPO_ROOT / "s2it_kaggle.ipynb")
    print("\nDone. Upload the archive as a Kaggle Dataset and attach Qwen2.5-7B-Instruct separately.")


if __name__ == "__main__":
    main()
