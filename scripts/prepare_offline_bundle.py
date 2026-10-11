#!/usr/bin/env python
"""Prepare language-tagged JSONL and Stanza parse cache for offline Kaggle use.

Run after installing this repository and downloading Stanza models locally:
  python scripts/prepare_offline_bundle.py --data-dir asqp_data --output-dir offline_bundle
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from asqp_s2it.data import read_jsonl
from asqp_s2it.language_prep import annotate_languages
from asqp_s2it.syntax import DependencyParser

LANGUAGES = {"en", "vi", "zh-hans", "ja", "de", "th", "fr"}
ALIASES = {"zh": "zh-hans", "zh-cn": "zh-hans", "zh-tw": "zh-hans", "zh-hant": "zh-hans"}


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-dir", required=True, type=Path)
    cli.add_argument("--output-dir", required=True, type=Path)
    cli.add_argument("--stanza-dir", default=os.environ.get("STANZA_RESOURCES_DIR"), type=Path)
    args = cli.parse_args()
    if args.stanza_dir is None:
        cli.error("Set --stanza-dir or STANZA_RESOURCES_DIR to your local Stanza resources")
    os.environ["STANZA_RESOURCES_DIR"] = str(args.stanza_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parser = DependencyParser("auto", resources_dir=str(args.stanza_dir.resolve()))

    for split in ("train", "dev", "test"):
        tagged_path = args.output_dir / f"{split}.jsonl"
        annotate_languages(args.data_dir / f"{split}.jsonl", tagged_path)
        records = read_jsonl(tagged_path)
        retained = [row for row in records if (row.language or "") in LANGUAGES or (row.language or "") in ALIASES]
        if not retained:
            raise ValueError(f"No supported languages found in {split}")
        with tagged_path.open("w", encoding="utf-8", newline="\n") as out:
            for record in retained:
                row = {
                    "instance_id": record.instance_id,
                    "text": record.text,
                    "language": ALIASES.get(record.language or "", record.language),
                    "quads": [quad.as_dict() for quad in record.quads],
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    cache_path = args.output_dir / "stanza_parses.jsonl"
    count = 0
    with cache_path.open("w", encoding="utf-8", newline="\n") as out:
        for split in ("train", "dev", "test"):
            for record in read_jsonl(args.output_dir / f"{split}.jsonl"):
                parsed = parser.parse(record.text, record.language)
                out.write(json.dumps({"text": record.text, "language": record.language,
                                      "words": parsed.words, "heads": parsed.heads},
                                     ensure_ascii=False) + "\n")
                count += 1
                if count % 500 == 0:
                    print(f"Parsed {count} sentences", flush=True)
    print(f"Offline bundle ready: {args.output_dir} ({count} cached parses)")


if __name__ == "__main__":
    main()
