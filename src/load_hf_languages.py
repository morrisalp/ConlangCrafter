#!/usr/bin/env python3
"""
Load all ConlangCrafter language sketches from HuggingFace into the local output directory,
so they can be used as input for downstream steps (e.g. translation).

Usage:
    python src/load_hf_language.py
    python src/load_hf_language.py --output-dir output
"""

import os
import json
import argparse

HF_DATASET = "malper/ConlangCrafter"


def save_language(row: dict, output_dir: str):
    language_id = row["language_id"]
    lang_dir = os.path.join(output_dir, "languages", language_id)
    memory_dir = os.path.join(lang_dir, "memory")

    for step in ("phonology", "grammar", "lexicon"):
        os.makedirs(os.path.join(memory_dir, step), exist_ok=True)

    for filename, field in [("phonology/phonology.txt", "phonology"),
                             ("grammar/grammar.txt", "grammar"),
                             ("lexicon/lexicon.csv", "lexicon")]:
        with open(os.path.join(memory_dir, filename), "w", encoding="utf-8") as f:
            f.write(row[field])
        with open(os.path.join(memory_dir, os.path.dirname(filename), "metadata.json"), "w") as f:
            json.dump({}, f)

    metadata = {
        "language_id": language_id,
        "source": HF_DATASET,
        "model": row.get("model", "unknown"),
        "steps": ["phonology", "grammar", "lexicon"],
        "custom_constraints": None,
        "parameters": {"temperature": None, "max_tokens": None},
    }
    with open(os.path.join(lang_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return lang_dir


def main():
    parser = argparse.ArgumentParser(
        description="Load all ConlangCrafter language sketches from HuggingFace"
    )
    parser.add_argument("--output-dir", default="output", help="Local output directory")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install the 'datasets' package: pip install datasets")

    print(f"Loading dataset from {HF_DATASET}...")
    ds = load_dataset(HF_DATASET, split="test")
    rows = list(ds)

    for row in rows:
        lang_dir = save_language(row, args.output_dir)
        print(f"Saved {row['language_id']} → {lang_dir}")

    print(f"\nLoaded {len(rows)} language(s). To translate, run:")
    print(f"  python src/run_pipeline.py --language-id <id> --steps translation")


if __name__ == "__main__":
    main()
