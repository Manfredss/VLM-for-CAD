#!/usr/bin/env python3
"""
prepare_rare_features.py

Select samples from the augmented dataset that contain rare features
(those under-represented or absent in the original training set).

Rare features (from train_v2.jsonl analysis):
  - Rectangular Hole Group:  0.0% in original (ABSENT)
  - Slotted Hole Group:      0.0% in original (ABSENT)
  - Rectangular Hole:         0.3% in original
  - Slotted Hole:             1.8% in original
  - Fillet Group:             4.6% in original

Strategy:
  1. From train_aug.jsonl, find samples that contain at least one rare feature.
  2. Also include a fraction of "common" samples for replay (preserving ability).
  3. Build a balanced post-training set + a small validation set.

Usage:
  python prepare_rare_features.py \
      --augmented data/train_aug.jsonl \
      --original  /tmp/train_v2.jsonl \
      --val_pool  data/val_aug.jsonl data/test_aug.jsonl \
      --output_train data/posttrain_rare.jsonl \
      --output_val   data/posttrain_rare_val.jsonl \
      --replay_ratio 0.15 \
      --seed 42
"""

import argparse
import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path


# Features considered rare in the original dataset
RARE_CATEGORIES = {
    "Rectangular Hole Group",
    "Slotted Hole Group",
    "Rectangular Hole",
    "Slotted Hole",
    "Fillet Group",
}


def parse_features(sample: dict) -> list:
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = content[0].get("text", "") if content else ""
            try:
                return json.loads(content)
            except (json.JSONDecodeError, TypeError):
                m = re.search(r"```json\s*([\s\S]*?)```", content)
                if m:
                    try:
                        return json.loads(m.group(1).strip())
                    except json.JSONDecodeError:
                        pass
    return []


def get_categories(sample: dict) -> set:
    return {f.get("category", "") for f in parse_features(sample)} - {""}


def has_rare_feature(sample: dict) -> bool:
    return bool(get_categories(sample) & RARE_CATEGORIES)


def load_jsonl(path: str) -> list:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return samples


def get_original_names(path: str) -> set:
    """Get dataitem_names from original dataset (without augmentation suffixes)."""
    names = set()
    for sample in load_jsonl(path):
        name = sample.get("dataitem_name", "")
        names.add(name)
        # Also add base name without rotation/flip suffixes
        base = re.sub(r'_(rot\d+|flip\w+)\.', '.', name)
        names.add(base)
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmented", required=True, help="Path to train_aug.jsonl")
    parser.add_argument("--original", required=True, help="Path to original train_v2.jsonl")
    parser.add_argument("--val_pool", nargs="+", required=True, help="Paths to val/test files for val set")
    parser.add_argument("--output_train", required=True)
    parser.add_argument("--output_val", required=True)
    parser.add_argument("--replay_ratio", type=float, default=0.15,
                        help="Fraction of common-feature samples to include for replay")
    parser.add_argument("--val_size", type=int, default=100,
                        help="Number of validation samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load augmented data
    aug_samples = load_jsonl(args.augmented)
    orig_names = get_original_names(args.original)

    print(f"Augmented samples: {len(aug_samples)}")
    print(f"Original sample names: {len(orig_names)}")
    print(f"Rare categories: {RARE_CATEGORIES}")

    # Separate augmented-only samples (not in original) that have rare features
    # Also collect samples that ARE augmentation variants with rare features
    rare_samples = []
    common_samples = []

    for sample in aug_samples:
        if has_rare_feature(sample):
            rare_samples.append(sample)
        else:
            common_samples.append(sample)

    print(f"\nSamples with rare features: {len(rare_samples)}")
    print(f"Samples with only common features: {len(common_samples)}")

    # Analyze rare sample distribution
    rare_cat_counts = Counter()
    for s in rare_samples:
        for cat in get_categories(s) & RARE_CATEGORIES:
            rare_cat_counts[cat] += 1
    print("\nRare feature sample counts:")
    for cat, cnt in sorted(rare_cat_counts.items(), key=lambda x: x[1]):
        print(f"  {cat}: {cnt} samples")

    # Select replay samples (common features from original to prevent forgetting)
    random.shuffle(common_samples)
    n_replay = int(len(rare_samples) * args.replay_ratio)
    replay_samples = common_samples[:n_replay]
    print(f"\nReplay samples (common features): {len(replay_samples)}")

    # Combine: all rare + replay
    train_samples = rare_samples + replay_samples
    random.shuffle(train_samples)
    print(f"Total post-training samples: {len(train_samples)}")

    # Build validation set from val_pool with rare features
    val_candidates = []
    for path in args.val_pool:
        val_candidates.extend(load_jsonl(path))

    # Deduplicate
    seen = set()
    unique_val = []
    for s in val_candidates:
        name = s.get("dataitem_name", "")
        if name not in seen:
            seen.add(name)
            unique_val.append(s)

    # Split into rare and common val samples
    val_rare = [s for s in unique_val if has_rare_feature(s)]
    val_common = [s for s in unique_val if not has_rare_feature(s)]
    random.shuffle(val_rare)
    random.shuffle(val_common)

    # Take mostly rare samples for val, with some common for balance
    n_val_rare = min(int(args.val_size * 0.7), len(val_rare))
    n_val_common = min(args.val_size - n_val_rare, len(val_common))
    val_samples = val_rare[:n_val_rare] + val_common[:n_val_common]
    random.shuffle(val_samples)

    print(f"Validation samples: {len(val_samples)} ({n_val_rare} rare + {n_val_common} common)")

    # Write outputs
    for path, samples in [(args.output_train, train_samples), (args.output_val, val_samples)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Final distribution report
    print("\n=== Post-training set distribution ===")
    all_cats = Counter()
    for s in train_samples:
        for f in parse_features(s):
            cat = f.get("category", "")
            if cat:
                all_cats[cat] += 1
    total = sum(all_cats.values())
    for cat, cnt in sorted(all_cats.items(), key=lambda x: -x[1]):
        marker = " [RARE]" if cat in RARE_CATEGORIES else ""
        print(f"  {cat}: {cnt} ({cnt/total*100:.1f}%){marker}")

    print(f"\nPost-training set saved to: {args.output_train}")
    print(f"Validation set saved to: {args.output_val}")


if __name__ == "__main__":
    main()
