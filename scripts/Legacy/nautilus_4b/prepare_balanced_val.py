#!/usr/bin/env python3
"""
prepare_balanced_val.py

Combine val_aug.jsonl + test_aug.jsonl and select a balanced validation set
whose per-category feature proportion matches that of train_aug.jsonl.

Strategy:
  1. Compute category proportions in training set (feature-level).
  2. Pool all candidate samples from val + test.
  3. For each sample, tag it with its "dominant" category (most features).
  4. Greedily select samples to approximate the target distribution.
  5. Output the selected samples as val_balanced.jsonl.

Usage:
  python prepare_balanced_val.py \
      --train data/train_aug.jsonl \
      --val   data/val_aug.jsonl \
      --test  data/test_aug.jsonl \
      --output data/val_balanced.jsonl \
      --target_size 400
"""

import argparse
import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_features(sample: dict) -> list:
    """Extract feature list from a sample's assistant message."""
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


def count_categories(samples: list) -> Counter:
    cats = Counter()
    for sample in samples:
        for feat in parse_features(sample):
            cat = feat.get("category", "")
            if cat:
                cats[cat] += 1
    return cats


def get_sample_categories(sample: dict) -> Counter:
    cats = Counter()
    for feat in parse_features(sample):
        cat = feat.get("category", "")
        if cat:
            cats[cat] += 1
    return cats


def load_jsonl(path: str) -> list:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train_aug.jsonl")
    parser.add_argument("--val", required=True, help="Path to val_aug.jsonl")
    parser.add_argument("--test", required=True, help="Path to test_aug.jsonl")
    parser.add_argument("--output", required=True, help="Output path for balanced val set")
    parser.add_argument("--target_size", type=int, default=400,
                        help="Target number of samples in balanced val set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load datasets
    train_samples = load_jsonl(args.train)
    val_samples = load_jsonl(args.val)
    test_samples = load_jsonl(args.test)

    print(f"Loaded: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    # Compute target proportions from training set
    train_cats = count_categories(train_samples)
    total_train_feats = sum(train_cats.values())
    target_proportions = {cat: cnt / total_train_feats for cat, cnt in train_cats.items()}

    print("\nTraining set category proportions:")
    for cat in sorted(target_proportions):
        print(f"  {cat}: {target_proportions[cat]:.4f} ({train_cats[cat]} features)")

    # Pool candidates and deduplicate by dataitem_name
    seen = set()
    candidates = []
    for sample in val_samples + test_samples:
        name = sample.get("dataitem_name", "")
        if name not in seen:
            seen.add(name)
            candidates.append(sample)
    random.shuffle(candidates)

    print(f"\nCandidate pool (deduplicated): {len(candidates)} samples")

    # Group candidates by dominant category
    cat_to_samples = defaultdict(list)
    for sample in candidates:
        sample_cats = get_sample_categories(sample)
        if sample_cats:
            dominant = sample_cats.most_common(1)[0][0]
            cat_to_samples[dominant].append(sample)

    # Compute per-category target sample counts
    # Use dominant category to allocate quota
    all_categories = sorted(target_proportions.keys())
    target_counts = {}
    remaining = args.target_size

    for cat in all_categories:
        target_counts[cat] = max(1, int(args.target_size * target_proportions.get(cat, 0)))

    # Normalize to target_size
    total_target = sum(target_counts.values())
    if total_target > 0:
        scale = args.target_size / total_target
        for cat in all_categories:
            target_counts[cat] = max(1, round(target_counts[cat] * scale))

    print("\nTarget sample allocation by dominant category:")
    for cat in all_categories:
        available = len(cat_to_samples.get(cat, []))
        print(f"  {cat}: target={target_counts.get(cat, 0)}, available={available}")

    # Select samples
    selected = []
    for cat in all_categories:
        pool = cat_to_samples.get(cat, [])
        n = min(target_counts.get(cat, 0), len(pool))
        selected.extend(pool[:n])

    # If we haven't reached target_size, fill from remaining candidates
    selected_names = {s.get("dataitem_name", "") for s in selected}
    if len(selected) < args.target_size:
        remaining_pool = [s for s in candidates if s.get("dataitem_name", "") not in selected_names]
        random.shuffle(remaining_pool)
        needed = args.target_size - len(selected)
        selected.extend(remaining_pool[:needed])

    # Trim if over target
    selected = selected[:args.target_size]
    random.shuffle(selected)

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in selected:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Report final distribution
    final_cats = count_categories(selected)
    total_final = sum(final_cats.values())
    print(f"\nSelected {len(selected)} samples with {total_final} total features")
    print("\nFinal vs Target proportions:")
    print(f"  {'Category':<30} {'Target':>8} {'Actual':>8} {'Diff':>8}")
    for cat in all_categories:
        t = target_proportions.get(cat, 0)
        a = final_cats.get(cat, 0) / total_final if total_final else 0
        print(f"  {cat:<30} {t:>8.4f} {a:>8.4f} {a-t:>+8.4f}")

    print(f"\nBalanced validation set saved to: {args.output}")


if __name__ == "__main__":
    main()
