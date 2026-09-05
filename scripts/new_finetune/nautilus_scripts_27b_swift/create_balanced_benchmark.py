#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from math import sqrt
from pathlib import Path


def load_raw_gt(raw_gt_path: Path) -> dict[str, dict]:
    data = json.loads(raw_gt_path.read_text(encoding='utf-8'))
    return {item['dataitem_name']: item for item in data if item.get('dataitem_name')}


def extract_categories_from_raw(item: dict) -> tuple[list[str], Counter]:
    categories = []
    counts = Counter()
    for task in item.get('tasks', []):
        for task_value in task.get('task_values', []):
            label = (task_value.get('value') or {}).get('label')
            if label:
                categories.append(label)
                counts[label] += 1
    return categories, counts


def load_swift4_dataset(dataset_path: Path) -> list[dict]:
    records = []
    with dataset_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            images = item.get('images') or []
            if not images:
                continue
            dataitem_name = Path(images[0]).name
            records.append({'dataitem_name': dataitem_name, 'line': line, 'item': item})
    return records


def build_candidates(swift_records: list[dict], raw_gt_by_name: dict[str, dict]) -> list[dict]:
    candidates = []
    for record in swift_records:
        dataitem_name = record['dataitem_name']
        raw_item = raw_gt_by_name.get(dataitem_name)
        if not raw_item:
            continue
        categories, object_counts = extract_categories_from_raw(raw_item)
        if not categories:
            continue
        candidates.append(
            {
                'dataitem_name': dataitem_name,
                'line': record['line'],
                'swift_item': record['item'],
                'raw_item': raw_item,
                'categories': sorted(set(categories)),
                'object_counts': object_counts,
                'total_objects': sum(object_counts.values()),
            }
        )
    return candidates


def score_candidate(candidate: dict, selected_presence: Counter, category_weights: dict[str, float]) -> float:
    score = 0.0
    for category in candidate['categories']:
        score += category_weights[category] / (1.0 + selected_presence[category])
    score += 0.05 * len(candidate['categories'])
    score -= 0.002 * max(0, candidate['total_objects'] - 12)
    return score


def select_balanced_subset(candidates: list[dict], benchmark_size: int) -> list[dict]:
    category_presence = Counter()
    for candidate in candidates:
        for category in candidate['categories']:
            category_presence[category] += 1

    category_weights = {
        category: 1.0 / sqrt(count)
        for category, count in category_presence.items()
        if count > 0
    }

    selected = []
    selected_names = set()
    selected_presence = Counter()

    while len(selected) < benchmark_size:
        best = None
        best_key = None
        for candidate in candidates:
            if candidate['dataitem_name'] in selected_names:
                continue
            score = score_candidate(candidate, selected_presence, category_weights)
            tie_break_key = (
                round(score, 8),
                len(candidate['categories']),
                -candidate['total_objects'],
                candidate['dataitem_name'],
            )
            if best is None or tie_break_key > best_key:
                best = candidate
                best_key = tie_break_key

        if best is None:
            break

        selected.append(best)
        selected_names.add(best['dataitem_name'])
        for category in best['categories']:
            selected_presence[category] += 1

    return sorted(selected, key=lambda item: item['dataitem_name'])


def summarize(records: list[dict]) -> dict:
    image_presence = Counter()
    object_counts = Counter()
    for record in records:
        for category in record['categories']:
            image_presence[category] += 1
        object_counts.update(record['object_counts'])
    return {
        'images': len(records),
        'image_presence': dict(sorted(image_presence.items())),
        'object_counts': dict(sorted(object_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_gt', required=True)
    parser.add_argument('--swift4_dataset', required=True)
    parser.add_argument('--benchmark_size', type=int, default=50)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--output_jsonl', required=True)
    parser.add_argument('--output_summary', default='')
    args = parser.parse_args()

    raw_gt_path = Path(args.raw_gt)
    swift4_dataset_path = Path(args.swift4_dataset)
    output_json = Path(args.output_json)
    output_jsonl = Path(args.output_jsonl)

    raw_gt_by_name = load_raw_gt(raw_gt_path)
    swift_records = load_swift4_dataset(swift4_dataset_path)
    candidates = build_candidates(swift_records, raw_gt_by_name)
    selected = select_balanced_subset(candidates, args.benchmark_size)

    output_json.write_text(
        json.dumps([record['raw_item'] for record in selected], ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    output_jsonl.write_text(
        '\n'.join(record['line'] for record in selected) + '\n',
        encoding='utf-8',
    )

    overall_summary = summarize(candidates)
    selected_summary = summarize(selected)
    summary = {
        'source_dataset': str(swift4_dataset_path),
        'benchmark_size': len(selected),
        'selection_method': 'inverse-frequency weighted greedy coverage on val_swift4',
        'overall_summary': overall_summary,
        'selected_summary': selected_summary,
        'selected_images': [record['dataitem_name'] for record in selected],
    }

    if args.output_summary:
        Path(args.output_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()