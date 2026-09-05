#!/usr/bin/env python3
"""
evaluate_checkpoints_benchmark.py

对 27B swift 训练产生的 checkpoint 做固定 50 张 benchmark 评估，
按 detection-level Precision / Recall / F1 排序，辅助替代 eval_loss 选模。

设计目标：
  - benchmark 固定且可复现
    - 默认优先使用 benchmarkdata_swift4.jsonl 中的 GT，无需额外 raw GT 文件
  - 统一使用 inference_swift.py 的模型加载 / 解析逻辑
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from inference_swift import load_model, run_inference_single, parse_output, _deduplicate


def format_checkpoint_label(checkpoint_path: Path, checkpoints_root: Path | None = None) -> str:
    checkpoint_path = Path(checkpoint_path)
    if checkpoints_root is not None:
        try:
            return str(checkpoint_path.relative_to(checkpoints_root))
        except ValueError:
            pass
    if checkpoint_path.parent.name:
        return f'{checkpoint_path.parent.name}/{checkpoint_path.name}'
    return checkpoint_path.name


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def normalize_box(box, image_size=None):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return []

    try:
        x1, y1, x2, y2 = [float(value) for value in box]
    except (TypeError, ValueError):
        return []

    max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if max_coord <= 1.5:
        x1, y1, x2, y2 = [value * 1000.0 for value in (x1, y1, x2, y2)]
    elif max_coord > 1000.0 and image_size:
        image_width, image_height = image_size
        if image_width > 0 and image_height > 0:
            x1 = x1 / image_width * 1000.0
            x2 = x2 / image_width * 1000.0
            y1 = y1 / image_height * 1000.0
            y2 = y2 / image_height * 1000.0

    normalized = [int(round(value)) for value in (x1, y1, x2, y2)]
    normalized = [max(0, min(1000, value)) for value in normalized]
    if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
        return []
    return normalized


def compute_match_score(iou_value: float, label_matched: bool) -> float:
    if iou_value < 0.4:
        return 0.0

    iou_reward = 1.0 if iou_value >= 0.8 else iou_value
    label_score = 1.0 if label_matched else 0.0
    return 0.5 * iou_reward + 0.5 * label_score


def load_benchmark(dataset_path: Path, image_dir: Path, benchmark_size: int):
    records = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            image_path = None

            images = item.get('images') or []
            if images:
                image_path = image_dir / Path(images[0]).name

            if image_path is None:
                for msg in item.get('messages', []):
                    content = msg.get('content', [])
                    if not isinstance(content, list):
                        continue
                    for part in content:
                        if isinstance(part, dict) and part.get('type') == 'image':
                            image_path = image_dir / Path(part['image']).name
                            break
                    if image_path:
                        break

            if not image_path or not image_path.exists():
                continue

            with Image.open(image_path) as image:
                image_size = image.size

            gt = []
            assistant = ''
            for msg in item.get('messages', []):
                if msg.get('role') == 'assistant':
                    assistant = msg.get('content', '')
                    break
            try:
                parsed = json.loads(assistant)
            except Exception:
                parsed = parse_output(assistant)
            for feature in parsed:
                bbox = normalize_box(feature.get('bbox', feature.get('bbox_2d', [])), image_size)
                if not bbox:
                    continue
                gt.append({
                    'category': feature.get('category', feature.get('label', '')),
                    'bbox': bbox,
                })
            records.append({
                'image_path': image_path,
                'dataitem_name': image_path.name,
                'gt': gt,
                'image_size': image_size,
            })

    records = sorted(records, key=lambda x: x['dataitem_name'])[:benchmark_size]
    return records


def evaluate_checkpoint(model_path, adapter_path, benchmark, max_new_tokens, iou_threshold, load_in_4bit):
    model, processor = load_model(model_path, str(adapter_path), load_in_4bit)
    total_tp = total_fp = total_fn = 0
    total_quality = 0.0
    match_count = 0

    for index, sample in enumerate(benchmark, start=1):
        print(f'Processing {index}/{len(benchmark)} image: {sample["dataitem_name"]}', flush=True)
        raw = run_inference_single(model, processor, sample['image_path'], max_new_tokens)
        preds = _deduplicate(parse_output(raw))

        pred_items = [
            {
                'category': item.get('category', item.get('label', '')),
                'bbox': normalize_box(item.get('bbox', item.get('bbox_2d', [])), sample['image_size']),
            }
            for item in preds
        ]
        pred_items = [item for item in pred_items if item['category'] and item['bbox']]

        gt_items = sample['gt']
        matched_gt_indices = set()

        for pred_item in pred_items:
            best_iou = -1.0
            best_gt_idx = -1

            for gt_index, gt_item in enumerate(gt_items):
                if gt_index in matched_gt_indices:
                    continue
                current_iou = calculate_iou(pred_item['bbox'], gt_item['bbox'])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = gt_index

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matched_gt_indices.add(best_gt_idx)
                target_gt = gt_items[best_gt_idx]
                label_matched = pred_item['category'] == target_gt['category']

                total_quality += compute_match_score(best_iou, label_matched)
                match_count += 1

                if label_matched:
                    total_tp += 1
                else:
                    total_fp += 1
                    total_fn += 1
            else:
                total_fp += 1

        total_fn += len(gt_items) - len(matched_gt_indices)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    avg_quality = total_quality / match_count if match_count else 0.0
    return {
        'checkpoint': str(adapter_path),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_quality': avg_quality,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'match_count': match_count,
        'benchmark_images': len(benchmark),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen3.5-27B')
    parser.add_argument('--checkpoints_root', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--val_dataset', type=str, default='/workspace/data/benchmarkdata_swift4.jsonl')
    parser.add_argument('--image_dir', type=str, default='/workspace/data/5k')
    parser.add_argument('--benchmark_size', type=int, default=50)
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument('--iou_threshold', type=float, default=0.4)
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--output_json', type=str, default='')
    args = parser.parse_args()

    checkpoints_root = Path(args.checkpoints_root)
    image_dir = Path(args.image_dir)
    benchmark = load_benchmark(Path(args.val_dataset), image_dir, args.benchmark_size)

    if args.checkpoint_path:
        candidates = [Path(args.checkpoint_path)]
    else:
        candidates = sorted(
            [p for p in checkpoints_root.iterdir() if p.is_dir() and p.name.startswith('checkpoint-')],
            key=lambda p: int(p.name.split('-')[-1])
        )
    if not candidates:
        raise SystemExit(f'No checkpoint-* directories found under {checkpoints_root}')

    results = []
    for checkpoint in candidates:
        checkpoint_label = format_checkpoint_label(checkpoint, checkpoints_root)
        print(f'=== Evaluating {checkpoint_label} on fixed {len(benchmark)} images ===', flush=True)
        score = evaluate_checkpoint(
            args.model_path,
            checkpoint,
            benchmark,
            args.max_new_tokens,
            args.iou_threshold,
            args.load_in_4bit,
        )
        results.append(score)
        print(json.dumps(score, ensure_ascii=False, indent=2), flush=True)

    results.sort(key=lambda x: x['f1'], reverse=True)
    print('\n=== Ranking by benchmark F1 ===', flush=True)
    for idx, result in enumerate(results, start=1):
        checkpoint_label = format_checkpoint_label(Path(result['checkpoint']), checkpoints_root)
        print(
            f"{idx}. {checkpoint_label} | "
            f"P={result['precision']:.3f} R={result['recall']:.3f} F1={result['f1']:.3f} "
            f"AQ={result['avg_quality']:.3f} TP={result['tp']} FP={result['fp']} FN={result['fn']}"
        , flush=True)

    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()