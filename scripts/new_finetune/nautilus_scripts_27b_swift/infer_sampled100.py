#!/usr/bin/env python3
"""
Run inference on the 100 sampled images using a trained checkpoint.
Outputs results.json in the format: [{dataitem_name, result: [json_string]}, ...]

Also computes P/R/F1 against ground truth if GT file is provided.

Usage:
  python3 infer_sampled100.py \
      --model_path Qwen/Qwen3.5-27B \
      --adapter_path /workspace/output/swift_27b/v26-.../checkpoint-400 \
      --gt_json /workspace/data/5k_10feats_v2_sampled_100.json \
      --image_dir /workspace/data/5k \
      --output_file /workspace/output/swift_27b/sampled100_results_ckpt400.json
"""

import argparse
import json
import time
import sys
from pathlib import Path

from inference_swift import load_model, run_inference_single, parse_output, _deduplicate


def normalize_box(box, image_size=None):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return []
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return []
    max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if max_coord <= 1.5:
        x1, y1, x2, y2 = [v * 1000.0 for v in (x1, y1, x2, y2)]
    elif max_coord > 1000.0 and image_size:
        w, h = image_size
        if w > 0 and h > 0:
            x1 = x1 / w * 1000.0
            x2 = x2 / w * 1000.0
            y1 = y1 / h * 1000.0
            y2 = y2 / h * 1000.0
    normed = [int(round(v)) for v in (x1, y1, x2, y2)]
    normed = [max(0, min(1000, v)) for v in normed]
    if normed[0] >= normed[2] or normed[1] >= normed[3]:
        return []
    return normed


def calculate_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_prf1(all_preds, all_gts, iou_threshold=0.4):
    tp = fp = fn = 0
    for preds, gts in zip(all_preds, all_gts):
        matched_gt = set()
        for pred in preds:
            best_iou = -1
            best_idx = -1
            for gi, gt in enumerate(gts):
                if gi in matched_gt:
                    continue
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi
            if best_iou >= iou_threshold and best_idx >= 0:
                matched_gt.add(best_idx)
                if pred['category'] == gts[best_idx]['category']:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            else:
                fp += 1
        fn += len(gts) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}


def load_gt(gt_json_path, image_dir):
    """Load ground truth from the raw annotation JSON."""
    from PIL import Image as PILImage
    with open(gt_json_path) as f:
        data = json.load(f)

    records = []
    for item in data:
        img_name = item['dataitem_name']
        img_path = Path(image_dir) / img_name
        if not img_path.exists():
            continue
        with PILImage.open(img_path) as img:
            img_size = img.size

        gts = []
        for task in item.get('tasks', []):
            for tv in task.get('task_values', []):
                bbox = normalize_box(tv['bbox'], img_size)
                if not bbox:
                    continue
                gts.append({
                    'category': tv['value']['label'],
                    'size': tv['value'].get('size', ''),
                    'bbox': bbox,
                })
        records.append({
            'dataitem_name': img_name,
            'image_path': img_path,
            'image_size': img_size,
            'gt': gts,
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen3.5-27B')
    parser.add_argument('--adapter_path', type=str, required=True)
    parser.add_argument('--gt_json', type=str, default='/workspace/data/5k_10feats_v2_sampled_100.json')
    parser.add_argument('--image_dir', type=str, default='/workspace/data/5k')
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument('--load_in_4bit', action='store_true')
    args = parser.parse_args()

    print(f'Loading GT from {args.gt_json}')
    records = load_gt(args.gt_json, args.image_dir)
    print(f'Loaded {len(records)} images with GT')

    print(f'Loading model with adapter: {args.adapter_path}')
    model, processor = load_model(args.model_path, args.adapter_path, args.load_in_4bit)

    results = []
    all_preds = []
    all_gts = []

    for i, rec in enumerate(records, 1):
        t0 = time.time()
        raw = run_inference_single(model, processor, rec['image_path'], args.max_new_tokens)
        parsed = _deduplicate(parse_output(raw))

        # Normalize predictions
        pred_items = []
        for item in parsed:
            bbox = normalize_box(
                item.get('bbox', item.get('bbox_2d', [])),
                rec['image_size']
            )
            cat = item.get('category', item.get('label', ''))
            if bbox and cat:
                pred_items.append({
                    'category': cat,
                    'size': item.get('size', ''),
                    'bbox': bbox,
                })

        all_preds.append(pred_items)
        all_gts.append(rec['gt'])

        # Format result as requested
        result_json = json.dumps(pred_items, ensure_ascii=False, indent=2)
        results.append({
            'dataitem_name': rec['dataitem_name'],
            'result': [result_json],
        })

        elapsed = time.time() - t0
        print(f'[{i}/{len(records)}] {rec["dataitem_name"]} | {len(pred_items)} preds / {len(rec["gt"])} gt | {elapsed:.1f}s')

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\nResults saved to {output_path}')

    # Compute and print metrics
    metrics = evaluate_prf1(all_preds, all_gts)
    print(f'\n=== Sampled-100 Metrics (IoU>=0.4) ===')
    print(f'Precision: {metrics["precision"]:.4f}')
    print(f'Recall:    {metrics["recall"]:.4f}')
    print(f'F1:        {metrics["f1"]:.4f}')
    print(f'TP={metrics["tp"]} FP={metrics["fp"]} FN={metrics["fn"]}')

    # Save metrics alongside results
    metrics_path = output_path.with_name(output_path.stem + '_metrics.json')
    metrics['checkpoint'] = args.adapter_path
    metrics['num_images'] = len(records)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Metrics saved to {metrics_path}')


if __name__ == '__main__':
    main()
