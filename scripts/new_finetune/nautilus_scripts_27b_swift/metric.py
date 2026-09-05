"""
IMMetric — 自定义评估指标（兼容 ms-swift 4.0.0.dev0）
适配 Qwen3.5-27B

兼容两种训练输出：
    1. 旧格式：```json ... ``` + bbox_2d
    2. v10 格式：纯 JSON + bbox
"""
import os
import re
import json
import logging
import numpy as np
import torch
from typing import Dict, List, Literal
from transformers import EvalPrediction, AutoTokenizer

try:
    from swift.plugin import metric_mapping
    _HAS_SWIFT_PLUGIN = True
except ImportError:
    _HAS_SWIFT_PLUGIN = False

logger = logging.getLogger(__name__)

_metric_model_path = os.environ.get('METRIC_MODEL', 'Qwen/Qwen3.5-27B')
_tokenizer = None


def _extract_json_candidates(text: str) -> List[str]:
    candidates = []

    # 兼容旧版 markdown 代码块
    pattern = re.compile(r'```(?:json)?\s*([\s\S]*?)```', re.MULTILINE)
    candidates.extend(m.group(1).strip() for m in pattern.finditer(text))

    stripped = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
    if stripped:
        candidates.append(stripped)

    # 直接提取首个 JSON array
    match = re.search(r'(\[[\s\S]*\])', stripped)
    if match:
        candidates.append(match.group(1).strip())

    # 去重并保序
    result = []
    seen = set()
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _normalize_items(items: List[Dict]) -> List[Dict]:
    normalized = []
    for item in items:
        bbox = item.get('bbox', item.get('bbox_2d'))
        category = item.get('category', item.get('label', ''))
        size = item.get('size', '')
        if bbox and category:
            normalized.append({
                'label': f'{category}_{size}',
                'bbox': bbox,
            })
    return normalized


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(
                _metric_model_path, trust_remote_code=True
            )
            logger.info(f"Metric tokenizer loaded from {_metric_model_path}")
        except Exception as e:
            logger.warning(f'Failed to load metric tokenizer: {e}. Token-level metric disabled.')
    return _tokenizer


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    x1_max = max(bbox1[0], bbox2[0])
    y1_max = max(bbox1[1], bbox2[1])
    x2_min = min(bbox1[2], bbox2[2])
    y2_min = min(bbox1[3], bbox2[3])
    if x2_min <= x1_max or y2_min <= y1_max:
        return 0.0
    intersection = (x2_min - x1_max) * (y2_min - y1_max)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def calculate_bbox_label_accuracy(predicted: List[Dict], ground_truth: List[Dict]) -> float:
    if not predicted or not ground_truth:
        return 0.0

    gt_items = []
    for item in ground_truth:
        if 'bbox' in item and 'label' in item:
            gt_items.append((item['bbox'], item['label']))

    total_reward = 0.0
    used_gt_indices = set()

    for pred_item in predicted:
        if 'bbox' not in pred_item or 'label' not in pred_item:
            continue

        pred_bbox = pred_item['bbox']
        pred_label = pred_item['label']

        best_iou = 0.0
        best_gt_idx = -1
        best_gt_label = None

        for i, (gt_bbox, gt_label) in enumerate(gt_items):
            if i in used_gt_indices:
                continue
            iou = calculate_iou(pred_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
                best_gt_label = gt_label

        if best_gt_idx != -1 and best_iou > 0.5:
            label_score = 1.0 if pred_label == best_gt_label else 0.0
            best_iou = 1.0 if best_iou >= 0.8 else best_iou
            combined_score = best_iou * 0.5 + label_score * 0.5
            total_reward += combined_score
            used_gt_indices.add(best_gt_idx)

    return total_reward / len(gt_items) if gt_items else 0.0


def compute_task_acc(preds, labels, *, acc_strategy='token',
                     is_encoder_decoder=False, cu_seqlens=None):
    if isinstance(preds, torch.Tensor):
        if torch.is_floating_point(labels):
            return {}
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
    if preds.ndim >= 2 and not is_encoder_decoder:
        labels = labels[..., 1:]
        preds = preds[..., :-1]
    if np.issubdtype(labels.dtype, np.floating) or preds.shape != labels.shape:
        return {}

    masks = labels != -100

    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return 0

    label_text = tokenizer.decode(labels[masks], skip_special_tokens=True)
    pred_text = tokenizer.decode(preds[masks], skip_special_tokens=True)

    pred_list = _extract_json_candidates(pred_text)
    label_list = _extract_json_candidates(label_text)

    scores = []
    if len(label_list) == len(pred_list):
        for ni, label in enumerate(label_list):
            try:
                label_data = _normalize_items(json.loads(label))
                pred_data = _normalize_items(json.loads(pred_list[ni]))
                if not label_data:
                    scores.append(1.0 if not pred_data else 0.0)
                elif not pred_data:
                    scores.append(0.0)
                elif len(label_data) != len(pred_data):
                    scores.append(0.0)
                else:
                    scores.append(calculate_bbox_label_accuracy(pred_data, label_data))
            except Exception:
                scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


def preprocess_logits_for_task(logits: torch.Tensor, labels: torch.Tensor):
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_task_metrics(eval_prediction: EvalPrediction, *,
                         acc_strategy='token',
                         is_encoder_decoder=False) -> Dict[str, float]:
    metric = compute_task_acc(
        eval_prediction.predictions,
        eval_prediction.label_ids,
        acc_strategy=acc_strategy,
        is_encoder_decoder=is_encoder_decoder
    )
    return {'IMMetric': metric}


if _HAS_SWIFT_PLUGIN:
    metric_mapping['IMMetric'] = (compute_task_metrics, preprocess_logits_for_task)
