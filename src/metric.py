from swift.plugin import metric_mapping
from transformers import EvalPrediction
import torch
from typing import Dict, List, Literal
from swift.utils import Serializer, get_current_device, get_logger
from swift.llm import Template
from swift.llm import get_template, get_model_tokenizer
import re
import json
import numpy as np


def extract_json_blocks(text: str):
    pattern = re.compile(r'json\s*([\s\S]*?)```', re.MULTILINE)
    matches = [m.group(1).strip() for m in pattern.finditer(text)]
    if not matches:
        raise ValueError("未找到 JSON 代码块")
    try:
        parsed_data = json.loads(matches[0])
        return parsed_data
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败: {e}")


logger = get_logger()

model, processor = get_model_tokenizer('/root/autodl-tmp/models/Qwen3-VL-4B-Instruct', load_model=False)
template = get_template('IMTemplate', processor)


def compute_task_acc(preds,
                     labels,
                     *,
                     acc_strategy: Literal['token', 'seq'] = 'token',
                     is_encoder_decoder: bool = False,
                     cu_seqlens=None) -> Dict[str, List[float]]:
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
    acc_list = (preds[masks] == labels[masks]).tolist()
    label = processor.tokenizer.decode(labels[masks], skip_special_tokens=True)
    pred = processor.tokenizer.decode(preds[masks], skip_special_tokens=True)
    pattern = re.compile(r'json\s*([\s\S]*?)```', re.MULTILINE)
    pred_list = [m.group(1).strip() for m in pattern.finditer(pred)]
    label_list = [m.group(1).strip() for m in pattern.finditer(label)]
    scores = []
    if len(label_list) == len(pred_list):
        for ni, label in enumerate(label_list):
            try:
                label_data = json.loads(label)
                pred_data = json.loads(pred_list[ni])
                label_data = [{"label": x["category"] + "_" + x["size"], "bbox_2d": x["bbox_2d"]} for x in label_data if x["bbox_2d"]]
                pred_data = [{"label": x["category"] + "_" + x["size"], "bbox_2d": x["bbox_2d"]} for x in pred_data if x["bbox_2d"]]
                if not label_data:
                    if pred_data:
                        scores.append(0)
                    else:
                        scores.append(1)
                else:
                    if not pred_data:
                        scores.append(0)
                    else:
                        if len(label_data) != len(pred_data):
                            scores.append(0)
                        else:
                            scores.append(
                                calculate_bbox_label_accuracy(pred_data, label_data)
                            )
            except:
                scores.append(0)

    if scores:
        scores_ = sum(scores) / len(scores)
    else:
        scores_ = 0

    return scores_


def preprocess_logits_for_task(logits: torch.Tensor, labels: torch.Tensor) -> List[str]:
    if isinstance(logits, (list, tuple)):
        logits = logits[0]

    pred_ids = logits.argmax(dim=-1)
    return pred_ids
    # return pred_ids, labels


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
        if 'bbox_2d' in item and 'label' in item:
            gt_items.append((item['bbox_2d'], item['label']))

    total_reward = 0.0
    used_gt_indices = set()

    for pred_item in predicted:
        if 'bbox_2d' not in pred_item or 'label' not in pred_item:
            continue

        pred_bbox = pred_item['bbox_2d']
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

                # 如果找到了匹配的ground truth（IoU > 0.5）
        if best_gt_idx != -1 and best_iou > 0.5:
            # 计算label得分
            label_score = 1.0 if pred_label == best_gt_label else 0.0
            best_iou = 1.0 if best_iou >= 0.8 else best_iou

            combined_score = best_iou * 0.5 + label_score * 0.5

            total_reward += combined_score
            used_gt_indices.add(best_gt_idx)

    return total_reward / len(gt_items) if gt_items else 0.0


def compute_task_metrics(eval_prediction: EvalPrediction,
                         *,
                         acc_strategy: Literal['token', 'seq'] = 'token',
                         is_encoder_decoder: bool = False) -> Dict[str, float]:
    metric = compute_task_acc(
        eval_prediction.predictions,
        eval_prediction.label_ids,
        acc_strategy=acc_strategy,
        is_encoder_decoder=is_encoder_decoder
    )
    # pred_ids, label_ids = eval_prediction.predictions

    # pred_texts = []
    # label_texts = []
    # print("test_len",len(pred_ids))
    # for pred_id, label_id in zip(pred_ids, label_ids):

    #     pred_segments = []
    #     current_segment = []

    #     for i, (p, l) in enumerate(zip(pred_id, label_id)):
    #         if l != -100:
    #             current_segment.append(p)
    #         elif current_segment:
    #             pred_segments.append(current_segment)
    #             current_segment = []

    #     if current_segment:
    #         pred_segments.append(current_segment)

    #     label_segments = []
    #     current_segment = []

    #     for i, (p, l) in enumerate(zip(pred_id, label_id)):
    #         if l != -100:
    #             current_segment.append(l)
    #         elif current_segment:
    #             label_segments.append(current_segment)
    #             current_segment = []

    #     if current_segment:
    #         label_segments.append(current_segment)

    #     if pred_segments and label_segments:
    #         # 解码预测
    #         last_pred = pred_segments[-1]
    #         valid_pred_ids = [x for x in last_pred if x not in [151655, 151656]]
    #         pred_text = processor.tokenizer.decode(valid_pred_ids, skip_special_tokens=True)
    #         pred_texts.append(pred_text)

    #         # 解码标签
    #         last_label = label_segments[-1]
    #         valid_label_ids = [x for x in last_label if x not in [151655, 151656]]
    #         label_text = processor.tokenizer.decode(valid_label_ids, skip_special_tokens=True)
    #         label_texts.append(label_text)
    #     scores = []
    #     called = False
    #     for ni,pred_text in enumerate(pred_texts):
    #         label_text = label_texts[ni]
    #         try:
    #             label = extract_json_blocks(label_text)
    #             label = [{"label":x["category"]+ "_" + x["size"],"bbox_2d":x["bbox_2d"]} for x in label if x["bbox_2d"]]
    #             pred = extract_json_blocks(pred_text)
    #             pred = [{"label":x["category"]+ "_" + x["size"],"bbox_2d":x["bbox_2d"]} for x in pred if x["bbox_2d"]]
    #             scores.append(
    #                 calculate_bbox_label_accuracy( pred, label)
    #             )
    #         except Exception as e:
    #             if not called:
    #                 print(str(e))
    #                 print(pred_text)
    #                 called = True
    #             scores.append(0)

    return {'IMMetric': metric}


# 注册到metric_mapping
metric_mapping['IMMetric'] = (compute_task_metrics, preprocess_logits_for_task)