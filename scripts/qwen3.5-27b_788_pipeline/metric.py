"""
IMMetric — CAD-aware eval metric for the multi-turn layout + feature task.

The previous implementation used one class-agnostic IoU matcher and raw string
size equality.  That is a poor fit for engineering drawings because group boxes
overlap child holes, layout boxes overlap features, and equivalent CAD
dimensions can be written with different symbols.  This plugin now delegates to
cad_metrics.py for class-aware matching, separate layout/feature reporting, and
normalized dimension scoring.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from transformers import AutoTokenizer, EvalPrediction

try:
    from cad_metrics import (
        DEFAULT_FEATURE_IOU,
        DEFAULT_LAYOUT_IOU,
        DEFAULT_STRICT_FEATURE_IOU,
        compute_metrics_from_text,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from cad_metrics import (
        DEFAULT_FEATURE_IOU,
        DEFAULT_LAYOUT_IOU,
        DEFAULT_STRICT_FEATURE_IOU,
        compute_metrics_from_text,
    )

try:
    from swift.trainers.mixin import eval_metrics_map
    _HAS_EVAL_METRICS_MAP = True
except ImportError:
    _HAS_EVAL_METRICS_MAP = False

try:
    from swift.plugin import metric_mapping
    _HAS_SWIFT_PLUGIN = True
except ImportError:
    _HAS_SWIFT_PLUGIN = False

logger = logging.getLogger(__name__)

_metric_model_path = os.environ.get("METRIC_MODEL", "Qwen/Qwen3.5-27B")
_tokenizer = None

LAYOUT_IOU_THRESHOLD = float(os.environ.get("METRIC_LAYOUT_IOU_THRESHOLD", DEFAULT_LAYOUT_IOU))
FEATURE_IOU_THRESHOLD = float(os.environ.get("METRIC_FEATURE_IOU_THRESHOLD", DEFAULT_FEATURE_IOU))
STRICT_FEATURE_IOU_THRESHOLD = float(
    os.environ.get("METRIC_STRICT_FEATURE_IOU_THRESHOLD", DEFAULT_STRICT_FEATURE_IOU)
)


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(
                _metric_model_path, trust_remote_code=True
            )
            logger.info("Metric tokenizer loaded from %s", _metric_model_path)
        except Exception as exc:
            logger.warning("Failed to load metric tokenizer: %s", exc)
    return _tokenizer


def compute_task_acc(preds, labels, *, acc_strategy="token",
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

    try:
        return compute_metrics_from_text(
            pred_text,
            label_text,
            layout_iou=LAYOUT_IOU_THRESHOLD,
            feature_iou=FEATURE_IOU_THRESHOLD,
            strict_feature_iou=STRICT_FEATURE_IOU_THRESHOLD,
        )
    except Exception as exc:
        logger.warning("CAD metric computation failed: %s", exc)
        return {"IMMetric": 0.0}


def preprocess_logits_for_task(logits: torch.Tensor, labels: torch.Tensor):
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_task_metrics(eval_prediction: EvalPrediction, *,
                         acc_strategy="token",
                         is_encoder_decoder=False) -> Dict[str, float]:
    result = compute_task_acc(
        eval_prediction.predictions,
        eval_prediction.label_ids,
        acc_strategy=acc_strategy,
        is_encoder_decoder=is_encoder_decoder,
    )
    if isinstance(result, dict):
        return result
    return {"IMMetric": result if isinstance(result, (int, float)) else 0.0}


if _HAS_EVAL_METRICS_MAP:
    from swift.metrics.acc import EvalMetrics

    class IMEvalMetrics(EvalMetrics):
        def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
            result = compute_task_acc(
                eval_prediction.predictions,
                eval_prediction.label_ids,
                is_encoder_decoder=self.trainer.is_encoder_decoder,
            )
            if isinstance(result, dict):
                return result
            return {"IMMetric": result if isinstance(result, (int, float)) else 0.0}

        def preprocess_logits_for_metrics(
            self, logits: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor:
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            return logits.argmax(dim=-1)

    eval_metrics_map["IMMetric"] = IMEvalMetrics

if _HAS_SWIFT_PLUGIN:
    metric_mapping["IMMetric"] = (compute_task_metrics, preprocess_logits_for_task)
