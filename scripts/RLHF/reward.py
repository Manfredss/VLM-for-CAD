"""
Programmatic reward function for GRPO/DPO training.

Designed as an ms-swift external_plugin. Computes a composite reward
from structured JSON predictions against ground truth annotations.

Reward components:
  R_format       — is the output valid JSON?
  R_category_f1  — category-level F1 (pred↔GT matched via IoU)
  R_bbox_iou     — mean IoU of matched pred-GT pairs
  R_size         — size string accuracy (exact or edit distance)
  R_completeness — recall: fraction of GT instances detected

Usage:
  # Standalone test
  python reward.py --config rlhf_config.yaml --pred '...' --gt '...'

  # As ms-swift external_plugin (registers IMRewardFunction)
  swift rlhf ... --external_plugins reward.py
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# =====================================================================
# IoU utility
# =====================================================================
def calculate_iou(a, b):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def normalized_edit_distance(a: str, b: str) -> float:
    """Normalized Levenshtein distance in [0, 1]. 0 = identical."""
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 1.0
    # Standard DP
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[lb] / max(la, lb)


# =====================================================================
# JSON parsing (mirrors inference_swift.py parse_output)
# =====================================================================
def parse_json_output(text: str):
    """Parse model output to list of dicts. Returns None if unparseable."""
    if not isinstance(text, str):
        return None
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

    # Markdown code block
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # Direct JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Extract [...] segment
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # Truncated JSON recovery
    last_brace = text.rfind("}")
    if last_brace >= 0:
        candidate = text[:last_brace + 1]
        first_bracket = candidate.find("[")
        if first_bracket >= 0:
            candidate = candidate[first_bracket:]
            if not candidate.rstrip().endswith("]"):
                candidate = candidate.rstrip().rstrip(",") + "]"
            try:
                obj = json.loads(candidate)
                if isinstance(obj, list):
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass

    return None


# =====================================================================
# Matching: greedy bipartite match pred↔GT by IoU
# =====================================================================
def match_predictions(pred_list, gt_list, iou_threshold=0.4):
    """
    Greedy matching of predictions to ground truth by IoU.

    Returns:
        matched: list of (pred_item, gt_item, iou) tuples
        unmatched_pred: list of pred items with no GT match
        unmatched_gt: list of GT items with no pred match
    """
    if not pred_list or not gt_list:
        return [], pred_list or [], gt_list or []

    # Compute IoU matrix
    scores = []
    for pi, p in enumerate(pred_list):
        p_bbox = p.get("bbox", [])
        if len(p_bbox) != 4:
            continue
        for gi, g in enumerate(gt_list):
            g_bbox = g.get("bbox", [])
            if len(g_bbox) != 4:
                continue
            iou = calculate_iou(p_bbox, g_bbox)
            if iou >= iou_threshold:
                scores.append((iou, pi, gi))

    # Greedy: sort by IoU descending, match greedily
    scores.sort(key=lambda x: -x[0])
    matched_p = set()
    matched_g = set()
    matched = []

    for iou, pi, gi in scores:
        if pi in matched_p or gi in matched_g:
            continue
        matched.append((pred_list[pi], gt_list[gi], iou))
        matched_p.add(pi)
        matched_g.add(gi)

    unmatched_pred = [p for i, p in enumerate(pred_list) if i not in matched_p]
    unmatched_gt = [g for i, g in enumerate(gt_list) if i not in matched_g]

    return matched, unmatched_pred, unmatched_gt


# =====================================================================
# Reward components
# =====================================================================
def reward_format(pred_list) -> float:
    """1.0 if valid parsed JSON list, 0.0 otherwise."""
    return 1.0 if pred_list is not None else 0.0


def reward_category_f1(matched, unmatched_pred, unmatched_gt,
                       category_bonus=None) -> float:
    """
    Category-aware F1.
    A match counts as TP only if categories agree.
    Applies per-category bonus weights to boost rare category importance.
    """
    if not matched and not unmatched_pred and not unmatched_gt:
        return 1.0  # both empty = perfect

    tp_weight = 0.0
    fp_weight = 0.0
    fn_weight = 0.0
    bonus = category_bonus or {}

    for pred, gt, iou in matched:
        cat_gt = gt.get("category", "")
        cat_pred = pred.get("category", "")
        w = bonus.get(cat_gt, 1.0)
        if cat_pred == cat_gt:
            tp_weight += w
        else:
            fp_weight += w
            fn_weight += w

    for p in unmatched_pred:
        w = bonus.get(p.get("category", ""), 1.0)
        fp_weight += w

    for g in unmatched_gt:
        w = bonus.get(g.get("category", ""), 1.0)
        fn_weight += w

    precision = tp_weight / (tp_weight + fp_weight) if (tp_weight + fp_weight) > 0 else 0.0
    recall = tp_weight / (tp_weight + fn_weight) if (tp_weight + fn_weight) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reward_bbox_iou(matched) -> float:
    """Mean IoU of matched pairs."""
    if not matched:
        return 0.0
    return sum(iou for _, _, iou in matched) / len(matched)


def reward_size_accuracy(matched, metric="edit_distance") -> float:
    """
    Size string accuracy across matched pairs.
    - exact: 1.0 if strings match, 0.0 otherwise
    - edit_distance: 1.0 - normalized_edit_distance
    """
    if not matched:
        return 0.0

    scores = []
    for pred, gt, _ in matched:
        pred_size = str(pred.get("size", "")).strip()
        gt_size = str(gt.get("size", "")).strip()

        # Skip if GT has no size (view categories)
        if not gt_size:
            continue

        if metric == "exact":
            scores.append(1.0 if pred_size == gt_size else 0.0)
        else:
            scores.append(1.0 - normalized_edit_distance(pred_size, gt_size))

    return sum(scores) / len(scores) if scores else 1.0


def reward_completeness(matched, unmatched_gt) -> float:
    """Recall: fraction of GT instances that were detected."""
    total_gt = len(matched) + len(unmatched_gt)
    if total_gt == 0:
        return 1.0
    return len(matched) / total_gt


# =====================================================================
# Composite reward
# =====================================================================
class DrawingRewardFunction:
    """
    Composite reward function for engineering drawing analysis.

    Loads configuration from YAML. All weights and thresholds are
    configurable without code changes.
    """

    def __init__(self, config=None, config_path=None):
        """
        Args:
            config: dict with reward config (takes precedence)
            config_path: path to rlhf_config.yaml
        """
        if config is None:
            if config_path is None:
                config_path = Path(__file__).parent / "rlhf_config.yaml"
            if _HAS_YAML and Path(config_path).exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
            else:
                config = {}

        rewards = config.get("rewards", {})
        self.w_format = rewards.get("format_valid", {}).get("weight", 0.10)
        self.w_cat = rewards.get("category_f1", {}).get("weight", 0.35)
        self.w_bbox = rewards.get("bbox_iou", {}).get("weight", 0.25)
        self.w_size = rewards.get("size_accuracy", {}).get("weight", 0.20)
        self.w_complete = rewards.get("completeness", {}).get("weight", 0.10)

        self.iou_threshold = rewards.get("category_f1", {}).get("iou_threshold", 0.4)
        self.size_metric = rewards.get("size_accuracy", {}).get("metric", "edit_distance")
        self.category_bonus = config.get("category_bonus", {})

    def __call__(self, prediction: str, ground_truth: str) -> float:
        """
        Score a (prediction, ground_truth) pair.

        Args:
            prediction: model's raw text output (may contain JSON)
            ground_truth: ground truth JSON string

        Returns:
            float reward in [0, 1]
        """
        pred_list = parse_json_output(prediction)
        gt_list = parse_json_output(ground_truth)

        # Format reward
        r_format = reward_format(pred_list)
        if pred_list is None:
            # Can't compute other rewards without valid predictions
            return self.w_format * 0.0

        if gt_list is None:
            gt_list = []

        # Match predictions to GT
        matched, unmatched_pred, unmatched_gt = match_predictions(
            pred_list, gt_list, self.iou_threshold
        )

        # Component rewards
        r_cat = reward_category_f1(matched, unmatched_pred, unmatched_gt,
                                   self.category_bonus)
        r_bbox = reward_bbox_iou(matched)
        r_size = reward_size_accuracy(matched, self.size_metric)
        r_complete = reward_completeness(matched, unmatched_gt)

        total = (
            self.w_format * r_format
            + self.w_cat * r_cat
            + self.w_bbox * r_bbox
            + self.w_size * r_size
            + self.w_complete * r_complete
        )

        return total

    def score_detailed(self, prediction: str, ground_truth: str) -> dict:
        """Like __call__ but returns per-component breakdown."""
        pred_list = parse_json_output(prediction)
        gt_list = parse_json_output(ground_truth)

        r_format = reward_format(pred_list)
        if pred_list is None:
            return {
                "total": 0.0,
                "format_valid": 0.0,
                "category_f1": 0.0,
                "bbox_iou": 0.0,
                "size_accuracy": 0.0,
                "completeness": 0.0,
                "n_pred": 0,
                "n_gt": len(gt_list) if gt_list else 0,
                "n_matched": 0,
            }

        if gt_list is None:
            gt_list = []

        matched, unmatched_pred, unmatched_gt = match_predictions(
            pred_list, gt_list, self.iou_threshold
        )

        r_cat = reward_category_f1(matched, unmatched_pred, unmatched_gt,
                                   self.category_bonus)
        r_bbox = reward_bbox_iou(matched)
        r_size = reward_size_accuracy(matched, self.size_metric)
        r_complete = reward_completeness(matched, unmatched_gt)

        total = (
            self.w_format * r_format
            + self.w_cat * r_cat
            + self.w_bbox * r_bbox
            + self.w_size * r_size
            + self.w_complete * r_complete
        )

        return {
            "total": round(total, 4),
            "format_valid": round(r_format, 4),
            "category_f1": round(r_cat, 4),
            "bbox_iou": round(r_bbox, 4),
            "size_accuracy": round(r_size, 4),
            "completeness": round(r_complete, 4),
            "n_pred": len(pred_list),
            "n_gt": len(gt_list),
            "n_matched": len(matched),
        }


# =====================================================================
# ms-swift compatible reward function wrapper
# =====================================================================
def compute_reward(prediction: str, ground_truth: str, **kwargs) -> float:
    """
    ms-swift compatible reward function.

    When used as --external_plugins, ms-swift calls this function
    with (prediction, ground_truth) strings.
    """
    config_path = kwargs.get("config_path", Path(__file__).parent / "rlhf_config.yaml")
    fn = DrawingRewardFunction(config_path=config_path)
    return fn(prediction, ground_truth)


# =====================================================================
# CLI for testing
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test reward function")
    parser.add_argument("--config", default=str(Path(__file__).parent / "rlhf_config.yaml"))
    parser.add_argument("--pred", type=str, help="Prediction JSON string")
    parser.add_argument("--gt", type=str, help="Ground truth JSON string")
    parser.add_argument("--pred_file", type=str, help="File with prediction JSON")
    parser.add_argument("--gt_file", type=str, help="File with ground truth JSON")
    parser.add_argument("--results_json", type=str,
                        help="Path to inference results JSON (batch scoring against GT)")
    parser.add_argument("--gt_json", type=str,
                        help="Path to 5k_15feats_with_view_v2.json (for batch scoring)")
    args = parser.parse_args()

    fn = DrawingRewardFunction(config_path=args.config)

    if args.pred and args.gt:
        result = fn.score_detailed(args.pred, args.gt)
        print(json.dumps(result, indent=2))

    elif args.pred_file and args.gt_file:
        with open(args.pred_file) as f:
            pred = f.read()
        with open(args.gt_file) as f:
            gt = f.read()
        result = fn.score_detailed(pred, gt)
        print(json.dumps(result, indent=2))

    elif args.results_json and args.gt_json:
        # Batch scoring: compare inference results against GT annotations
        with open(args.results_json) as f:
            results = json.load(f)
        with open(args.gt_json) as f:
            gt_data = json.load(f)

        # Build GT lookup
        gt_by_name = {item["dataitem_name"]: item for item in gt_data}

        scores = []
        for entry in results:
            name = entry["dataitem_name"]
            if name not in gt_by_name:
                continue
            pred_json = json.dumps(entry["result"], ensure_ascii=False)
            # Reconstruct GT JSON from annotations
            gt_item = gt_by_name[name]
            gt_features = []
            for task in gt_item.get("tasks", []):
                for tv in task.get("task_values", []):
                    val = tv["value"]
                    gt_features.append({
                        "category": val["label"],
                        "size": val.get("size", ""),
                        "bbox": tv["bbox"],
                    })
            gt_json_str = json.dumps(gt_features, ensure_ascii=False)
            detail = fn.score_detailed(pred_json, gt_json_str)
            detail["image"] = name
            scores.append(detail)

        # Summary
        if scores:
            avg = {k: sum(s[k] for s in scores) / len(scores)
                   for k in ["total", "category_f1", "bbox_iou",
                             "size_accuracy", "completeness"]}
            print(f"Batch reward scores ({len(scores)} images):")
            for k, v in avg.items():
                print(f"  {k}: {v:.4f}")
        else:
            print("No matching images found.")
    else:
        print("Usage:")
        print("  python reward.py --pred '<json>' --gt '<json>'")
        print("  python reward.py --results_json results.json --gt_json gt.json")
