"""
Composite reward function for GRPO training on Siemens 788 drawings.

Extended from scripts/RLHF/reward.py with:
  - 15 layout/view + 10 feature category support (incl. hole groups)
  - Cross-view consistency reward using containment instead of raw IoU
  - Configurable via rlhf_config.yaml

Reward components:
  R_format        — is the output valid JSON?
  R_category_f1   — category-level F1 (class-aware pred↔GT IoU matching)
  R_bbox_iou      — mean IoU of matched pred-GT pairs
  R_size          — normalized CAD dimension accuracy/similarity
  R_completeness  — recall: fraction of GT instances detected
  R_cross_view    — bonus for cross-view consistency (features in correct views)

Usage:
  # Standalone test
  python reward.py --pred '<json>' --gt '<json>'

  # As ms-swift external_plugin
  swift rlhf ... --external_plugins reward.py
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from cad_metrics import (  # noqa: E402
    ALL_CATEGORIES,
    center_in_box,
    containment_ratio,
    extract_items,
    score_items,
    summarize_stats,
)

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
# Categories (must match prepare_dataset_swift.py)
# =====================================================================
VIEW_CATEGORIES = {
    "Title Block", "Notes", "Revision Table", "Bill of Materials",
    "Orthographic Projection - Front View",
    "Orthographic Projection - Top View",
    "Orthographic Projection - Left View",
    "Orthographic Projection - Right View",
    "Orthographic Projection - Bottom View",
    "Orthographic Projection - Rear View",
    "Isometric View", "Flat Pattern View",
    "Section View", "Detail View", "Auxiliary View",
}

FEATURE_CATEGORIES = {
    "Round Hole", "Rectangular Hole", "Threaded Hole", "Slotted Hole",
    "Round Hole Group", "Rectangular Hole Group", "Slotted Hole Group",
    "Fillet", "Bending", "Silver Plating",
}

# Features typically found in specific views
FEATURE_VIEW_AFFINITY = {
    "Bending": {"Orthographic Projection - Front View", "Flat Pattern View",
                 "Section View", "Orthographic Projection - Right View",
                 "Orthographic Projection - Left View", "Orthographic Projection - Top View"},
    "Silver Plating": {"Orthographic Projection - Front View", "Flat Pattern View",
                       "Orthographic Projection - Top View",
                       "Orthographic Projection - Bottom View",
                       "Isometric View"},
}


# =====================================================================
# JSON parsing
# =====================================================================
def parse_json_output(text: str):
    """Parse model output to list of dicts. Returns None if unparseable."""
    if not isinstance(text, str):
        return None
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, list):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

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
    if not pred_list or not gt_list:
        return [], pred_list or [], gt_list or []

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

    scores.sort(key=lambda x: -x[0])
    matched_p, matched_g = set(), set()
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
    return 1.0 if pred_list is not None else 0.0


def reward_category_f1(matched, unmatched_pred, unmatched_gt,
                       category_bonus=None) -> float:
    if not matched and not unmatched_pred and not unmatched_gt:
        return 1.0

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
    if not matched:
        return 0.0
    return sum(iou for _, _, iou in matched) / len(matched)


def reward_size_accuracy(matched, metric="edit_distance") -> float:
    if not matched:
        return 0.0
    scores = []
    for pred, gt, _ in matched:
        pred_size = str(pred.get("size", "")).strip()
        gt_size = str(gt.get("size", "")).strip()
        if not gt_size:
            continue
        if metric == "exact":
            scores.append(1.0 if pred_size == gt_size else 0.0)
        else:
            scores.append(1.0 - normalized_edit_distance(pred_size, gt_size))
    return sum(scores) / len(scores) if scores else 1.0


def reward_completeness(matched, unmatched_gt) -> float:
    total_gt = len(matched) + len(unmatched_gt)
    if total_gt == 0:
        return 1.0
    return len(matched) / total_gt


def reward_cross_view_consistency(feature_items, view_items) -> float:
    """Bonus for features detected inside or near appropriate view types.

    For each feature with known view affinity (e.g., Bending in Flat Pattern View),
    check whether its center is inside, or most of the feature is contained by,
    a compatible view. IoU is not suitable here because a small feature inside a
    large view has very small IoU even when it is perfectly placed.
    """
    if not feature_items:
        return 1.0
    if not view_items:
        return 0.0

    score = 0.0
    total = 0
    for feat in feature_items:
        cat = feat.get("category", "")
        if cat not in FEATURE_VIEW_AFFINITY:
            continue
        fbox = feat.get("bbox", [])
        if len(fbox) != 4:
            continue
        total += 1
        compatible_views = FEATURE_VIEW_AFFINITY[cat]
        for view in view_items:
            vcat = view.get("category", "")
            if vcat not in compatible_views:
                continue
            vbox = view.get("bbox", [])
            if len(vbox) != 4:
                continue
            if center_in_box(fbox, vbox) or containment_ratio(fbox, vbox) >= 0.50:
                score += 1.0
                break

    return score / total if total > 0 else 1.0


# =====================================================================
# Composite reward
# =====================================================================
class DrawingRewardFunction:
    """Composite reward function for engineering drawing analysis.

    Weights configurable via YAML. Supports scoring both single-pass
    (combined views+features) and multi-pass (initial + refined) outputs.
    """

    def __init__(self, config=None, config_path=None):
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
        self.w_cat = rewards.get("category_f1", {}).get("weight", 0.30)
        self.w_bbox = rewards.get("bbox_iou", {}).get("weight", 0.25)
        self.w_size = rewards.get("size_accuracy", {}).get("weight", 0.20)
        self.w_complete = rewards.get("completeness", {}).get("weight", 0.10)
        self.w_crossview = rewards.get("cross_view_consistency", {}).get("weight", 0.05)

        self.iou_threshold = rewards.get("category_f1", {}).get("iou_threshold", 0.4)
        self.size_metric = rewards.get("size_accuracy", {}).get("metric", "edit_distance")
        self.category_bonus = config.get("category_bonus", {})

    def _score_combined(self, pred_items, gt_items):
        """Score a single prediction list (views + features combined) against GT."""
        if pred_items is None:
            return {
                "total": self.w_format * 0.0,
                "format_valid": 0.0,
                "category_f1": 0.0,
                "bbox_iou": 0.0,
                "size_accuracy": 0.0,
                "completeness": 0.0,
                "cross_view": 0.0,
            }

        pred_items = extract_items(pred_items)
        gt_items = extract_items(gt_items or [])

        if not pred_items and not gt_items:
            return {
                "total": 1.0,
                "format_valid": 1.0,
                "category_f1": 1.0,
                "bbox_iou": 1.0,
                "size_accuracy": 1.0,
                "completeness": 1.0,
                "cross_view": 1.0,
                "n_pred": 0,
                "n_gt": 0,
                "n_matched": 0,
            }

        combined_stats = score_items(
            pred_items,
            gt_items,
            ALL_CATEGORIES,
            self.iou_threshold,
            score_size=True,
            category_weights=self.category_bonus,
            count_unexpected=True,
        )
        feature_stats = score_items(
            pred_items,
            gt_items,
            FEATURE_CATEGORIES,
            self.iou_threshold,
            score_size=True,
            category_weights=self.category_bonus,
            count_unexpected=False,
        )
        combined_summary = summarize_stats(combined_stats, ALL_CATEGORIES)
        feature_summary = summarize_stats(feature_stats, FEATURE_CATEGORIES)

        r_format = 1.0
        r_cat = combined_summary["weighted_f1"]
        r_bbox = combined_summary["mean_iou"]
        if self.size_metric == "exact":
            r_size = feature_summary["size_accuracy"]
        else:
            r_size = feature_summary["size_similarity"]
        r_complete = combined_summary["weighted_recall"]

        # Cross-view: features against predicted views
        pred_features = [p for p in pred_items
                        if p.get("category", "") in FEATURE_CATEGORIES]
        pred_views = [p for p in pred_items
                     if p.get("category", "") in VIEW_CATEGORIES]
        gt_views = [g for g in gt_items if g.get("category", "") in VIEW_CATEGORIES]
        if not gt_views and not pred_views:
            r_crossview = 1.0
        else:
            r_crossview = reward_cross_view_consistency(pred_features, pred_views)

        total = (
            self.w_format * r_format
            + self.w_cat * r_cat
            + self.w_bbox * r_bbox
            + self.w_size * r_size
            + self.w_complete * r_complete
            + self.w_crossview * r_crossview
        )

        return {
            "total": total,
            "format_valid": r_format,
            "category_f1": r_cat,
            "bbox_iou": r_bbox,
            "size_accuracy": r_size,
            "completeness": r_complete,
            "cross_view": r_crossview,
            "n_pred": len(pred_items),
            "n_gt": len(gt_items),
            "n_matched": combined_stats["match_count"],
        }

    def __call__(self, prediction: str, ground_truth: str) -> float:
        """Score a (prediction, ground_truth) pair. Returns float reward in [0,1]."""
        pred_items = parse_json_output(prediction)
        gt_items = parse_json_output(ground_truth)

        if gt_items is None:
            gt_items = []

        result = self._score_combined(pred_items, gt_items)
        return result["total"]

    def score_detailed(self, prediction: str, ground_truth: str) -> dict:
        """Like __call__ but returns per-component breakdown."""
        pred_items = parse_json_output(prediction)
        gt_items = parse_json_output(ground_truth)

        if gt_items is None:
            gt_items = []

        result = self._score_combined(pred_items, gt_items)
        for k, v in result.items():
            if isinstance(v, float):
                result[k] = round(v, 4)
        return result

    def score_agentic(self, initial_pred: str, refined_pred: str,
                      ground_truth: str) -> dict:
        """Score an agentic (multi-pass) refinement.

        Returns base score on refined output plus improvement delta from initial.
        """
        base = self.score_detailed(refined_pred, ground_truth)
        initial = self.score_detailed(initial_pred, ground_truth)

        delta = base["total"] - initial["total"]
        # Reward = refined quality + improvement bonus (clipped to [-0.2, 0.2])
        improvement_bonus = max(-0.2, min(0.2, delta))
        agentic_total = base["total"] + 0.5 * improvement_bonus

        return {
            "initial_score": initial["total"],
            "refined_score": base["total"],
            "delta": round(delta, 4),
            "improvement_bonus": round(improvement_bonus, 4),
            "agentic_total": round(max(0.0, min(1.0, agentic_total)), 4),
            "initial_details": initial,
            "refined_details": base,
        }


# =====================================================================
# ms-swift compatible reward function
# =====================================================================
def compute_reward(prediction: str, ground_truth: str, **kwargs) -> float:
    """ms-swift compatible reward function for GRPO."""
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
    parser.add_argument("--initial_pred", type=str,
                        help="Initial (pre-refinement) prediction for agentic scoring")
    parser.add_argument("--mode", choices=["single", "agentic"], default="single")
    args = parser.parse_args()

    fn = DrawingRewardFunction(config_path=args.config)

    if args.mode == "agentic" and args.initial_pred and args.pred and args.gt:
        result = fn.score_agentic(args.initial_pred, args.pred, args.gt)
        print(json.dumps(result, indent=2))
    elif args.pred and args.gt:
        result = fn.score_detailed(args.pred, args.gt)
        print(json.dumps(result, indent=2))
    else:
        print("Usage:")
        print("  python reward.py --pred '<json>' --gt '<json>'")
        print("  python reward.py --mode agentic --initial_pred '<json>' --pred '<json>' --gt '<json>'")
