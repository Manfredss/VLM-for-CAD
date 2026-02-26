"""
Evaluate and compare model performance on the validation set.

Two modes
---------
  summary   Parse trainer_state.json and print training-curve metrics.  (No GPU)
  eval      Run inference on val set; compute feature-detection metrics. (GPU required)

Usage examples
--------------
  # Training summary — no GPU needed
  python src/evaluate.py --mode summary \\
      --trainer_state outputs/qwen3-vl-3b-lora/checkpoint-1383/trainer_state.json

  # Full comparison: base model vs fine-tuned (use --max_samples for a quick check)
  python src/evaluate.py --mode eval \\
      --val_json data/val.json \\
      --image_root data/IM_D03_PT_5K \\
      --base_model merve/qwen3-vl-3b-llava-1pct \\
      --finetuned_adapter outputs/qwen3-vl-3b-lora/final \\
      --output outputs/eval_results.json \\
      --max_samples 50

  # Fine-tuned only (skip base model)
  python src/evaluate.py --mode eval \\
      --val_json data/val.json \\
      --image_root data/IM_D03_PT_5K \\
      --finetuned_adapter outputs/qwen3-vl-3b-lora/final \\
      --output outputs/eval_results.json
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Category mappings ─────────────────────────────────────────────────────────

CATEGORIES_ZH = ["螺纹孔", "圆角", "矩形孔", "圆孔", "长圆孔"]
CAT_DISPLAY = {
    "螺纹孔": "threaded_hole (螺纹孔)",
    "圆角":   "fillet       (圆角)",
    "矩形孔": "rect_hole    (矩形孔)",
    "圆孔":   "round_hole   (圆孔)",
    "长圆孔": "slot_hole    (长圆孔)",
}

# English category names (sample.json format) -> Chinese for metrics
CAT_EN_TO_ZH = {
    "Threaded Hole":     "螺纹孔",
    "Fillet":            "圆角",
    "Rectangular Hole":  "矩形孔",
    "Round Hole":        "圆孔",
    "Slotted Hole":      "长圆孔",
}

# ── Feature parsing ───────────────────────────────────────────────────────────

# Matches lines like:
#   - 螺纹孔（M8），位置坐标：[1426, 1127, 1529, 1248]
# Skips group entries (name contains "组").
_FEAT_RE = re.compile(
    r"-\s+"                             # bullet
    r"([^\（\(]+?)"                     # feature name (Chinese)
    r"[（\(]([^\）\)]+?)[）\)]"         # spec inside brackets
    r"[，,]\s*位置坐标[：:]\s*"         # coord label
    r"[\[（\(]"                         # opening bracket
    r"(\d+)[,，]\s*(\d+)[,，]\s*(\d+)[,，]\s*(\d+)"  # x1 y1 x2 y2
    r"[\]）\)]"                         # closing bracket
)

# Matches category headers like 【螺纹孔检测结果】
_CAT_RE = re.compile(r"【(.+?)检测结果】")


def parse_features(text: str) -> list:
    """
    Extract individual feature detections from a model-output string.

    Handles two formats:
    1. JSON array (new format):
       [{"category": "Round Hole", "size": "...", "bbox": [x1,y1,x2,y2]}, ...]
    2. Chinese structured text (old format):
       【螺纹孔检测结果】\n  - 螺纹孔（M8），位置坐标：[...]

    Returns a list of dicts::
        {"category": str, "spec": str, "bbox": [x1, y1, x2, y2]}

    Group entries are skipped (English "Group" suffix or Chinese "组").
    """
    features = []

    # --- Try JSON format first ---
    text_stripped = text.strip()
    if text_stripped.startswith("["):
        try:
            dets = json.loads(text_stripped)
            if isinstance(dets, list):
                for det in dets:
                    if not isinstance(det, dict):
                        continue
                    category = det.get("category", "")
                    if "Group" in category:
                        continue  # skip group aggregates
                    zh_cat = CAT_EN_TO_ZH.get(category)
                    if zh_cat is None:
                        continue  # unknown / unsupported category
                    spec = det.get("size", "")
                    bbox = det.get("bbox", [])
                    if len(bbox) == 4:
                        features.append({"category": zh_cat, "spec": spec, "bbox": list(bbox)})
                return features
        except (json.JSONDecodeError, ValueError):
            pass  # fall through to regex parser

    # --- Fallback: old Chinese structured-text format ---
    current_cat = None
    for line in text.splitlines():
        cat_m = _CAT_RE.search(line)
        if cat_m:
            cat_zh = cat_m.group(1)
            current_cat = next(
                (c for c in CATEGORIES_ZH if c in cat_zh), cat_zh
            )
            continue

        feat_m = _FEAT_RE.search(line)
        if feat_m:
            name = feat_m.group(1).strip()
            if "组" in name:
                continue  # skip group/aggregate rows

            spec = feat_m.group(2).strip()
            x1, y1, x2, y2 = (
                int(feat_m.group(3)),
                int(feat_m.group(4)),
                int(feat_m.group(5)),
                int(feat_m.group(6)),
            )

            cat = current_cat
            if cat is None:
                cat = next((c for c in CATEGORIES_ZH if c in name), None)

            features.append({"category": cat, "spec": spec, "bbox": [x1, y1, x2, y2]})

    return features


# ── Geometry helpers ──────────────────────────────────────────────────────────

def compute_iou(box_a: list, box_b: list) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── Matching and metrics ──────────────────────────────────────────────────────

def match_features(gt: list, pred: list, iou_thr: float) -> tuple:
    """
    Greedy IoU matching between ground-truth and predicted feature lists
    (both pre-filtered to a single category).

    Returns (tp, fp, fn, matched_ious_list).
    """
    used_pred = set()
    matched_ious = []

    for gt_feat in gt:
        best_iou = iou_thr  # minimum threshold to count as TP
        best_j = -1
        for j, pred_feat in enumerate(pred):
            if j in used_pred:
                continue
            iou = compute_iou(gt_feat["bbox"], pred_feat["bbox"])
            if iou >= best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            used_pred.add(best_j)
            matched_ious.append(best_iou)

    tp = len(matched_ious)
    fn = len(gt) - tp
    fp = len(pred) - len(used_pred)
    return tp, fp, fn, matched_ious


def compute_detection_metrics(
    all_gt: list, all_pred: list, iou_thr: float = 0.5
) -> dict:
    """
    Compute precision / recall / F1 and mean IoU per category and overall.

    Args:
        all_gt:   list of per-sample GT feature lists
        all_pred: list of per-sample predicted feature lists
        iou_thr:  IoU threshold for TP (default 0.5)

    Returns:
        dict keyed by category name (+ "overall"), each with keys:
        precision, recall, f1, mean_iou, tp, fp, fn
    """
    keys = CATEGORIES_ZH + ["overall"]
    acc = {k: {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0, "iou_count": 0}
           for k in keys}

    count_errors = []  # |pred_count - gt_count| per sample

    for gt_feats, pred_feats in zip(all_gt, all_pred):
        count_errors.append(abs(len(pred_feats) - len(gt_feats)))

        for cat in CATEGORIES_ZH:
            gt_cat = [f for f in gt_feats if f["category"] == cat]
            pred_cat = [f for f in pred_feats if f["category"] == cat]
            tp, fp, fn, ious = match_features(gt_cat, pred_cat, iou_thr)
            acc[cat]["tp"] += tp
            acc[cat]["fp"] += fp
            acc[cat]["fn"] += fn
            acc[cat]["iou_sum"] += sum(ious)
            acc[cat]["iou_count"] += len(ious)

        tp, fp, fn, ious = match_features(gt_feats, pred_feats, iou_thr)
        acc["overall"]["tp"] += tp
        acc["overall"]["fp"] += fp
        acc["overall"]["fn"] += fn
        acc["overall"]["iou_sum"] += sum(ious)
        acc["overall"]["iou_count"] += len(ious)

    results = {}
    for key, s in acc.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        miou = s["iou_sum"] / s["iou_count"] if s["iou_count"] > 0 else 0.0
        results[key] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "mean_iou": round(miou, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    results["_count_mae"] = round(
        sum(count_errors) / len(count_errors) if count_errors else 0.0, 2
    )
    return results


# ═════════════════════════════════════════════════════════════════════════════
# MODE: summary
# ═════════════════════════════════════════════════════════════════════════════

def mode_summary(trainer_state_path: str, output: Optional[str]):
    """Parse trainer_state.json and print a training-metrics summary table."""
    with open(trainer_state_path, encoding="utf-8") as f:
        state = json.load(f)

    log = state["log_history"]

    # Separate train-loss entries from eval-loss entries
    train_entries = [e for e in log if "loss" in e and "eval_loss" not in e]
    eval_entries = [e for e in log if "eval_loss" in e]

    best_step = state.get("best_global_step")
    best_eval = state.get("best_metric")
    total_steps = state.get("global_step", 0)
    total_epochs = state.get("num_train_epochs", "?")

    # Build a step → train-loss lookup for gap analysis at eval points
    step_to_train = {e["step"]: e["loss"] for e in train_entries}

    # Find the closest train entry to each eval step
    train_steps_sorted = sorted(step_to_train.keys())

    def nearest_train_loss(eval_step):
        closest = min(train_steps_sorted, key=lambda s: abs(s - eval_step))
        return step_to_train[closest]

    # ── Print summary ────────────────────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print("  TRAINING SUMMARY")
    print(sep)
    print(f"  Trainer state : {trainer_state_path}")
    print(f"  Epochs        : {total_epochs}  |  Total steps : {total_steps}")
    print(f"  Best checkpoint: step {best_step}  (eval_loss = {best_eval:.4f})")
    print(sep)

    # ── Training-loss milestones ─────────────────────────────────────────────
    first_loss = train_entries[0]["loss"] if train_entries else float("nan")
    print(f"\n--- TRAINING LOSS {'-'*54}")
    print(f"  {'Step':>6}  {'Epoch':>6}  {'Loss':>8}  {'vs Initial':>12}")
    milestones = set()
    # Always include first and last; show every complete epoch boundary
    for e in train_entries:
        ep = round(e["epoch"], 6)
        if ep <= 0.03 or abs(ep - round(ep)) < 0.025 or e["step"] == total_steps:
            milestones.add(e["step"])
    for e in sorted([x for x in train_entries if x["step"] in milestones],
                    key=lambda x: x["step"]):
        delta = (e["loss"] - first_loss) / first_loss * 100
        marker = " <init" if e["step"] == train_entries[0]["step"] else ""
        print(f"  {e['step']:>6}  {e['epoch']:>6.2f}  {e['loss']:>8.4f}  {delta:>+11.1f}%{marker}")

    # ── Eval-loss table ──────────────────────────────────────────────────────
    first_eval = eval_entries[0]["eval_loss"] if eval_entries else float("nan")
    print(f"\n--- VALIDATION LOSS {'-'*53}")
    print(f"  {'Step':>6}  {'Epoch':>6}  {'EvalLoss':>10}  {'vs 1st Eval':>12}  "
          f"{'TrainLoss':>10}  {'Gap':>8}")
    for e in eval_entries:
        delta = (e["eval_loss"] - first_eval) / first_eval * 100
        tl = nearest_train_loss(e["step"])
        gap = e["eval_loss"] - tl
        marker = " * BEST" if e["step"] == best_step else ""
        print(
            f"  {e['step']:>6}  {e['epoch']:>6.2f}  {e['eval_loss']:>10.4f}"
            f"  {delta:>+11.1f}%  {tl:>10.4f}  {gap:>+8.4f}{marker}"
        )

    # ── Quick health summary ─────────────────────────────────────────────────
    if eval_entries:
        last_eval = eval_entries[-1]["eval_loss"]
        improvement = (first_eval - last_eval) / first_eval * 100
        best_gap = best_eval - nearest_train_loss(best_step)
        print(f"\n--- QUICK STATS {'-'*57}")
        print(f"  Initial eval loss : {first_eval:.4f}")
        print(f"  Best eval loss    : {best_eval:.4f}  (step {best_step})")
        print(f"  Total improvement : {improvement:+.1f}% vs first eval")
        overfitting = "[healthy]" if best_gap < 0.05 else "[possible overfitting]"
        print(f"  Train-eval gap @best: {best_gap:+.4f}  {overfitting}")

    print(f"\n{sep}\n")

    # ── Persist results ──────────────────────────────────────────────────────
    summary = {
        "trainer_state": trainer_state_path,
        "total_steps": total_steps,
        "num_epochs": total_epochs,
        "best_step": best_step,
        "best_eval_loss": best_eval,
        "first_train_loss": first_loss,
        "last_train_loss": train_entries[-1]["loss"] if train_entries else None,
        "first_eval_loss": first_eval,
        "eval_history": [
            {
                "step": e["step"],
                "epoch": round(e["epoch"], 3),
                "eval_loss": e["eval_loss"],
                "train_loss_nearby": nearest_train_loss(e["step"]),
            }
            for e in eval_entries
        ],
    }

    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Summary saved to {output}")

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# MODE: eval — model loading and inference
# ═════════════════════════════════════════════════════════════════════════════

def load_model_for_eval(model_path: str, adapter_path: Optional[str],
                        dtype_str: str = "bf16"):
    """
    Load a model for evaluation.

    If adapter_path is given, model_path is the base model and adapter_path
    holds the LoRA adapter (merged before inference).
    If adapter_path is None, model_path is treated as a full/merged model.
    """
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import PeftModel

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }

    if adapter_path:
        logger.info(f"Loading base model: {model_path}")
        model = AutoModelForVision2Seq.from_pretrained(model_path, **kwargs)
        logger.info(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        if getattr(model, "_hf_peft_config_loaded", False):
            model._hf_peft_config_loaded = False
        try:
            processor = AutoProcessor.from_pretrained(
                adapter_path, trust_remote_code=True
            )
        except Exception:
            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
    else:
        logger.info(f"Loading merged/full model: {model_path}")
        model = AutoModelForVision2Seq.from_pretrained(model_path, **kwargs)
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

    model.eval()
    return model, processor


def run_inference_single(model, processor, image_path: str, prompt: str,
                         system_prompt: str, min_pixels: int,
                         max_pixels: int, max_new_tokens: int = 1024) -> str:
    """Run inference on a single image and return the decoded output."""
    import torch
    from qwen_vl_utils import process_vision_info

    # Correct pixel formula: n_patches * 28 * 28
    min_pix = min_pixels * 28 * 28
    max_pix = max_pixels * 28 * 28

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{os.path.abspath(image_path)}",
                    "min_pixels": min_pix,
                    "max_pixels": max_pix,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def evaluate_model_on_val(
    model,
    processor,
    val_data: list,
    image_root: str,
    system_prompt: str,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
    label: str,
    iou_thr: float,
) -> dict:
    """
    Run inference over the val set and compute detection metrics.

    Returns a dict with keys: label, metrics, per_sample (list of raw results).
    """
    image_root = Path(image_root)
    all_gt, all_pred = [], []
    per_sample = []

    for i, sample in enumerate(val_data):
        image_name = sample.get("image", "")
        # Support full relative paths (e.g. "data/IM_D03_PT_5K/x.png") and
        # bare filenames joined with image_root.
        image_path = Path(image_name)
        if not image_path.is_absolute() and not image_path.exists():
            image_path = image_root / image_name
        if not image_path.exists():
            logger.warning(f"[{i+1}/{len(val_data)}] Image not found: {image_path}")
            continue

        # Support both "conversations" (plural) and "conversation" (singular)
        convs = sample.get("conversations") or sample.get("conversation", [])

        def _turn_role(t):
            return t.get("role", t.get("from", ""))

        def _turn_content(t):
            v = t.get("content", t.get("value", ""))
            if isinstance(v, list):
                v = json.dumps(v, ensure_ascii=False)
            return v

        user_prompt = next(
            (_turn_content(c) for c in convs if _turn_role(c) in ("user", "human")),
            "请识别这张图纸中的所有工件特征。",
        )
        # Strip <image> placeholder — inference adds the image as a content part
        user_prompt = user_prompt.replace("<image>", "").strip()
        gt_text = next(
            (_turn_content(c) for c in convs if _turn_role(c) in ("assistant", "qwen")),
            "",
        )

        logger.info(f"[{i+1}/{len(val_data)}] {label}: {image_name}")

        try:
            pred_text = run_inference_single(
                model, processor, str(image_path), user_prompt,
                system_prompt, min_pixels, max_pixels, max_new_tokens,
            )
        except Exception as e:
            logger.error(f"  Inference failed: {e}")
            pred_text = ""

        gt_feats = parse_features(gt_text)
        pred_feats = parse_features(pred_text)
        all_gt.append(gt_feats)
        all_pred.append(pred_feats)
        per_sample.append({
            "image": image_name,
            "gt_text": gt_text,
            "pred_text": pred_text,
            "gt_count": len(gt_feats),
            "pred_count": len(pred_feats),
        })

    metrics = compute_detection_metrics(all_gt, all_pred, iou_thr)
    return {"label": label, "n_samples": len(all_gt),
            "metrics": metrics, "per_sample": per_sample}


# ── Comparison report printer ─────────────────────────────────────────────────

def print_comparison(results: list, iou_thr: float):
    """Print a side-by-side comparison table for all evaluated models."""
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  DETECTION METRICS COMPARISON  (IoU threshold = {iou_thr})")
    print(sep)

    # Header
    labels = [r["label"] for r in results]
    n_models = len(labels)
    col_w = 26

    header = f"  {'Category':<22}"
    for lbl in labels:
        header += f"  {lbl[:col_w]:<{col_w}}"
    print(header)
    sub = f"  {'':22}"
    for _ in labels:
        sub += f"  {'Prec':>6} {'Rec':>6} {'F1':>6} {'mIoU':>6}"
    print(sub)
    print("  " + "-" * (22 + n_models * (col_w + 2)))

    row_keys = CATEGORIES_ZH + ["overall"]
    for key in row_keys:
        disp = CAT_DISPLAY.get(key, key) if key != "overall" else f"{'---'} OVERALL {'---'}"
        if key == "overall":
            print("  " + "-" * (22 + n_models * (col_w + 2)))
        row = f"  {disp:<22}"
        for r in results:
            m = r["metrics"].get(key, {})
            row += (f"  {m.get('precision', 0):>6.3f}"
                    f" {m.get('recall', 0):>6.3f}"
                    f" {m.get('f1', 0):>6.3f}"
                    f" {m.get('mean_iou', 0):>6.3f}")
        print(row)

    # Count MAE row
    print("  " + "-" * (22 + n_models * (col_w + 2)))
    row = f"  {'Count MAE':<22}"
    for r in results:
        mae = r["metrics"].get("_count_mae", 0)
        row += f"  {mae:>{col_w}.2f}"
    print(row)

    # Sample count
    row = f"  {'Samples':<22}"
    for r in results:
        row += f"  {r['n_samples']:>{col_w}}"
    print(row)

    # Delta section (only when two models are compared)
    if len(results) == 2:
        print(f"\n{'--- IMPROVEMENT (fine-tuned vs base) ':-<80}")
        base_m = results[0]["metrics"]
        ft_m = results[1]["metrics"]
        print(f"  {'Category':<22}  {'dPrec':>8}  {'dRecall':>8}  {'dF1':>8}  {'dmIoU':>8}")
        for key in row_keys:
            if key == "overall":
                print("  " + "-" * 60)
            b = base_m.get(key, {})
            f = ft_m.get(key, {})
            disp = CAT_DISPLAY.get(key, key) if key != "overall" else "OVERALL"
            dp = f.get("precision", 0) - b.get("precision", 0)
            dr = f.get("recall", 0) - b.get("recall", 0)
            df = f.get("f1", 0) - b.get("f1", 0)
            di = f.get("mean_iou", 0) - b.get("mean_iou", 0)
            print(f"  {disp:<22}  {dp:>+8.3f}  {dr:>+8.3f}  {df:>+8.3f}  {di:>+8.3f}")
        dcount = results[0]["metrics"].get("_count_mae", 0) - \
                 results[1]["metrics"].get("_count_mae", 0)
        print(f"  {'Count MAE reduction':<22}  {dcount:>+8.2f}")

    print(f"\n{sep}\n")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare base/fine-tuned Qwen3-VL on drawing val set"
    )
    parser.add_argument(
        "--mode", choices=["summary", "eval"], default="summary",
        help="'summary': parse training logs (no GPU). 'eval': run model inference.",
    )

    # summary mode
    parser.add_argument(
        "--trainer_state",
        default="outputs/qwen3-vl-3b-lora/checkpoint-1383/trainer_state.json",
        help="Path to trainer_state.json (used in summary mode).",
    )

    # eval mode
    parser.add_argument("--val_json", default="data/val.json")
    parser.add_argument("--image_root", default="data/IM_D03_PT_5K")
    parser.add_argument(
        "--base_model", default=None,
        help="Base model path or HF repo. Omit to skip base-model evaluation.",
    )
    parser.add_argument(
        "--finetuned_adapter", default=None,
        help="Path to LoRA adapter dir (outputs/qwen3-vl-3b-lora/final). "
             "Omit to skip fine-tuned evaluation.",
    )
    parser.add_argument(
        "--finetuned_merged", default=None,
        help="Path to already-merged fine-tuned model (alternative to --finetuned_adapter).",
    )
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Cap the number of val samples evaluated (None = all 409).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--min_pixels", type=int, default=256,
                        help="Min image resolution in 28×28 patches.")
    parser.add_argument("--max_pixels", type=int, default=1280,
                        help="Max image resolution in 28×28 patches.")
    parser.add_argument(
        "--iou_thr", type=float, default=0.5,
        help="IoU threshold for TP detection matching.",
    )
    parser.add_argument(
        "--system_prompt", type=str,
        default=(
            "你是一个专业的工业图纸分析助手。你的任务是识别和分析工程图纸中的工件特征，"
            "包括但不限于：螺纹孔、圆角、矩形孔、圆孔、长圆孔、倒角、槽等特征。"
            "对于每种特征，请输出其类型、规格尺寸（如M8、R3）以及在图纸中的位置坐标。"
            "请根据图纸内容给出准确、完整的分析结果。"
        ),
    )
    parser.add_argument("--output", default=None,
                        help="Path to save JSON results. Auto-set if not given.")

    args = parser.parse_args()

    # ── Auto output path ─────────────────────────────────────────────────────
    if args.output is None:
        args.output = (
            "outputs/summary_metrics.json"
            if args.mode == "summary"
            else "outputs/eval_comparison.json"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ── SUMMARY mode ─────────────────────────────────────────────────────────
    if args.mode == "summary":
        mode_summary(args.trainer_state, args.output)
        return

    # ── EVAL mode ────────────────────────────────────────────────────────────
    with open(args.val_json, encoding="utf-8") as f:
        val_data = json.load(f)

    if args.max_samples:
        val_data = val_data[: args.max_samples]
        logger.info(f"Capped to {args.max_samples} samples.")

    all_results = []

    # -- Base model -----------------------------------------------------------
    if args.base_model:
        logger.info("▶ Evaluating BASE model ...")
        model, processor = load_model_for_eval(
            args.base_model, adapter_path=None, dtype_str=args.dtype
        )
        result = evaluate_model_on_val(
            model, processor, val_data,
            args.image_root, args.system_prompt,
            args.min_pixels, args.max_pixels, args.max_new_tokens,
            label="Base model", iou_thr=args.iou_thr,
        )
        all_results.append(result)
        # Free GPU memory before loading next model
        import torch
        del model
        torch.cuda.empty_cache()

    # -- Fine-tuned model -----------------------------------------------------
    if args.finetuned_adapter or args.finetuned_merged:
        if args.finetuned_adapter:
            if args.base_model is None:
                parser.error("--base_model is required when using --finetuned_adapter")
            logger.info("▶ Evaluating FINE-TUNED model (LoRA adapter) ...")
            model, processor = load_model_for_eval(
                args.base_model, adapter_path=args.finetuned_adapter,
                dtype_str=args.dtype,
            )
        else:
            logger.info("▶ Evaluating FINE-TUNED model (merged) ...")
            model, processor = load_model_for_eval(
                args.finetuned_merged, adapter_path=None, dtype_str=args.dtype
            )
        result = evaluate_model_on_val(
            model, processor, val_data,
            args.image_root, args.system_prompt,
            args.min_pixels, args.max_pixels, args.max_new_tokens,
            label="Fine-tuned", iou_thr=args.iou_thr,
        )
        all_results.append(result)

    if not all_results:
        logger.error("No models evaluated. Provide --base_model and/or --finetuned_adapter.")
        sys.exit(1)

    # -- Print comparison table -----------------------------------------------
    print_comparison(all_results, args.iou_thr)

    # -- Save results ---------------------------------------------------------
    serialisable = []
    for r in all_results:
        serialisable.append({
            "label": r["label"],
            "n_samples": r["n_samples"],
            "metrics": r["metrics"],
            # omit per_sample raw text to keep file size manageable
            "per_sample_counts": [
                {"image": s["image"], "gt_count": s["gt_count"],
                 "pred_count": s["pred_count"]}
                for s in r.get("per_sample", [])
            ],
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
