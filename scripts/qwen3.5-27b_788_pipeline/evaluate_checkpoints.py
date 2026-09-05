"""
evaluate_checkpoints.py — periodic generative eval of surviving checkpoints (improvement #6).

Why this exists:
  ms-swift's during-training eval is teacher-forced (cheap but optimistic).
  Real generation can drift, so we want a true F1 number per checkpoint.

What it does:
  - Finds all checkpoint-N under the latest version dir
  - Loads base model + the FIRST adapter once
  - For subsequent checkpoints, swaps the LoRA adapter via PEFT
    (avoids re-loading the 54 GB base each time)
  - Runs multi-turn generation on a val subset (default 20 images)
  - Scores predictions with the same CAD-aware logic as metric.py
  - Reports layout F1, feature F1, size similarity, and CADScore per checkpoint

Use this AFTER training to pick the actual-best ckpt by real F1, vs trusting
the teacher-forced eval_loss / IMMetric numbers from training.

Usage:
  python evaluate_checkpoints.py \
      --output-dir /workspace/output/swift_27b_788_view_7feats_improve \
      --val-jsonl /workspace/data/val_view_7feats_improve.jsonl \
      --image-dir /workspace/data/simens_7feats \
      --n-samples 20

  # Restrict to specific ckpts:
  python evaluate_checkpoints.py --output-dir <dir> --ckpts checkpoint-300,checkpoint-400
"""
import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import cad_metrics as cm  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =====================================================================
# Categories (must match prepare_dataset_swift.py / inference_swift.py)
# =====================================================================
VIEW_CATEGORIES = sorted([
    "Title Block", "Notes",
    "Orthographic Projection - Front View",
    "Orthographic Projection - Top View",
    "Orthographic Projection - Bottom View",
    "Orthographic Projection - Left View",
    "Orthographic Projection - Right View",
    "Orthographic Projection - Rear View",
    "Isometric View", "Flat Pattern View",
    "Section View", "Detail View", "Auxiliary View",
])
FEATURE_CATEGORIES = sorted([
    "Round Hole", "Rectangular Hole", "Threaded Hole", "Slotted Hole",
    "Round Hole Group", "Rectangular Hole Group", "Slotted Hole Group",
    "Fillet", "Bending", "Silver Plating",
])

IOU_THRESHOLD = 0.4


# =====================================================================
# Metric helpers (inlined from metric.py / local scoring)
# =====================================================================
def calc_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    A = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    B = max(0, bx2 - bx1) * max(0, by2 - by1)
    u = A + B - inter
    return inter / u if u > 0 else 0.0


def norm_box(b):
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return []
    try:
        x1, y1, x2, y2 = [float(v) for v in b]
    except (TypeError, ValueError):
        return []
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1, y1, x2, y2 = [v * 1000 for v in (x1, y1, x2, y2)]
    n = [int(round(v)) for v in (x1, y1, x2, y2)]
    n = [max(0, min(1000, v)) for v in n]
    if n[0] >= n[2] or n[1] >= n[3]:
        return []
    return n


def parse_json_arrays(text: str):
    """Find ALL top-level [...] arrays in the text and concatenate.
    Robust to multi-turn naked JSON output."""
    items = []
    text = text.strip()
    # Try whole-text JSON first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # Code-block matches
    for m in re.finditer(r'```json\s*([\s\S]*?)```', text):
        try:
            parsed = json.loads(m.group(1).strip())
            if isinstance(parsed, list):
                items.extend(parsed)
        except Exception:
            continue
    if items:
        return items
    # Naked top-level arrays
    for m in re.finditer(r'\[\s*(?:\{[\s\S]*?\}\s*,?\s*)*\]', text):
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                items.extend(parsed)
        except Exception:
            continue
    return items


def extract(items):
    out = []
    for it in items:
        cat = it.get("category", it.get("label", ""))
        bb = norm_box(it.get("bbox_2d", it.get("bbox", [])))
        sz = it.get("size", "")
        if cat and bb:
            out.append({"category": cat, "bbox": bb, "size": sz})
    return out


def score(pred_items, gt_items, score_size=True):
    s = {"tp": 0, "fp": 0, "fn": 0,
         "iou_sum": 0.0, "match": 0,
         "size_correct": 0, "size_total": 0,
         "per_cat": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})}
    if not gt_items and not pred_items:
        return s
    matched_gt = set()
    for p in pred_items:
        best_iou, best_idx, best_gt = -1.0, -1, None
        for i, g in enumerate(gt_items):
            if i in matched_gt:
                continue
            iou = calc_iou(p["bbox"], g["bbox"])
            if iou > best_iou:
                best_iou, best_idx, best_gt = iou, i, g
        if best_iou >= IOU_THRESHOLD and best_idx >= 0:
            matched_gt.add(best_idx)
            s["iou_sum"] += best_iou
            s["match"] += 1
            if p["category"] == best_gt["category"]:
                s["tp"] += 1
                s["per_cat"][p["category"]]["tp"] += 1
                if score_size:
                    s["size_total"] += 1
                    if p["size"] == best_gt["size"]:
                        s["size_correct"] += 1
            else:
                s["fp"] += 1
                s["fn"] += 1
                s["per_cat"][p["category"]]["fp"] += 1
                s["per_cat"][best_gt["category"]]["fn"] += 1
        else:
            s["fp"] += 1
            s["per_cat"][p["category"]]["fp"] += 1
    for i, g in enumerate(gt_items):
        if i not in matched_gt:
            s["fn"] += 1
            s["per_cat"][g["category"]]["fn"] += 1
    return s


def f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def merge(total, s):
    for k in ["tp", "fp", "fn", "iou_sum", "match", "size_correct", "size_total"]:
        total[k] += s[k]
    for c, v in s["per_cat"].items():
        for kk in ["tp", "fp", "fn"]:
            total["per_cat"][c][kk] += v[kk]


# =====================================================================
# Generation (similar to inference_swift.py but stripped down)
# =====================================================================
def _resolve_device_map():
    v = os.environ.get("INFERENCE_DEVICE_MAP", "auto").strip()
    if not v or v.lower() == "auto":
        return "auto"
    if v.lower() in ("none", "manual"):
        return None
    if v.lower() == "cpu" or v.lower().startswith("cuda:"):
        return {"": v}
    return v


def load_base_with_first_adapter(model_path, adapter_path):
    from transformers import AutoProcessor
    from peft import PeftModel
    try:
        from transformers import AutoModelForImageTextToText as _Cls
    except ImportError:
        from transformers import AutoModelForCausalLM as _Cls

    logger.info(f"Loading processor: {model_path}")
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    logger.info(f"Loading base model: {model_path}")
    os.environ.setdefault("INFERENCE_MODEL_PATH", model_path)
    kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    dm = _resolve_device_map()
    if dm is not None:
        kwargs["device_map"] = dm
    try:
        import flash_attn  # noqa
        kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        kwargs["attn_implementation"] = "sdpa"
    model = _Cls.from_pretrained(model_path, **kwargs)

    name = Path(adapter_path).name
    logger.info(f"Loading first adapter: {adapter_path}  (name='{name}')")
    model = PeftModel.from_pretrained(model, str(adapter_path), adapter_name=name)
    model.set_adapter(name)
    if dm is None:
        model = model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, proc


def swap_adapter(model, adapter_path):
    name = Path(adapter_path).name
    if name not in model.peft_config:
        logger.info(f"Loading adapter: {adapter_path}")
        model.load_adapter(str(adapter_path), adapter_name=name)
    model.set_adapter(name)


def gen_one_turn(model, processor, messages, max_new_tokens):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs,
                           videos=video_inputs if video_inputs else None,
                           return_tensors="pt")
    except ImportError:
        from PIL import Image
        paths = [c.get("image", "").replace("file://", "")
                 for m in messages if isinstance(m.get("content"), list)
                 for c in m["content"] if c.get("type") == "image"]
        imgs = [Image.open(p).convert("RGB") for p in paths]
        inputs = processor(text=[text], images=imgs if imgs else None, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
            repetition_penalty=1.05,
        )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_ids, skip_special_tokens=True)


def gen_two_turn(model, processor, image_path, sys_msg_content, user1_text, user2_text, max_new_tokens):
    msgs = [
        {"role": "system", "content": sys_msg_content},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": user1_text},
        ]},
    ]
    raw_v = gen_one_turn(model, processor, msgs, max_new_tokens)
    msgs.append({"role": "assistant", "content": raw_v})
    msgs.append({"role": "user", "content": user2_text})
    raw_f = gen_one_turn(model, processor, msgs, max_new_tokens)
    return raw_v, raw_f


# =====================================================================
# Main
# =====================================================================
def latest_version_dir(output_dir: Path) -> Path:
    versions = [p for p in output_dir.glob("v*-*") if p.is_dir()]
    if not versions:
        raise FileNotFoundError(f"No v*-* under {output_dir}")
    return max(versions, key=lambda p: p.stat().st_mtime)


def load_val_subset(val_jsonl: Path, n: int, seed: int):
    rng = random.Random(seed)
    items = []
    with open(val_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    rng.shuffle(items)
    return items[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Top-level output dir containing v*-* subdirs")
    ap.add_argument("--val-jsonl", required=True, type=Path,
                    help="Val JSONL (multi-turn format) for ground truth")
    ap.add_argument("--image-dir", required=True, type=Path,
                    help="Local dir with the PNG images referenced by val JSONL")
    ap.add_argument("--model-path", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--version-dir", default=None,
                    help="Specific v*-* dir name; defaults to most recent")
    ap.add_argument("--ckpts", default=None,
                    help="Comma-separated checkpoint names (e.g., 'checkpoint-300,checkpoint-400'); "
                         "defaults to ALL surviving checkpoint-* in the version dir")
    ap.add_argument("--n-samples", type=int, default=20,
                    help="Number of val images to generate on per checkpoint")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--layout-iou", type=float, default=cm.DEFAULT_LAYOUT_IOU)
    ap.add_argument("--feature-iou", type=float, default=cm.DEFAULT_FEATURE_IOU)
    ap.add_argument("--strict-feature-iou", type=float, default=cm.DEFAULT_STRICT_FEATURE_IOU)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="Output JSON path; defaults to <version_dir>/checkpoint_eval.json")
    args = ap.parse_args()

    if args.version_dir:
        vdir = args.output_dir / args.version_dir
    else:
        vdir = latest_version_dir(args.output_dir)
    logger.info(f"version dir: {vdir}")

    if args.ckpts:
        ckpts = [vdir / c.strip() for c in args.ckpts.split(",")]
    else:
        ckpts = sorted(
            [p for p in vdir.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1]),
        )
    if not ckpts:
        raise FileNotFoundError("No checkpoints to evaluate")
    logger.info(f"Will evaluate {len(ckpts)} checkpoints: {[c.name for c in ckpts]}")

    val_items = load_val_subset(args.val_jsonl, args.n_samples, args.seed)
    logger.info(f"Selected {len(val_items)} val samples (seed={args.seed})")

    out_path = Path(args.out) if args.out else (vdir / "checkpoint_eval.json")

    # Load base + first adapter (slow); subsequent ckpts swap via PEFT
    model, processor = load_base_with_first_adapter(args.model_path, ckpts[0])

    all_results = {}
    for ci, ckpt in enumerate(ckpts):
        logger.info(f"\n=== [{ci+1}/{len(ckpts)}] {ckpt.name} ===")
        if ci > 0:
            swap_adapter(model, ckpt)

        view_total = cm.empty_stats()
        feat_total = cm.empty_stats()
        strict_feat_total = cm.empty_stats()

        t0 = time.time()
        for i, rec in enumerate(val_items):
            image_name = Path(rec["images"][0]).name
            local_img = args.image_dir / image_name
            if not local_img.exists():
                logger.warning(f"  skipping (image not found): {local_img}")
                continue
            sys_msg = rec["messages"][0]["content"]
            u1 = rec["messages"][1]["content"].replace("<image>", "")
            u2 = rec["messages"][3]["content"]
            try:
                raw_v, raw_f = gen_two_turn(
                    model, processor, str(local_img), sys_msg, u1, u2, args.max_new_tokens
                )
            except Exception as e:
                logger.error(f"  [{i+1}] {image_name}: gen failed: {e}")
                continue

            # Ground truth
            try:
                gt_v = json.loads(rec["messages"][2]["content"])
            except Exception:
                gt_v = []
            try:
                gt_f = json.loads(rec["messages"][4]["content"])
            except Exception:
                gt_f = []

            pred_v = cm.extract_items(cm.parse_json_arrays(raw_v))
            pred_f = cm.extract_items(cm.parse_json_arrays(raw_f))
            _, view_stats, feat_stats, _, strict_feat_stats = cm.compute_split_metrics_from_items(
                pred_v,
                pred_f,
                cm.extract_items(gt_v),
                cm.extract_items(gt_f),
                layout_iou=args.layout_iou,
                feature_iou=args.feature_iou,
                strict_feature_iou=args.strict_feature_iou,
            )
            cm.merge_stats(view_total, view_stats)
            cm.merge_stats(feat_total, feat_stats)
            cm.merge_stats(strict_feat_total, strict_feat_stats)

            elapsed = time.time() - t0
            avg = elapsed / (i + 1)
            eta = avg * (len(val_items) - i - 1)
            logger.info(f"  [{i+1}/{len(val_items)}] {image_name} | "
                        f"v={len(pred_v)} f={len(pred_f)} | {elapsed/60:.1f}m elapsed | ETA {eta/60:.0f}m")

        combined_total = cm.add_stats(view_total, feat_total)
        metrics = cm.flatten_metrics(view_total, feat_total, combined_total, strict_feat_total)

        per_cat_views = {}
        for c in sorted(cm.VIEW_CATEGORIES):
            s = view_total["per_cat"].get(c, {"tp": 0, "fp": 0, "fn": 0})
            _, _, ff = cm.prf(s["tp"], s["fp"], s["fn"])
            per_cat_views[c] = {"tp": s["tp"], "fp": s["fp"], "fn": s["fn"], "f1": ff}
        per_cat_feats = {}
        for c in sorted(cm.FEATURE_CATEGORIES):
            s = feat_total["per_cat"].get(c, {"tp": 0, "fp": 0, "fn": 0})
            _, _, ff = cm.prf(s["tp"], s["fp"], s["fn"])
            per_cat_feats[c] = {"tp": s["tp"], "fp": s["fp"], "fn": s["fn"], "f1": ff}

        result = {
            "checkpoint": ckpt.name,
            "n_samples": len(val_items),
            "views": {
                "P": metrics["layout_precision"],
                "R": metrics["layout_recall"],
                "F1": metrics["layout_f1"],
                "mean_iou": metrics["layout_bbox_iou_mean"],
            },
            "features": {
                "P": metrics["feature_precision"],
                "R": metrics["feature_recall"],
                "F1": metrics["feature_f1"],
                "strict_F1": metrics["feature_f1_strict"],
                "mean_iou": metrics["feature_bbox_iou_mean"],
                "size_acc": metrics["size_accuracy"],
                "size_similarity": metrics["size_similarity"],
            },
            "overall": {
                "P": metrics["detection_precision"],
                "R": metrics["detection_recall"],
                "F1": metrics["detection_f1"],
                "CADScore": metrics["cad_extraction_score"],
            },
            "per_cat_views": per_cat_views,
            "per_cat_features": per_cat_feats,
        }
        all_results[ckpt.name] = result
        logger.info(
            "  -> layout F1=%.4f, feature F1=%.4f, size_sim=%.4f, CADScore=%.4f",
            metrics["layout_f1"],
            metrics["feature_f1"],
            metrics["size_similarity"],
            metrics["cad_extraction_score"],
        )

        # Persist after each checkpoint so partial results survive interruption
        with open(out_path, "w") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Final summary
    print("\n=== SUMMARY (sorted by CADScore, descending) ===")
    print(
        f"{'ckpt':<20s} {'layout F1':>10s} {'feature F1':>11s} "
        f"{'size sim':>10s} {'CADScore':>10s}"
    )
    ranked = sorted(all_results.items(),
                    key=lambda kv: -kv[1]["overall"]["CADScore"])
    for name, r in ranked:
        print(
            f"{name:<20s} {r['views']['F1']:>10.4f} "
            f"{r['features']['F1']:>11.4f} "
            f"{r['features']['size_similarity']:>10.4f} "
            f"{r['overall']['CADScore']:>10.4f}"
        )
    winner = ranked[0][0]
    print(f"\nWinner by CADScore: {winner}")
    print(f"Use --adapter_path {vdir / winner} for inference.")
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
