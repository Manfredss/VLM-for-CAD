"""Post-hoc scorer: load an inference results JSON (with `result_views` and
`result_features` already parsed) and a test JSONL (with GT in assistant
messages), print real F1 — same logic as evaluate_checkpoints.py but skips
the inference step.

Usage:
    python score_results_json.py \\
        --results scripts/qwen3.5-27b_788_with_view/results_..._swa.json \\
        --test_jsonl scripts/qwen3.5-27b_788_with_view_improve/dataset/test_view_7feats_improve.jsonl
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


VIEW_CATEGORIES = sorted([
    "Title Block", "Notes",
    "Isometric View", "Flat Pattern View",
    "Detail View", "Section View", "Auxiliary View",
    "Orthographic Projection - Front View",
    "Orthographic Projection - Top View",
    "Orthographic Projection - Bottom View",
    "Orthographic Projection - Left View",
    "Orthographic Projection - Right View",
    "Orthographic Projection - Rear View",
])
FEATURE_CATEGORIES = sorted([
    "Round Hole", "Rectangular Hole", "Threaded Hole", "Slotted Hole",
    "Round Hole Group", "Rectangular Hole Group", "Slotted Hole Group",
    "Fillet", "Bending", "Silver Plating",
])
IOU_THRESHOLD = 0.4


def calc_iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0: return 0.0
    A = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    B = max(0, bx2 - bx1) * max(0, by2 - by1)
    u = A + B - inter
    return inter / u if u > 0 else 0.0


def norm_box(b):
    if not isinstance(b, (list, tuple)) or len(b) != 4: return []
    try:
        x1, y1, x2, y2 = [float(v) for v in b]
    except (TypeError, ValueError):
        return []
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1, y1, x2, y2 = [v * 1000 for v in (x1, y1, x2, y2)]
    n = [int(round(v)) for v in (x1, y1, x2, y2)]
    n = [max(0, min(1000, v)) for v in n]
    if n[0] >= n[2] or n[1] >= n[3]: return []
    return n


def parse_json_arrays(text: str):
    items = []
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list): return parsed
    except Exception:
        pass
    for m in re.finditer(r'```json\s*([\s\S]*?)```', text):
        try:
            parsed = json.loads(m.group(1).strip())
            if isinstance(parsed, list): items.extend(parsed)
        except Exception:
            continue
    if items: return items
    for m in re.finditer(r'\[\s*(?:\{[\s\S]*?\}\s*,?\s*)*\]', text):
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list): items.extend(parsed)
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
    if not gt_items and not pred_items: return s
    matched_gt = set()
    for p in pred_items:
        best_iou, best_idx, best_gt = -1.0, -1, None
        for i, g in enumerate(gt_items):
            if i in matched_gt: continue
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--test_jsonl", required=True, type=Path)
    args = ap.parse_args()

    results = json.load(open(args.results))
    by_name = {r["dataitem_name"]: r for r in results}
    print(f"Loaded {len(results)} prediction records")

    # Map test JSONL by image filename
    gts = []
    with open(args.test_jsonl) as f:
        for line in f:
            d = json.loads(line)
            imgs = d.get("images", [])
            if not imgs: continue
            img_name = Path(imgs[0]).name
            msgs = d.get("messages", [])
            asst = [m for m in msgs if m.get("role") == "assistant"]
            if len(asst) < 2: continue
            gt_v = parse_json_arrays(asst[0].get("content", ""))
            gt_f = parse_json_arrays(asst[1].get("content", ""))
            gts.append({"name": img_name,
                        "gt_views": extract(gt_v),
                        "gt_feats": extract(gt_f)})
    print(f"Loaded {len(gts)} GT records")

    view_total = {"tp": 0, "fp": 0, "fn": 0,
                  "iou_sum": 0.0, "match": 0,
                  "size_correct": 0, "size_total": 0,
                  "per_cat": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})}
    feat_total = {"tp": 0, "fp": 0, "fn": 0,
                  "iou_sum": 0.0, "match": 0,
                  "size_correct": 0, "size_total": 0,
                  "per_cat": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})}

    matched, unmatched = 0, 0
    for gt in gts:
        rec = by_name.get(gt["name"])
        if not rec:
            unmatched += 1
            continue
        matched += 1
        pred_v = extract(rec.get("result_views", []))
        pred_f = extract(rec.get("result_features", []))
        merge(view_total, score(pred_v, gt["gt_views"], score_size=False))
        merge(feat_total, score(pred_f, gt["gt_feats"], score_size=True))

    print(f"Matched {matched} preds with GT, {unmatched} GT not in preds")

    Pv, Rv, Fv = f1(view_total["tp"], view_total["fp"], view_total["fn"])
    Pf, Rf, Ff = f1(feat_total["tp"], feat_total["fp"], feat_total["fn"])
    cb_tp = view_total["tp"] + feat_total["tp"]
    cb_fp = view_total["fp"] + feat_total["fp"]
    cb_fn = view_total["fn"] + feat_total["fn"]
    Pc, Rc, Fc = f1(cb_tp, cb_fp, cb_fn)
    iou_v = view_total["iou_sum"] / view_total["match"] if view_total["match"] else 0.0
    iou_f = feat_total["iou_sum"] / feat_total["match"] if feat_total["match"] else 0.0
    size_acc = feat_total["size_correct"] / feat_total["size_total"] if feat_total["size_total"] else 0.0

    print()
    print(f"=== Overall ===")
    print(f"Views    | P={Pv:.4f} R={Rv:.4f} F1={Fv:.4f} mIoU={iou_v:.4f}")
    print(f"Features | P={Pf:.4f} R={Rf:.4f} F1={Ff:.4f} mIoU={iou_f:.4f} size_acc={size_acc:.4f}")
    print(f"Combined | P={Pc:.4f} R={Rc:.4f} F1={Fc:.4f}")

    print()
    print("=== Per-class views ===")
    for c in VIEW_CATEGORIES:
        s = view_total["per_cat"].get(c, {"tp": 0, "fp": 0, "fn": 0})
        _, _, ff = f1(s["tp"], s["fp"], s["fn"])
        print(f"  {c:46s} tp={s['tp']:3d} fp={s['fp']:3d} fn={s['fn']:3d} F1={ff:.4f}")

    print()
    print("=== Per-class features ===")
    for c in FEATURE_CATEGORIES:
        s = feat_total["per_cat"].get(c, {"tp": 0, "fp": 0, "fn": 0})
        _, _, ff = f1(s["tp"], s["fp"], s["fn"])
        print(f"  {c:24s} tp={s['tp']:3d} fp={s['fp']:3d} fn={s['fn']:3d} F1={ff:.4f}")


if __name__ == "__main__":
    main()
