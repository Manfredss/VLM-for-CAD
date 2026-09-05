"""
score_predictions.py — Step 2 of the DPO pipeline.

For each sampled prediction (from generate_predictions.py), compute the
quality score against ground truth from the train JSONL. The score uses
the same metric formulation as metric.py:
   quality = mean(0.5 * iou_reward + 0.5 * label_match) over IoU-matched pairs
   minus a soft penalty for FP and FN.

Output: same JSONL but with `quality_views`, `quality_features` and
`quality_total` per sample.
"""
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


def calc_iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter <= 0: return 0.0
    A = max(0,ax2-ax1)*max(0,ay2-ay1); B = max(0,bx2-bx1)*max(0,by2-by1)
    u = A + B - inter
    return inter/u if u > 0 else 0.0


def norm_box(b):
    if not isinstance(b,(list,tuple)) or len(b) != 4: return []
    try: x1,y1,x2,y2 = [float(v) for v in b]
    except: return []
    if max(abs(x1),abs(y1),abs(x2),abs(y2)) <= 1.5:
        x1,y1,x2,y2 = [v*1000 for v in (x1,y1,x2,y2)]
    n = [int(round(v)) for v in (x1,y1,x2,y2)]
    n = [max(0,min(1000,v)) for v in n]
    if n[0]>=n[2] or n[1]>=n[3]: return []
    return n


def parse_json_list(text: str):
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list): return obj
    except Exception: pass
    m = re.search(r"```json\s*([\s\S]+?)```", text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, list): return obj
        except Exception: pass
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, list): return obj
        except Exception: pass
    return []


def extract(items):
    out = []
    for it in items:
        cat = it.get("category", it.get("label",""))
        bb = norm_box(it.get("bbox_2d", it.get("bbox",[])))
        sz = it.get("size","")
        if cat and bb:
            out.append({"category": cat, "bbox": bb, "size": sz})
    return out


def quality(pred_items, gt_items, iou_thr=0.4, fp_penalty=0.05, fn_penalty=0.05):
    """Score in [0, 1] — higher is better.

    base_quality = sum(0.5*iou_reward + 0.5*label_match) over matched pairs / max(len(pred), len(gt))
    Then subtract small penalties for unmatched pred (FP) and unmatched gt (FN).
    """
    if not gt_items and not pred_items:
        return 1.0
    matched_gt = set()
    base = 0.0
    matches = 0
    fp = 0
    for p in pred_items:
        best_iou, best_idx, best_gt = -1.0, -1, None
        for i, g in enumerate(gt_items):
            if i in matched_gt: continue
            iou = calc_iou(p["bbox"], g["bbox"])
            if iou > best_iou:
                best_iou, best_idx, best_gt = iou, i, g
        if best_iou >= iou_thr and best_idx >= 0:
            matched_gt.add(best_idx)
            iou_r = 1.0 if best_iou >= 0.8 else best_iou
            lab_m = 1.0 if p["category"] == best_gt["category"] else 0.0
            base += 0.5 * iou_r + 0.5 * lab_m
            matches += 1
        else:
            fp += 1
    fn = len(gt_items) - len(matched_gt)
    denom = max(len(pred_items), len(gt_items), 1)
    score = base / denom
    score -= fp_penalty * fp / denom
    score -= fn_penalty * fn / denom
    return max(0.0, min(1.0, score))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="from generate_predictions.py")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # Load ground truth from train JSONL keyed by image filename
    gt_by_image = {}
    with open(args.train_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            img = Path(rec["images"][0]).name
            msgs = rec["messages"]
            try:
                gt_v = json.loads(msgs[2]["content"]) if len(msgs) >= 3 else []
            except Exception: gt_v = []
            try:
                gt_f = json.loads(msgs[4]["content"]) if len(msgs) >= 5 else []
            except Exception: gt_f = []
            gt_by_image[img] = (gt_v if isinstance(gt_v, list) else [],
                                gt_f if isinstance(gt_f, list) else [])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.samples) as f, open(out_path, "w") as out:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            img = rec["image"]
            if img not in gt_by_image:
                continue
            gt_v, gt_f = gt_by_image[img]
            gt_v_e = extract(gt_v); gt_f_e = extract(gt_f)
            pred_v = extract(parse_json_list(rec.get("raw_views","")))
            pred_f = extract(parse_json_list(rec.get("raw_features","")))
            qv = quality(pred_v, gt_v_e)
            qf = quality(pred_f, gt_f_e)
            rec["quality_views"] = qv
            rec["quality_features"] = qf
            rec["quality_total"] = (qv + qf) / 2.0
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"scored -> {out_path}")


if __name__ == "__main__":
    main()
