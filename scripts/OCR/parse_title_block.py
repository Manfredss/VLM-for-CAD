"""
parse_title_block.py — Heuristic structured-field extraction from Title Block OCR.

Reads `ocr_title_block` (from ocr_extract.py) and tries to extract common
fields by looking for known label tokens (CN + EN) and grabbing the value
text on the same line or the line directly to the right.

This is a rough heuristic; it works on consistent layouts and degrades on
unusual ones. For more robust extraction, feed the raw OCR lines into an LLM
with a structured-extraction prompt.
"""
import argparse
import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# (canonical_field, list of label patterns that match)
LABEL_RULES = [
    ("part_number",   [r"零件号", r"图号", r"part\s*no", r"part\s*number", r"drawing\s*no"]),
    ("name",          [r"名称", r"零件名称", r"part\s*name", r"description"]),
    ("material",      [r"材料", r"材质", r"material", r"matl"]),
    ("scale",         [r"比例", r"scale"]),
    ("date",          [r"日期", r"date"]),
    ("drawn_by",      [r"制图", r"设计", r"drawn", r"drawn\s*by", r"by"]),
    ("checked_by",    [r"审核", r"校对", r"checked", r"checked\s*by"]),
    ("approved_by",   [r"批准", r"approved", r"approved\s*by"]),
    ("revision",      [r"版本", r"修订", r"rev\b", r"revision"]),
    ("weight",        [r"重量", r"质量", r"weight", r"mass"]),
    ("sheet",         [r"图幅", r"页", r"sheet", r"page"]),
    ("project",       [r"项目", r"project"]),
]


def _line_center(bbox_polygon):
    """PaddleOCR bbox is a 4-point polygon [[x,y], ...]. Return (cx, cy)."""
    if not bbox_polygon or len(bbox_polygon) < 1:
        return None
    xs = [p[0] for p in bbox_polygon if isinstance(p, (list, tuple)) and len(p) >= 2]
    ys = [p[1] for p in bbox_polygon if isinstance(p, (list, tuple)) and len(p) >= 2]
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _line_right_x(bbox_polygon):
    if not bbox_polygon:
        return None
    xs = [p[0] for p in bbox_polygon if isinstance(p, (list, tuple)) and len(p) >= 2]
    return max(xs) if xs else None


def _strip_label(text: str, label_pattern: str) -> str:
    """Remove the label and any trailing colon / space from `text`."""
    pat = re.compile(label_pattern + r"[:：\s]*", re.IGNORECASE)
    return pat.sub("", text, count=1).strip()


def extract_fields(ocr_lines):
    """ocr_lines: list of {bbox, text, conf} from PaddleOCR.
    Returns dict of field -> value."""
    if not ocr_lines:
        return {}

    # Build line objects with center coords for adjacency lookups
    enriched = []
    for ln in ocr_lines:
        text = (ln.get("text") or "").strip()
        if not text:
            continue
        center = _line_center(ln.get("bbox"))
        right_x = _line_right_x(ln.get("bbox"))
        enriched.append({"text": text, "center": center, "right_x": right_x,
                         "raw": ln})

    fields = {}
    used_idxs = set()
    for field, label_pats in LABEL_RULES:
        for i, ln in enumerate(enriched):
            if i in used_idxs:
                continue
            for pat in label_pats:
                m = re.search(pat, ln["text"], flags=re.IGNORECASE)
                if not m:
                    continue
                # Try value on same line first (text after the label)
                same = _strip_label(ln["text"], pat)
                # If "label only" (no remaining content), look right on same row
                if not same:
                    if ln["center"] is None:
                        continue
                    cy = ln["center"][1]
                    rx = ln["right_x"] or 0
                    candidates = []
                    for j, other in enumerate(enriched):
                        if j == i or j in used_idxs:
                            continue
                        if other["center"] is None:
                            continue
                        ocy = other["center"][1]
                        ocx = other["center"][0]
                        if abs(ocy - cy) <= 25 and ocx > rx:  # roughly same row, to the right
                            candidates.append((ocx - rx, j, other["text"]))
                    if candidates:
                        candidates.sort(key=lambda t: t[0])
                        used_idxs.add(candidates[0][1])
                        fields[field] = candidates[0][2]
                else:
                    fields[field] = same
                used_idxs.add(i)
                break
            if field in fields:
                break

    return fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path,
                    help="JSON from ocr_extract.py (has ocr_title_block per record)")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--key", default="ocr_title_block",
                    help="Field name in the input JSON to parse from")
    args = ap.parse_args()

    src = json.load(open(args.input))
    n_with_fields = 0
    out = []
    for rec in src:
        rec_out = dict(rec)
        all_fields = []
        regions = rec.get(args.key, [])
        for region in regions:
            lines = region.get("lines", []) if isinstance(region, dict) else []
            f = extract_fields(lines)
            all_fields.append(f)
        # If single Title Block, flatten; else keep list
        if len(all_fields) == 1:
            rec_out["title_block_fields"] = all_fields[0]
            if all_fields[0]:
                n_with_fields += 1
        elif len(all_fields) > 1:
            rec_out["title_block_fields_list"] = all_fields
            if any(all_fields):
                n_with_fields += 1
        out.append(rec_out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote {args.output}")
    logger.info(f"Records with at least one field extracted: {n_with_fields}/{len(src)}")

    if n_with_fields > 0:
        # Print sample
        for rec in out:
            if rec.get("title_block_fields"):
                logger.info(f"\nSample — {rec.get('dataitem_name')}:")
                for k, v in rec["title_block_fields"].items():
                    logger.info(f"  {k}: {v}")
                break


if __name__ == "__main__":
    main()
