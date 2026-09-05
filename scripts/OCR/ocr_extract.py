"""
ocr_extract.py — Extract text from detected regions via PaddleOCR.

Reads an inference results JSON (output of inference_swift.py — has
`result_views` per record with bboxes in normalized 0-1000 coords),
crops each image to the predicted bbox of each requested region
(default: Title Block), runs PaddleOCR, and appends raw line-level
OCR text to the record.

Output: a new JSON with the same records plus `ocr_<region_snake>` fields.

Run on CPU; ~1-3 s per Title Block crop.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _denorm_bbox(norm_bbox, img_w, img_h, padding=0.0):
    """0-1000 normalized [x1,y1,x2,y2] -> pixel coords with optional padding."""
    if not norm_bbox or len(norm_bbox) != 4:
        return None
    x1, y1, x2, y2 = norm_bbox
    px1 = x1 * img_w / 1000
    py1 = y1 * img_h / 1000
    px2 = x2 * img_w / 1000
    py2 = y2 * img_h / 1000
    # Apply padding (relative to bbox size)
    if padding > 0:
        bw = px2 - px1
        bh = py2 - py1
        px1 -= bw * padding
        py1 -= bh * padding
        px2 += bw * padding
        py2 += bh * padding
    px1 = max(0, int(round(px1)))
    py1 = max(0, int(round(py1)))
    px2 = min(img_w, int(round(px2)))
    py2 = min(img_h, int(round(py2)))
    if px2 <= px1 or py2 <= py1:
        return None
    return (px1, py1, px2, py2)


def _normalize_paddle_result(result):
    """PaddleOCR return shape varies by version. Normalize to a flat list of
    {bbox, text, conf} dicts. Handles:
      - v2.x: [[ [bbox_polygon, (text, conf)], ... ]]
      - v3.x: [ { "rec_texts": [...], "rec_scores": [...], "rec_polys": [...]} ]
              or dict-like OCRResult with the same keys
    Returns [] on any structural mismatch.
    """
    if not result:
        return []
    if not isinstance(result, list) or len(result) == 0:
        return []
    first = result[0]
    if first is None:
        return []

    # v3.x: dict-like OCRResult with rec_texts / rec_scores / rec_polys (or rec_boxes)
    try:
        keys = list(first.keys()) if hasattr(first, "keys") else None
    except Exception:
        keys = None
    if keys and ("rec_texts" in keys or "rec_text" in keys):
        texts = first.get("rec_texts") or first.get("rec_text") or []
        scores = first.get("rec_scores") or first.get("rec_score") or []
        polys = (first.get("rec_polys") or first.get("rec_boxes")
                 or first.get("dt_polys") or [])
        out = []
        for i, t in enumerate(texts):
            try:
                conf = float(scores[i]) if i < len(scores) else 0.0
            except Exception:
                conf = 0.0
            try:
                bbox = polys[i].tolist() if hasattr(polys[i], "tolist") else list(polys[i])
            except Exception:
                bbox = []
            out.append({"bbox": bbox, "text": str(t), "conf": conf})
        return out

    # v2.x: list of [bbox_polygon, (text, conf)] tuples
    if isinstance(first, list):
        out = []
        for line in first:
            if not line:
                continue
            try:
                bbox = line[0]
                txt_conf = line[1]
                if isinstance(txt_conf, (list, tuple)) and len(txt_conf) >= 2:
                    text, conf = txt_conf[0], float(txt_conf[1])
                else:
                    text, conf = str(txt_conf), 0.0
                out.append({"bbox": bbox, "text": text, "conf": conf})
            except Exception:
                continue
        return out
    return []


def _make_key(region: str) -> str:
    return "ocr_" + region.lower().replace(" ", "_").replace("-", "_")


def main():
    ap = argparse.ArgumentParser(description="Run PaddleOCR on detected regions")
    ap.add_argument("--input", required=True, type=Path,
                    help="Input results JSON (from inference_swift.py)")
    ap.add_argument("--image-dir", required=True, type=Path,
                    help="Local dir holding the source PNGs")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output augmented JSON")
    ap.add_argument("--regions", nargs="+", default=["Title Block"],
                    help="View categories to OCR (e.g., 'Title Block' 'Notes')")
    ap.add_argument("--padding", type=float, default=0.05,
                    help="Bbox padding fraction (each side); e.g. 0.05 = +5%%")
    ap.add_argument("--upscale", type=float, default=1.0,
                    help="Upscale crop by this factor before OCR (helps on small text)")
    ap.add_argument("--lang", default="ch",
                    help="PaddleOCR language code; 'ch' = CN+EN bundle")
    ap.add_argument("--use-gpu", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N records (0 = all)")
    args = ap.parse_args()

    # Lazy-import — heavy dependency, give a clearer error if missing
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        logger.error("paddleocr not installed. See scripts/OCR/README.md for install.")
        sys.exit(1)
    try:
        from PIL import Image
    except ImportError:
        logger.error("Pillow not installed. pip install Pillow")
        sys.exit(1)
    try:
        import numpy as np
    except ImportError:
        logger.error("numpy not installed. pip install numpy")
        sys.exit(1)

    src = json.load(open(args.input))
    if args.limit > 0:
        src = src[: args.limit]
    logger.info(f"Loaded {len(src)} records from {args.input}")

    logger.info(f"Initializing PaddleOCR (lang={args.lang}) — first run downloads ~50 MB of models")
    # API differs across versions; try a few combos.
    init_attempts = [
        # v3.x: use_textline_orientation, no use_gpu
        dict(use_textline_orientation=True, lang=args.lang),
        # v2.x with use_gpu
        dict(use_angle_cls=True, lang=args.lang, use_gpu=args.use_gpu),
        # v2.x without use_gpu
        dict(use_angle_cls=True, lang=args.lang),
        # minimal
        dict(lang=args.lang),
    ]
    ocr = None
    last_err = None
    for attempt in init_attempts:
        try:
            ocr = PaddleOCR(**attempt)
            logger.info(f"PaddleOCR initialized with kwargs={list(attempt.keys())}")
            break
        except Exception as e:
            last_err = e
            continue
    if ocr is None:
        logger.error(f"PaddleOCR init failed all attempts: {last_err}")
        sys.exit(1)
    logger.info("PaddleOCR ready")

    out = []
    img_cache = {}

    n_total_crops = 0
    n_total_lines = 0
    for ri, rec in enumerate(src):
        rec_out = dict(rec)
        image_name = rec.get("dataitem_name")
        if not image_name:
            out.append(rec_out)
            continue
        local_path = args.image_dir / image_name
        if not local_path.exists():
            logger.warning(f"[{ri+1}/{len(src)}] image not found: {local_path}")
            out.append(rec_out)
            continue

        try:
            img = img_cache.get(image_name)
            if img is None:
                img = Image.open(local_path).convert("RGB")
                # don't cache big images — cap cache size
                if len(img_cache) < 8:
                    img_cache[image_name] = img
            W, H = img.size
        except Exception as e:
            logger.warning(f"[{ri+1}/{len(src)}] cannot open {local_path}: {e}")
            out.append(rec_out)
            continue

        for region in args.regions:
            entries = [d for d in rec.get("result_views", [])
                       if d.get("category") == region]
            results_for_region = []
            for entry in entries:
                bbox_norm = entry.get("bbox")
                pix = _denorm_bbox(bbox_norm, W, H, padding=args.padding)
                if pix is None:
                    results_for_region.append({"bbox_pixel": None,
                                               "lines": [],
                                               "_warn": "invalid bbox"})
                    continue
                crop = img.crop(pix)
                if args.upscale and abs(args.upscale - 1.0) > 1e-3:
                    new_size = (int(crop.width * args.upscale),
                                int(crop.height * args.upscale))
                    crop = crop.resize(new_size, Image.LANCZOS)
                arr = np.array(crop)
                lines = []
                # Try a sequence of call styles that work across PaddleOCR versions
                for call in (
                    lambda: ocr.predict(arr),         # v3.x preferred
                    lambda: ocr.ocr(arr, cls=True),   # v2.x with cls
                    lambda: ocr.ocr(arr),             # v2.x / v3.x without cls
                ):
                    try:
                        raw = call()
                        lines = _normalize_paddle_result(raw)
                        if lines is not None:
                            break
                    except (TypeError, AttributeError):
                        continue
                    except Exception as e:
                        logger.warning(f"  OCR failed on {image_name}/{region}: {e}")
                        lines = []
                        break
                results_for_region.append({
                    "bbox_norm": bbox_norm,
                    "bbox_pixel": list(pix),
                    "lines": lines,
                })
                n_total_crops += 1
                n_total_lines += len(lines)
            rec_out[_make_key(region)] = results_for_region

        out.append(rec_out)
        if (ri + 1) % 10 == 0 or ri + 1 == len(src):
            logger.info(f"[{ri+1}/{len(src)}] {image_name} | "
                        f"crops so far {n_total_crops}, lines {n_total_lines}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")
    logger.info(f"Total: {n_total_crops} crops, {n_total_lines} OCR'd lines")


if __name__ == "__main__":
    main()
