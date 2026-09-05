"""
extract_gdt.py — Extract GD&T annotations from engineering drawings via Claude Opus 4.7.

Covers three annotation categories:
  1. Surface roughness  (ISO 1302 / ASME Y14.36) — Ra, Rz, etc.
  2. Geometric tolerances (ISO 1101 / ASME Y14.5) — feature control frames
     (flatness, position, perpendicularity, runout, …)
  3. Datums — reference datum labels (A, B, C, …)

Reads either:
  (a) a single image          (--image)
  (b) a directory of images   (--image-dir + --glob)
  (c) an inference results JSON  (--input) — augments each record with `gdt`

Output: JSON list, each record has `dataitem_name` and `gdt` (list of annotations).

Authentication:
  Uses OpenRouter as the routing layer to Claude.
  Set OPENROUTER_API_KEY (preferred) or ANTHROPIC_API_KEY in env.

Resumability:
  Each completed record is appended to the output as it finishes.
  Re-running with the same --output skips already-processed images.
"""
import argparse
import base64
import json
import logging
import os
import sys
import time
from io import BytesIO
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL   = "anthropic/claude-opus-4-7"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
MAX_IMAGE_DIM   = 4000
MAX_IMAGE_BYTES = 4_500_000
MAX_RETRIES     = 4

SYSTEM_PROMPT = """You are an expert mechanical engineer reading engineering drawings (ISO/GB standards).

Your task: extract every GD&T annotation visible in the drawing. Return a single JSON object.

══════════════════════════════════════════════════
CATEGORY 1 — SURFACE ROUGHNESS (ISO 1302 / ASME Y14.36)
══════════════════════════════════════════════════
Symbols: check-mark / V-shaped triangles, often with Ra/Rz/Rmax values in μm.
- Parameters: Ra (most common), Rz, Rmax, Rt, Rp, Rq, Rsm.
  If only a number appears next to the triangle with no prefix, assume Ra.
- Process variants:
    plain V            → process = "any"
    V with horiz. bar  → process = "machined"
    V with circle      → process = "no_machining"
- Scope:
    "general" — applies to all otherwise-unmarked surfaces (near Title Block,
                often with "(rest)" / "其他" / "其余" / "ALL OVER" annotation)
    "surface" — attached to a specific edge or face via leader line

══════════════════════════════════════════════════
CATEGORY 2 — GEOMETRIC TOLERANCES (ISO 1101 / ASME Y14.5)
══════════════════════════════════════════════════
These appear as Feature Control Frames: a rectangular box divided into cells:
  [geometric symbol | tolerance value | (material condition) | datum refs…]

Recognised symbols (use exactly these English names):
  Form        : flatness, straightness, circularity, cylindricity
  Orientation : angularity, perpendicularity, parallelism
  Location    : position, concentricity, symmetry
  Profile     : profile_of_line, profile_of_surface
  Runout      : circular_runout, total_runout

Material condition modifiers (appear as circled letters):
  ⓜ → "MMC"   (maximum material condition)
  ⓛ → "LMC"   (least material condition)
  ⓢ → "RFS"   (regardless of feature size — default, often omitted)
  null if not shown.

Datum references: letters (A, B, C, …) in the rightmost cells of the frame.
Tolerance value: the numeric string in the second cell, e.g. "0.05".
Tolerance unit: always "mm" unless otherwise marked.

══════════════════════════════════════════════════
CATEGORY 3 — DATUMS (ISO 5459 / ASME Y14.5)
══════════════════════════════════════════════════
Datum feature symbols: a filled triangle attached to a surface or feature,
with a square box containing a letter (A, B, C, …).
Capture each unique datum label once, with a brief context of where it is.

══════════════════════════════════════════════════
CATEGORY 4 — DIMENSIONAL TOLERANCES
══════════════════════════════════════════════════
Any dimension annotated with an explicit tolerance. Three forms:

  a) Symmetric bilateral — ± symbol:
       25 ±0.1      → nominal="25", upper="+0.1", lower="-0.1"

  b) Asymmetric bilateral — stacked +/- values:
       25 +0.05/-0.02  → nominal="25", upper="+0.05", lower="-0.02"
       25 +0.1/0       → nominal="25", upper="+0.1",  lower="0"

  c) ISO fit / tolerance code appended to the nominal:
       Ø25 H7          → nominal="25", fit_code="H7", upper=null, lower=null
       Ø25 h6          → nominal="25", fit_code="h6"

dimension_type: "linear" | "diameter" | "radius" | "angular"
  - Use "diameter" when the nominal has a Ø prefix.
  - Use "radius"   when the nominal has an R prefix.
  - Use "angular"  when the unit is ° (degrees).
  - Use "linear"   otherwise.

unit: "mm" for lengths, "°" for angles.

Extract EVERY dimensioned feature that carries a tolerance or fit code.
Do NOT extract plain nominal dimensions that have no tolerance annotation.

══════════════════════════════════════════════════
OUTPUT FORMAT (strict JSON, no prose outside the object)
══════════════════════════════════════════════════
{
  "gdt_annotations": [
    {
      "category": "roughness",
      "parameter": "Ra",
      "value": "1.6",
      "unit": "μm",
      "process": "machined",
      "scope": "general",
      "context": "general (rest), shown above title block"
    },
    {
      "category": "geometric_tolerance",
      "symbol": "flatness",
      "tolerance_value": "0.05",
      "tolerance_unit": "mm",
      "datum_references": [],
      "material_condition": null,
      "context": "top mounting face"
    },
    {
      "category": "geometric_tolerance",
      "symbol": "position",
      "tolerance_value": "0.1",
      "tolerance_unit": "mm",
      "datum_references": ["A", "B"],
      "material_condition": "MMC",
      "context": "4× Ø8 bolt holes"
    },
    {
      "category": "datum",
      "label": "A",
      "context": "bottom face"
    },
    {
      "category": "dimensional_tolerance",
      "dimension_type": "linear",
      "nominal_value": "25",
      "upper_deviation": "+0.05",
      "lower_deviation": "-0.02",
      "fit_code": null,
      "unit": "mm",
      "context": "slot width"
    },
    {
      "category": "dimensional_tolerance",
      "dimension_type": "diameter",
      "nominal_value": "20",
      "upper_deviation": null,
      "lower_deviation": null,
      "fit_code": "H7",
      "unit": "mm",
      "context": "main bore"
    }
  ]
}

Rules:
- Return ALL annotations of all four categories.
- value / tolerance_value / nominal_value: numeric string only ("1.6", not "Ra 1.6").
- datum_references: [] (empty list) when no datum is referenced.
- upper_deviation / lower_deviation: include the sign ("+0.05", "-0.02", "0").
  Set to null only when fit_code is given instead.
- fit_code: ISO tolerance code string (e.g. "H7", "h6", "k5") or null.
- If a symbol is illegible, omit that annotation entirely. Do not invent values.
- If no GD&T annotations are visible at all, return {"gdt_annotations": []}.
- Do not return any text outside the JSON object."""

USER_PROMPT = """Analyze this engineering drawing and extract every GD&T annotation
(surface roughness, geometric tolerances, datums, dimensional tolerances with ± or fit codes).

Return only the JSON object described in the system prompt."""


def _encode_image(path: Path) -> tuple[str, str]:
    from PIL import Image
    img = Image.open(path)
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    raw = buf.getvalue()
    if len(raw) > MAX_IMAGE_BYTES:
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=88)
        raw = buf.getvalue()
        return base64.b64encode(raw).decode("ascii"), "image/jpeg"
    return base64.b64encode(raw).decode("ascii"), "image/png"


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def _call_api_anthropic(client, model: str, b64: str, media_type: str) -> dict:
    """Direct Anthropic SDK path (sk-ant-... keys)."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=8192,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": USER_PROMPT},
                ]}],
            )
            text = resp.content[0].text
            try:
                return _parse_json_response(text)
            except json.JSONDecodeError as e:
                last_err = f"JSON parse error: {e}; head: {text[:200]!r}"
                if attempt >= 1:
                    return {"_error": last_err}
        except Exception as e:
            last_err = f"API error: {type(e).__name__}: {e}"
            sleep = 2 ** attempt
            logger.warning(f"  attempt {attempt+1}/{MAX_RETRIES} failed: {last_err}; sleeping {sleep}s")
            time.sleep(sleep)
    return {"_error": last_err or "unknown failure"}


def _call_api_openai(client, model: str, b64: str, media_type: str) -> dict:
    """OpenRouter / OpenAI-compatible path."""
    data_url = f"data:{media_type};base64,{b64}"
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=8192,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": USER_PROMPT},
                    ]},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            try:
                return _parse_json_response(text)
            except json.JSONDecodeError as e:
                last_err = f"JSON parse error: {e}; head: {text[:200]!r}"
                if attempt >= 1:
                    return {"_error": last_err}
        except Exception as e:
            last_err = f"API error: {type(e).__name__}: {e}"
            sleep = 2 ** attempt
            logger.warning(f"  attempt {attempt+1}/{MAX_RETRIES} failed: {last_err}; sleeping {sleep}s")
            time.sleep(sleep)
    return {"_error": last_err or "unknown failure"}


def main():
    ap = argparse.ArgumentParser(description="Extract GD&T annotations via Claude Opus 4.7")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image",     type=Path, help="Single image path")
    g.add_argument("--image-dir", type=Path, help="Directory containing images")
    g.add_argument("--input",     type=Path,
                   help="Inference results JSON (uses dataitem_name per record)")
    ap.add_argument("--glob",   default="*.png",
                    help="Glob pattern when using --image-dir (default: *.png)")
    ap.add_argument("--image-resolve-dir", type=Path,
                    help="When using --input, resolve images from this dir")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output JSON path")
    ap.add_argument("--limit",  type=int, default=0,
                    help="Process at most N images (0 = all). Useful for smoke tests.")
    ap.add_argument("--model",
                    default=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
                    help=f"Model slug (default: {DEFAULT_MODEL})")
    ap.add_argument("--base-url",
                    default=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
                    help=f"OpenAI-compatible endpoint (default: {DEFAULT_BASE_URL})")
    args = ap.parse_args()

    if args.input and args.image_dir:
        args.image_resolve_dir = args.image_dir
        args.image_dir = None

    api_key = (os.environ.get("OPENROUTER_API_KEY")
               or os.environ.get("ANTHROPIC_API_KEY"))
    if not api_key:
        logger.error("Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY in env.")
        sys.exit(1)
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        logger.error("pip install Pillow")
        sys.exit(1)

    # Choose SDK based on key type
    use_anthropic_sdk = api_key.startswith("sk-ant-")
    if use_anthropic_sdk:
        try:
            import anthropic as _anthropic
        except ImportError:
            logger.error("pip install anthropic")
            sys.exit(1)
        client = _anthropic.Anthropic(api_key=api_key)
        _call_api = _call_api_anthropic
        # For direct Anthropic, model slug has no "anthropic/" prefix
        if args.model.startswith("anthropic/"):
            args.model = args.model.split("/", 1)[1]
        logger.info(f"Backend  : Anthropic SDK (direct)")
    else:
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("pip install openai")
            sys.exit(1)
        client = OpenAI(api_key=api_key, base_url=args.base_url)
        _call_api = _call_api_openai
        logger.info(f"Backend  : OpenRouter / OpenAI-compat ({args.base_url})")
    logger.info(f"Model    : {args.model}")

    # Build items list
    if args.input:
        src_dir = args.image_resolve_dir or Path(".")
        src = json.load(open(args.input))
        items = [(r.get("dataitem_name") or r.get("image"), src_dir / (r.get("dataitem_name") or r.get("image", "")))
                 for r in src if r.get("dataitem_name") or r.get("image")]
    elif args.image_dir:
        items = [(p.name, p) for p in sorted(args.image_dir.glob(args.glob))]
    else:
        items = [(args.image.name, args.image)]

    if args.limit > 0:
        items = items[:args.limit]
    if not items:
        logger.error("No images to process.")
        sys.exit(1)

    # Resume support
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_names = set()
    existing = []
    if args.output.exists():
        try:
            existing = json.load(open(args.output))
            done_names = {r.get("dataitem_name") for r in existing if r.get("dataitem_name")}
            logger.info(f"Resume: skipping {len(done_names)} already-processed images")
        except Exception:
            existing = []

    out = list(existing)
    n_done = n_err = 0
    for i, (name, path) in enumerate(items, 1):
        if name in done_names:
            continue
        if not path.exists():
            logger.warning(f"[{i}/{len(items)}] missing: {path}")
            out.append({"dataitem_name": name, "gdt": [], "_error": f"image not found: {path}"})
            n_err += 1
        else:
            t0 = time.time()
            b64, mt = _encode_image(path)
            result = _call_api(client, args.model, b64, mt)  # _call_api set above
            elapsed = time.time() - t0
            if "_error" in result:
                logger.warning(f"[{i}/{len(items)}] {name} | ERROR: {result['_error']} | {elapsed:.1f}s")
                out.append({"dataitem_name": name, "gdt": [], "_error": result["_error"]})
                n_err += 1
            else:
                annotations = result.get("gdt_annotations", [])
                by_cat = {}
                for a in annotations:
                    by_cat[a.get("category", "?")] = by_cat.get(a.get("category", "?"), 0) + 1
                summary = ", ".join(f"{v} {k}" for k, v in sorted(by_cat.items()))
                logger.info(f"[{i}/{len(items)}] {name} | {summary or '0 annotations'} | {elapsed:.1f}s")
                out.append({"dataitem_name": name, "gdt": annotations})
                n_done += 1

        with open(args.output, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info(f"Done. Wrote {args.output}")
    logger.info(f"  processed OK : {n_done}")
    logger.info(f"  errors       : {n_err}")
    logger.info(f"  total records: {len(out)}")


if __name__ == "__main__":
    main()
