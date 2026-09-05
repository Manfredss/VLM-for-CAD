"""
extract_roughness.py — Extract surface roughness specs from engineering drawings via Claude Opus 4.7.

Reads either:
  (a) a single image (--image)
  (b) a directory of images (--image-dir + glob pattern)
  (c) an inference results JSON (--input) and augments each record with `roughness`

Output: JSON list of records, each with `dataitem_name` and `roughness` field.
The roughness field is a list of structured specs (parameter, value, scope, context).

Cost note: Opus 4.7 is the most expensive Claude model. Each engineering drawing
takes ~$0.10-0.30 depending on size. Budget accordingly for large batches.

Authentication:
  Uses OpenRouter (https://openrouter.ai) as the routing layer to Claude.
  Set OPENROUTER_API_KEY (preferred) or ANTHROPIC_API_KEY env var with an
  OpenRouter `sk-or-v1-...` key.
  Override the endpoint with --base-url and the model slug with --model.

Resumability:
  Each completed record is appended to the output as it finishes. Re-running with
  the same --output skips already-processed images.
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

DEFAULT_MODEL = "anthropic/claude-opus-4.7"   # OpenRouter slug — verify at https://openrouter.ai/models
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
MAX_IMAGE_DIM = 4000        # downscale longest side to this if larger
MAX_IMAGE_BYTES = 4_500_000 # leave headroom under the 5 MB API ceiling
MAX_RETRIES = 4

SYSTEM_PROMPT = """You are an expert mechanical engineer reading Siemens-style engineering drawings.

Your task: extract every surface roughness specification visible in the drawing.

Surface roughness conventions (ISO 1302 / ASME Y14.36):
- Symbols are check-mark / V-shaped triangles, often nested with values like:
    Ra 1.6, Ra 3.2, Ra 6.3, Ra 0.8, Rz 25, Rmax 12.5
- "Ra" (arithmetic average, in μm) is by far the most common parameter.
- Other parameters: Rz, Rmax, Rt, Rp, Rq, Rsm. If only a number is shown next to the
  triangle without a parameter prefix, default the parameter to "Ra".
- Three symbol variants:
    · Plain V                = Any surface texture allowed (rare; usually treat as Ra)
    · V with horizontal bar  = Material removal required (machined)
    · V with circle          = No material removal allowed (as-cast / forged)
- A "general" roughness applies to all otherwise-unmarked surfaces. It typically
  appears in or near the Title Block, often with a "(rest)" annotation, parenthetical
  symbols, or wording like "ALL OVER", "其他", "其余", "Rest".
- A "surface-specific" roughness is attached via a leader line to one specific edge or face.

Output format (strict JSON, no prose):
{
  "roughness_specs": [
    {
      "parameter": "Ra",
      "value": "1.6",
      "unit": "μm",
      "scope": "general" | "surface",
      "process": "machined" | "no_machining" | "any",
      "context": "<brief location/description, e.g. 'top flange edge', 'general (rest)', 'inside bore'>"
    }
  ]
}

Rules:
- value: numeric only as a string ("1.6", not "Ra 1.6").
- If the symbol has only a parameter and no value (rare), set value to "" and put a
  note in context.
- Multiple specs are common; return ALL of them, including the general one.
- If no roughness annotation is visible at all, return {"roughness_specs": []}.
- Do not invent values. If a symbol is illegible, omit it.
- Do not return any text outside the JSON object."""

USER_PROMPT = """Analyze this engineering drawing and extract every surface roughness specification.

Return only the JSON object described in the system prompt."""


def _encode_image(path: Path) -> tuple[str, str]:
    """Returns (base64_data, media_type). Resizes if too large for the API."""
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
        # PNG too big — re-encode as JPEG
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=88)
        raw = buf.getvalue()
        return base64.b64encode(raw).decode("ascii"), "image/jpeg"
    return base64.b64encode(raw).decode("ascii"), "image/png"


def _call_api(client, model: str, b64: str, media_type: str) -> dict:
    """Single API call with retries. Returns parsed dict; on persistent failure,
    returns {"_error": "<reason>"}.

    Uses the OpenAI chat-completions schema (works for OpenRouter and any
    OpenAI-compatible endpoint).
    """
    data_url = f"data:{media_type};base64,{b64}"
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": USER_PROMPT},
                    ]},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            # Be tolerant of fenced JSON.
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:].strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                last_err = f"JSON parse error: {e}; response head: {text[:200]!r}"
                if attempt >= 1:
                    return {"_error": last_err}
        except Exception as e:
            last_err = f"API error: {type(e).__name__}: {e}"
            sleep = 2 ** attempt
            logger.warning(f"  attempt {attempt+1}/{MAX_RETRIES} failed: {last_err}; sleeping {sleep}s")
            time.sleep(sleep)
    return {"_error": last_err or "unknown failure"}


def _iter_inputs(args):
    """Yield (image_name, full_path) tuples based on args."""
    if args.image:
        p = Path(args.image)
        yield p.name, p
        return
    if args.image_dir:
        d = Path(args.image_dir)
        for p in sorted(d.glob(args.glob)):
            yield p.name, p
        return
    if args.input:
        src = json.load(open(args.input))
        for rec in src:
            name = rec.get("dataitem_name") or rec.get("image")
            if not name:
                continue
            local = Path(args.image_dir or ".") / name
            yield name, local


def main():
    ap = argparse.ArgumentParser(description="Extract surface roughness via Claude Opus 4.7")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=Path, help="Single image path")
    g.add_argument("--image-dir", type=Path, help="Directory containing images")
    g.add_argument("--input", type=Path,
                   help="Inference results JSON (uses dataitem_name field per record)")
    ap.add_argument("--glob", default="*.png",
                    help="Glob pattern when using --image-dir (default: *.png)")
    ap.add_argument("--image-resolve-dir", type=Path,
                    help="When using --input, resolve images from this dir instead of cwd")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output JSON path")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N images (0 = all). Useful for cost-controlled smoke tests.")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
                    help=f"Model slug (default: {DEFAULT_MODEL}). Override at https://openrouter.ai/models if the slug changed.")
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
                    help=f"OpenAI-compatible endpoint (default: {DEFAULT_BASE_URL})")
    args = ap.parse_args()

    # If --input was used but --image-dir also given, image-dir is the resolve dir.
    if args.input and args.image_dir:
        args.image_resolve_dir = args.image_dir
        args.image_dir = None

    api_key = (os.environ.get("OPENROUTER_API_KEY")
               or os.environ.get("ANTHROPIC_API_KEY"))
    if not api_key:
        logger.error("Set OPENROUTER_API_KEY (preferred) or ANTHROPIC_API_KEY in env.")
        sys.exit(1)
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed. pip install openai")
        sys.exit(1)
    try:
        from PIL import Image  # noqa: F401  - imported in _encode_image; check early
    except ImportError:
        logger.error("Pillow not installed. pip install Pillow")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    logger.info(f"Endpoint: {args.base_url}")
    logger.info(f"Model:    {args.model}")

    # Build list of (name, path)
    if args.input and args.image_resolve_dir:
        # Manual iteration since _iter_inputs uses args.image_dir
        src = json.load(open(args.input))
        items = []
        for rec in src:
            name = rec.get("dataitem_name") or rec.get("image")
            if name:
                items.append((name, args.image_resolve_dir / name))
    else:
        items = list(_iter_inputs(args))

    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        logger.error("No images to process. Check your --image / --image-dir / --input.")
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

    # Process
    out = list(existing)
    n_done = n_err = 0
    for i, (name, path) in enumerate(items, 1):
        if name in done_names:
            continue
        if not path.exists():
            logger.warning(f"[{i}/{len(items)}] missing: {path}")
            out.append({"dataitem_name": name, "roughness": [],
                        "_error": f"image not found: {path}"})
            n_err += 1
        else:
            t0 = time.time()
            b64, mt = _encode_image(path)
            result = _call_api(client, args.model, b64, mt)
            elapsed = time.time() - t0
            if "_error" in result:
                logger.warning(f"[{i}/{len(items)}] {name} | ERROR: {result['_error']} | {elapsed:.1f}s")
                out.append({"dataitem_name": name, "roughness": [],
                            "_error": result["_error"]})
                n_err += 1
            else:
                specs = result.get("roughness_specs", [])
                logger.info(f"[{i}/{len(items)}] {name} | {len(specs)} roughness spec(s) | {elapsed:.1f}s")
                out.append({"dataitem_name": name, "roughness": specs})
                n_done += 1

        # Persist after every record (resumability)
        with open(args.output, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info(f"Done. Wrote {args.output}")
    logger.info(f"  processed OK : {n_done}")
    logger.info(f"  errors       : {n_err}")
    logger.info(f"  total records: {len(out)}")


if __name__ == "__main__":
    main()
