"""
Convert swift infer output (JSONL) to sample.json format.

Swift writes one JSON object per line:
  {"dataitem_name": ..., "images": [...], "messages": [...], "response": "..."}

This script parses each "response" as a detection list and outputs a JSON array
matching outputs/results.json:
  [{"dataitem_name": ..., "result": [...], "raw": "..."}, ...]

Usage:
    python src/postprocess_swift_infer.py \
        --input  output/infer_results.jsonl \
        --output output/infer_results_sample.json
"""

import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_response(resp: str):
    """
    Try to parse the model response string as a JSON detection list.
    Returns (detections, raw_string_if_failed).
    """
    text = resp.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result, None
    except (json.JSONDecodeError, ValueError):
        pass
    return [], resp


def main():
    parser = argparse.ArgumentParser(
        description="Convert swift infer JSONL to sample.json format"
    )
    parser.add_argument(
        "--input", required=True,
        help="Swift infer output JSONL (e.g. output/infer_results.jsonl)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON file in sample.json format",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    results = []
    n_parse_fail = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {lineno}: JSON parse error — {e}")
                continue

            name   = item.get("dataitem_name", "")
            images = item.get("images", [])
            image  = images[0] if images else ""

            # User prompt — take from original messages if present
            messages = item.get("messages", [])
            user_content = next(
                (m["content"] for m in messages if m.get("role") == "user"), ""
            )

            response = item.get("response", "")
            detections, raw = parse_response(response)

            # Normalize field names: bbox_2d -> bbox (for results.json compatibility)
            normalized = [
                {
                    "category": d.get("category", d.get("label", "")),
                    "size":     d.get("size", ""),
                    "bbox":     d.get("bbox", d.get("bbox_2d", [])),
                }
                for d in detections
            ]

            entry = {
                "dataitem_name": name,
                "result": normalized,
                "raw": response,
            }
            if raw is not None:
                n_parse_fail += 1
                n_parse_fail += 1

            results.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(results)} results to {output_path}")
    if n_parse_fail:
        logger.warning(
            f"{n_parse_fail}/{len(results)} responses could not be parsed as JSON"
            f" -- raw output preserved in 'raw' field"
        )


if __name__ == "__main__":
    main()
