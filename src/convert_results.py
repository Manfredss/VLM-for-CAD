"""Convert infer_results_v7.jsonl to results.json format."""
import json
import re
import sys

input_path = sys.argv[1] if len(sys.argv) > 1 else "output/infer_results_v7.jsonl"
output_path = sys.argv[2] if len(sys.argv) > 2 else "output/infer_results_v7.json"

results = []
with open(input_path, "r") as f:
    data = json.load(f)

for item in data:

    # Parse the response: extract JSON from markdown code block
    response = item["response"]
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", response, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = response.strip()

    try:
        detections = json.loads(json_str)
    except json.JSONDecodeError:
        detections = []

    # Rename bbox_2d -> bbox in each detection
    for det in detections:
        if "bbox_2d" in det:
            det["bbox"] = det.pop("bbox_2d")

    # Build raw string (compact, matching results.json format)
    raw = json.dumps(detections, ensure_ascii=False, separators=(", ", ": "))

    results.append({
        "dataitem_name": item["dataitem_name"],
        "result": detections,
        "raw": raw,
    })

with open(output_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Converted {len(results)} items -> {output_path}")
