#!/usr/bin/env python3
"""Run inference on sampled 100 images. Saves every 5 images."""

import argparse
import json
import time
from pathlib import Path

from inference_swift import load_model, run_inference_single, parse_output, _deduplicate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--gt_json", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--base_model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    args = parser.parse_args()

    print(f"Loading ground truth: {args.gt_json}")
    with open(args.gt_json) as f:
        gt_data = json.load(f)
    print(f"  {len(gt_data)} samples")

    records = []
    for item in gt_data:
        img_name = item["dataitem_name"]
        img_path = Path(args.image_dir) / img_name
        if img_path.exists():
            records.append({
                "dataitem_name": img_name,
                "image_path": str(img_path),
                "ground_truth": item,
            })
    print(f"  {len(records)} images found")

    print("Loading model...")
    model, processor = load_model(args.adapter_path, args.base_model)

    results = []
    t0 = time.time()

    for i, rec in enumerate(records, 1):
        print(f"[{i}/{len(records)}] {rec['dataitem_name']}...", end=" ", flush=True)
        try:
            raw = run_inference_single(model, processor, rec["image_path"],
                                       args.max_new_tokens)
            features = parse_output(raw)
            features = _deduplicate(features)
            print(f"{len(features)} features")
        except Exception as e:
            print(f"ERROR: {e}")
            raw = ""
            features = []

        results.append({
            "dataitem_name": rec["dataitem_name"],
            "result": features,
            "raw_output": raw,
        })

        if i % 5 == 0 or i == len(records):
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f'  [save] {i}/{len(records)} written')

    elapsed = time.time() - t0
    print(f"\nDone! {len(results)} results in {elapsed:.0f}s ({elapsed/len(results):.1f}s/img)")


if __name__ == "__main__":
    main()
