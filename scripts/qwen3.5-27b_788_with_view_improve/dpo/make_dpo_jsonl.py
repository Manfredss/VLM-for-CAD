"""
make_dpo_jsonl.py — Step 3 of the DPO pipeline.

Group scored samples by image, pick (highest_quality, lowest_quality) per image,
form pairs, and write ms-swift DPO JSONL format.

Two records per image (skipping pairs where the score gap < threshold):
  - one for the views turn (single-turn DPO on step1)
  - one for the features turn (single-turn DPO on step2, with chosen step1 in context)

ms-swift DPO format (single-turn variant):
  {
    "messages": [{"role":"system","content":...},
                 {"role":"user","content":...},
                 {"role":"assistant","content":<chosen>}],
    "rejected_response": "<rejected>",
    "images": [...]   # only on the views record
  }
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="from score_predictions.py")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min_score_gap", type=float, default=0.05,
                    help="Skip pairs where chosen-rejected gap < this (per turn)")
    ap.add_argument("--target", choices=["views", "features", "both"], default="both")
    args = ap.parse_args()

    # Load train JSONL keyed by image to get the prompts
    train_by_img = {}
    with open(args.train_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            img = Path(rec["images"][0]).name
            train_by_img[img] = rec

    # Group scored samples by image
    samples_by_img = defaultdict(list)
    with open(args.scored) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            samples_by_img[rec["image"]].append(rec)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_views = 0
    n_feats = 0
    n_skipped = 0

    with open(out_path, "w") as out:
        for img, samples in samples_by_img.items():
            if img not in train_by_img:
                continue
            tr = train_by_img[img]
            sys_msg = tr["messages"][0]
            user1 = tr["messages"][1]
            asst1_gt = tr["messages"][2]
            user2 = tr["messages"][3]

            # Views pair
            if args.target in ("views", "both"):
                samples.sort(key=lambda s: s.get("quality_views", 0.0))
                lo, hi = samples[0], samples[-1]
                if hi["quality_views"] - lo["quality_views"] >= args.min_score_gap and hi.get("raw_views") and lo.get("raw_views"):
                    rec = {
                        "messages": [
                            sys_msg,
                            user1,
                            {"role": "assistant", "content": hi["raw_views"]},
                        ],
                        "rejected_response": lo["raw_views"],
                        "images": tr["images"],
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_views += 1
                else:
                    n_skipped += 1

            # Features pair (single-turn DPO conditioned on the GT views as turn-1)
            if args.target in ("features", "both"):
                samples.sort(key=lambda s: s.get("quality_features", 0.0))
                lo, hi = samples[0], samples[-1]
                if hi["quality_features"] - lo["quality_features"] >= args.min_score_gap and hi.get("raw_features") and lo.get("raw_features"):
                    rec = {
                        "messages": [
                            sys_msg,
                            user1,
                            asst1_gt,
                            user2,
                            {"role": "assistant", "content": hi["raw_features"]},
                        ],
                        "rejected_response": lo["raw_features"],
                        "images": tr["images"],
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_feats += 1
                else:
                    n_skipped += 1

    print(f"DPO pairs written -> {out_path}")
    print(f"  views pairs:    {n_views}")
    print(f"  features pairs: {n_feats}")
    print(f"  skipped (gap<{args.min_score_gap}): {n_skipped}")


if __name__ == "__main__":
    main()
