# DPO pipeline (improvement E)

Preference fine-tuning on top of the SFT-trained model, using `metric.py`'s
`quality_sum` (0.5·IoU + 0.5·label_match) as the implicit preference signal.

## Pipeline overview

```
  SFT adapter (e.g., ckpt-600 or swa_adapter)
        |
        v
  [1] generate K predictions per train image with sampling
        |
        v
  [2] score each prediction vs ground truth -> quality_sum
        |
        v
  [3] form (chosen, rejected) pairs (highest- vs lowest-scoring sample)
        |
        v
  [4] swift rlhf --rlhf_type dpo --adapters <SFT adapter> --dataset dpo.jsonl
        |
        v
  DPO-tuned adapter
```

We split each multi-turn sample into TWO independent DPO records (one for
the views turn, one for the features turn). This is simpler than full
multi-turn DPO and lets us upweight whichever turn is weaker.

## Running it

All of these run on the training pod (the same one that has the SFT
adapter and the venv with cu13 ld path):

```bash
cd /workspace/scripts_27b_788_with_view_improve/dpo

# 1) sample K=4 predictions per train image with temperature
python generate_predictions.py \
    --adapter /workspace/output/swift_27b_788_view_7feats_improve/v0-*/swa_adapter \
    --train_jsonl /workspace/data/train_view_7feats_improve.jsonl \
    --image_dir /workspace/data/simens_7feats \
    --output /workspace/output/dpo_samples.jsonl \
    --k 4 --temperature 0.8

# 2) score each sample against ground truth
python score_predictions.py \
    --samples /workspace/output/dpo_samples.jsonl \
    --train_jsonl /workspace/data/train_view_7feats_improve.jsonl \
    --output /workspace/output/dpo_scored.jsonl

# 3) form (chosen, rejected) pairs into ms-swift DPO format
python make_dpo_jsonl.py \
    --scored /workspace/output/dpo_scored.jsonl \
    --train_jsonl /workspace/data/train_view_7feats_improve.jsonl \
    --output /workspace/data/dpo_view_7feats.jsonl \
    --min_score_gap 0.1

# 4) run DPO (a few hundred steps usually enough to shift behavior)
bash train_dpo.sh
```

## Tunables to know about
- `--k` (samples per image): 4 is a good start. More = better pair quality, more compute.
- `--temperature`: 0.8 gives diverse samples; lower (0.3-0.5) = less spread.
- `--min_score_gap` (in step 3): drop pairs where chosen and rejected are within X of
  each other. Forces DPO to learn from clear preferences. 0.1 is conservative.
- DPO `beta` in `train_dpo.sh`: 0.1 is the standard default for ms-swift; lower (0.05)
  gives stronger updates, higher (0.3) gives more conservative updates.

## What to expect
DPO on top of SFT typically gives **+1-3 F1** on detection-style tasks where
the metric is non-differentiable (we can't directly back-prop through IoU).
The biggest gains usually appear on:
- Fixing systematic FP categories (e.g., the Bending over-prediction we saw earlier)
- Tightening bbox quality on borderline cases (since chosen samples have higher IoU)
- Removing parse failures (since chosen samples are valid JSON more often)

## Caveats
- DPO on small datasets (~600 train samples × K=4 = ~2400 pairs) is borderline. Watch
  for overfitting; eval on held-out val set after every ~50 steps.
- The implicit reward is "metric quality", which doesn't penalize over-generation. May
  need to add a length penalty in step 3 if DPO model becomes verbose.
- For a stronger reward, consider running this in a loop: DPO → infer → re-score →
  form new pairs → DPO again (poor man's iterative RL).
