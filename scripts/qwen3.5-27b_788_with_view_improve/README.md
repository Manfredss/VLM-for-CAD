# 788_with_view — improved variant (A + B + C + E)

Successor to `scripts/qwen3.5-27b_788_with_view`. Same task (multi-turn views + 7 features),
same data source. Adds four orthogonal improvements:

- **A. Data:** class-balanced oversampling for tail labels in `prepare_dataset_swift.py`.
- **B. Prompt + decoding:** sharper STEP1 rules for the weak categories (Notes bbox, Rear View,
  rare views), plus post-hoc category snapping in `inference_swift.py` (closest-string match
  if the model emits an out-of-vocab category — cheap constrained-decoding substitute that
  doesn't require touching the generation loop).
- **C. Training-procedure:** **Stochastic Weight Averaging (SWA)** of the top-N LoRA
  checkpoints (`swa.py`). Simpler than in-loop EMA, doesn't require modifying ms-swift's
  trainer, applied as a post-training step. Empirically gives +1-2 F1 on the kind of LoRA
  training that has noisy late-epoch loss.
- **E. RL / preference:** end-to-end DPO scaffolding under `dpo/`. Uses the existing
  `metric.py` quality score as the preference signal — generate predictions from two
  checkpoints, score each, form (chosen, rejected) pairs, run `swift rlhf` with `--rlhf_type dpo`.

## Files
- `prepare_dataset_swift.py` — adds `OVERSAMPLE_MAP` for tail classes; refines STEP1 prompt.
- `inference_swift.py` — same prompts as `prepare`; adds `--snap_categories` flag (default on)
  for closest-match category recovery.
- `train_swift.sh` — points at this folder's scripts and a separate output dir
  (`/workspace/output/swift_27b_788_view_7feats_improve`); same hyperparams as the
  3A100 retry (image=2560, max_length=10240, LoRA rank 64, etc.).
- `swa.py` — averages the top-N adapters by `eval_loss` into a new `swa_adapter/` directory
  next to the checkpoints. Use that adapter at inference time.
- `dpo/` — DPO pipeline (see `dpo/README.md`).

## Suggested workflow
1. Generate JSONLs with oversampling: `python prepare_dataset_swift.py`
2. Upload them and the scripts to the pod, run `bash train_swift.sh`.
3. After training, run `python swa.py --output-dir <output_dir>` to produce the averaged adapter.
4. Inference: point `--adapter_path` at either the best ckpt or the SWA adapter; with
   `--snap_categories` on by default.
5. Optionally run the DPO pipeline (`dpo/README.md`) on top of the SFT model for further gains.
