"""
swa.py — Stochastic Weight Averaging of LoRA checkpoints (improvement C).

After training finishes, this script averages the top-N LoRA adapters by
eval_loss into a new `swa_adapter/` directory next to the checkpoints.
Use that adapter at inference time. Often gives +1-2 F1 with no retraining.

This is a post-training step (no in-loop EMA needed — sidesteps ms-swift
trainer surgery). It only averages the LoRA `adapter_model.safetensors`
(the trainable weights), so it works regardless of base-model size.

Usage:
  python swa.py --output-dir /workspace/output/swift_27b_788_view_7feats_improve
  python swa.py --output-dir <dir> --top-n 3 --version-dir v0-...
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def latest_version_dir(output_dir: Path) -> Path:
    versions = [p for p in output_dir.glob("v*-*") if p.is_dir()]
    if not versions:
        raise FileNotFoundError(f"No v*-* version dir under {output_dir}")
    return max(versions, key=lambda p: p.stat().st_mtime)


def topn_by_eval_loss(version_dir: Path, n: int):
    """Return list of (ckpt_dir_name, eval_loss) for the top-N checkpoints
    (lowest eval_loss). Reads the latest checkpoint's trainer_state.json
    log_history (each ckpt's eval_loss is keyed by step)."""
    ckpts = sorted(
        version_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-* under {version_dir}")
    state = json.loads((ckpts[-1] / "trainer_state.json").read_text())
    step_to_loss = {
        int(e["step"]): float(e["eval_loss"])
        for e in state.get("log_history", [])
        if "eval_loss" in e
    }
    losses = []
    for c in ckpts:
        step = int(c.name.split("-")[-1])
        if step in step_to_loss:
            losses.append((c.name, step_to_loss[step]))
    losses.sort(key=lambda kv: kv[1])
    return losses[:n]


def average_adapters(version_dir: Path, ckpt_names, out_name="swa_adapter"):
    """Average adapter_model.safetensors across the listed checkpoints.
    Saves to <version_dir>/<out_name>/."""
    out_dir = version_dir / out_name
    out_dir.mkdir(exist_ok=True)

    # Use the first ckpt's adapter_config.json + tokenizer-related files as the template.
    template_ckpt = version_dir / ckpt_names[0]
    for fname in ("adapter_config.json", "additional_config.json", "README.md"):
        src = template_ckpt / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)

    # Load and average safetensors
    keys_template = None
    avg = None
    n = len(ckpt_names)
    for name in ckpt_names:
        path = version_dir / name / "adapter_model.safetensors"
        sd = load_file(str(path))
        if keys_template is None:
            keys_template = set(sd.keys())
            avg = {k: v.float() / n for k, v in sd.items()}
        else:
            if set(sd.keys()) != keys_template:
                raise RuntimeError(f"Adapter key mismatch between checkpoints: "
                                   f"{ckpt_names[0]} vs {name}")
            for k in avg:
                avg[k] = avg[k] + sd[k].float() / n

    # Cast back to original dtype (assume bf16 for our setup; sniff from any ckpt)
    sample_sd = load_file(str(version_dir / ckpt_names[0] / "adapter_model.safetensors"))
    sample_dtype = next(iter(sample_sd.values())).dtype
    avg = {k: v.to(sample_dtype) for k, v in avg.items()}
    save_file(avg, str(out_dir / "adapter_model.safetensors"))
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="SWA over top-N LoRA checkpoints by eval_loss")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Top-level output dir (e.g., /workspace/output/swift_27b_788_view_7feats_improve)")
    ap.add_argument("--version-dir", default=None,
                    help="Specific v*-* dir; defaults to most recently modified")
    ap.add_argument("--top-n", type=int, default=3,
                    help="How many top checkpoints (by eval_loss) to average")
    ap.add_argument("--out-name", default="swa_adapter",
                    help="Output subdir name under the version dir")
    args = ap.parse_args()

    if args.version_dir:
        version_dir = args.output_dir / args.version_dir
    else:
        version_dir = latest_version_dir(args.output_dir)
    print(f"version dir: {version_dir}")

    topn = topn_by_eval_loss(version_dir, args.top_n)
    print(f"top-{args.top_n} by eval_loss:")
    for name, loss in topn:
        print(f"  {name:<24s}  eval_loss={loss:.6f}")

    if not topn:
        raise RuntimeError("No checkpoints with eval_loss recorded; nothing to average.")

    out_dir = average_adapters(version_dir, [name for name, _ in topn], args.out_name)
    sample_dtype = next(iter(load_file(str(out_dir / "adapter_model.safetensors")).values())).dtype
    avg_size = (out_dir / "adapter_model.safetensors").stat().st_size / 1e6
    print(f"\nSWA adapter saved -> {out_dir}")
    print(f"  size: {avg_size:.1f} MB, dtype: {sample_dtype}")
    print(f"  inference: --adapter_path {out_dir}")


if __name__ == "__main__":
    main()
