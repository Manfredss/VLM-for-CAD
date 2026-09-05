#!/usr/bin/env python3
"""Keep only the top-N checkpoints by eval_loss in the latest version dir.

Polls the output dir, reads each checkpoint's trainer_state.json, ranks by
eval_loss (lower is better), and removes checkpoints outside the top-N.

Guards against deleting the most-recently-modified checkpoint (in-progress
save) and the trainer's best_model_checkpoint (always #1 in our ranking
when metric_for_best_model=eval_loss, but defended against drift).
"""
import argparse
import json
import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger("janitor")


def latest_version_dir(output_dir: Path):
    versions = [p for p in output_dir.glob("v*-*") if p.is_dir()]
    if not versions:
        return None
    return max(versions, key=lambda p: p.stat().st_mtime)


def collect_ckpt_losses(version_dir: Path):
    ckpts = sorted(
        version_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not ckpts:
        return {}, None
    state_path = ckpts[-1] / "trainer_state.json"
    if not state_path.exists():
        return {}, None
    state = json.loads(state_path.read_text())
    step_to_loss = {
        int(e["step"]): float(e["eval_loss"])
        for e in state.get("log_history", [])
        if "eval_loss" in e
    }
    losses = {}
    for ckpt in ckpts:
        step = int(ckpt.name.split("-")[-1])
        if step in step_to_loss:
            losses[ckpt.name] = step_to_loss[step]
    bmc = state.get("best_model_checkpoint")
    bmc_name = Path(bmc).name if bmc else None
    return losses, bmc_name


def prune_once(output_dir: Path, keep_n: int, dry_run: bool):
    vdir = latest_version_dir(output_dir)
    if vdir is None:
        return
    losses, bmc = collect_ckpt_losses(vdir)
    if len(losses) <= keep_n:
        return
    ranked = sorted(losses.items(), key=lambda kv: kv[1])
    keep = {name for name, _ in ranked[:keep_n]}
    by_step = sorted(losses.keys(), key=lambda n: int(n.split("-")[-1]))
    keep.add(by_step[-1])
    if bmc:
        keep.add(bmc)
    for name, loss in losses.items():
        if name in keep:
            continue
        target = vdir / name
        logger.info(f"prune {target.name} eval_loss={loss:.6f}")
        if not dry_run:
            try:
                shutil.rmtree(target)
            except OSError as e:
                logger.warning(f"rmtree failed for {target}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--keep-n", type=int, default=5)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [janitor] %(levelname)s %(message)s",
    )
    logger.info(
        f"start dir={args.output_dir} keep_n={args.keep_n} "
        f"interval={args.interval}s dry_run={args.dry_run}"
    )
    while True:
        try:
            prune_once(args.output_dir, args.keep_n, args.dry_run)
        except Exception:
            logger.exception("prune_once failed")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
