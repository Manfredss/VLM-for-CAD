# Siemens 788 Pipeline — Full SFT → GRPO → Agentic RL

Multi-turn Qwen3.5-27B LoRA training for industrial drawing feature detection.
15 layout/view categories + 10 feature categories (incl. hole groups). Designed for
**Nautilus (Kubernetes)** with A100-80GB / H100 GPUs.

## Quick Start (Nautilus)

```bash
# 1. Launch training pod (3× A100-80GB)
kubectl apply -f k8s/nautilus/train-27b-3a100.yaml

# 2. Shell in
kubectl exec -it -n nsf-maica deploy/wenfeiqwen3-5-27b-3a100-a -- bash

# 3. On the pod: activate env, prepare data, run SFT
source /workspace/venv/bin/activate
cd /workspace/scripts_27b_788_pipeline

python prepare_dataset_swift.py \
    --image_dir /workspace/data/simens_7feats \
    --deploy_image_dir /workspace/data/simens_7feats

# Copy JSONLs to /workspace/data/
cp dataset/train_view_7feats_pipeline.jsonl /workspace/data/
cp dataset/val_view_7feats_pipeline.jsonl /workspace/data/
cp dataset/test_view_7feats_pipeline.jsonl /workspace/data/

# Train
TRAIN_CUDA_VISIBLE_DEVICES=0,1,2 bash train_sft.sh

# 4. SWA (post-training weight averaging)
python swa.py --output-dir /workspace/output/swift_27b_788_pipeline --top-n 3

# 5. Evaluate checkpoints (find best by CADScore)
python evaluate_checkpoints.py \
    --output-dir /workspace/output/swift_27b_788_pipeline \
    --val-jsonl /workspace/data/val_view_7feats_pipeline.jsonl \
    --image-dir /workspace/data/simens_7feats \
    --n-samples 50

# 6. Inference
python inference_swift.py \
    --adapter_path /workspace/output/swift_27b_788_pipeline/v0-*/swa_adapter \
    --test_jsonl /workspace/data/test_view_7feats_pipeline.jsonl \
    --image_dir /workspace/data/simens_7feats \
    --output_file /workspace/output/results_sft_swa.json
```

## Pipeline Stages

```
Stage 1 (SFT):        prepare_dataset → train_sft → swa → evaluate_checkpoints
Stage 2 (DPO):        generate → score → make_pairs → train_dpo          [optional]
Stage 3 (GRPO):       train_grpo                                          [optional]
Stage 4 (Agentic RL): agentic_rl_inference → train_agentic_grpo          [optional]
```

### Stage 1: SFT (always run first)

```bash
# Data prep with oversampling for tail classes
python prepare_dataset_swift.py

# Train (3 GPUs recommended)
TRAIN_CUDA_VISIBLE_DEVICES=0,1,2 bash train_sft.sh
```

Key hyperparameters (all overridable via env):

| Param | Default | Notes |
|---|---|---|
| `LORA_RANK` | 64 | LoRA rank |
| `LORA_ALPHA` | 128 | Usually 2× rank |
| `LEARNING_RATE` | 2e-5 | LLM LR |
| `VIT_LR` | 3e-5 | Vision tower LR (higher to learn drawing-specific features) |
| `ALIGNER_LR` | 2e-5 | Vision-language aligner LR |
| `IMAGE_MAX_TOKEN_NUM` | 4096 | Resolution budget |
| `MAX_LENGTH` | 12288 | Max sequence length |
| `NUM_TRAIN_EPOCHS` | 6 | Full passes |
| `PER_DEVICE_TRAIN_BATCH_SIZE` | 2 | Per GPU |
| `GRADIENT_ACCUMULATION_STEPS` | 2 | Effective batch = 2 × 2 × 3 = 12 |
| `DEEPSPEED_CONFIG` | zero3 | DeepSpeed stage |

### Stage 2: DPO (optional, improves quality further)

```bash
# Set SFT adapter
export SFT_ADAPTER=/workspace/output/swift_27b_788_pipeline/v0-*/swa_adapter

# Generate K=4 predictions per image with temperature
python dpo/generate_predictions.py \
    --adapter "${SFT_ADAPTER}" \
    --train_jsonl /workspace/data/train_view_7feats_pipeline.jsonl \
    --image_dir /workspace/data/simens_7feats \
    --output /workspace/output/dpo/samples.jsonl \
    --k 4 --temperature 0.8

# Score each prediction
python dpo/score_predictions.py \
    --samples /workspace/output/dpo/samples.jsonl \
    --train_jsonl /workspace/data/train_view_7feats_pipeline.jsonl \
    --output /workspace/output/dpo/scored.jsonl

# Form (chosen, rejected) pairs
python dpo/make_dpo_jsonl.py \
    --scored /workspace/output/dpo/scored.jsonl \
    --train_jsonl /workspace/data/train_view_7feats_pipeline.jsonl \
    --output /workspace/data/dpo_view_7feats_pipeline.jsonl

# Train DPO
TRAIN_CUDA_VISIBLE_DEVICES=0,1 ADAPTER="${SFT_ADAPTER}" \
DPO_DATASET=/workspace/data/dpo_view_7feats_pipeline.jsonl \
bash dpo/train_dpo.sh
```

### Stage 3: GRPO (reward-based RL)

Uses a composite reward function: format (10%) + class-aware category F1 (30%) +
bbox IoU (25%) + normalized CAD dimension similarity (20%) + completeness (10%) +
cross-view consistency (5%).

```bash
SFT_CHECKPOINT="${SFT_ADAPTER}" \
TRAIN_CUDA_VISIBLE_DEVICES=0,1 \
TRAIN_DATASET=/workspace/data/train_view_7feats_pipeline.jsonl \
bash train_grpo.sh
```

GRPO-specific params: `NUM_SAMPLES=4`, `TEMPERATURE=0.7`, `KL_COEFF=0.05`, `BETA=0.04`.

### Stage 4: Agentic RL (self-refinement, experimental)

The model learns to critique and correct its own mistakes through a 3-pass loop:

```
Pass 1: detect views → detect features
Pass 2: self-critique (review output against image, flag issues)
Pass 3: refinement (correct flagged issues, output final list)
```

Reward = base_score(refined) + 0.5 × improvement_delta.

```bash
# Generate agentic training data
CUDA_VISIBLE_DEVICES=0,1 python agentic_rl_inference.py \
    --adapter_path "${SFT_ADAPTER}" \
    --test_jsonl /workspace/data/train_view_7feats_pipeline.jsonl \
    --image_dir /workspace/data/simens_7feats \
    --output_file /workspace/output/agentic_train_results.json \
    --temperature 0.7

# Train
SFT_CHECKPOINT="${SFT_ADAPTER}" \
TRAIN_CUDA_VISIBLE_DEVICES=0,1 \
bash train_agentic_grpo.sh
```

Note: `MAX_LENGTH=16384` for agentic training (longer multi-turn context).
LR is lower (`2e-6`) with fewer epochs (`1`) to avoid catastrophic forgetting.

## Files

| File | Purpose |
|---|---|
| `train_sft.sh` | SFT training (merged best practices) |
| `train_grpo.sh` | GRPO training with composite reward |
| `train_agentic_grpo.sh` | Agentic GRPO with self-refinement reward |
| `run_pipeline.sh` | Full orchestrator (`bash run_pipeline.sh all`) |
| `prepare_dataset_swift.py` | JSON → multi-turn JSONL with oversampling |
| `inference_swift.py` | Multi-turn inference with category snapping |
| `optimizer.py` | Hierarchical LR (ViT/Aligner/LLM) |
| `metric.py` | CAD-aware eval during training |
| `reward.py` | Composite reward for GRPO (6 components) |
| `keep_best_n_ckpts.py` | Background checkpoint janitor |
| `swa.py` | Stochastic Weight Averaging (post-training) |
| `evaluate_checkpoints.py` | Generative CADScore eval to rank checkpoints |
| `score_results_json.py` | Post-hoc CAD-aware scoring of results |
| `cad_metrics.py` | Shared class-aware matching and dimension scoring |
| `agentic_rl_inference.py` | Multi-pass self-refinement inference |
| `rlhf_config.yaml` | Reward weights + RL hyperparameters |
| `788_11Feats_View.json` | Annotation data source |
| `dpo/` | DPO pipeline scripts |

## GPU Requirements

| Stage | Min GPUs | Recommended | GPU Type |
|---|---|---|---|
| SFT | 2 | 3 | A100-80GB / H100 |
| SWA / eval | 1 | 1 | Any with ≥16GB VRAM |
| DPO | 2 | 2 | A100-80GB |
| GRPO | 2 | 2 | A100-80GB / H100 |
| Agentic RL | 2 | 2 | A100-80GB / H100 |
| Inference | 1 | 2 (split) | A100 / L40 / RTX 6000 Ada |

For single-GPU inference, use `--load_in_4bit` to fit the 27B model in 24GB.
For 2-GPU split, set `INFERENCE_DEVICE_MAP=split`.

### Nautilus Pod Specs

Training pod (3× A100): `k8s/nautilus/train-27b-3a100.yaml`
```yaml
resources:
  limits:
    memory: 192Gi
    nvidia.com/gpu: 2    # or 3 for larger pods
```

Inference pod (2× GPU): `k8s/nautilus/infer-27b-hr.yaml`
```yaml
resources:
  limits:
    memory: 64Gi
    nvidia.com/gpu: 2
```

## Data Layout on Nautilus

```
/workspace/
├── data/
│   ├── simens_7feats/          # PNG images
│   ├── train_view_7feats_pipeline.jsonl
│   ├── val_view_7feats_pipeline.jsonl
│   └── test_view_7feats_pipeline.jsonl
├── scripts_27b_788_pipeline/   # This directory
├── output/
│   ├── swift_27b_788_pipeline/       # SFT checkpoints + SWA
│   ├── swift_27b_788_pipeline_grpo/  # GRPO output
│   └── swift_27b_788_pipeline_agentic_grpo/  # Agentic RL output
├── logs/                       # Training + janitor logs
├── venv/                       # Python venv with ms-swift + deps
└── .cache/
    ├── huggingface/
    └── triton/
```

## Scoring Results

After inference, score against ground truth:

```bash
python score_results_json.py \
    --results /workspace/output/results_sft_swa.json \
    --test_jsonl /workspace/data/test_view_7feats_pipeline.jsonl
```

Outputs: layout F1, feature F1, strict feature F1, mean IoU, normalized size
accuracy/similarity, per-class F1, combined F1, and CADScore.

## Reward Function Tuning

Edit `rlhf_config.yaml` to adjust reward weights and category bonuses.
Test changes locally:

```bash
python reward.py --pred '[{"category":"Round Hole","bbox":[100,200,150,250],"size":"Ø18"}]' \
    --gt '[{"category":"Round Hole","bbox":[100,200,150,250],"size":"Ø18"}]'

# Agentic mode: compare initial vs refined
python reward.py --mode agentic \
    --initial_pred '<initial_json>' --pred '<refined_json>' --gt '<gt_json>'
```

## Common Issues

**Training OOM**: Reduce `IMAGE_MAX_TOKEN_NUM` (2560 instead of 4096) or
`PER_DEVICE_TRAIN_BATCH_SIZE` (1 instead of 2).

**Checkpoint disk usage**: The janitor (`keep_best_n_ckpts.py`) runs in background
and keeps only the top-N checkpoints by eval loss. Adjust `KEEP_BEST_N` (default 5).

**Resume training**: Set `RESUME_FROM_CHECKPOINT=/path/to/checkpoint-N` and
optionally `RESUME_ONLY_MODEL=true` to load only weights (skip optimizer state).

**ms-swift not found**: Ensure `/workspace/venv/bin` is first on PATH.
The training scripts do this automatically if the venv exists.
