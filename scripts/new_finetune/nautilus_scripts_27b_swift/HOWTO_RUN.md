# How to Run v16 Fine-tuning (Qwen3.5-27B on Industrial Drawings)

## What You Need

### Code
Just this folder: `nautilus_scripts_27b_swift/`

### Data files (get these separately)
```
train_augmented.jsonl       # training set (~6450 samples, swift JSONL format)
val_swift4.jsonl            # validation set (~614 samples)
benchmarkdata_sampled50.jsonl  # benchmark eval set (49 images)
```

### Images
A directory of `.png` images referenced by the JSONL files.
On the pod these should live at `/workspace/data/5k/` or `/workspace/data/images/`.

### Model
`Qwen/Qwen3.5-27B` — downloaded automatically from HuggingFace during training.
Make sure the pod has internet access or the model is already cached at
`/workspace/.cache/huggingface/`.

---

## Folder Structure (what goes where on the pod)

```
/workspace/
├── scripts_27b_swift/          ← upload this entire folder here
│   ├── run_v16.sh              ← ENTRY POINT — edit this to change hyperparams
│   ├── run_pipeline.sh         ← main pipeline (deps, GPU check, train)
│   ├── train_swift.sh          ← ms-swift sft launch command
│   ├── metric.py               ← custom P/R/F1 eval metric (required)
│   ├── optimizer.py            ← layered-LR optimizer (required)
│   ├── watch_checkpoints_and_benchmark.sh  ← benchmark watcher
│   ├── evaluate_checkpoints_benchmark.py   ← per-checkpoint evaluation
│   ├── inference_swift.py      ← batch inference
│   └── fix_data_paths.py       ← auto-fixes Mac paths in JSONL
│
├── data/
│   ├── train_augmented.jsonl
│   ├── val_swift4.jsonl
│   ├── benchmarkdata_sampled50.jsonl
│   └── 5k/                     ← image directory (or symlink)
│       ├── image1.png
│       └── ...
│
├── output/swift_27b/           ← checkpoints saved here (auto-created)
├── logs/                       ← training logs (auto-created)
└── .cache/huggingface/         ← model cache (auto-created)
```

---

## Step-by-Step: Running on a Kubernetes Pod

### Step 1 — Upload scripts to the pod

```bash
POD_NAME="your-pod-name"
NS="your-namespace"

kubectl cp nautilus_scripts_27b_swift/ $NS/$POD_NAME:/workspace/scripts_27b_swift/
```

### Step 2 — Upload data

```bash
kubectl cp train_augmented.jsonl $NS/$POD_NAME:/workspace/data/train_augmented.jsonl
kubectl cp val_swift4.jsonl $NS/$POD_NAME:/workspace/data/val_swift4.jsonl
kubectl cp benchmarkdata_sampled50.jsonl $NS/$POD_NAME:/workspace/data/benchmarkdata_sampled50.jsonl
```

Upload your image directory (can be slow for large datasets):
```bash
kubectl cp /your/local/images/ $NS/$POD_NAME:/workspace/data/5k/
```

### Step 3 — Edit `run_v16.sh` to match your paths

Open `/workspace/scripts_27b_swift/run_v16.sh` on the pod and confirm:
```bash
export TRAIN_FILE=/workspace/data/train_augmented.jsonl
export VAL_FILE=/workspace/data/val_swift4.jsonl
export BENCHMARK_FILE=/workspace/data/benchmarkdata_sampled50.jsonl
export TRAIN_CUDA_VISIBLE_DEVICES=0,1   # adjust to your GPU count
```

For a single GPU, set:
```bash
export TRAIN_CUDA_VISIBLE_DEVICES=0
export GRADIENT_ACCUMULATION_STEPS=6   # keep effective batch size = 12
```

### Step 4 — Start training

```bash
kubectl exec -n $NS $POD_NAME -- bash -lc '
  mkdir -p /workspace/logs
  nohup bash /workspace/scripts_27b_swift/run_v16.sh \
    > /workspace/logs/swift_27b_v16.runner.log 2>&1 &
  echo "Launched PID $!"
'
```

### Step 5 — Monitor

```bash
# Training progress (loss, step, ETA)
kubectl exec -n $NS $POD_NAME -- tail -f /workspace/logs/train_swift_v16.log

# Pipeline log (dependency install, GPU detection)
kubectl exec -n $NS $POD_NAME -- tail -f /workspace/logs/train_swift_v16_runner.log

# GPU utilization
kubectl exec -n $NS $POD_NAME -- nvidia-smi
```

### Step 6 — (Optional) Run benchmark watcher on a separate pod

If you have a second pod with a free GPU:
```bash
kubectl exec -n $NS $BENCH_POD -- bash -lc '
  nohup bash -c '"'"'
    export CUDA_VISIBLE_DEVICES=0
    export OUTPUT_DIR=/workspace/output/swift_27b
    export MODEL_PATH=Qwen/Qwen3.5-27B
    export VAL_DATASET=/workspace/data/benchmarkdata_sampled50.jsonl
    export IMAGE_DIR=/workspace/data/5k
    export BENCHMARK_POLL_INTERVAL=300
    export RESULTS_JSONL=/workspace/output/swift_27b/benchmark_results_v16.jsonl
    bash /workspace/scripts_27b_swift/watch_checkpoints_and_benchmark.sh
  '"'"' > /workspace/logs/train_swift_v16_benchmark.log 2>&1 &
  echo "Benchmark watcher PID $!"
'
```

Results will print as each checkpoint is evaluated:
```
1. v23-.../checkpoint-400 | P=0.85 R=0.83 F1=0.84 AQ=0.91
```

### Step 7 — Run inference after training

```bash
kubectl exec -n $NS $POD_NAME -- bash -lc '
  python3 /workspace/scripts_27b_swift/inference_swift.py \
    --adapter_path /workspace/output/swift_27b/<run_tag>/final_adapter \
    --image_dir /workspace/data/5k \
    --output_file /workspace/output/inference_results_v16.json
'
```

---

## Key Hyperparameters (v16)

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3.5-27B |
| LoRA rank / alpha | **32 / 128** |
| target_modules | all-linear |
| LLM LR | 2e-5, cosine |
| ViT LR | 1e-5 |
| Aligner LR | 2e-5 |
| Batch size | 2/GPU × 2 GPUs × 3 grad_accum = **12 effective** |
| Epochs | 3 |
| Image tokens | 1560 |
| max_length | 3860 |
| Thinking mode | Disabled |

All of these are set in `run_v16.sh` and can be changed there.

---

## Output Format

Model outputs a JSON array, one object per detected feature:
```json
[
  {"category": "Threaded Hole", "size": "M6", "bbox": [100, 200, 300, 400]},
  {"category": "Round Hole",    "size": "ø10", "bbox": [500, 600, 700, 800]}
]
```

`bbox` values are normalized to **0–1000** (not pixels).

Categories: `Threaded Hole`, `Round Hole`, `Fillet`, `Slotted Hole`, `Rectangular Hole` (and `* Group` variants).

---

## Common Issues

| Problem | Fix |
|---------|-----|
| `val_augmented.jsonl` crash | Use `val_swift4.jsonl` instead — val_augmented is in old FSDP format |
| Flash Attention install fails | Normal on CUDA mismatch; pipeline auto-falls back to SDPA |
| `cosine_with_min_lr` + DeepSpeed error | Use `cosine` scheduler (already set) |
| ms-swift not found | Install from git: `pip install "ms-swift @ git+https://github.com/modelscope/ms-swift.git@main"` |
| transformers version error | Need `>=5.2.0`: `pip install "transformers>=5.2.0"` |
