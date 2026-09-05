# Inference Steps — Siemens 788 Pipeline (Nautilus)

## Pod

- YAML: `retrain model/k8s/nautilus/infer-27b-hr.yaml`
- Deploy: `kubectl apply -f k8s/nautilus/infer-27b-hr.yaml`
- Name: `wenfei-infer-27b-hr` (namespace: `nsf-maica`)
- GPUs: 2× NVIDIA L40 (46 GB each)

## What was on the pod already

| Path | Contents |
|---|---|
| `/workspace/venv/` | Python venv with ms-swift 4.1.3, transformers, peft |
| `/workspace/scripts_27b_788_with_view_improve/` | inference_swift.py, prepare_dataset_swift.py, train_swift.sh, etc. |
| `/workspace/data/simens_7feats/` | 1266 PNG images |
| `/workspace/data/test_view_7feats_improve.jsonl` | Test JSONL (5-message multi-turn format) |
| `/workspace/output/swift_27b_788_view_7feats_improve/v0-20260503-150530/swa_adapter/` | SWA adapter (best checkpoint) |

## Steps run on the pod

### 1. Shell into the pod
```bash
kubectl exec -it -n nsf-maica deploy/wenfei-infer-27b-hr -- bash
```

### 2. Activate venv
```bash
source /workspace/venv/bin/activate
```

### 3. Create 2-sample test file
```bash
head -2 /workspace/data/test_view_7feats_improve.jsonl > /workspace/output/test_2samples.jsonl
```

### 4. Run inference
```bash
CUDA_VISIBLE_DEVICES=0,1 \
python3 /workspace/scripts_27b_788_with_view_improve/inference_swift.py \
    --adapter_path /workspace/output/swift_27b_788_view_7feats_improve/v0-20260503-150530/swa_adapter \
    --test_jsonl /workspace/output/test_2samples.jsonl \
    --image_dir /workspace/data/simens_7feats \
    --output_file /workspace/output/test_inference_2samples.json \
    --max_new_tokens 3072
```

### 5. Full test set inference
```bash
CUDA_VISIBLE_DEVICES=0,1 \
python3 /workspace/scripts_27b_788_with_view_improve/inference_swift.py \
    --adapter_path /workspace/output/swift_27b_788_view_7feats_improve/v0-20260503-150530/swa_adapter \
    --test_jsonl /workspace/data/test_view_7feats_improve.jsonl \
    --image_dir /workspace/data/simens_7feats \
    --output_file /workspace/output/test_inference_full.json \
    --max_new_tokens 3072
```

### 6. Score results (if score_results_json.py available)
```bash
python3 score_results_json.py \
    --results /workspace/output/test_inference_full.json \
    --test_jsonl /workspace/data/test_view_7feats_improve.jsonl
```

## Key details

- **No `--load_in_4bit`**: bitsandbytes is not installed. Device map defaults to `"auto"`, splitting the 27B model across both L40 GPUs (~56 GB VRAM total).
- **flash_attn**: Installed but broken (CUDA symbol mismatch) → falls back to SDPA.
- **Adapter**: SWA (Stochastic Weight Averaging) of top-3 checkpoints from v0 run.
- **ms-swift version**: 4.1.3
