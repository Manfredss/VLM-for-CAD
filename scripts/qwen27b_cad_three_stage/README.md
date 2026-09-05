# Qwen 27B：二维 CAD 三阶段微调流程

这套流程用于把 5K-10 特征、5K-15 layout/view 和 Siemens 788 等来源合并为一个可审计的数据集，并训练一个模型完成全部 19 种结构特征的提取。默认基座是 `Qwen/Qwen3.6-27B`；如果现有训练镜像尚未支持 Qwen3.6，可通过 `MODEL=Qwen/Qwen3.5-27B` 回退。

核心设计不是把旧版“两步 Prompt”继续加长，而是把不同视觉尺度的任务明确拆开：

1. Step 1 只定位通用视图区和文档区，不判断 Front/Top 等方向。
2. Step 2 在整图关系上判断投影法，并对每个 `View Region` 分类。
3. Step 3 在整图或高分辨率视图裁剪中穷尽提取指定范围的结构特征。

生产候选建议使用 `hybrid`：整图检测负责跨视图和大范围特征，视图裁剪负责圆孔、腰孔、倒角、圆角等小目标，最后做确定性坐标映射和去重。

## 文件说明

- `cad_schema.py`：唯一的类别、显式 alias、JSON Schema、Prompt 和坐标变换定义。
- `build_merged_dataset.py`：合并各来源、按零件族切分、生成三阶段对话与视图裁剪。
- `audit_dataset.py`：检查跨 split 泄漏、图片可访问性、schema、scope 和类别分布。
- `preflight.py`：离线检查 Qwen3.6 依赖、CUDA 数量与 bf16 能力，不自动安装软件。
- `metrics.py`：三阶段匹配、scope 感知、尺寸与跨阶段约束指标的统一实现。
- `optimizer.py`：兼容 ms-swift 4.1 与旧注册 API 的 ViT/Aligner/LLM 分层学习率。
- `train_27b.sh`：Phase A/Phase B 两阶段 LoRA SFT。
- `infer_three_stage.py`：三阶段 full/crops/hybrid 推理、断点续跑和 JSONL 原子保存。
- `evaluate_predictions.py`：按阶段、类别和整图完全匹配口径评测。
- `run_pipeline.sh`：`prepare/audit/preflight/train/infer/eval/all` 统一入口。

本流程不默认运行 SWA、DPO、GRPO 或其他 RL。当前数据规模下，先把数据边界、真实生成评测和多尺度推理做好，收益更稳定。

## 统一输出协议

Step 1：

```json
{
  "regions": [
    {"region_id": "r001", "category": "View Region", "bbox": [80, 90, 620, 700]},
    {"region_id": "d001", "category": "Title Block", "bbox": [700, 760, 990, 990]}
  ]
}
```

Step 2：

```json
{
  "projection_method": "first_angle",
  "views": [
    {"region_id": "r001", "category": "Orthographic Projection - Front View"}
  ]
}
```

Step 3：

```json
{
  "feature_scope": ["Round Hole", "Slotted Hole", "Chamfer"],
  "features": [
    {"region_id": "r001", "category": "Round Hole", "size": "Ø8", "bbox": [120, 140, 190, 210]}
  ]
}
```

`feature_scope` 表示该样本中“标注是穷尽的类别集合”。它不是推理模式字段。某来源没有标注 Slotted Hole 时，不能把该来源中未出现的 Slotted Hole 当成负样本；构建器会用 scope 避免这种错误监督。

全部结构类别为：Threaded Hole、Fillet、Round Hole、Pin Hole、Chamfer、Counterbore Hole、Rectangular Hole、Slotted Hole 及对应 Group，再加 Threaded Shaft、Bending、Silver Plating，共 19 类。

## 1. 环境与模型

Qwen3.6 的建议训练镜像至少包含：

- `ms-swift >= 4.1.3`
- `transformers >= 5.0.0.dev0`
- `qwen-vl-utils >= 0.0.14`
- `decord`
- `peft`、`deepspeed`、`torch`、`Pillow`、`packaging`

示例：

```bash
python -m pip install -U \
  'ms-swift>=4.1.3' \
  'transformers>=5.0.0.dev0' \
  'qwen-vl-utils>=0.0.14' \
  decord peft deepspeed Pillow packaging
```

Qwen3.6 官方 Transformers 加载入口是 `AutoModelForMultimodalLM`。推理脚本优先使用该类，并只为 Qwen3.5 兼容场景回退到旧的多模态 Auto 类。Qwen3.6 默认会 thinking；本流程训练和推理均显式设置 `enable_thinking=False`，防止思维文本污染严格 JSON。

准备数据后运行严格检查：

```bash
EXPECTED_GPUS=3 bash run_pipeline.sh preflight
```

如果使用已经验证过的 Qwen3.5 镜像：

```bash
MODEL=Qwen/Qwen3.5-27B bash run_pipeline.sh preflight
```

不要为了满足版本号盲目升级一个已经工作的 Qwen3.5 训练环境；应为 3.6 单独固化镜像和依赖 lock。

## 2. 合并数据

必须同时区分“本机源图片路径”和“GPU 训练容器看到的部署路径”。下面是完整命令模板：

```bash
python build_merged_dataset.py \
  --json-10 /data/labels/5k_10feats.json \
  --json-15 /data/labels/5k_15feats_with_view.json \
  --json-788 /data/labels/788_11Feats_View.json \
  --image-dir-10 /data/images/IM_D03_PT_5k_Augmented \
  --image-dir-15 /data/images/IM_D03_PT_5K \
  --image-dir-788 /data/images/siemens_788 \
  --deploy-dir-10 /workspace/data/IM_D03_PT_5k_Augmented \
  --deploy-dir-15 /workspace/data/IM_D03_PT_5K \
  --deploy-dir-788 /workspace/data/siemens_788 \
  --projection-10 unknown \
  --projection-15 unknown \
  --projection-788 first_angle \
  --output-dir ./dataset \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
  --seed 42 \
  --strict-missing-images \
  --materialize-crops \
  --crop-padding 0.12 \
  --max-empty-crops-per-image 1 \
  --crop-dir ./dataset/view_crops \
  --crop-deploy-dir /workspace/data/cad_view_crops
```

未显式传 `--scope-*` 时，构建器使用三种来源各自经过审计的默认 scope；不要统一传 `auto`，否则可能把来源未标注的类别加入负样本空间。只有在人工核对标注协议后，才用逗号分隔的明确类别覆盖某个 source scope。

当前 Siemens 788 来源按已核验的数据约定使用 `first_angle`，两套 5K 默认 `unknown`。对新来源，只有工程标准或标题栏信息已经人工确认投影法，才改为 `first_angle` 或 `third_angle`；不确定时必须保留 `unknown`，不要把历史 Prompt 中固定的投影法继续传播到其他数据。

数据切分以 `family_id` 为单位，旋转/翻转/增强版本只能进入原图所在的 train split；增强只在切分后用于 train。family 规则覆盖 `rot5/rot45/rot355` 等任意数值角度，并去除 Siemens 文件名前不稳定的三位数据集行号（例如 `226_A7...`/`227_A7...`），防止近重复图跨 split。构建器还会合并同一基础图片的互补 annotation scope，避免“未标注类别被解释为不存在”。

主要产物：

- `merged_annotations.json`、`manifest.jsonl`、`stats.json`
- `train_three_stage.jsonl`、`val_three_stage.jsonl`、`test_three_stage.jsonl`
- `train_ground_truth.jsonl`、`val_ground_truth.jsonl`、`test_ground_truth.jsonl`
- `train_view_crops.jsonl`、`crop_manifest.jsonl`

`train_three_stage.jsonl` 的完整整图样本是 Step1 → Step2 → Step3 连续多轮对话；没有 view 标注或增强派生样本可以是 Step3-only。ms-swift 文件只含 `messages/images`，审计 metadata 与规范化标签位于对应的 `*_ground_truth.jsonl` 和 manifest 中。`train_view_crops.jsonl` 是独立的高分辨率 Step3 样本，GT bbox 是裁剪局部 0–1000 坐标。

默认每张整图最多保留 1 个“无特征”负 crop，所有有正目标的 crop 都保留，避免空视图淹没训练。可通过 `--max-empty-crops-per-image N` 调整，`-1` 才表示保留全部空 crop。若使用 `--no-materialize-crops`，实际训练文件 `train_view_crops.jsonl` 会保持为空，计划写入 `train_view_crops_plan.jsonl`，因此 Phase B 也必须关闭 crop 数据。

也可以用统一入口。路径与容器部署路径全部由环境变量覆盖：

```bash
JSON_10=/data/labels/5k_10feats.json \
JSON_15=/data/labels/5k_15feats_with_view.json \
JSON_788=/data/labels/788.json \
IMAGE_DIR_10=/data/images/IM_D03_PT_5k_Augmented \
IMAGE_DIR_15=/data/images/IM_D03_PT_5K \
IMAGE_DIR_788=/data/images/788 \
DEPLOY_DIR_10=/workspace/data/IM_D03_PT_5k_Augmented \
DEPLOY_DIR_15=/workspace/data/IM_D03_PT_5K \
DEPLOY_DIR_788=/workspace/data/788 \
CROP_DEPLOY_DIR=/workspace/data/cad_view_crops \
bash run_pipeline.sh prepare
```

`run_pipeline.sh` 默认以 `IMAGE_WORKERS=8` 并发读取图片尺寸，并用
`CROP_WORKERS=8` 并发生成确定性视图裁剪，适合 Nautilus CephFS；本地磁盘或
I/O 受限环境可将两者设为 `1`。并发结果按输入顺序回收，不会改变 family split、
manifest 或训练 JSONL。

训练 JSONL 只保留 ms-swift 需要的 `messages/images`；`family_id/sources/annotation_scope/stage1/stage2/stage3` 等审计字段保存在 `*_ground_truth.jsonl` 和 manifest，不会触发 ms-swift 的未知字段兼容问题。训练前仍应在目标 ms-swift 4.1.3 镜像中运行 preflight。

## 3. 数据审计

```bash
python audit_dataset.py \
  --train ./dataset/train_ground_truth.jsonl \
  --val ./dataset/val_ground_truth.jsonl \
  --test ./dataset/test_ground_truth.jsonl \
  --strict \
  --require-images \
  --json-out ./dataset/audit_report.json
```

或者：

```bash
bash run_pipeline.sh audit
```

严格审计失败时不要开始训练。至少要处理：family/近重复跨 split、图片不存在、非法 bbox、未知类别、feature 超出 scope、重复 region_id、crop 坐标或映射错误。test 在最终验收前应锁定，不能用于 checkpoint 选择。

## 4. 两阶段 27B LoRA SFT

默认 LoRA 配置为 rank 32、alpha 64、dropout 0.05。完成 `--materialize-crops` 后设置 `USE_CROP_DATA=true`，训练才会同时读取整图三阶段对话与高分辨率 crop Step3：

```bash
MODEL=Qwen/Qwen3.6-27B \
DATA_DIR=$PWD/dataset \
PHASE_A_USE_CROP_DATA=false \
PHASE_B_USE_CROP_DATA=true \
CUDA_VISIBLE_DEVICES=0,1,2 \
bash train_27b.sh both
```

Phase A 的目标是先学会稳定协议和类别边界：

- `freeze_vit=true`
- 只读取整图连续多轮，`PHASE_A_USE_CROP_DATA=false`
- LLM LR `1e-5`
- Aligner LR `5e-6`
- 默认 1 epoch

```bash
bash train_27b.sh phase_a
```

Phase B 从 Phase A adapter 继续，用很低学习率打开视觉 LoRA：

- `freeze_vit=false`
- 加入高分辨率视图裁剪，`PHASE_B_USE_CROP_DATA=true`
- ViT LR `1.5e-6`
- Aligner LR `4e-6`
- LLM LR `6e-6`
- 默认 1.5 epoch

```bash
PHASE_A_ADAPTER=/output/phase_a/checkpoint-XXX \
PHASE_B_USE_CROP_DATA=true \
bash train_27b.sh phase_b
```

该 curriculum 防止数量更多的 crop Step3 样本在早期淹没 Step1/Step2 协议学习。若尚未执行 `--materialize-crops`，必须设置 `PHASE_B_USE_CROP_DATA=false`；`run_pipeline.sh` 会根据 `MATERIALIZE_CROPS` 自动设置。`USE_CROP_DATA=true/false` 仍可作为同时覆盖两个阶段的兼容开关，但标准流程不建议使用。默认最大文本长度为 16384，整图图片 token 预算为 4096。显存不足时先减小 per-device batch 并增大梯度累积，不要先把图片压缩到看不见小孔标注。

Nautilus A100 任务应在 CPU Pod 完成数据审计和依赖检查，上卡后直接训练，避免把昂贵的 import/preflight 放在 GPU Pod。`gpu_guarded_train.sh` 会每 30 秒读取显存占用和计算利用率；启动宽限期后，如果两项都连续低于阈值，脚本以状态 42 退出，让 Kubernetes 立即释放 GPU。单卡 80 GB 的保守启动配置如下；视图 crop 仍保留局部细节，因此优先把整图图片 token 从 3072 降到 2048：

```bash
RUN_PREFLIGHT=false \
SKIP_SWIFT_HELP=true \
USE_LOGITS_TO_KEEP=true \
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
MAX_LENGTH=8192 \
IMAGE_MAX_TOKEN_NUM=2048 \
GRADIENT_ACCUMULATION_STEPS=8 \
GPU_GUARD_MIN_PERCENT=50 \
GPU_GUARD_GRACE_SECONDS=90 \
GPU_GUARD_LOW_SECONDS=120 \
bash gpu_guarded_train.sh both
```

`RUN_PREFLIGHT=false` 仅用于已经在相同镜像、相同依赖和同一份数据上通过静态及 GPU preflight 的正式 Job；首次部署不能跳过验证。守卫把“显存占用或计算利用率至少一项达到阈值”视为有效负载，不会通过无意义的显存占位来伪造利用率。

断点恢复：

```bash
RESUME_FROM_CHECKPOINT_A=/output/phase_a/checkpoint-XXX bash train_27b.sh phase_a
RESUME_FROM_CHECKPOINT_B=/output/phase_b/checkpoint-YYY bash train_27b.sh phase_b
```

训练 loss 只用于早停参考。生产 checkpoint 必须在固定 validation 子集上执行真实 greedy generation 后按 Round Hole、Slotted Hole、View、macro-F1、JSON 合法率和约束违规率共同选择。不要默认对 adapter 做 SWA，也不要在 SFT 尚未稳定时加入 DPO/GRPO。

## 5. 三阶段推理

先在少量 validation 图片上做 smoke test：

```bash
python infer_three_stage.py \
  --model-path Qwen/Qwen3.6-27B \
  --adapter-path /output/phase_b/checkpoint-XXX \
  --input-jsonl ./dataset/val_three_stage.jsonl \
  --output-file ./output/val_predictions.jsonl \
  --feature-mode hybrid \
  --limit 10
```

三种 Step3 模式：

- `full`：只在整图上提取，速度较快，作为对照基线。
- `crops`：只对 Step2 视图区裁剪提取，小特征更清楚；局部 bbox 自动映射回整图。
- `hybrid`：整图与 crops 都运行，按“类别 + IoU”识别同一实例并优先保留 crop 结果，推荐生产候选。即使 full/crop OCR 尺寸不一致也不会重复报同一个孔；冲突会写入 diagnostics，便于复核。

P2 两步协议把布局定位、投影法和视图语义合并为第一次生成，第二次生成全部特征；最终仍写出标准 `step1/step2/step3/result`，因此可直接复用现有评测：

```bash
python infer_adapters.py \
  --model-path Qwen/Qwen3.6-27B \
  --adapter ckpt3000=/output/phase_b/checkpoint-3000 \
  --input-jsonl ./dataset/val_three_stage.jsonl \
  --output-dir ./output/p2 \
  --protocol-mode two_stage \
  --feature-mode full \
  --image-max-tokens 2048 \
  --save-raw
```

多卡吞吐评测采用独立单卡分片，每张卡加载完整 27B，避免 80GB 卡切分后单卡显存低于 50%。各 worker 设置相同 `--num-shards N` 和不同 `--shard-index 0..N-1`；完成后用 `merge_sharded_predictions.py` 合并并检查期望行数。不要让多个 worker 写同一个未带 shard 后缀的文件。

如果项目标准或图纸元数据已经确认投影法，可以覆盖模型判断：

```bash
python infer_three_stage.py \
  --model-path Qwen/Qwen3.6-27B \
  --adapter-path /output/phase_b/checkpoint-XXX \
  --image-dir /workspace/incoming_drawings \
  --output-file ./output/predictions.jsonl \
  --feature-mode hybrid \
  --projection-method first_angle
```

只有验证过时才使用该覆盖；默认 `auto` 会保留模型的 `first_angle/third_angle/unknown` 判断。

推理逐图原子写入 JSONL。重复执行同一命令时，`status=ok` 的图片会跳过，失败图片会重试；`--overwrite` 才会从头生成。默认不保存原始生成文本，需要诊断 JSON 解析时加 `--save-raw`。类别修复只使用 `cad_schema.py` 中显式 alias，不使用模糊相似度 snapping。

## 6. 评测

```bash
python evaluate_predictions.py \
  --ground-truth ./dataset/test_ground_truth.jsonl \
  --predictions ./output/test_predictions.jsonl \
  --output ./output/test_metrics.json
```

统一入口默认先执行可续跑推理（已有成功行会跳过），再评测：

```bash
MODEL=Qwen/Qwen3.6-27B \
ADAPTER_PATH=/output/phase_b/checkpoint-XXX \
DATASET_DIR=$PWD/dataset \
PREDICTIONS=$PWD/output/test_predictions.jsonl \
bash run_pipeline.sh eval
```

建议同时报告：

- Step1 region、Step2 view/projection、Step3 feature 的实例级 P/R/F1。
- Round Hole、Slotted Hole、View 单类指标和 macro-F1。
- IoU@0.4/0.5/0.6、数量误差、Group 完整率。
- 每张图完全匹配率、JSON/schema 合法率、约束违规率。
- GT 尺寸为空时模型擅自填写尺寸的 `empty_gt_size_hallucination_rate`。
- 按 source domain、projection、图纸质量和特征稀有度切片的结果。
- bootstrap 95% 置信区间。

脚本的 `FocusScore` 将 Round/Slotted/View 完全匹配与 macro-F1 作为主项，同时保留 strict-IoU、原始 JSON 合法率和跨阶段约束合法率作为门禁；尺寸幻觉率单独报告，不能被较高检测 F1 掩盖。

## 7. 推荐验收顺序

1. 在目标训练镜像运行 `prepare`、严格 `audit` 和 `preflight`。
2. 用旧 27B checkpoint 在无泄漏 validation/test 上重新建立 full 基线。
3. 训练 Phase A，只在 validation 运行真实生成；确认协议合法率和 feature macro-F1 上升。
4. 训练 Phase B；确认小目标与 View 指标上升且没有明显灾难性遗忘。
5. 在同一个 validation 集比较 `full`、`crops`、`hybrid`，固定 crop padding 和去重阈值。
6. 选定唯一 checkpoint 和推理模式后，只运行一次锁定 test。
7. 将失败样本按“漏检、错类、尺寸、bbox、投影/视图、Group”归因，进入下一轮主动标注，而不是直接增加 epoch。

`bash run_pipeline.sh all` 会依次执行准备、审计、环境检查和两阶段 27B 训练，但故意不会自动选“最新 checkpoint”后查看 test。训练完成后，先用 validation 真实生成选定 `ADAPTER_PATH`，再单独运行 `infer`/`eval`。首次落地建议逐阶段运行并检查每个产物。
