# 本地 3B 模版（Qwen3-VL-3B + 自研训练循环）

> 这是项目最早的一条路径：`src/` 下的自研 HuggingFace 训练循环，配 `configs/train_config.yaml`，
> 目标是在单张 RTX 5070 Ti (16GB) 上跑通，也能提交到 Nautilus。2026-02 ~ 03 期间使用。
>
> 后续所有实验都转到了 ms-swift + 27B 基座，见 [`scripts/README.md`](../scripts/README.md)。
> 本文保留下来是因为数据格式说明、显存参考和踩坑记录仍然适用。
>
> 分步命令另见 [`PIPELINE.txt`](PIPELINE.txt)（Linux）、[`PIPELINE_WSL.txt`](PIPELINE_WSL.txt)、[`PIPELINE_WINDOWS.txt`](PIPELINE_WINDOWS.txt)。

---

## 模型选择

| 模型 | 参数量 | 来源 | 说明 |
|------|--------|------|------|
| Qwen3-VL-3B | 3B | HuggingFace / ModelScope | 默认，16GB 显存即可训练 |

模型默认从 **ModelScope** 下载（国内速度快），可在 `train_config.yaml` 中切换为 HuggingFace。

## 数据格式

训练数据为 JSON 文件。**直接使用 Label Studio 导出的 JSON 即可**，程序会自动检测并转换。也支持其他通用格式。

### 格式 1: Label Studio JSON 导出（主要格式）

从 Label Studio 导出时选择 **JSON** 格式，直接作为 `train.json` 使用：

```json
[
    {
        "id": 1,
        "data": {
            "image": "/data/upload/1/drawing_001.png"
        },
        "annotations": [
            {
                "result": [
                    {
                        "type": "textarea",
                        "value": {"text": ["检测到以下特征：..."]},
                        "from_name": "answer",
                        "to_name": "image"
                    },
                    {
                        "type": "rectanglelabels",
                        "value": {
                            "x": 15.2, "y": 22.5,
                            "width": 8.0, "height": 10.3,
                            "rectanglelabels": ["螺纹孔"],
                            "original_width": 1920,
                            "original_height": 1080
                        },
                        "from_name": "label",
                        "to_name": "image"
                    }
                ]
            }
        ]
    }
]
```

**支持的 Label Studio 标注类型：**
- `textarea` — 文本描述/回答（作为模型的目标输出）
- `rectanglelabels` — 矩形框标注（坐标自动转换为像素值）
- `polygonlabels` — 多边形标注
- `choices` — 分类标签

**图片路径自动处理：** Label Studio 的路径（如 `/data/upload/1/xxx.png` 或 `/data/local-files/?d=xxx`）会被自动清理，只保留文件名部分，然后在 `image_root` 目录下查找。

**自定义 prompt：** 如果 Label Studio 的 `data` 中有 `prompt` / `question` / `instruction` 字段，会自动作为用户提问；否则使用默认提问。

> 完整示例见 `data/example_label_studio_export.json`

### 格式 2: 对话格式
```json
[
    {
        "image": "drawing_001.png",
        "conversations": [
            {"role": "user", "content": "请识别这张图纸中的工件特征。"},
            {"role": "assistant", "content": "检测到以下特征：..."}
        ]
    }
]
```

### 格式 3: 简单问答格式
```json
[
    {
        "image": "drawing_001.png",
        "question": "请识别这张图纸中的工件特征。",
        "answer": "检测到以下特征：..."
    }
]
```

### 格式 4: 带检测框的结构化格式
```json
[
    {
        "image": "drawing_001.png",
        "question": "请识别图中的特征并给出位置。",
        "answer": "描述...",
        "features": [
            {"label": "螺纹孔", "bbox": [100, 200, 150, 250]},
            {"label": "倒角", "bbox": [300, 100, 350, 150]}
        ]
    }
]
```

**图片字段名** 自动识别: `image`, `image_path`, `img`, `img_path`, `file_name`, `filename`

**角色名** 自动识别: `user/human`, `assistant/gpt/bot/model`

## 当前训练状态（IM_D03_PT_5K 数据集）

### 训练配置

| 项目 | 参数 |
|---|---|
| 基础模型 | merve/qwen3-vl-3b-llava-1pct |
| 数据集 | 4,092 张工业图纸（训练 3,683 / 验证 409） |
| LoRA | r=32，alpha=64，目标模块：q/k/v/o/gate/up/down proj |
| 训练轮次 | 5 epoch（第一阶段 3 epoch，lr=1e-4 cosine；第二阶段续训 2 epoch，lr=5e-5 固定） |
| 最优检查点 | **checkpoint-1200**（step 1200，epoch 2.60，验证损失 0.3792） |

### 训练损失

训练损失从 **1.1098 → 0.3655（下降 67.1%）**，验证损失从 **0.4309 → 0.3792（下降 12.0%）**。
训练集与验证集损失差仅 +0.018，无过拟合。续训 2 epoch 未改善最优验证损失，模型已触及当前数据量上限。

### 检测性能（50 张验证图，IoU@0.5）

| 模型 | F1 | 精确率 | 召回率 | 特征数量 MAE |
|---|---|---|---|---|
| 基础模型（未微调） | 0.000 | 0.000 | 0.000 | 4.40 |
| **微调模型（checkpoint-1200）** | **0.049** | **0.052** | **0.046** | **0.92** |

放宽至 IoU@0.25 时 F1 提升至 **0.146**（提升 3 倍），说明预测框位置方向正确但精度尚不足。
当前瓶颈为训练数据量，建议扩充至 10,000+ 张再训练。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt

# Flash Attention 2（推荐，需要单独安装）
pip install flash-attn --no-build-isolation
```

### 2. 准备数据

#### 从 Label Studio 导出

1. 在 Label Studio 项目中完成标注
2. 点击 **Export** → 选择 **JSON** 格式 → 下载
3. 将导出的 JSON 文件重命名为 `train.json`（可选拆分 `val.json`）
4. 将标注用的图片复制到 `data/images/` 目录

```
data/
├── train.json          # Label Studio 导出的 JSON（直接使用）
├── val.json            # 可选，验证集
└── images/
    ├── drawing_001.png
    ├── drawing_002.png
    └── ...
```

> **注意：** Label Studio 中的图片路径（如 `/data/upload/1/drawing_001.png`）会被自动清理，
> 程序会在 `image_root`（默认 `data/images/`）下查找 `drawing_001.png`。
> 只需确保图片文件名和 Label Studio 中上传的文件名一致即可。

#### Label Studio 标注模板建议

推荐在 Label Studio 中使用以下 labeling config：

```xml
<View>
  <Image name="image" value="$image"/>

  <!-- 文本框：描述图纸中的工件特征 -->
  <TextArea name="answer" toName="image"
            placeholder="请描述图纸中的工件特征..."
            rows="6" maxSubmissions="1" editable="true"/>

  <!-- 可选：矩形框标注特征位置 -->
  <RectangleLabels name="label" toName="image">
    <Label value="螺纹孔" background="#FF0000"/>
    <Label value="倒角" background="#00FF00"/>
    <Label value="圆角" background="#0000FF"/>
    <Label value="尺寸标注" background="#FFA500"/>
    <Label value="形位公差" background="#800080"/>
    <Label value="表面粗糙度" background="#FF69B4"/>
    <Label value="槽" background="#00CED1"/>
    <Label value="孔" background="#FFD700"/>
  </RectangleLabels>
</View>
```

### 3. 修改配置

编辑 `configs/train_config.yaml`，主要需要确认：
- `dataset.train_json`: 训练数据路径
- `dataset.image_root`: 图片根目录
- `model.source`: 模型下载源（`modelscope` 或 `huggingface`）

### 4. 开始训练

```bash
# 单卡训练
python src/train.py --config configs/train_config.yaml

# 多卡训练
NUM_GPUS=4 bash scripts/run_train.sh

# 覆盖配置参数
python src/train.py --config configs/train_config.yaml \
    --training.num_train_epochs 5 \
    --training.learning_rate 5e-5
```

### 5. 评估模型

```bash
# 训练曲线摘要（无需 GPU）
python src/evaluate.py --mode summary \
    --trainer_state outputs/qwen3-vl-3b-lora/checkpoint-1383/trainer_state.json

# 检测指标对比（需要 GPU；--max_samples 50 快速验证，去掉则跑全部 409 张）
python src/evaluate.py --mode eval \
    --val_json data/val.json \
    --image_root data/IM_D03_PT_5K \
    --base_model merve/qwen3-vl-3b-llava-1pct \
    --finetuned_adapter outputs/qwen3-vl-3b-lora/checkpoint-1200 \
    --iou_thr 0.5 \
    --max_samples 50 \
    --output outputs/eval_results.json
```

评估脚本输出：精确率 / 召回率 / F1 / 平均 IoU（按类别及总体），以及两模型对比 delta 表。

### 6. 推理测试

```bash
# 单张图推理（使用最优 LoRA 检查点）
python src/inference.py \
    --model_path outputs/qwen3-vl-3b-lora/checkpoint-1200 \
    --base_model merve/qwen3-vl-3b-llava-1pct \
    --image test_drawing.png

# 批量推理
python src/inference.py \
    --model_path outputs/qwen3-vl-3b-lora/checkpoint-1200 \
    --base_model merve/qwen3-vl-3b-llava-1pct \
    --image_dir test_images/ \
    --output results.json
```

### 7. 合并 LoRA 权重（可选，用于部署）

合并后模型为独立完整模型，无需基础模型即可直接加载，适合部署至生产环境或上传到 HuggingFace Hub。
推理和评估无需提前合并，脚本内部会自动完成合并操作。

```bash
python src/merge_lora.py \
    --base_model merve/qwen3-vl-3b-llava-1pct \
    --adapter_path outputs/qwen3-vl-3b-lora/checkpoint-1200 \
    --output_path outputs/qwen3-vl-3b-merged
```

## Nautilus 部署

### 1. 构建并推送 Docker 镜像

```bash
cd "retrain model"
docker build -f docker/Dockerfile -t <your-registry>/industrial-mind-train:latest .
docker push <your-registry>/industrial-mind-train:latest
```

### 2. 创建 PVC 并上传数据

```bash
# 创建 PVC
kubectl apply -f k8s/training_job.yaml  # PVC 定义在文件底部

# 上传数据到 PVC（可通过临时 Pod 或 kubectl cp）
```

### 3. 提交训练任务

编辑 `k8s/training_job.yaml` 中的 `TODO` 项，然后：

```bash
kubectl apply -f k8s/training_job.yaml
kubectl logs -f job/industrial-mind-train
```

## 显存需求参考（Qwen3-VL-3B）

| 配置 | 预计显存 |
|------|----------|
| LoRA r=64, bs=1, grad_accum=8 | ~14GB |
| LoRA r=64, bs=1, grad_accum=8 + gradient_checkpointing | ~11GB |
| LoRA r=32, bs=1 + gradient_checkpointing | ~10GB |

**RTX 5070 Ti (16GB)** 可以胜任。也支持 A100 / H100 等更高端 GPU。

## 关键超参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lora.r` | 32 | LoRA 秩，越大表达能力越强，显存越多 |
| `lora.lora_alpha` | 64 | 通常设为 2×r |
| `training.learning_rate` | 1e-4 | LoRA 微调常用 1e-4 ~ 2e-4 |
| `training.num_train_epochs` | 5 | 数据量小则适当多训几轮 |
| `training.gradient_accumulation_steps` | 8 | 等效 batch_size = bs × accum = 8 |
| `dataset.max_pixels` | 1280 | 图片最大分辨率（单位：28×28 patch 数），1280 ≈ 1002×1002 像素 |

> **注意：** `min_pixels` / `max_pixels` 的单位是 **patch 数**，实际像素 = n × 28 × 28。
> 例如 `max_pixels=1280` → 1280 × 784 = 1,003,520 像素（约 1002×1002）。

## 已知问题与修复记录

| 问题 | 影响 | 状态 |
|---|---|---|
| `dataset.py` 像素公式错误（`n*n*4` 应为 `n*28*28`） | 训练时图片分辨率过高 6.5 倍，OOM 风险 | 已修复 |
| `dataset.py` 标签边界用文本解码定位，VLM 视觉 token 导致偏移 | 损失计算范围错误 | 已修复 |
| `train.py` 无验证集时仍设 `load_best_model_at_end=True` | Trainer 报错 | 已修复 |
| `dataset.py` `image_grid_thw` 被错误 squeeze，collator 产生 0 维张量 | 训练崩溃 | 已修复 |
| `merge_lora.py` `save_pretrained` 触发 transformers PEFT bug | 合并后无法保存 | 已修复 |
| `inference.py` 像素公式错误（`n*n*4` 应为 `n*28*28`） | 推理分辨率与训练不一致 | 已修复 |
