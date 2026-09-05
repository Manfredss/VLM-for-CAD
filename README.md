# VLM for CAD — 工业图纸特征识别

用视觉语言模型从二维工程图纸中定位、分类并提取结构特征（孔类、圆角倒角、折弯镀银、
视图布局元素等），输出结构化 JSON。基座从 Qwen3-VL-3B 一路做到 Qwen3.6-27B，
训练方式为 LoRA SFT，后期叠加 SWA / DPO / GRPO。

## 两条路径

仓库里并存两套东西，用途不同：

| | 本地 3B 模版 | ms-swift 实验线 |
|---|---|---|
| 位置 | `src/` + `configs/` | `scripts/` |
| 框架 | 自研 HuggingFace 训练循环 | [ms-swift](https://github.com/modelscope/ms-swift) |
| 基座 | Qwen3-VL-3B | Qwen3.5-27B / Qwen3.6-27B |
| 硬件 | 单卡 RTX 5070 Ti (16GB) | Nautilus 集群，A100-80G / H100 |
| 时间 | 2026-02 ~ 03 | 2026-03 至今 |
| 状态 | 冻结，作为可独立跑通的参考实现 | **主线** |

新工作走 `scripts/`。3B 模版的完整用法见 [`docs/local-3b-template.md`](docs/local-3b-template.md)。

## 目录结构

```
.
├── scripts/              # 实验线：17 个独立实验目录，见 scripts/README.md
├── src/                  # 本地 3B 模版：dataset / train / inference / evaluate / merge_lora
├── configs/              # 3B 模版的训练配置
├── k8s/
│   ├── training_job.yaml # 3B 模版的 K8s Job
│   └── nautilus/         # 89 个 Nautilus 作业清单（训练 / 推理 / 评测 / 数据准备 / 探针）
├── docker/               # 镜像构建
├── docs/
│   ├── local-3b-template.md      # 3B 模版完整文档：数据格式、快速开始、显存、踩坑
│   ├── PIPELINE*.txt             # 分步命令（Linux / WSL / Windows）
│   ├── prompts/                  # 5K 数据集的单步与两步 prompt
│   └── notes/                    # 改进方向与 RLHF 设计笔记
└── requirements.txt
```

## 从哪开始看

1. [`scripts/README.md`](scripts/README.md) — 实验索引：17 个目录各是什么、时间线、主线演进路径。
2. [`scripts/qwen27b_cad_three_stage/README.md`](scripts/qwen27b_cad_three_stage/README.md) — 最新的三阶段方案，19 类特征。
3. [`scripts/qwen3.5-27b_788_pipeline/README.md`](scripts/qwen3.5-27b_788_pipeline/README.md) — SFT → SWA → GRPO → Agentic RL 全流水线。
4. [`docs/local-3b-template.md`](docs/local-3b-template.md) — 数据格式与本地单卡训练。

## Nautilus 集群

作业清单在 `k8s/nautilus/`，命名规则是 `<动作>-<模型>-<实验>-<日期>.yaml`：

| 前缀 | 用途 |
|------|------|
| `train-` | 训练作业 |
| `infer-` | 批量推理 |
| `eval-` / `benchmark-` | 评测与基准 |
| `build-` / `prepare-` / `seed-` / `materialize-` | 数据集与脚本准备 |
| `inspect-` / `probe-` / `profile-` | 排查用的临时 Pod |
| `prewarm-` | 抢占 GPU 节点、预热镜像 |

```bash
# kubeconfig 不在仓库里，需要自己配置到 .kube/config
kubectl --kubeconfig .kube/config apply -f k8s/nautilus/train-qwen36-27b-p3-r3-feature-20260804.yaml
kubectl --kubeconfig .kube/config -n nsf-maica get pods -o wide
```

## 仓库里没有什么

这是公开仓库，以下内容**一律不入库**（规则见 [`.gitignore`](.gitignore)，采用默认拒绝 + 白名单放行）：

- **图纸与数据集** —— 西门子工程图纸及其派生的标注、JSONL 数据集、推理结果。属于客户资料。
- **训练产物** —— checkpoint、adapter、合并后的权重、预测结果转储。
- **凭据** —— `.kube/config`（Nautilus kubeconfig）、`api_key.txt`。

因此 clone 下来无法直接开跑，需要自备数据和集群凭据。各实验目录的 README 里写了预期的数据
路径和格式。新增文件类型时，往 `.gitignore` 的白名单区加 `!` 规则，不要动开头那行 `*`。

## 依赖

```bash
pip install -r requirements.txt
```

`scripts/` 下的实验跑在 ModelScope 官方 ms-swift 镜像里，依赖以镜像为准，不走这份
`requirements.txt`；具体镜像 tag 见对应的 `k8s/nautilus/*.yaml`。
