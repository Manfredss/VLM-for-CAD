# 实验目录索引

每个子目录是一次独立的训练/推理实验，自带 `prepare_dataset_swift.py` → `train_swift.sh` →
`inference_swift.py` → `evaluate_*.py` 这一套脚本。目录之间**刻意不共享代码**：一次实验跑完后
其脚本就冻结，下一次实验整目录复制再改，这样任何一次历史结果都能原样复现。

数据集、图纸、checkpoint 和推理结果都不入库（见根目录 `.gitignore`），只保留代码、prompt 和文档。

## 时间线

| 时间 | 目录 | 基座 | 任务 | 状态 |
|------|------|------|------|------|
| 2026-02 ~ 03 | [`Legacy/`](Legacy/) | Qwen3-VL-4B / 3.5-27B | 最早的 ms-swift 启动脚本与 Nautilus 作业 | 归档 |
| 2026-03 | [`new_finetune/`](new_finetune/) | Qwen3.5-27B | 10 特征 prompt，多规格 GPU 作业模板 | 归档 |
| 2026-03 | [`qwen3.5-4b_old_prompt_new_data/`](qwen3.5-4b_old_prompt_new_data/) | Qwen3.5-4B | 新数据 + 旧 prompt，做 prompt 消融的对照组 | 归档 |
| 2026-03 | [`qwen3.5-4b_new_prompt_new_data/`](qwen3.5-4b_new_prompt_new_data/) | Qwen3.5-4B | 新数据 + 新 prompt，消融实验组 | 归档 |
| 2026-03 | [`qwen3.5-27b_5k_view/`](qwen3.5-27b_5k_view/) | Qwen3.5-27B | 5K 图纸，15 视图/布局 + 10 特征，单步 prompt | 归档 |
| 2026-03 | [`qwen3.5-27b_5k_view_v1/`](qwen3.5-27b_5k_view_v1/) | Qwen3.5-27B | 同上，类别再平衡后重训 | 归档 |
| 2026-03 | [`qwen3.5-27b_5k_view_multi_step/`](qwen3.5-27b_5k_view_multi_step/) | Qwen3.5-27B | 拆成两步 prompt（先视图后特征） | 归档 |
| 2026-03 | [`qwen3.5-27b_5k_view_multi_step_v1/`](qwen3.5-27b_5k_view_multi_step_v1/) | Qwen3.5-27B | 两步 prompt 调优版 | 归档 |
| 2026-03 | [`RAG/`](RAG/) | — | 检索相似图纸做 few-shot 示例注入 | 探索性 |
| 2026-03 | [`RLHF/`](RLHF/) | Qwen3.5-27B | GRPO 偏好训练脚手架 | 探索性 |
| 2026-04 | [`qwn3.5-27b_788_silver_plate_bend/`](qwn3.5-27b_788_silver_plate_bend/) | Qwen3.5-27B | 西门子 788 图纸，7 特征（含折弯、镀银） | 归档 |
| 2026-04 ~ 05 | [`qwen3.5-27b_788_with_view/`](qwen3.5-27b_788_with_view/) | Qwen3.5-27B | 788 + 多轮视图检测，7 特征带分组 | 归档 |
| 2026-05 | [`qwen3.5-27b_788_with_view_improve/`](qwen3.5-27b_788_with_view_improve/) | Qwen3.5-27B | 上一条 + 尾类过采样 / 类别吸附 / SWA / DPO | **产出当时最优 ckpt** |
| 2026-05 | [`OCR/`](OCR/) | PaddleOCR | 在检出的 Title Block / Notes 区域上做文字提取 | 工具 |
| 2026-05 | [`Opus4.7/`](Opus4.7/) | Claude Opus 4.7 | 用视觉模型提 GD&T 与粗糙度，做标注/对照 | 工具 |
| 2026-05 ~ 09 | [`qwen3.5-27b_788_pipeline/`](qwen3.5-27b_788_pipeline/) | Qwen3.5-27B | 完整 SFT → SWA → GRPO → Agentic RL 流水线 | 活跃 |
| 2026-07 ~ 08 | [`qwen27b_cad_three_stage/`](qwen27b_cad_three_stage/) | Qwen3.6-27B | 三阶段拆解，19 类特征，多尺度 hybrid 推理 | 活跃 |

## 主线

真正的演进主线是这一条，其余是分支或工具：

```
5k_view  →  5k_view_multi_step  →  788_silver_plate_bend  →  788_with_view
                                                                  ↓
                                                        788_with_view_improve
                                                          ↓                ↓
                                                 788_pipeline      qwen27b_cad_three_stage
                                                 (SFT→RL 纵深)     (任务拆解 + 类别扩到 19)
```

- **`788_with_view_improve`** 是 7 特征 + 视图任务上验证过的最好一版，四个正交改动：
  尾类过采样、prompt 收紧 + 越界类别就近吸附、top-N checkpoint 权重平均（SWA）、DPO 脚手架。
- **`788_pipeline`** 沿 improve 往 RL 方向做纵深：SFT → SWA → GRPO → Agentic RL，
  奖励函数复用 `metric.py` 的质量分。
- **`qwen27b_cad_three_stage`** 换了思路——不再把两步 prompt 越写越长，而是按视觉尺度拆开：
  Step 1 只定位视图区/文档区，Step 2 判断投影法并给每个视图区分类，Step 3 在整图或高分辨率
  裁剪里穷尽提特征。生产候选走 `hybrid`：整图负责跨视图大范围特征，裁剪负责圆孔、腰孔、
  倒角、圆角等小目标，最后做确定性坐标映射与去重。类别、alias、JSON Schema、prompt 和坐标
  变换全部收敛到 `cad_schema.py` 一个文件。

## 各目录里的常见文件

| 文件 | 作用 |
|------|------|
| `prepare_dataset_swift.py` | 原始标注 → ms-swift 会话格式 JSONL，负责 train/val/test 切分与过采样 |
| `train_swift.sh` | ms-swift LoRA SFT 启动脚本，含超参与多卡设置 |
| `optimizer.py` | ViT / Aligner / LLM 分层学习率，注册进 ms-swift |
| `metric.py` / `metrics.py` / `cad_metrics.py` | 匹配逻辑与 CADScore，训练、评测、RL 奖励共用 |
| `inference_swift.py` | 批量推理，支持断点续跑 |
| `evaluate_checkpoints.py` | 扫描各 checkpoint，按 CADScore 挑最优 |
| `keep_best_n_ckpts.py` | 清理 checkpoint，省 PVC 空间 |
| `swa.py` | 按 `eval_loss` 取 top-N adapter 做权重平均 |

## 跑之前

集群访问、PVC 布局和常见坑见仓库根目录 [`README.md`](../README.md)，
Nautilus 作业清单在 [`k8s/nautilus/`](../k8s/nautilus/)。
