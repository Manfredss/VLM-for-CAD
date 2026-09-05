# GD&T 自动提取工具 — 详细文档

## 一、任务背景

本工具从西门子风格的工程图纸（PNG 图片）中，自动提取 **GD&T（几何尺寸与公差）** 标注。调用 Claude Opus 4.7 视觉模型，无需训练数据，直接零样本识别。

提取的标注分为四类：

| 类别 | 内容 | 标准 |
|---|---|---|
| 表面粗糙度 | Ra、Rz 等符号及数值 | ISO 1302 / ASME Y14.36 |
| 几何公差 | 特征控制框（平面度、位置度、垂直度等） | ISO 1101 / ASME Y14.5 |
| 基准 | 基准字母标注（A、B、C…） | ISO 5459 |
| 尺寸公差 | ±、+/-、ISO 配合代号（H7、h6 等） | 通用 |

---

## 二、文件结构

```
scripts/Opus4.7/
├── extract_gdt.py        # 主脚本（GD&T 全类别提取）
├── extract_roughness.py  # 旧脚本（仅提取粗糙度，保留向后兼容）
├── gdt_smoke.json        # 烟雾测试占位文件（待填充 API 结果）
├── roughness_smoke.json  # 旧版粗糙度占位文件
├── requirements.txt      # Python 依赖
└── README.md             # 英文简要说明
```

---

## 三、图片存储位置

### 本地（开发机）
```
scripts/qwn3.5-27b_788_silver_plate_bend/simens_7feats/
```
共 **1266 张** PNG 图片，命名格式：
```
001_A7E0018072440_00_page_001.png
NNN_<零件号>_<版本>_page_<页码>.png
```

### Nautilus 集群（PVC）
挂载路径：`/workspace/data/simens_7feats/`
- PVC 名称：`qwf-workspace`（1 TiB，RWX，rook-cephfs）
- 与本地路径内容一致，可通过 `kubectl cp` 上传/下载

### 测试集
```
/workspace/data/test_view_7feats_improve.jsonl   # 80 张图片的测试集（带标注路径）
```

---

## 四、代码功能说明（extract_gdt.py）

### 4.1 整体流程

```
输入图片
    ↓ 压缩/编码（PIL → base64 PNG/JPEG）
    ↓ 调用 Claude Opus 4.7 API（单次多模态请求）
    ↓ 解析 JSON 响应
    ↓ 写入输出文件（逐条追加，支持续跑）
输出 JSON
```

### 4.2 图片预处理

- 若图片最长边 > 4000px，等比缩放至 4000px
- 先尝试 PNG 编码；若 > 4.5 MB，改用 JPEG（quality=88）
- 编码为 base64 后嵌入 API 请求

### 4.3 API 调用

支持两种鉴权路径：

| 密钥格式 | 后端 | SDK |
|---|---|---|
| `sk-ant-...` | Anthropic 直连 | `anthropic` Python SDK |
| `sk-or-v1-...` | OpenRouter 路由 | `openai` Python SDK（兼容接口） |

- 失败自动重试（指数退避，最多 4 次）
- `max_tokens=8192`（GD&T 输出较粗糙度长）

### 4.4 断点续跑

每处理完一张图片，立即将结果写入输出 JSON。下次以相同 `--output` 参数运行时，自动跳过已处理的图片（根据 `dataitem_name` 去重）。

---

## 五、System Prompt 说明

模型收到的系统提示包含四个类别的详细说明：

### 类别 1：表面粗糙度
- 识别 V 形三角符号（带/不带横线/圆圈）
- 提取参数（Ra、Rz、Rmax 等）和数值（μm）
- 区分"通用粗糙度"（near Title Block，含"其余/REST"字样）和"表面粗糙度"（引线指向具体面）
- 输出字段：`parameter`、`value`、`unit`、`process`（machined/no_machining/any）、`scope`（general/surface）、`context`

### 类别 2：几何公差
- 识别特征控制框（矩形框，格子结构）
- 支持 14 种符号：平面度、直线度、圆度、圆柱度、角度、垂直度、平行度、位置度、同轴度、对称度、线轮廓度、面轮廓度、圆跳动、全跳动
- 提取：公差值、公差单位（默认 mm）、基准引用字母、材料条件修饰符（MMC/LMC/RFS）
- 输出字段：`symbol`、`tolerance_value`、`tolerance_unit`、`datum_references`、`material_condition`、`context`

### 类别 3：基准
- 识别填充三角形 + 方框字母的基准符号
- 每个唯一基准字母记录一次
- 输出字段：`label`、`context`

### 类别 4：尺寸公差
- 识别带公差的标注尺寸：
  - 对称式：`25 ±0.1` → `upper="+0.1"`, `lower="-0.1"`
  - 非对称式：`25 +0.05/-0.02`
  - 单向式：`25 +0.1/0`
  - ISO 配合代号：`Ø25 H7` → `fit_code="H7"`
- **不提取**纯标称尺寸（无公差的尺寸）
- 输出字段：`dimension_type`（linear/diameter/radius/angular）、`nominal_value`、`upper_deviation`、`lower_deviation`、`fit_code`、`unit`、`context`

---

## 六、输出格式

每张图片对应一条记录：

```json
[
  {
    "dataitem_name": "001_A7E0018072440_00_page_001.png",
    "gdt": [
      {
        "category": "roughness",
        "parameter": "Ra",
        "value": "3.2",
        "unit": "μm",
        "process": "machined",
        "scope": "general",
        "context": "标题栏上方，含'其余'字样"
      },
      {
        "category": "roughness",
        "parameter": "Ra",
        "value": "1.6",
        "unit": "μm",
        "process": "machined",
        "scope": "surface",
        "context": "顶面法兰铣削面"
      },
      {
        "category": "geometric_tolerance",
        "symbol": "flatness",
        "tolerance_value": "0.05",
        "tolerance_unit": "mm",
        "datum_references": [],
        "material_condition": null,
        "context": "顶部安装面"
      },
      {
        "category": "geometric_tolerance",
        "symbol": "position",
        "tolerance_value": "0.1",
        "tolerance_unit": "mm",
        "datum_references": ["A", "B"],
        "material_condition": "MMC",
        "context": "4× Ø8 螺栓孔"
      },
      {
        "category": "datum",
        "label": "A",
        "context": "底面"
      },
      {
        "category": "dimensional_tolerance",
        "dimension_type": "diameter",
        "nominal_value": "20",
        "upper_deviation": null,
        "lower_deviation": null,
        "fit_code": "H7",
        "unit": "mm",
        "context": "主孔"
      }
    ]
  }
]
```

若某张图片失败（图片缺失、API 错误），记录包含 `"_error"` 字段，`gdt` 为空列表。

---

## 七、安装与环境

```bash
pip install -r requirements.txt
# requirements.txt 内容：
#   openai>=1.30.0
#   Pillow>=10.0
#   anthropic   （若使用 Anthropic 直连，需额外安装）
```

设置 API 密钥（二选一）：
```bash
export ANTHROPIC_API_KEY=sk-ant-...    # Anthropic 直连（推荐）
export OPENROUTER_API_KEY=sk-or-v1-... # OpenRouter 路由
```

---

## 八、运行方式

### 8.1 单张图片测试
```bash
cd scripts/Opus4.7

python extract_gdt.py \
    --image ../qwn3.5-27b_788_silver_plate_bend/simens_7feats/001_A7E0018072440_00_page_001.png \
    --output gdt_single_test.json
```

### 8.2 批量处理目录（成本控制：先跑 5 张）
```bash
python extract_gdt.py \
    --image-dir ../qwn3.5-27b_788_silver_plate_bend/simens_7feats \
    --glob "*.png" \
    --output gdt_all.json \
    --limit 5
```

### 8.3 处理全部 1266 张图片
```bash
python extract_gdt.py \
    --image-dir ../qwn3.5-27b_788_silver_plate_bend/simens_7feats \
    --glob "*.png" \
    --output gdt_all.json
# 预计耗时：~4-6 小时（顺序调用，无并发）
# 预计费用：~$200-$400（Opus 4.7，约 $0.15-0.30/张）
```

### 8.4 在 Nautilus 集群上运行（Inference Pod 中）
```bash
# 1. 进入 pod
kubectl -n nsf-maica exec -it <pod名> -- bash

# 2. 安装依赖（若 venv 中未安装）
pip install openai Pillow anthropic

# 3. 设置密钥
export ANTHROPIC_API_KEY=sk-ant-...

# 4. 运行
python /workspace/scripts_gdt/extract_gdt.py \
    --image-dir /workspace/data/simens_7feats \
    --glob "*.png" \
    --output /workspace/output/gdt_all.json \
    --limit 10    # 先跑 10 张验证
```

### 8.5 续跑（断点恢复）
直接以相同 `--output` 参数重新运行，脚本会自动跳过已完成的图片：
```bash
python extract_gdt.py \
    --image-dir ../qwn3.5-27b_788_silver_plate_bend/simens_7feats \
    --glob "*.png" \
    --output gdt_all.json   # 与上次相同路径即可
```

---

## 九、费用估算

| 场景 | 图片数 | 单张费用 | 总费用 |
|---|---|---|---|
| 烟雾测试 | 5 | ~$0.20 | ~$1 |
| 测试集 | 80 | ~$0.20 | ~$16 |
| 全量 | 1266 | ~$0.20 | ~$250 |

> Opus 4.7 为 Anthropic 最贵模型。若预算有限，可用 `--model anthropic/claude-sonnet-4-6` 降低成本（准确率略低）。

---

## 十、与现有推理结果联用

`gdt_smoke.json` 目前是占位文件（所有 `gdt` 为空列表）。运行脚本后会填充实际提取结果。

若要将 GD&T 结果与视图/特征检测结果（Qwen3.5-27B 推理输出）合并：
```python
import json

views = {r["dataitem_name"]: r for r in json.load(open("results_v4_ckpt350.json"))}
gdts  = {r["dataitem_name"]: r for r in json.load(open("gdt_all.json"))}

merged = []
for name, vr in views.items():
    record = dict(vr)
    record["gdt"] = gdts.get(name, {}).get("gdt", [])
    merged.append(record)

json.dump(merged, open("results_with_gdt.json", "w"), ensure_ascii=False, indent=2)
```

---

## 十一、常见问题

**Q：为什么不用已有的 Qwen3.5-27B 微调模型来提取 GD&T？**
A：GD&T 符号（特别是几何公差框、粗糙度符号）体积小、密度高，需要极强的符号识别能力。Opus 4.7 零样本能力更强；微调 Qwen 需要有标注的 GD&T 训练数据，目前没有。

**Q：图片里没有 GD&T 怎么办？**
A：返回 `{"gdt_annotations": []}`，记录中 `gdt` 字段为空列表，不报错。

**Q：API 调用失败怎么办？**
A：自动重试 4 次（指数退避）。若仍失败，记录 `_error` 字段，下次续跑时会重新处理该图片。
