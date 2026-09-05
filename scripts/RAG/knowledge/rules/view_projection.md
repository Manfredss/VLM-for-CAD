---
categories: [Orthographic Projection - Front View, Orthographic Projection - Top View, Orthographic Projection - Left View, Orthographic Projection - Right View]
step: 1
keywords: [view, projection, front, top, left, right, third-angle, 视图, 投影, 正视图, 俯视图, 左视图, 右视图, 第三角]
---

# 第三角投影法视图位置规则

所有标注统一遵循第三角投影法。视图位置关系如下：

```
              ┌──────────┐
              │ 俯视图    │
              │ Top View  │
              └──────────┘
                   ↑
┌──────────┐ ┌──────────┐ ┌──────────┐
│ 右视图    │ │ 正视图    │ │ 左视图    │
│Right View│ │Front View│ │ Left View│
└──────────┘ └──────────┘ └──────────┘
```

## 判断步骤

1. **找正视图**：通常是最大的视图，位于图纸中央偏左
2. **正视图正上方** → 俯视图 (Top View)
3. **正视图右侧** → 左视图 (Left View)
4. **正视图左侧** → 右视图 (Right View)

## 注意事项

- 不要混淆左视图和右视图的位置：在第三角投影法中，左视图在右边，右视图在左边
- 正视图通常包含最多的尺寸标注
- 有些图纸可能只有2-3个投影视图，不一定有全部四个
