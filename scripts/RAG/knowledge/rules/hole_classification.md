---
categories: [Round Hole, Threaded Hole, Pin Hole, Counterbore Hole, Rectangular Hole, Round Hole Group, Threaded Hole Group, Pin Hole Group, Counterbore Hole Group]
step: 2
keywords: [hole, M, H7, H6, G6, thread, pin, counterbore, tolerance, 孔, 螺纹, 销, 沉头, 公差]
---

# 孔类分类优先级

当一个孔的类型不确定时，按以下优先级判断：

## 分类决策树

```
尺寸标注包含 M（如 M8、M6）？
  ├── 是 → 螺纹孔 (Threaded Hole)
  └── 否 → 标注包含公差代号（H7、H6、G6）？
              ├── 是 → 销孔 (Pin Hole)
              └── 否 → 标注包含两个不同直径？
                        ├── 是 → 沉头孔 (Counterbore Hole)
                        └── 否 → 轮廓为四边形？
                                  ├── 是 → 矩形孔 (Rectangular Hole)
                                  └── 否 → 圆孔 (Round Hole)
```

## 尺寸格式示例

| 类型 | 尺寸示例 | 组标注示例 |
|------|---------|-----------|
| 圆孔 | 18, Ø12, Ø24 DP15 | 4x18, 2-Ø12 |
| 螺纹孔 | M8, M6 DP20 | 2xM8, 4-M6 |
| 销孔 | Ø8H7, Ø6H7 DP20 | 2-Ø8H7 |
| 沉头孔 | Ø14 DP10 Ø9, Ø21 Ø14.5 | 2-Ø14 DP10 Ø9 |
| 矩形孔 | 20x123.5, 50x160 | — |

## 常见混淆

- **圆孔 vs 螺纹孔**：关键看有没有 M 前缀。螺纹线在视图中可能不明显
- **圆孔 vs 销孔**：关键看有没有 H7/H6 公差。销孔通常较小，用于定位
- **圆孔 vs 沉头孔**：沉头孔有阶梯结构（双直径），标注包含两个 Ø 值
