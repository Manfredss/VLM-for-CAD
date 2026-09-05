---
categories: [Round Hole Group, Threaded Hole Group, Pin Hole Group, Counterbore Hole Group, Fillet Group, Chamfer Group]
step: 2
keywords: [group, 组, multiple, Nx, 2x, 3x, 4x, 统一标注, instance]
---

# 组 (Group) 检测规则

## 定义

组 = 多个相同尺寸特征由统一标注描述。

## 检测要求

当检测到组时，必须同时检测：
1. **每个单实例**（各自有独立 bbox）
2. **组本身**（bbox 覆盖所有实例）

## 标注格式

- `N x 尺寸`：如 4x18、2xM8、3xR5
- `N - 尺寸`：如 2-Ø8H7、2-C1、2-Ø14 DP10 Ø9

## Size 字段规则

| 层级 | size 格式 | 示例 |
|------|----------|------|
| 单实例 | 不含数量前缀 | "18"、"M8"、"R5" |
| 组 | 必须含数量前缀 | "4x18"、"2xM8"、"3xR5" |

## 常见错误

- 只检测组而遗漏单实例
- 单实例的 size 误写了数量前缀（如 "4x18" 应为 "18"）
- 组的 bbox 未覆盖所有实例
