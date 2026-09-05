---
categories: [Section View, Auxiliary View, Bill of Materials, Threaded Shaft, Isometric View, Revision Table]
step: 0
keywords: [section, auxiliary, BOM, bill, materials, threaded shaft, isometric, revision, 剖视图, 辅助视图, 材料清单, 螺纹轴, 等轴测, 修改表]
---

# 稀有类别识别指南

以下类别在训练数据中出现较少，需特别注意：

## 剖视图 (Section View)
- **出现频率**：极低（~11 实例）
- **关键特征**：剖面线（平行斜线阴影）、标有剖切线（如 A-A、B-B）
- **区别于普通视图**：有阴影线填充的区域

## 辅助视图 (Auxiliary View)
- **出现频率**：极低（~45 实例）
- **关键特征**：不在标准正交位置、有方向箭头和标识字母
- **用途**：展示倾斜面的真实形状

## 材料清单 (Bill of Materials)
- **出现频率**：极低（~16 实例）
- **位置**：通常位于标题栏上方
- **特征**：表格形式，列出零件编号、名称、数量
- **仅在装配图中出现**

## 螺纹轴 (Threaded Shaft)
- **出现频率**：极低（~59 实例）
- **区别于螺纹孔**：螺纹轴是外螺纹（凸出的），螺纹孔是内螺纹（凹进的）
- **尺寸格式**：M + 直径（如 M10、M14）

## 等轴测图 (Isometric View)
- **位置**：通常在图纸右上角或空白区域
- **特征**：三维透视效果，同时显示三个面
- **不用于尺寸标注**

## 修改表 (Revision Table)
- **位置**：通常在图纸右上角或标题栏上方
- **特征**：包含 REV、DATE、DESCRIPTION 等列
- **容易与 Notes 混淆**：修改表是表格格式，Notes 是文字列表
