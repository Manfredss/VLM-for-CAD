"""
修复数据集中的图片路径 — 将 Mac 本地路径替换为 Pod 内路径
适配 JSONL 格式（每行一个 JSON）

用法:
  python fix_data_paths.py \
      --input_dir /workspace/data \
      --image_dir /workspace/data/images \
      --output_dir /workspace/data
"""

import json
import os
import argparse
from pathlib import Path


def fix_paths_jsonl(jsonl_path, image_dir, output_path):
    """修复单个 JSONL 文件中的图片路径"""
    fixed = 0
    missing = 0
    total = 0

    with open(jsonl_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            total += 1

            # 1) 修复顶层 images 字段（ms-swift 常用）
            images = sample.get("images")
            if isinstance(images, list):
                for idx, img_path in enumerate(images):
                    if isinstance(img_path, str) and img_path:
                        img_name = Path(img_path).name
                        new_path = str(Path(image_dir) / img_name)
                        if os.path.exists(new_path):
                            images[idx] = new_path
                            fixed += 1
                        else:
                            images[idx] = new_path
                            missing += 1

            # 2) 修复 messages 里 user.content(list) 的 image 项（兼容旧格式）
            messages = sample["messages"]
            for msg in messages:
                if msg["role"] == "user" and isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item.get("type") == "image":
                            # 提取文件名（去掉 Mac 路径前缀）
                            img_name = Path(item["image"]).name
                            new_path = str(Path(image_dir) / img_name)
                            if os.path.exists(new_path):
                                item["image"] = new_path
                                fixed += 1
                            else:
                                item["image"] = new_path
                                missing += 1

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return total, fixed, missing


def main():
    parser = argparse.ArgumentParser(description="Fix image paths from Mac local to Pod paths (JSONL)")
    parser.add_argument("--input_dir", type=str, default="/workspace/data",
                        help="Directory containing JSONL dataset files")
    parser.add_argument("--image_dir", type=str, default="/workspace/data/images",
                        help="Directory containing the images on the pod")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as input, overwrite)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"WARNING: no jsonl files found in {input_dir}")
        return

    for src in jsonl_files:
        fname = src.name
        if src.exists():
            # 如果输出和输入是同一文件，先写到临时文件再替换
            dst = output_dir / fname
            if src == dst:
                tmp = dst.with_suffix(".jsonl.tmp")
                total, fixed, missing = fix_paths_jsonl(src, args.image_dir, tmp)
                tmp.rename(dst)
            else:
                total, fixed, missing = fix_paths_jsonl(src, args.image_dir, dst)
            print(f"{fname}: {total} samples, {fixed} paths fixed, {missing} images not found on disk (OK if not uploaded yet)")
        else:
            print(f"WARNING: {src} not found, skipping")


if __name__ == "__main__":
    main()
