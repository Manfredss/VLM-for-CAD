#!/usr/bin/env python3
"""
convert_jsonl_format.py

把旧格式 JSONL（user content 是 array）转换成 ms-swift 4.0 dev0 格式：
  - user content 改为字符串，<image> 占位符插在文字前面
  - 图片路径统一放到顶层 images 字段

旧格式：
  {"messages": [
    {"role": "user", "content": [
      {"type": "text", "text": "分析图纸："},
      {"type": "image", "image": "/path/img.png"}
    ]},
    {"role": "assistant", "content": "..."}
  ]}

新格式：
  {"messages": [
    {"role": "user", "content": "<image>分析图纸："},
    {"role": "assistant", "content": "..."}
  ],
  "images": ["/path/img.png"]}

用法：
  python3 convert_jsonl_format.py --input /workspace/data/train.jsonl --output /workspace/data/train_swift4.jsonl
  python3 convert_jsonl_format.py --input /workspace/data/val.jsonl   --output /workspace/data/val_swift4.jsonl
"""

import json
import argparse
from pathlib import Path


def convert_sample(sample: dict) -> dict:
    messages = sample.get("messages", [])
    new_messages = []
    images = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, list):
            # user 消息：content 是 array，提取 text 和 image
            text_parts = []
            for item in content:
                if item.get("type") == "image":
                    images.append(item["image"])
                    text_parts.insert(0, "<image>")  # <image> 放在文字前面
                elif item.get("type") == "text":
                    text_parts.append(item["text"])
            new_content = "".join(text_parts)
        else:
            # system / assistant 消息：content 已经是字符串
            new_content = content

        new_messages.append({"role": role, "content": new_content})

    new_sample = {"messages": new_messages}
    if images:
        new_sample["images"] = images

    # 保留其他字段，但排除 objects（会被 ms-swift 误解析为 grounding bbox）
    skip_keys = {"messages", "objects"}
    for k, v in sample.items():
        if k not in skip_keys:
            new_sample[k] = v

    return new_sample


def convert_file(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    total = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                new_sample = convert_sample(sample)
                fout.write(json.dumps(new_sample, ensure_ascii=False) + "\n")
                total += 1
            except Exception as e:
                print(f"[WARN] skipping line: {e}")
                skipped += 1

    print(f"Done: {total} samples converted, {skipped} skipped")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入 JSONL 路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    args = parser.parse_args()
    convert_file(args.input, args.output)
