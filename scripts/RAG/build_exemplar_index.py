"""
Build FAISS index of training image embeddings for RAG exemplar retrieval.

Encodes all training images using CLIP or DINOv2, stores embeddings in FAISS,
and saves metadata (image name, GT annotations per step) as JSONL.

Usage:
  # Using existing training JSONL
  python build_exemplar_index.py \
    --gt_json /workspace/data/5k_15feats_with_view_v2.json \
    --image_dir /workspace/data/5k \
    --output_dir /workspace/data/rag_index \
    --encoder openai/clip-vit-large-patch14

  # Quick test with subset
  python build_exemplar_index.py \
    --gt_json /workspace/data/5k_15feats_with_view_v2.json \
    --image_dir /workspace/data/5k \
    --output_dir /workspace/data/rag_index \
    --max_images 100
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VIEW_CATEGORIES = {
    "Title Block", "Notes", "Revision Table", "Bill of Materials",
    "Orthographic Projection - Front View",
    "Orthographic Projection - Top View",
    "Orthographic Projection - Left View",
    "Orthographic Projection - Right View",
    "Isometric View", "Auxiliary View", "Section View",
}

FEATURE_CATEGORIES = {
    "Threaded Hole", "Threaded Hole Group",
    "Round Hole", "Round Hole Group",
    "Pin Hole", "Pin Hole Group",
    "Counterbore Hole", "Counterbore Hole Group",
    "Rectangular Hole",
    "Fillet", "Fillet Group",
    "Chamfer", "Chamfer Group",
    "Threaded Shaft",
}


def normalize_bbox(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(1000, int(round(x1 / img_w * 1000)))),
        max(0, min(1000, int(round(y1 / img_h * 1000)))),
        max(0, min(1000, int(round(x2 / img_w * 1000)))),
        max(0, min(1000, int(round(y2 / img_h * 1000)))),
    ]


def load_encoder(encoder_name, device):
    """Load vision encoder (CLIP or DINOv2)."""
    import torch

    if "clip" in encoder_name.lower():
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(encoder_name).to(device)
        processor = CLIPProcessor.from_pretrained(encoder_name)
        mode = "clip"
    elif "dino" in encoder_name.lower():
        from transformers import AutoModel, AutoImageProcessor
        model = AutoModel.from_pretrained(encoder_name).to(device)
        processor = AutoImageProcessor.from_pretrained(encoder_name)
        mode = "dino"
    else:
        from transformers import AutoModel, AutoImageProcessor
        model = AutoModel.from_pretrained(encoder_name).to(device)
        processor = AutoImageProcessor.from_pretrained(encoder_name)
        mode = "generic"

    model.eval()
    return model, processor, mode


def encode_image(model, processor, mode, image_path, device):
    """Encode a single image to normalized embedding."""
    import torch
    from PIL import Image

    img = Image.open(image_path).convert("RGB")

    if mode == "clip":
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
    else:
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0]

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32").flatten()


def main():
    parser = argparse.ArgumentParser(description="Build RAG exemplar FAISS index")
    parser.add_argument("--gt_json", required=True,
                        help="Path to annotation JSON")
    parser.add_argument("--image_dir", required=True,
                        help="Directory with training images")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for index and metadata")
    parser.add_argument("--encoder", default="openai/clip-vit-large-patch14",
                        help="Vision encoder model name")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for encoding (1 is safest)")
    parser.add_argument("--max_images", type=int, default=0,
                        help="Limit to N images (0 = all)")
    args = parser.parse_args()

    import torch
    import faiss
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}, Encoder: {args.encoder}")

    # Load GT
    with open(args.gt_json) as f:
        gt_data = json.load(f)
    logger.info(f"Loaded {len(gt_data)} annotations")

    image_dir = Path(args.image_dir)
    available = set(f for f in os.listdir(image_dir) if f.endswith(".png"))

    valid_items = [item for item in gt_data if item["dataitem_name"] in available]
    if args.max_images > 0:
        valid_items = valid_items[:args.max_images]
    logger.info(f"Processing {len(valid_items)} images")

    # Load encoder
    model, processor, mode = load_encoder(args.encoder, device)

    # Encode all images
    embeddings = []
    metadata = []
    t_start = time.time()

    for i, item in enumerate(valid_items):
        name = item["dataitem_name"]
        img_path = str(image_dir / name)

        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            logger.warning(f"Cannot open {name}: {e}")
            continue

        # Encode
        try:
            emb = encode_image(model, processor, mode, img_path, device)
            embeddings.append(emb)
        except Exception as e:
            logger.warning(f"Encoding failed for {name}: {e}")
            continue

        # Extract GT per step
        views, features = [], []
        categories = set()
        for task in item.get("tasks", []):
            for tv in task.get("task_values", []):
                val = tv["value"]
                label = val["label"]
                bbox = normalize_bbox(tv["bbox"], img_w, img_h)
                categories.add(label)
                if label in VIEW_CATEGORIES:
                    views.append({"category": label, "bbox": bbox})
                elif label in FEATURE_CATEGORIES:
                    features.append({
                        "category": label,
                        "size": val.get("size", ""),
                        "bbox": bbox,
                    })

        metadata.append({
            "image_name": name,
            "categories": sorted(categories),
            "gt_step1": views,
            "gt_step2": features,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(valid_items) - i - 1) / rate
            logger.info(f"[{i+1}/{len(valid_items)}] {rate:.1f} img/s, ETA {int(eta)}s")

    # Build FAISS index
    embeddings_np = np.stack(embeddings)
    dim = embeddings_np.shape[1]
    logger.info(f"Building FAISS index: {len(embeddings)} vectors, dim={dim}")

    # Use IndexFlatIP (inner product = cosine similarity for normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_np)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "index.faiss"
    metadata_path = output_dir / "metadata.jsonl"

    faiss.write_index(index, str(index_path))
    logger.info(f"Saved FAISS index: {index_path} ({index.ntotal} vectors)")

    with open(metadata_path, "w", encoding="utf-8") as f:
        for meta in metadata:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    logger.info(f"Saved metadata: {metadata_path}")

    elapsed = time.time() - t_start
    logger.info(f"Done in {elapsed:.1f}s")

    # Print config snippet
    print(f"\n# Add to rag_config.yaml:")
    print(f"exemplar_index: {index_path}")
    print(f"exemplar_metadata: {metadata_path}")
    print(f"encoder: {args.encoder}")


if __name__ == "__main__":
    main()
