"""
Adaptive RAG retriever for engineering drawing analysis.

Three knowledge stores (all optional):
  1. Visual exemplars  — FAISS index of training image embeddings
  2. Category rules    — Markdown files with disambiguation rules
  3. Standards         — Engineering standards documents (ISO/ASME)

Each store is enabled/disabled via rag_config.yaml.
The retriever degrades gracefully — if a store is unavailable, it's skipped.

Usage:
  from rag_retriever import DrawingRAG

  rag = DrawingRAG("rag_config.yaml")
  context = rag.retrieve(image_path, step=1)
  context = rag.retrieve(image_path, step=2, model_prediction=[...])
"""

import json
import logging
import os
import re
from pathlib import Path

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

logger = logging.getLogger(__name__)


# =====================================================================
# Store 1: Visual Exemplars (FAISS + CLIP/DINOv2)
# =====================================================================
class ExemplarStore:
    """
    Retrieves visually similar training images and their GT annotations.
    Uses CLIP or DINOv2 embeddings indexed in FAISS.
    """

    def __init__(self, index_path, metadata_path, encoder_name, top_k=3):
        import faiss
        import numpy as np

        self.top_k = top_k
        self.encoder_name = encoder_name

        logger.info(f"Loading FAISS index: {index_path}")
        self.index = faiss.read_index(str(index_path))

        logger.info(f"Loading metadata: {metadata_path}")
        self.metadata = []
        with open(metadata_path) as f:
            for line in f:
                self.metadata.append(json.loads(line.strip()))

        self._encoder = None
        self._processor = None
        self._device = None

    def _load_encoder(self):
        """Lazy-load the vision encoder."""
        if self._encoder is not None:
            return

        import torch

        if "clip" in self.encoder_name.lower():
            from transformers import CLIPModel, CLIPProcessor
            self._encoder = CLIPModel.from_pretrained(self.encoder_name)
            self._processor = CLIPProcessor.from_pretrained(self.encoder_name)
        elif "dino" in self.encoder_name.lower():
            from transformers import AutoModel, AutoImageProcessor
            self._encoder = AutoModel.from_pretrained(self.encoder_name)
            self._processor = AutoImageProcessor.from_pretrained(self.encoder_name)
        else:
            from transformers import AutoModel, AutoImageProcessor
            self._encoder = AutoModel.from_pretrained(self.encoder_name)
            self._processor = AutoImageProcessor.from_pretrained(self.encoder_name)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._encoder = self._encoder.to(self._device)
        self._encoder.eval()

    def _encode_image(self, image_path):
        """Encode a single image to embedding vector."""
        import torch
        import numpy as np
        from PIL import Image

        self._load_encoder()

        img = Image.open(image_path).convert("RGB")

        if "clip" in self.encoder_name.lower():
            inputs = self._processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self._encoder.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        else:
            inputs = self._processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._encoder(**inputs)
            emb = outputs.last_hidden_state[:, 0]  # CLS token
            emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb.cpu().numpy().astype("float32")

    def query(self, image_path, step=None, top_k=None):
        """
        Find top-k visually similar training images.

        Returns list of metadata dicts with keys:
          image_name, gt_step1, gt_step2, categories, distance
        """
        k = top_k or self.top_k
        emb = self._encode_image(image_path)

        distances, indices = self.index.search(emb, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = dict(self.metadata[idx])
            meta["distance"] = float(dist)
            results.append(meta)

        return results


# =====================================================================
# Store 2: Category Rules (keyword-based retrieval)
# =====================================================================
class RuleStore:
    """
    Retrieves disambiguation rules from markdown files.

    Rules are organized by category/topic. Each .md file has a YAML
    frontmatter with metadata for targeted retrieval.

    File format:
      ---
      categories: [Chamfer, Fillet, Chamfer Group, Fillet Group]
      step: 2
      keywords: [chamfer, fillet, R, C, angle, 45]
      ---
      Rule content here...
    """

    def __init__(self, rules_dir, retrieval="keyword"):
        self.rules_dir = Path(rules_dir)
        self.retrieval = retrieval
        self.rules = []
        self._load_rules()

    def _load_rules(self):
        if not self.rules_dir.exists():
            logger.warning(f"Rules directory not found: {self.rules_dir}")
            return

        for md_file in sorted(self.rules_dir.glob("*.md")):
            content = md_file.read_text(encoding="utf-8")

            # Parse frontmatter
            meta = {"categories": [], "step": 0, "keywords": []}
            fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
            if fm_match:
                if _HAS_YAML:
                    try:
                        meta.update(yaml.safe_load(fm_match.group(1)) or {})
                    except yaml.YAMLError:
                        pass
                body = content[fm_match.end():]
            else:
                body = content

            self.rules.append({
                "file": md_file.name,
                "categories": set(meta.get("categories", [])),
                "step": meta.get("step", 0),
                "keywords": [k.lower() for k in meta.get("keywords", [])],
                "content": body.strip(),
            })

        logger.info(f"Loaded {len(self.rules)} rule files from {self.rules_dir}")

    def query_by_categories(self, categories: set, max_rules=5) -> list:
        """Retrieve rules relevant to the given categories."""
        scored = []
        for rule in self.rules:
            overlap = len(categories & rule["categories"])
            if overlap > 0:
                scored.append((overlap, rule["content"]))

        scored.sort(key=lambda x: -x[0])
        return [content for _, content in scored[:max_rules]]

    def query_by_step(self, step: int, max_rules=3) -> list:
        """Retrieve rules relevant to the given step."""
        results = []
        for rule in self.rules:
            if rule["step"] == step or rule["step"] == 0:
                results.append(rule["content"])
        return results[:max_rules]

    def query_by_keywords(self, text: str, max_rules=3) -> list:
        """Keyword matching against rule keywords."""
        text_lower = text.lower()
        scored = []
        for rule in self.rules:
            score = sum(1 for kw in rule["keywords"] if kw in text_lower)
            if score > 0:
                scored.append((score, rule["content"]))
        scored.sort(key=lambda x: -x[0])
        return [content for _, content in scored[:max_rules]]


# =====================================================================
# Store 3: Standards Knowledge (dense retrieval)
# =====================================================================
class StandardsStore:
    """
    Dense retrieval over engineering standards documents.

    Chunks markdown documents and retrieves via sentence-transformer
    embeddings. Falls back to keyword search if sentence-transformers
    is unavailable.
    """

    def __init__(self, docs_dir, chunk_size=512, overlap=50):
        self.docs_dir = Path(docs_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []
        self._embeddings = None
        self._model = None
        self._load_documents()

    def _load_documents(self):
        if not self.docs_dir.exists():
            logger.warning(f"Standards directory not found: {self.docs_dir}")
            return

        for md_file in sorted(self.docs_dir.glob("*.md")):
            content = md_file.read_text(encoding="utf-8")
            # Remove frontmatter
            content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)

            # Chunk by paragraphs first, then by size
            paragraphs = re.split(r"\n\n+", content)
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                    self.chunks.append({
                        "source": md_file.name,
                        "content": current_chunk.strip(),
                    })
                    # Overlap: keep last N chars
                    current_chunk = current_chunk[-self.overlap:] + "\n\n" + para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para

            if current_chunk.strip():
                self.chunks.append({
                    "source": md_file.name,
                    "content": current_chunk.strip(),
                })

        logger.info(f"Loaded {len(self.chunks)} chunks from {self.docs_dir}")

    def _ensure_embeddings(self):
        if self._embeddings is not None:
            return True

        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            logger.warning("sentence-transformers not installed, "
                           "falling back to keyword search for standards")
            return False

        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [c["content"] for c in self.chunks]
        self._embeddings = self._model.encode(texts, normalize_embeddings=True)
        return True

    def query(self, query_text, top_k=2):
        """Retrieve most relevant standard chunks."""
        if not self.chunks:
            return []

        if self._ensure_embeddings():
            import numpy as np
            q_emb = self._model.encode([query_text], normalize_embeddings=True)
            scores = np.dot(self._embeddings, q_emb.T).flatten()
            top_idx = scores.argsort()[-top_k:][::-1]
            return [self.chunks[i]["content"] for i in top_idx if scores[i] > 0.2]
        else:
            # Keyword fallback
            query_words = set(query_text.lower().split())
            scored = []
            for chunk in self.chunks:
                words = set(chunk["content"].lower().split())
                overlap = len(query_words & words)
                if overlap > 0:
                    scored.append((overlap, chunk["content"]))
            scored.sort(key=lambda x: -x[0])
            return [c for _, c in scored[:top_k]]


# =====================================================================
# Main RAG Retriever
# =====================================================================
class DrawingRAG:
    """
    Adaptive RAG for engineering drawing analysis.

    Combines three knowledge stores. Each is optional and configured
    via rag_config.yaml. Degrades gracefully if stores are missing.
    """

    def __init__(self, config=None, config_path=None):
        if config is None:
            if config_path is None:
                config_path = Path(__file__).parent / "rag_config.yaml"
            if _HAS_YAML and Path(config_path).exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f) or {}
            else:
                config = {}

        self.config = config
        self.stores = {}
        self.max_context_tokens = config.get("max_context_tokens", 1024)
        self.context_position = config.get("context_position", "prepend")

        # Store 1: Visual exemplars
        index_path = config.get("exemplar_index")
        metadata_path = config.get("exemplar_metadata")
        if index_path and metadata_path and Path(index_path).exists():
            try:
                self.stores["exemplars"] = ExemplarStore(
                    index_path=index_path,
                    metadata_path=metadata_path,
                    encoder_name=config.get("encoder", "openai/clip-vit-large-patch14"),
                    top_k=config.get("exemplar_top_k", 3),
                )
            except Exception as e:
                logger.warning(f"Failed to load exemplar store: {e}")

        # Store 2: Category rules
        rules_dir = config.get("rules_dir")
        if rules_dir:
            # Resolve relative to config file location
            rules_path = Path(rules_dir)
            if not rules_path.is_absolute():
                rules_path = Path(__file__).parent / rules_path
            if rules_path.exists():
                self.stores["rules"] = RuleStore(
                    rules_dir=rules_path,
                    retrieval=config.get("rule_retrieval", "keyword"),
                )

        # Store 3: Standards
        standards_dir = config.get("standards_dir")
        if standards_dir and Path(standards_dir).exists():
            self.stores["standards"] = StandardsStore(
                docs_dir=standards_dir,
                chunk_size=config.get("standards_chunk_size", 512),
            )

        active = list(self.stores.keys())
        logger.info(f"DrawingRAG initialized with stores: {active or ['none']}")

    def retrieve(self, image_path: str, step: int,
                 model_prediction: list = None) -> str:
        """
        Build retrieval context for a given image and step.

        Args:
            image_path: path to the query image
            step: 1 (views/layout) or 2 (features)
            model_prediction: optional first-pass predictions for
                              targeted rule retrieval

        Returns:
            context string to inject into the prompt
        """
        contexts = []

        # 1. Visual exemplars
        if "exemplars" in self.stores:
            try:
                similar = self.stores["exemplars"].query(image_path, step=step)
                if similar:
                    formatted = self._format_exemplars(similar, step)
                    if formatted:
                        contexts.append(formatted)
            except Exception as e:
                logger.warning(f"Exemplar retrieval failed: {e}")

        # 2. Category rules
        if "rules" in self.stores:
            if model_prediction:
                categories = {d.get("category", "") for d in model_prediction
                              if d.get("category")}
                rules = self.stores["rules"].query_by_categories(categories)
            else:
                rules = self.stores["rules"].query_by_step(step)

            if rules:
                contexts.append(self._format_rules(rules))

        # 3. Standards
        if "standards" in self.stores:
            query = ("projection views layout third-angle" if step == 1
                     else "hole chamfer fillet dimensions tolerance")
            std = self.stores["standards"].query(
                query, top_k=self.config.get("standards_top_k", 2)
            )
            if std:
                contexts.append(self._format_standards(std))

        # Enforce context budget
        result = "\n\n".join(contexts) if contexts else ""
        max_chars = self.max_context_tokens * 4  # rough char estimate
        if len(result) > max_chars:
            result = result[:max_chars] + "\n..."

        return result

    def augment_prompt(self, prompt: str, context: str) -> str:
        """Inject retrieved context into a prompt."""
        if not context:
            return prompt
        if self.context_position == "append":
            return prompt + "\n\n" + context
        return context + "\n\n" + prompt

    def _format_exemplars(self, exemplars, step):
        lines = ["参考示例（类似图纸的标注结果）："]
        max_chars = self.config.get("exemplar_max_chars", 2000)
        total = 0

        for ex in exemplars:
            key = "gt_step1" if step == 1 else "gt_step2"
            gt = ex.get(key, ex.get("gt", []))
            entry = f"\n示例 {ex.get('image_name', '?')}:\n" + \
                    json.dumps(gt, ensure_ascii=False)
            if total + len(entry) > max_chars:
                break
            lines.append(entry)
            total += len(entry)

        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_rules(self, rules):
        return "相关判别规则：\n\n" + "\n\n---\n\n".join(rules)

    def _format_standards(self, chunks):
        return "相关工程标准：\n\n" + "\n\n".join(chunks)
