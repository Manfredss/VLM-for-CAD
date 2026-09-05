# OCR — Extract text from detected regions (Title Block, Notes, ...)

After view-detection inference, each record has bboxes for `Title Block`, `Notes`,
the orthographic projection views, etc. This pipeline:

1. Crops each image to the predicted bbox (with small padding)
2. Runs **PaddleOCR** on the crop (CN + EN mixed-language, with angle classifier)
3. Appends the raw line-level OCR output to the record
4. (Optional) Parses the Title Block OCR into structured fields like part number,
   material, scale, drawing number

OCR runs **on CPU** by default — no GPU needed. ~1-3 s per Title Block crop on a
modern laptop.

## Install

PaddleOCR has CPU and GPU paddlepaddle wheels. Pick one:

```bash
# CPU (recommended for OCR):
pip install paddlepaddle==2.6.* paddleocr==2.7.*

# GPU (only if you want it on a CUDA box):
pip install paddlepaddle-gpu==2.6.* paddleocr==2.7.*
```

First run will auto-download the OCR models (~50 MB total) into `~/.paddleocr/`.
For Chinese + English, use `lang='ch'` (the 'ch' bundle includes English).

## Usage

### 1. Raw OCR extraction (line-level text per region)

```bash
python scripts/OCR/ocr_extract.py \
    --input results_788_view_7feats_3a100.json \
    --image-dir scripts/qwn3.5-27b_788_silver_plate_bend/simens_7feats \
    --output results_788_view_7feats_3a100_with_ocr.json \
    --regions "Title Block" "Notes"
```

Per record, this adds `ocr_title_block` and `ocr_notes` fields, each a list of
`{text, conf, bbox}` from PaddleOCR.

Other knobs:
- `--padding 0.05` (5 % bbox padding on each side)
- `--upscale 2` (upscale crop 2× before OCR; helps on small Title Blocks)
- `--lang ch` (default; CN + EN mixed)
- `--limit 10` (test on first 10 records before running on the full set)

### 2. Structured field extraction from Title Block (optional)

After raw OCR, parse Title Block lines into named fields:

```bash
python scripts/OCR/parse_title_block.py \
    --input results_788_view_7feats_3a100_with_ocr.json \
    --output results_788_view_7feats_3a100_titleblock_fields.json
```

This adds a `title_block_fields` dict per record:
```json
"title_block_fields": {
  "part_number": "A7E0018074050",
  "name": "...",
  "material": "...",
  "scale": "1:1",
  "date": "...",
  "drawn_by": "...",
  "revision": "..."
}
```

The parser uses heuristics — looks for known label tokens (零件号 / Part No, 材料 / Material, 比例 / Scale, etc.) and grabs the adjacent value text. Coverage depends on how clean the Title Block layout is. For drawings where the heuristics fail, you can:
- Tune the label list in `parse_title_block.py`
- Or feed the raw `ocr_title_block` text into an LLM (Qwen, Claude, GPT) with a
  structured-extraction prompt — much more robust to layout variation but slower

## How it composes with the detection pipeline

```
inference_swift.py
  → results_..._test_results.json   (with bbox per region)
       │
       ▼
ocr_extract.py
  → results_..._with_ocr.json       (+ ocr_title_block, +ocr_notes per record)
       │
       ▼
parse_title_block.py (optional)
  → results_..._titleblock_fields.json   (+ structured fields per record)
```

## Caveats

- OCR quality depends on **bbox accuracy**. If the model's predicted Title Block
  bbox is slightly off (e.g. clips the right edge), text in that area is missed.
  `--padding 0.05` helps absorb small bbox errors.
- Engineering drawings often have **stamps and watermarks overlapping** the Title
  Block. Those get OCR'd too — filter post-hoc by font size or position if needed.
- For non-Title-Block regions (Notes, view areas), OCR works fine but the
  structured-extraction step in `parse_title_block.py` is Title-Block-specific.
- PaddleOCR's `lang='ch'` model handles English but not German. If you have
  German text in stamps, switch to `lang='german'` for those (separate run).
