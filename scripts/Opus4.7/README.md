# GD&T extraction via Claude Opus 4.7

Pipeline that reads Siemens-style engineering drawings and extracts all
GD&T (Geometric Dimensioning & Tolerancing) annotations using the Claude
Opus 4.7 vision model.

## Why a separate model and not the fine-tuned Qwen?

GD&T symbols are small, dense, and use specialised iconography. Reading
them reliably needs strong OCR + symbol understanding without per-class
training data. Opus 4.7 handles this zero-shot; fine-tuning the existing
detector would require labelled GD&T data we don't have.

## Three annotation categories

| Category | Examples |
|---|---|
| **Surface roughness** | Ra 1.6 μm (general/rest), Rz 25 on bore |
| **Geometric tolerances** | flatness 0.05, position ⌀0.1 MMC /A/B |
| **Datums** | datum A (bottom face), datum B (left edge) |

## Install

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-v1-...
```

The script uses OpenRouter as the routing layer (OpenAI-compatible API).

## Usage

### Smoke test — single image
```bash
python extract_gdt.py \
    --image path/to/drawing.png \
    --output gdt_single.json
```

### Directory of images
```bash
python extract_gdt.py \
    --image-dir /path/to/drawings \
    --glob "*.png" \
    --output gdt_all.json \
    --limit 5     # cost-controlled smoke test
```

### Augment an inference results JSON
```bash
python extract_gdt.py \
    --input results_788_view_7feats_v3_improve_ckpt350.json \
    --image-resolve-dir /path/to/drawing/images \
    --output results_with_gdt.json
```

## Output schema

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
        "context": "general (rest), above title block"
      },
      {
        "category": "roughness",
        "parameter": "Ra",
        "value": "1.6",
        "unit": "μm",
        "process": "machined",
        "scope": "surface",
        "context": "top flange milled face"
      },
      {
        "category": "geometric_tolerance",
        "symbol": "flatness",
        "tolerance_value": "0.05",
        "tolerance_unit": "mm",
        "datum_references": [],
        "material_condition": null,
        "context": "top mounting face"
      },
      {
        "category": "geometric_tolerance",
        "symbol": "position",
        "tolerance_value": "0.1",
        "tolerance_unit": "mm",
        "datum_references": ["A", "B"],
        "material_condition": "MMC",
        "context": "4× Ø8 bolt holes"
      },
      {
        "category": "datum",
        "label": "A",
        "context": "bottom face"
      }
    ]
  }
]
```

## Cost / runtime

- Each drawing: ~5K input tokens + ~400 output tokens ≈ **$0.15–0.40**
  (GD&T output is larger than roughness-only).
- Sequential calls, no parallelism. For 80 drawings: ~15 min, ~$20–30.
- Script is resumable: re-running with the same `--output` skips done images.

## Model choice

Default is `anthropic/claude-opus-4-7` via OpenRouter. Override with:
```bash
python extract_gdt.py --model anthropic/claude-opus-4-7 ...
```
Find current slugs at https://openrouter.ai/models.
Sonnet is cheaper but less reliable on dense industrial diagrams.

## Geometric tolerance symbols (reference)

| Symbol name | ISO character |
|---|---|
| flatness | ⏥ |
| straightness | — |
| circularity | ○ |
| cylindricity | ⌭ |
| angularity | ∠ |
| perpendicularity | ⊥ |
| parallelism | ∥ |
| position | ⊕ |
| concentricity | ◎ |
| symmetry | ≡ |
| profile_of_line | ⌒ |
| profile_of_surface | ⌓ |
| circular_runout | ↗ |
| total_runout | ⌰ |
