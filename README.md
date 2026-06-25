# Food Security AI System

## End-to-End Pipeline for Detecting Buildings on Agricultural Land

An 8-step ML pipeline that compares multi-temporal satellite imagery to detect where buildings have been constructed on agricultural land — combined with a trained Random Forest tile classifier validated on the KEMET1 Egypt dataset.

---

## KEMET1 Classifier — Final Results

The KEMET1 dataset covers 75 tile locations × 4 time periods across Egypt's agricultural zones. A Random Forest classifier is trained on per-tile spectral index statistics to detect encroachment.

| Metric | Value |
|--------|-------|
| **Pooled AUC** (all 201 pairs) | **0.990** |
| Val AUC (post-processing) | 0.963 |
| Test AUC (post-processing) | 0.988 |
| Recall | **1.000** — zero missed encroachments |
| Precision | 0.667 |
| F2 score | 0.909 |
| False Positives (val / test) | **3 per split** |
| False Negatives | **0** |
| Threshold (F2-optimal) | 0.29 |

### Key improvements over baseline

| Version | Test AUC | FP (test) | FN |
|---------|----------|-----------|----|
| v3 — RF depth=10 (baseline) | 0.747 | 22 | 0 |
| v4 + `t1_is_pos` feature | 0.907 | 10 | 0 |
| v4 + temporal consistency | **0.988** | **3** | **0** |

`t1_is_pos` — binary prior-label feature (#1 importance at 0.077) that flags pos→pos tile pairs, eliminating the most common FP source.  
Temporal consistency — majority filter: if ≥2 of 3 consecutive pairs for a tile score above threshold, all scores are multiplied by 0.6 (seasonal dampening).

---

## Quick Start — KEMET1 Classifier

```bash
# Install dependencies
pip install -r requirements.txt

# Train the classifier (skip cross-validation for speed)
python train_classifier.py --no-cv

# Evaluate on all splits with temporal consistency
python evaluate.py

# Inference — single pair
python predict.py \
    --t1 data/KEMET1_split/val/T1_2022_tile_4_neg.tif \
    --t2 data/KEMET1_split/val/T2_2023_tile_4_pos.tif

# Inference — all 3 pairs for a tile (enables temporal consistency)
python predict.py \
    --t1 T1_2022_tile_39_neg.tif --t2 T2_2023_tile_39_neg.tif \
    --t1 T2_2023_tile_39_neg.tif --t2 T3_2024_tile_39_neg.tif \
    --t1 T3_2024_tile_39_neg.tif --t2 T4_2025_tile_39_neg.tif

# JSON output for downstream processing
python predict.py --t1 T1.tif --t2 T2.tif --json

# View interactive dashboard
open training_results.html
```

### predict.py options

```
--t1 PATH          "Before" image path (repeat for multiple pairs)
--t2 PATH          "After"  image path (repeat, must match --t1 count)
--model PATH       Model bundle .pkl (default: weights/encroachment_classifier_rf.pkl)
--t1-is-pos        Force t1_is_pos=True; default: inferred from filename (*_pos.tif)
--no-consistency   Disable temporal consistency dampening
--json             Machine-readable JSON output
```

Exit codes: `0` = no encroachment, `1` = encroachment detected, `2` = error.

---

## Full Pipeline — 8 Steps

| # | Step | Model / Tool | Output |
|---|------|-------------|--------|
| 01 | Data Acquisition | Google Earth Engine | GeoTIFF (T1 & T2) |
| 02 | Cloud Detection | U-Net + ResNet34 | Cloud probability + binary mask |
| 03 | Cloud Removal | OpenCV Telea | Cloud-free GeoTIFFs |
| 04 | Spectral Indices | NumPy | NDVI, NDBI, MNDWI, SAVI, BSI, NDWI |
| 05 | Change Detection | ChangeFormer **or RF (KEMET1 mode)** | Binary change map / tile score |
| 06 | Agriculture Seg. | SegFormer-B4 | Farmland mask |
| 07 | Building Detection | SAM + YOLOv8-seg | Building mask + polygons |
| 08 | Final Output | OpenCV + GeoPandas | Colored PNG, GeoTIFF, GeoJSON, report |

### Quick Start — Full Pipeline

```bash
# Run with test data (no model weights needed for fallback mode)
python run.py --test

# Run with real satellite images
python run.py --t1 data/raw/T1/image.tif --t2 data/raw/T2/image.tif

# Run with KEMET1 pre-computed tiles (uses RF classifier)
python run.py --kemet1 --t1 T1_tile.tif --t2 T2_tile.tif

# Download from Google Earth Engine
python run.py --gee
```

### Output Color Legend

- **RED**: Buildings detected on farmland (encroachment)
- **YELLOW**: Vegetation change (no building)
- **GREEN**: Stable agricultural land

### Model Weights

Download pretrained weights to the `weights/` directory:

| Weight | Source | Filename |
|--------|--------|----------|
| KEMET1 RF classifier | Trained locally (`python train_classifier.py`) | `weights/encroachment_classifier_rf.pkl` |
| ChangeFormer | [wgcban/ChangeFormer](https://github.com/wgcban/ChangeFormer) | `weights/ChangeFormer_LEVIR.pth` |
| SAM (vit_b) | [fbaipublicfiles](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | `weights/sam_vit_b_01ec64.pth` |
| SegFormer-B4 | Auto-downloads from HuggingFace | — |
| YOLOv8-seg | Auto-downloads from Ultralytics | — |

---

## Project Structure

```
├── config/settings.py           # All configurations
├── data/
│   └── KEMET1_split/            # 75 tiles × 4 time periods
│       ├── train/               # 45 tiles (60%)
│       ├── val/                 # 11 tiles (15%)
│       ├── test/                # 11 tiles (15%)
│       └── unlabelled/          # 8 tiles (10%)
├── src/
│   ├── utils/                   # Geo I/O, tiling, logging
│   ├── step_01_data_acquisition/
│   ├── step_02_cloud_detection/
│   ├── step_03_cloud_removal/
│   ├── step_04_spectral_indices/
│   ├── step_05_change_detection/
│   ├── step_06_agriculture_segmentation/
│   ├── step_07_building_detection/
│   ├── step_08_final_output/
│   └── temporal/                # Year-over-year comparison logic
├── weights/
│   └── encroachment_classifier_rf.pkl  # Trained RF bundle
├── train_classifier.py          # KEMET1 RF training
├── evaluate.py                  # Per-tile evaluation with temporal consistency
├── predict.py                   # Standalone inference script
├── training_results.html        # Interactive results dashboard
├── evaluation_report.html       # Per-tile prediction report
├── pipeline.py                  # 8-step pipeline orchestrator
├── run.py                       # CLI entry point
└── requirements.txt
```

### Recommended Datasets

| Dataset | Use Case | Source |
|---------|----------|--------|
| KEMET1 | Egypt tile classification (included) | — |
| LEVIR-CD | Change detection training | [justchenhao/LEVIR](https://justchenhao.github.io/LEVIR/) |
| CloudSEN12 | Cloud detection training | [Zenodo](https://zenodo.org/record/7431205) |
| SpaceNet v2 | Building detection | [spacenet.ai](https://spacenet.ai/) |
| ADE20K | Agriculture segmentation | HuggingFace (auto) |
