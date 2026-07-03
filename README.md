# Food Security AI System

## End-to-End Pipeline for Detecting Agricultural Land Encroachment in Egypt

An 8-step ML pipeline that compares multi-temporal Sentinel-2 satellite imagery to detect where buildings have been constructed on agricultural land — validated with the **KEMET1 BeforeAfter** Random Forest classifier trained on 300 Nile Delta sites.

---

## KEMET1 BeforeAfter Classifier — Results

Trained on 300 co-registered Sentinel-2 tile pairs (before: 2024, after: 2025) covering the Nile Delta. Detects agricultural-land encroachment using 46 spectral-change features derived from NDVI, NDBI, MNDWI, and five additional bands.

**Auto-labelling:** `pct_conv > 0.02576 OR max_cluster_ha > 3.13` → 78 positive / 222 negative sites  
**Split:** 70% train / 15% val / 15% test (stratified, seed=42)

| Metric | Value | Note |
|--------|-------|------|
| **Val AUC** | **0.9444** | Primary / conservative metric |
| Test AUC | ~1.000 | Inflated — see circularity note |
| Ablation Val AUC (44 features) | 0.8763 | Without pct_conv & pct_new |
| Val Confusion @ 0.40 | TN=29 FP=4 FN=2 TP=10 | |
| Test Confusion @ 0.40 | TN=34 FP=0 FN=0 TP=11 | |
| Decision threshold | 0.40 | Optimised for high recall |

> **Circularity note:** auto-labels were generated from `pct_conv` and `max_cluster_ha`, both of which are in the 46-feature vector. The ablation model (44 features, removing `pct_conv` and `pct_new`) gives val AUC = **0.8763** — the conservative lower bound on true generalisation.

### KEMET1 BeforeAfter — Quick Start

```powershell
# Install dependencies
pip install -r requirements.txt

# Train RF on all 300 BA sites
python train_ba_classifier.py

# Evaluate on held-out test split (reproduces dashboard)
python evaluate_ba.py

# Ablation: retrain without circular features
python ablation_no_circular.py

# Single-site inference -> HTML report
python run_inference.py --before data/KEMET1_BeforeAfter/site0_before_2024.tif `
                        --after  data/KEMET1_BeforeAfter/site0_after_2025.tif `
                        --site site0

# Batch: all 78 positive sites -> outputs/ + results/encroachment_summary.html
python batch_report.py

# Bake geocoded place names into reports (server-side, no JS loading)
python geocode_reports.py (Get-ChildItem outputs\*_report.html).FullName

# View interactive dashboards
start results\training_results.html
start results\encroachment_summary.html
```

---

## KEMET1 Tile Classifier (legacy multi-temporal)

The original classifier operates on per-tile spectral statistics across 4 time periods (T1–T4) with temporal consistency post-processing.

| Metric | Value |
|--------|-------|
| Pooled AUC (all 201 pairs) | 0.990 |
| Val AUC (post-processing) | 0.963 |
| Test AUC (post-processing) | 0.988 |
| Recall | 1.000 |
| F2 score | 0.909 |
| False Positives (val / test) | 3 per split |

```bash
python train_classifier.py --no-cv
python evaluate.py
python predict.py --t1 T1.tif --t2 T2.tif
```

---

## Full Pipeline — 8 Steps

| # | Step | Model / Tool | Output |
|---|------|-------------|--------|
| 01 | Data Acquisition | Sentinel-2 via GEE | GeoTIFFs (T1, T2) |
| 02 | Cloud Detection | U-Net + ResNet34 (38-Cloud) | Binary cloud mask |
| 03 | Cloud Removal | OpenCV Telea inpainting | Clean GeoTIFF |
| 04 | Spectral Indices | NumPy (NDVI, NDBI, MNDWI) | Index rasters |
| 05 | Change Detection | ChangeFormer (LEVIR-CD) | Binary change map |
| 06 | Agriculture Segmentation | SegFormer-B4 (ADE20K) | Agriculture mask |
| 07 | Building Detection | SAM (SpaceNet v2) | Building footprints |
| 08 | Final Output | Spatial intersection | Colored map + JSON report |

**Color coding:** Red = buildings on farmland | Yellow = changed vegetation | Green = stable agriculture

```bash
# Run full pipeline on a tile pair
python pipeline.py full --before T1.tif --after T2.tif

# Run BeforeAfter inference only
python pipeline.py beforeafter --before site0_before_2024.tif \
                               --after  site0_after_2025.tif \
                               --site-name site0
```

---

## Repository Structure

```
models/
  ba_rf_model.pkl          # KEMET1 BeforeAfter RF (sklearn 1.7.2)
  ba_rf_ablation.pkl       # Ablation model (44 features)
  encroachment_classifier_rf.pkl  # Legacy tile classifier
data/
  KEMET1_BeforeAfter/      # 300 site pairs (site{N}_before/after_*.tif)
outputs/                   # Generated per-site HTML reports (gitignored)
src/                       # Pipeline step modules (steps 01-08)
results/
  training_results.html    # Interactive evaluation dashboard
  encroachment_summary.html# Master Egypt map of all detected sites
  ablation_results.json    # Ablation study output
  viz/                     # Sample site imagery and change maps
docs/
  KEMET1_Final_Report.pdf  # Full GP report
  GP_Presentation.pptx     # Defence presentation (13 slides)
  datasets_reference.md    # Dataset sources and notes
```

---

## Requirements

```
rasterio, numpy, scikit-learn, scipy, torch, transformers
fpdf2, requests, folium, leaflet (via CDN)
```

See `requirements.txt` for pinned versions.
