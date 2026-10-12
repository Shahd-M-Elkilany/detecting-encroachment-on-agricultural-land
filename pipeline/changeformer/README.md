# ChangeFormer on the KEMET1 sites

The 8-step pipeline was designed with ChangeFormer, a Siamese transformer for building change detection, as its
Step 05. This folder tests that design on the project's 300 sites with the published ChangeFormerV6 checkpoint
trained on LEVIR-CD (`best_ckpt.pt`, change-class F1 ≈ 0.90 on the LEVIR-CD test set).

![Sentinel-2 pairs and ChangeFormer's change probability for three test sites](../../docs/figures/changeformer_examples.png)

## What was run

1. **Colour images.** ChangeFormer takes RGB pairs, but the KEMET1 tiles carry six spectral indices.
   [`download_s2.py`](download_s2.py) rebuilt the composites the tiles were made from (Sentinel-2 L2A, 1 July to
   15 September, scenes under 20 % cloud, SCL cloud mask, median) on the exact pixel grid of every tile: 600
   five-band GeoTIFFs. NDVI recomputed from them correlates with the original tiles at a median r = 0.95 (5th
   percentile 0.81, [`results/s2_download_log.csv`](../../results/s2_download_log.csv)); the residual difference
   comes from Google's reprocessing of the Sentinel-2 archive since the tiles were exported, not from misalignment
   (on the site checked, shifting the grid by a pixel in any direction lowers the correlation).
2. **Checkpoint check.** On the LEVIR-CD sample pairs shipped with the ChangeFormer code, the checkpoint scores F1
   0.912 against the labels and reproduces the authors' published predictions exactly (F1 1.000), so the model
   and pre-processing are loaded correctly.
3. **Inference** with [`run_changeformer.py`](run_changeformer.py) on every pair: 2.62 km tiles resized to
   ChangeFormer's 256 × 256 input (x1), and upsampled 2× (all 300 sites) and 4× (the 45 test sites; one site
   takes about a minute on a laptop CPU) so that fields and buildings cover more pixels, closer to LEVIR-CD's
   0.5 m resolution. Each site is scored by its mean change probability and by the changed farmland area.
4. **Pipeline with ChangeFormer in Step 05**: `evaluate_pipeline.py score --changeformer ...` replaces the
   spectral change rule with ChangeFormer's map and keeps Steps 06 to 08.

Everything is scored against the same labels, development sites and test sites as the other models by
[`../compare_models.py`](../compare_models.py) ([`results/model_comparison.json`](../../results/model_comparison.json)).

## Result

| ChangeFormer configuration | Development AUC (255 sites) | Test AUC (45 sites) |
| --- | ---: | ---: |
| x1, best of four input / score choices (chosen on development) | 0.54 | 0.55 (95 % CI 0.38 to 0.73) |
| x2, mean change probability | 0.46 | |
| x4, mean change probability, fixed in advance | | 0.51 (95 % CI 0.31 to 0.72) |
| 8-step pipeline with ChangeFormer in Step 05 | 0.50 | 0.49 |

ChangeFormer does not separate encroached from intact sites in any configuration. At native resolution it marks
pixels as changed at only 14 of the 300 sites, and inside the pipeline it produces red regions at 6.

The reason is the domain gap, not a fault in the code: LEVIR-CD pairs are 0.5 m aerial photographs of US
suburbs, where a house covers hundreds of pixels; at Sentinel-2's 10 m a house is smaller than one pixel and a
new building cluster is a few blurred pixels inside a field mosaic. Upsampling enlarges the pixels but adds no
detail.

What could make a deep change detector work here:

- **Train or fine-tune on Sentinel-2 change pairs** (for example OSCD, or the KEMET1 pairs themselves once hand
  labels exist), with all 13 bands rather than RGB;
- **Very-high-resolution imagery** (under 1 m) for the sites flagged by the classifier, where pretrained
  building-change models are in their element.

## Reproducing

Needs an Earth Engine project, PyTorch, `timm`, `einops`, the ChangeFormer code and the checkpoint:

```bash
python download_s2.py --project <ee-project> --tiles ../../data/tiles --out ../../data/s2_tiles
python run_changeformer.py sanity --code ChangeFormer-main --ckpt best_ckpt.pt
python run_changeformer.py run --code ChangeFormer-main --ckpt best_ckpt.pt \
    --s2 ../../data/s2_tiles --tiles ../../data/tiles --out ../../data/changeformer pct_x1 fixed_x1 pct_x2
cp ../../data/changeformer/changeformer_scores.csv ../../results/
cd .. && python evaluate_pipeline.py score --tiles ../data/tiles --changeformer ../data/changeformer \
    --variant pct_x1 --out ../results/pipeline_changeformer_pct_x1.csv
python compare_models.py
```

The 4× run was restricted to the test sites with `--sites` and a file of site names; the downloaded images (about 300 MB) and
the change maps are not stored in the repository.
