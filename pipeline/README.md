# The 8-step pipeline

KEMET1's second model. Where the [tile classifier](../README.md#model-1-the-tile-classifier) gives one
probability per site, this pipeline maps change pixel by pixel and outputs a red / yellow / green map:

- **red**: farmland that changed and now shows a built-up signal (likely encroachment)
- **yellow**: vegetation loss with no built-up signal (often harvest or fallow)
- **green**: stable farmland

It was designed by the KEMET1 team; the original design document is [`docs/design.md`](docs/design.md) and the
datasets behind each pretrained model are listed in [`docs/datasets.md`](docs/datasets.md).

![The pipeline's map next to the 2025 image for three test sites](../docs/figures/pipeline_examples.png)

## The eight steps

| Step | Task | Model it was designed around | What runs on the KEMET1 index tiles |
| --- | --- | --- | --- |
| 01 | Data acquisition | Google Earth Engine (Sentinel-2 L2A) | Reads the before and after GeoTIFFs |
| 02 | Cloud detection | U-Net + ResNet34 (38-Cloud) | Skipped: the tiles are cloud-free composites |
| 03 | Cloud removal | OpenCV Telea inpainting | Skipped |
| 04 | Spectral indices | NDVI, NDBI, MNDWI band maths | Taken directly from bands 0 to 2 |
| 05 | Change detection | ChangeFormer, Siamese transformer (LEVIR-CD) | Mean absolute change across the six bands, above half the tile maximum |
| 06 | Agriculture mask | SegFormer-B4 (ADE20K) | NDVI before > 0.2 |
| 07 | Building detection | YOLOv8-seg, refined with SAM | NDBI rise > 0.05 inside changed farmland, morphologically cleaned |
| 08 | Final output | OpenCV + Folium | Unchanged: map, regions, GeoJSON, interactive map, JSON report |

The deep models expect raw Sentinel-2 bands or RGB. The project's 300 tiles carry six spectral indices instead,
which those models were never trained on, so on this data each step uses the spectral rule shown in the last
column. Every result below comes from that configuration. With raw imagery and the weights in `weights/`, the
same code runs the deep models.

ChangeFormer was also tested on its own and as Step 05, on Sentinel-2 colour images downloaded for all 300 sites:
it scores at chance (test AUC 0.55 alone, 0.49 inside the pipeline), because a model trained on 0.5 m aerial
photographs does not see buildings at 10 m. See [`changeformer/README.md`](changeformer/README.md).

## How well it ranks sites

Scored against the same automatic labels and the same 45 held-out test sites as the classifier, using the red
area per site as the score ([`results/pipeline_evaluation.json`](../results/pipeline_evaluation.json)):

| | 8-step pipeline | Tile classifier |
| --- | ---: | ---: |
| ROC AUC, all 300 sites | 0.75 | 0.90 (cross-validated) |
| ROC AUC, 45 test sites | 0.81 (95 % CI 0.67 to 0.93) | 0.97 |
| Test precision / recall at the development-chosen threshold | 0.38 / 0.91 (≥ 1.8 ha red) | 0.59 / 0.91 (Medium risk) |

![ROC curves of both models on the test sites](../docs/figures/two_models_roc.png)

The classifier is the better way to decide **which** sites to inspect. The pipeline is useful for **where** to
look: it outlines the changed fields, and because it reacts to local change rather than tile-wide statistics it
sometimes catches a small conversion the classifier misses (site299 above). Its main weakness is harvest:
fields that lost their crop and look brighter in NDBI turn red, as in site150, so it raises 16 false alarms on
the 34 negative test sites at the threshold above.

## Running it

From the repository root, after `pip install -e ".[dev]"`:

```bash
cd pipeline
python run.py --test                                  # synthetic before/after pair through all 8 steps

python run_full_pipeline.py --site site264            # a KEMET1 tile pair from ../data/tiles
python run_full_pipeline.py --before b.tif --after a.tif --site mysite

# Step 05 replaced by the tile classifier: change pixels kept only when the site is Medium risk or above
python run.py --t1 ../data/tiles/site264_before_2024.tif --t2 ../data/tiles/site264_after_2025.tif \
              --precomputed --kemet1

python pipeline.py beforeafter --before b.tif --after a.tif --site-name mysite   # classifier report only
```

Outputs go to `outputs/` (one folder per site for `run_full_pipeline.py`): `final_colored.png` and `.tif`,
`encroachment.geojson`, `interactive_map.html`, `report.json` and before/after chips for each region.

To reproduce the evaluation:

```bash
python evaluate_pipeline.py score --tiles ../data/tiles    # about 2 minutes; writes ../results/pipeline_scores.csv
python evaluate_pipeline.py report                         # writes ../results/pipeline_evaluation.json
python evaluate_pipeline.py figures --tiles ../data/tiles  # the two figures above
```

### With raw Sentinel-2 imagery and the deep models

```bash
pip install -r requirements-deep.txt
python run.py --t1 before_raw.tif --t2 after_raw.tif     # bands B2, B3, B4, B8, B11, B12
python run.py --gee                                      # download from Earth Engine (set the project in config/settings.py)
```

Weights go in `weights/`: `cloud_unet_resnet34.pth` (Step 02), `ChangeFormer_LEVIR.pth` with the ChangeFormer code
on the Python path (Step 05) and `sam_vit_b_01ec64.pth` (Step 07); SegFormer downloads itself and YOLOv8 reads
`yolov8m-seg.pt` from this folder. Any step whose model is missing falls back to its spectral rule and says so in
the log. `run.py --register` and `--temporal` keep an archive of dated images and compare each new one with the
image from about a year earlier, so seasonal change is not mistaken for building.

## Limitations

- **Pretrained deep models do not transfer.** They were trained elsewhere (Landsat clouds, 0.5 m aerial change
  pairs, natural-image segmentation) and not fine-tuned on Nile Delta imagery; ChangeFormer, the one tested, scores
  at chance on these sites.
- **Close to the labelling rule.** Steps 06 and 07 use NDVI and NDBI thresholds similar to those that define the
  automatic labels, so agreement with the labels overstates real accuracy, for this model even more than for the
  classifier.
- **Harvest looks like building** at 10 m in two dates, the same confusion the classifier has, but stronger.

## Layout

```
run.py                  command line: test, single pair, classifier mode, Earth Engine, temporal archive
run_full_pipeline.py    the pipeline on KEMET1 index tiles (one function per stage, no files written by analyse())
pipeline.py             FoodSecurityPipeline: the eight steps, classifier mode, temporal mode
evaluate_pipeline.py    scores the pipeline against the labels and draws its figures
compare_models.py       scores every model (classifiers, pipeline, ChangeFormer) on the same sites
changeformer/           Sentinel-2 colour download and the ChangeFormer experiment
config/settings.py      thresholds, model names and paths for every step
src/step_01 ... step_08 one module per step
src/temporal/           dated image archive for the temporal mode
docs/                   the team's design document and dataset notes
```
