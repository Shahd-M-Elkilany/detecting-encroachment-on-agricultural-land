# Archive

Earlier stages of the project, kept for reference. Nothing here is maintained or
tested, and scripts may need path changes to run: they were written to run from
the repository root before it was reorganised. The maintained code is the
`encroachment` package at the top level.

| Folder | What it was | Replaced by |
| --- | --- | --- |
| `v1_scripts/` | The first before/after Random Forest: training, evaluation, site reports, web app and backend upload as separate scripts | The `encroachment` package and the `kemet1` command |
| `multitemporal_classifier/` | A classifier over 75 tiles at four dates (2022-2025), labelled visually by one annotator | Not carried forward; its tiles and model are not in this repository |
| `pipeline_8_step/` | An end-to-end deep-learning pipeline (cloud detection, cloud removal, SegFormer agriculture segmentation, SAM building detection) using pretrained weights without fine-tuning on Nile Delta data | Not carried forward |
| `delta_dataset/` | Tools to export and organise a larger multi-year Nile Delta tile set from Google Earth Engine | Not used for the results in the README |

`requirements-archive.txt` lists the heavier dependencies these scripts need
(PyTorch, transformers, Earth Engine).

The v1 results quoted in the project report are kept in `../results/v1/`.
