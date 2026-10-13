# Model card: KEMET1 v2 encroachment model

This card covers Model 1, the tile classifier. Model 2, the 8-step pipeline, is described and evaluated in
[`../pipeline/README.md`](../pipeline/README.md).

| | |
| --- | --- |
| File | `models/encroachment_logreg.joblib` (scikit-learn 1.7.2 pipeline), metadata in `models/encroachment_logreg.json` |
| Task | Probability that farmland in a 2.6 km Sentinel-2 tile was converted to built-up land between 2024 and 2025 |
| Input | 44 statistics of the change in six spectral indices (`encroachment/features.py`) |
| Model | `StandardScaler` + `LogisticRegression(C=1.0)` |
| Training data | 255 development sites (67 positive); 45 test sites held out |
| Labels | Automatic, from a spectral threshold rule (`docs/DATA.md`) |
| Licence of data | Sentinel-2 data: Copernicus open data licence |

## Performance

ROC AUC against the automatic labels (`results/evaluation.json`):

| Evaluation | AUC |
| --- | ---: |
| Random 5-fold CV ×10, all sites | 0.90 ± 0.04 |
| Spatial 5-fold CV (regions held out), mean | 0.86 (0.70 to 0.94) |
| Held-out test, 45 sites | 0.97 (95 % CI 0.91 to 1.00) |

Alarm tiers on the test sites: High (probability ≥ 0.51) precision 0.90, recall 0.82; Medium or above
(≥ 0.16) precision 0.59, recall 0.91. On cross-validated development predictions the High tier gives
precision 0.77 at recall 0.55.

## Intended use

Ranking Nile Delta sites for follow-up inspection on very-high-resolution imagery or in the field, and as a
documented baseline for the same dataset.

## Not intended for

- Deciding that a specific building is illegal. The labels describe spectral change, not legal status or
  confirmed construction.
- Areas outside the Nile Delta, other sensors or resolutions, or periods other than 2024 to 2025, without
  re-evaluation.

## Known failure modes

- Tile-wide harvest or fallow between the two dates (false alarms).
- A few hectares of conversion masked by vegetation gains elsewhere in the tile (misses).
- The north-eastern region is the weakest under spatial cross-validation (AUC 0.70).

## Reproduce

```bash
pip install -e ".[dev]"
kemet1 train && kemet1 evaluate
```

Training is deterministic; a test checks that refitting reproduces the shipped coefficients.
