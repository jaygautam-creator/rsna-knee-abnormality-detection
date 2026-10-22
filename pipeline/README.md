# pipeline/

| Path | Contents |
|---|---|
| `base/infer-v90.ipynb` | The submitted v90 inference notebook (outputs stripped, S6 parity rows removed). Base for every variant |
| `base/kernel-metadata.json` | Kaggle metadata: T4 x2, internet off, pinned docker image, the 5 public datasets it mounts |
| `base/part_c_s6fall.py` | Readable copy of cell 10 (Part C, S6 all-data blend) |
| `base/fastread.py` | Readable copy of the fast DICOM reader that cells 2 and 4 embed as base64 |
| `variants/*.json` | One config per submission variant (see below) |
| `eval/gold58-eval.ipynb` | The evaluation notebook: the v90 chain on 58 labelled + 46 other training studies, dumping every component |
| `local/` | **Gitignored.** `s6_parity.json` with the 3 public-test reference rows |

## Variant config

```json
{
  "name": "v94",
  "description": "...",
  "stage3_raptor_weight": {"Medial OA": 0.5, "Lateral OA": 0.15, "PF OA": 0.5, "Synovitis": 0.5, "Baker's": 0.5},
  "w_s6": 0.35,
  "s6_timing": {"S6_STOP_H": 8.60, "S6_KILL_H": 8.72, "S6_MIN_LEFT_H": 0.25},
  "provenance": {"kernel": "...", "version": 1, "submission": 57522189, "submitted_notebook_sha256": "..."}
}
```

* `stage3_raptor_weight` replaces `W_RAPTOR_C` whole. Findings not named use 0.5 (Raptor half and R6 equal).
* `w_s6` is a scalar in (0, 0.5) or a dict naming all 12 findings with values in [0, 0.75).
* `s6_timing` is optional and must keep `S6_STOP_H < S6_KILL_H < 8.75`.
* `provenance.submitted_notebook_sha256` fingerprints the notebook that was actually pushed. Rebuilding a config
  reproduces the same settings; `tests/test_variants.py` checks the round trip. v95 was built by a second agent with
  different code for the same settings, so its code differs while its constants match.

## Mounted datasets

`goodpjw2008/rsna-knee-2-5d-convnext-reader`, `heliosli/rsna-knee-r6-frozen-fe055406-w1`,
`nartaa/rsna-knee-publication-swa-weights-20261007`, `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`,
`nartaa/rsna-knee-s6fall-weights`, plus the competition data.
