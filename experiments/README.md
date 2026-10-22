# experiments/ (archive)

Source of the Kaggle kernels and Colab trainers we wrote earlier in the competition, kept for reference. These are
**not maintained**: they assume Kaggle mount paths and private datasets from that time, and several were dead ends.
Each kernel is a `# %%`-cell Python file; `python -m knee.tools.py2nb source.py out.ipynb` turns it into a notebook.

Kernels that differed from a sibling only by a fold or shard number were dropped. `tools/make_shards.py` shows how
the shard copies were generated.

## kernels/

| Group | Kernels | Notes |
|---|---|---|
| Probes | `dicom_probe`, `side_probe`, `submit_baseline`, `io_probe_336`, `compute_probe_336` | DICOM header survey, which-knee detection, test-set I/O budget, 336 px step-time ablation |
| Volume caches | `cache_volumes`, `cache_volumes_24`, `cache_volumes_lat`, `cache_volumes_336` | Decode the train set once. `_lat` adds laterality canonicalisation; 336 px needed 3 shards (35.8 GB) |
| Report labels | `label_reports`, `label_reports_v3`, `label_sweep_v4`, `label_relabel_v4`, `label_relabel_v7_colab` | LLM soft labels from the reports (logit-scored yes/no questions) and targeted relabels of weak findings |
| Image models | `train_v0` … `train_v7_*` | 2.5D backbones distilled from report soft labels. v3 trains in chunked subprocesses (memory leak), v5 fixes laterality, v6/v7 go to 336 px with frozen folds |
| Later trainers | `train_v9_specialist`, `train_v10_triplane`, `train_v11_specialist`, `train_sota_v1`, `train_sota_384_v1`, `train_coronal_mcl_v1`, `train_0970_master` | Specialist and tri-plane attempts. None beat the public checkpoints; `train_0970_master` trained at chance (see lessons) |
| Inference | `infer_v0` … `infer_v6`, `infer_v3_cpu`, `infer_v4_ens`, `infer_v18`, `infer_v19` | Our own inference kernels up to the in-house best (0.906, `infer_v5`) |

## colab/

Colab trainers for the specialist and DINOv2 plane-routed models, plus notebook generators. They read Kaggle
credentials from a local file at runtime; no credentials are stored here.

## tools/ and models/

* `tools/check_decode_parity.py`: train-time and test-time DICOM decoding must agree (orientation, slice order).
* `tools/laterality_check.py`: verifies the canonicalisation decision for every plane and side.
* `tools/make_shards.py`: generates the 336 px cache shard kernels from one template.
* `models/joint_specialist.py`: the joint-specialist architecture (ROI crop, slice attention, cross-label head).
