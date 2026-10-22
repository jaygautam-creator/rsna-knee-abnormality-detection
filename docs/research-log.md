> Working notes written during the competition (August to October 2026), kept as written. Paths such as
> `notebooks/<kernel>/` now live in `experiments/kernels/<kernel>/`. Later sections supersede earlier ones;
> the final pipeline is described in [architecture.md](architecture.md).

# RSNA Knee Abnormality Detection — Strategy

Deadline **2026-10-22** (~11 weeks). 281 teams. $77,000 incl. a separate **Efficiency Prize** track.
Top public LB today: **0.932**.

## Confirmed rules

**Metric:** macro-averaged AUC ROC — the plain mean of the 12 per-condition AUCs.
Every condition counts equally regardless of prevalence, so the rare findings
carry the same weight as `Effusion`. Since it is rank-based, only the *ordering*
of our scores matters — no threshold tuning, and calibration matters only where
it changes rank.

**Code requirements:** submission via notebook, **≤ 9 h runtime** (not 12),
**internet disabled**, `submission.csv`, and freely/publicly available external
data and pretrained models are allowed. So ImageNet/DINOv2 weights are fine but
must be attached as a Kaggle Dataset or Model rather than downloaded at runtime.

### The Efficiency Prize is won on runtime, not accuracy

```
Efficiency = AUC / (Benchmark - maxAUC)  +  RuntimeSeconds / 32400      (minimise)
```

`Benchmark` is `sample_submission.csv` (AUC 0.5) and `maxAUC` is the best private
score, so the denominator is negative — roughly `0.5 - 0.93 = -0.43`. That makes
the accuracy term about `-2.3 x AUC`, while runtime contributes at most `1.0`
(32,400 s is exactly the 9 h cap).

Differentiating: **+0.01 AUC buys you 0.023.** Cutting runtime from 9 h to 1 h
buys you **0.889** — equivalent to gaining *0.38 AUC*. Runtime dominates by more
than an order of magnitude.

So the Efficiency submission should be a deliberately lean single model that
clears the benchmark and finishes in well under an hour. Do not enter the
ensemble there — TTA and multi-model averaging are nearly free in AUC terms and
catastrophic in efficiency terms.

## The thing that actually defines this competition

`train.csv` has 4,407 studies. **Only 58 carry labels. All 4,407 carry a radiology report.**

| | count |
|---|---|
| studies | 4,407 |
| with all 12 labels | **58** |
| with a report | 4,407 |
| series | 24,371 (mean 5.5/study) |

Reports span ~10+ languages (en 1695, es 299, de 251, tr 233, ru 220, nl 101, fr 81, it 44, …).
`test.csv` contains **StudyInstanceUID only — no reports at test time.**

So the competition is not "train a classifier on labelled MRIs." It is:

```
reports (4,407, multilingual)  --text model-->  labels for 4,407 studies
                                                     |
                       DICOM volumes (4,407) --------+--> image model --> test predictions
                                                     |
                    58 expert labels ----------------+--> validate / calibrate
```

**Whoever extracts the best labels from the reports wins.** The image model is
downstream of label quality — a great architecture trained on noisy labels loses
to a good architecture trained on clean ones.

### Evidence the reports carry the signal

A crude v0 regex labeler (`src/labels/rules_en.py`), English only, vs. the 28
English expert-labelled studies:

| cond | acc | prec | rec |
|---|---|---|---|
| ACL | 0.857 | 0.73 | **1.00** |
| MCL | 0.857 | 0.56 | **1.00** |
| Medial Meniscus | 0.821 | 0.71 | **1.00** |
| Lateral Meniscus | 0.821 | 0.71 | 0.91 |
| Medial OA | 0.821 | 1.00 | 0.29 |
| Lateral OA | 0.821 | 1.00 | 0.17 |
| PF OA | 0.821 | 0.67 | 0.75 |
| Effusion | 0.714 | 0.67 | 0.93 |
| Synovitis | 0.714 | 0.62 | 0.50 |
| Baker's | 0.929 | 0.80 | 0.80 |
| Contusion | 0.714 | 0.56 | 0.90 |
| Fracture | 0.929 | 0.88 | 0.88 |
| **mean** | **0.818** | | |

Recall is 1.00 on the tear findings — the reports *state* the findings plainly;
my regex is just over-firing (precision) and my OA compartment patterns are too
narrow (recall 0.17–0.29). **0.82 from an afternoon of regex is a floor, not a
ceiling.** An LLM labeler should reach the high 0.9s.

### Measured: LLM labeller vs regex (v9, Qwen3-4B-Instruct, 2x T4)

Scored on the same 58 studies, on **AUC** — the competition metric:

| subset | regex | LLM | gain |
|---|---|---|---|
| all 58 | 0.6287 | **0.8448** | +0.216 |
| English (28) | 0.7844 | 0.8461 | +0.062 |
| **non-English (30)** | **0.5051** | **0.8484** | **+0.343** |

The headline is the third row. **The regex is at chance on non-English reports,
and 61% of the corpus is non-English.** The LLM shows no language gap at all
(0.846 vs 0.848). Public notebooks are using hand-built multilingual lexicons;
this is where they will be weakest and where our edge comes from.

Per-condition gain, largest first: Medial OA +0.382, Lateral OA +0.282, Medial
Meniscus +0.249, Baker's +0.248, ACL +0.242, MCL +0.206, PF OA +0.189, Fracture
+0.188, Lateral Meniscus +0.185, Synovitis +0.177, Effusion +0.169, Contusion
+0.077. The two largest gains are exactly the conditions the questions were
written to fix — explicit laterality worked.

Weakest absolute AUCs are **Effusion 0.717, Contusion 0.738, Synovitis 0.740** —
the severity-threshold conditions, where the experts' bar differs most from the
report prose. These are the targets for a v2 labeller.

### The 58 gold studies are abnormality-enriched — do not read prevalence off them

Positive rate in the 58 gold studies vs the labeller's rate over all 4,407:
Fracture **4.4x**, Synovitis **3.4x**, ACL **2.9x**, Lateral Meniscus **2.7x**.

Decomposing that against calibration on the *same* 58 (LLM rate / gold rate)
separates two different effects:

| condition | calib on the same 58 | enrichment of the 58 |
|---|---|---|
| Fracture | 1.06x (good) | 4.40x |
| Lateral Meniscus | 1.00x (good) | 2.66x |
| ACL | 0.92x (good) | 2.94x |
| Contusion | **1.95x over-calls** | 0.87x |
| MCL | **1.67x over-calls** | 1.41x |
| Effusion | **1.43x over-calls** | 1.02x |
| Synovitis | **0.74x under-calls** | 3.37x |

So Fracture/ACL/Lateral Meniscus are labelled *well* — those gaps are the 58
being a curated, abnormality-rich sample, not a labelling failure. The 58 are
**not** a random draw and must never be used to estimate prevalence or to set
priors. (The competition also warns prevalence differs across train, public LB
and private LB, so nothing should depend on it.)

The genuine label problems are **Contusion (1.95x), MCL (1.67x), Effusion
(1.43x) over-calling and Synovitis (0.74x) under-calling** — and note the three
worst-calibrated conditions are three of the four worst AUCs. Since AUC is
rank-based, miscalibration alone cannot hurt it; the damage comes from the
float32 saturation ties (Effusion had 11 distinct values across 58 studies).
That makes a testable prediction: **the margin fix should lift Effusion and
Contusion specifically.**

v2 prompt targets: Contusion's "answer No if the oedema is degenerative" clause
and Effusion's "more than a trace" clause are both being ignored.

Caveat worth knowing: the 58 labels are not verbatim report transcription. At least one
study whose report describes a moderate effusion is labelled `Effusion = 0`. The expert labels apply a stricter threshold than the report
prose. So the 58 are a **calibration set**, not just a test set — we fit the
decision thresholds on them.

### Duplicate reports — a CV leak waiting to happen

4,407 studies contain only **4,257 unique reports**. 54 report texts are shared
by more than one study, covering 204 studies; the largest single group is **37
studies sharing one identical report**.

Since our labels are *derived from the report*, every study in a group gets
byte-identical targets. Split such a group across folds and the model scores on
validation rows whose labels it already memorised from training. A 37-study
group alone is enough to visibly inflate CV.

**Therefore: fold assignment must be grouped on a hash of the normalised report,
not on StudyInstanceUID.** (Where gold labels exist for duplicate groups they do
not conflict, so the duplicates are genuine repeats, not annotation noise.)

## Compute — the 570 GB problem

The development laptop has 8 GB RAM and little free disk, so local training is off the table.

It does not matter. This is a **Kaggle Code competition** — test DICOMs only exist
inside the Kaggle notebook environment. You must submit a notebook regardless.
So the data never touches your laptop:

- **Kaggle Notebooks** — free 30 h/week GPU (T4×2 or P100), and the 570 GB dataset
  is already mounted at `/kaggle/input`. This is the primary compute.
- **Google Colab free** — T4, for overflow experiments only. No dataset mount, so
  it is only useful for work on artifacts we export (cached tensors, label CSVs).
- **Laptop** — CSVs, report/label work, code authoring, notebook push via
  `kaggle kernels push`. All of it CPU-light. Already working.

Local disk stays under ~1 GB the whole project.

## Plan

### Phase 1 — Labels (weeks 1–2) · highest leverage, runs free
1. Translate-free multilingual labeling with an open-weight LLM (Qwen2.5-32B /
   Gemma-3-27B) running **on Kaggle GPUs from the Kaggle Models hub** — $0, no
   API key, no internet needed at inference (satisfies code-comp rules).
2. Constrained JSON output: 12 fields, each a probability, not a hard 0/1.
   Soft targets beat hard ones for a downstream AUC metric.
3. Ensemble the LLM with the regex rules; the regex gives high-precision seeds.
4. Fit per-condition thresholds/calibration on the 58. Report per-condition AUC.
5. Ship `train_labels_v1.csv` — 4,407 × 12 soft labels. **Everything downstream
   depends on this file.**

### Phase 2 — Image model (weeks 3–7)
Target the metric, and note the Efficiency track rewards small/fast models.

- **Preprocessing**: DICOM → per-series volumes, resample to isotropic, crop to
  the knee joint, cache as `uint8`/`fp16` `.npy` in a Kaggle Dataset.

  **Measured, not assumed** (`notebooks/dicom_probe`): a header survey of
  **2,000 of the 24,371 series returned 100% uncompressed Explicit VR Little
  Endian — 0% compressed**. The data description promises "a mix of transfer
  syntaxes (uncompressed Explicit VR LE, JPEG Lossless, JPEG 2000, Implicit VR
  LE)"; for the *training* set that is simply not true, and the plan should
  follow the measurement, not the prose. (Test-set syntaxes are unverifiable
  until submission, so inference code must still handle compressed input — just
  do not let it drive the training schedule.)

  Decode runs at **2.7 ms/slice**, so caching the entire training set costs
  **1.3 h on one core, ~0.3 h on four**. I had budgeted a week for this assuming
  JPEG 2000 dominated; it does not, and preprocessing is *not* the bottleneck.
  That means we can afford to re-cache with different crops/resolutions rather
  than committing to one representation up front.

  Geometry from the same probe: rows/cols 192–1280 (median 456), 18–160 slices
  per series (median 30), slice thickness 0.6–5 mm, all MONOCHROME2, dtypes
  **mixed uint16/int16** — so intensity normalisation must be per-series, not
  global.

  Tag availability (12 series): `ImagePositionPatient`, `ImageOrientationPatient`,
  `SliceLocation`, `PixelSpacing`, `SliceThickness` all present 12/12, so proper
  geometric slice ordering is possible — do **not** order by `InstanceNumber`.
  But `Laterality` is present only 9/12 and `RescaleSlope`/`Intercept` only 3/12,
  so left/right normalisation needs an image-based fallback.
- **Architecture**: per-series 2.5D encoder → attention pooling over slices →
  attention pooling over series → 12 sigmoid heads.
  - Encoder: `convnext_small` / `efficientnetv2_s` (ImageNet), or a video model
    (`X3D`, `MViT`) on the slice axis. Start 2.5D — it is what wins RSNA comps.
  - Condition the series pooling on `Anatomical_Plane` + `Fluid_Sensitive`
    (note: `Fat_Suppression` is **identical** to `Fluid_Sensitive` in the data —
    one is redundant, drop it).
  - Sagittal → cruciates/menisci, Coronal → collaterals/compartment OA,
    Axial → PF OA/synovitis. The plane embedding matters.
- **Laterality: do NOT use horizontal-flip augmentation.** Four of the twelve
  targets are side-specific — `Medial OA` / `Lateral OA` and `Medial Meniscus` /
  `Lateral Meniscus`. A horizontal flip swaps medial and lateral, so the default
  "random hflip" in every vision pipeline would silently relabel a third of the
  targets. It is the single easiest way to destroy this model while every loss
  curve still looks healthy.

  For the same reason laterality must be *normalised*, not augmented away. Medial
  means "toward the midline", so which image side is medial depends on whether
  the knee is left or right:
  - `ImageOrientationPatient[0:3]` gives the patient-space direction of
    increasing column; DICOM patient +x points to the patient's **left**.
  - A right knee's medial side faces the patient's left, a left knee's faces
    the patient's right.
  - Combine the two to flip every study into one canonical orientation.

  `Laterality` is present in only ~9/12 sampled series, so it needs a fallback:
  the sign of `ImagePositionPatient[0]` (right knees sit at negative patient x).
  Both tags survived the 86-tag allowlist.

- **Loss**: soft-target BCE on the 4,407 LLM labels, then fine-tune on the 58.
- **Validation**: grouped CV by site (report language is a proxy for site) —
  the LB is a different site mix, so random CV will lie to you.

## Leaderboard results so far

| date | submission | model | public LB |
|---|---|---|---|
| 2026-08-09 | Knee Infer v0 — v1 | resnet18, 16 slices, attn pool | 0.826 |
| 2026-08-11 | **Knee Infer v2 — v3** | convnext_nano, 24 slices, two-head, flip TTA, `v3_fold0` | **0.896** |

Top public LB: **0.932**. Gap: **0.036**.

**The local metric is pessimistic.** `v3_fold0` measured `report_auc` 0.8694
locally but scored **0.896** on the LB — about +0.027 in the *good* direction.
That is expected: the local number grades against LLM-derived report labels while
the LB grades against real expert labels, so the image model is generalising past
noise in its own training targets. Consequence: **do not over-trust small local
deltas** when choosing what to ship; local is a weak-but-directional ranker.

**What earned +0.070 is confounded.** Backbone (resnet18 -> convnext_nano), slices
(16 -> 24), the two-head architecture, and flip TTA all changed at once — *and* the
checkpoint went from undertrained to fully annealed (513 steps @ 0.7608 -> 4,400
steps @ 0.8694). Training duration is the most likely single driver. Not worth an
ablation to separate; the direction is clear.

### Available but unsubmitted

`artifacts/train_v3_f1/v3_fold1.pt` — fold 1, fully trained (4,400 steps, 8.0 h),
**never submitted**. Slightly stronger than fold 0 on report_auc (0.8780 vs
0.8694) and, more importantly, **complementary per-condition**:

| condition | fold 0 | fold 1 |
|---|---|---|
| Fracture | 0.8674 | **0.8961** |
| Medial OA | 0.8807 | **0.9230** |
| Lateral OA | 0.8693 | **0.8920** |
| Baker's | **0.8706** | 0.8469 |
| Contusion | **0.8916** | 0.8521 |

So a **2-fold ensemble (f0 + f1, flip TTA)** is the best submission obtainable with
no further training — inference only. Expect **+0.005–0.015**, i.e. ~0.90–0.91.

**Reaching 0.95 is not on the current path** — it is above the 0.932 top public LB.
Ensembling folds buys hundredths; the honest route to 0.93+ is *better labels*
(the v4 relabel) feeding a retrain, per this document's core thesis. Rough ladder:
f0+f1 ensemble ~0.90–0.91 (today, inference only) -> +fold 2 ~0.905–0.915 (8 h
training) -> +v4 relabel ~0.91–0.925 (relabel + retrain).

### Phase 3 — Ensemble + submission (weeks 8–11)
Multi-seed / multi-encoder ensemble and TTA for the main track — no threshold
calibration, since the metric is rank-based and thresholds do not move AUC.

A **separate, deliberately lean single-model notebook** for the Efficiency Prize,
optimised for wall-clock (small encoder, fewer slices, no TTA, no ensemble). Per
the arithmetic above it should target minutes, not hours; sacrificing 0.02 AUC to
save 6 h is a large net win on that track.

Both must run offline in ≤ 9 h with weights attached as Kaggle Datasets.

## Repo

```
knee/
  data/               # CSVs only — never the DICOMs
  src/labels/         # report -> label mining
  src/data/           # DICOM -> cached volumes (runs on Kaggle)
  src/models/
  notebooks/          # pushed to Kaggle via `kaggle kernels push`
```

## Open risks
- Label quality is the whole ballgame; 58 studies is a thin calibration set.
- DICOM decode throughput inside the 30 h/week GPU budget.
- LB at 0.932 on day 3 of a 3-month competition is suspiciously high — worth
  reading the discussion forum for a leak before over-investing in architecture.
- Compute is now the binding constraint, not ideas. Kaggle GPU quota is spent
  (resets ~2026-08-15) and free Colab caps out mid-run, which is why the v4
  relabel is checkpointed to Drive and resumed across sessions
  (`notebooks/label_relabel_v4_colab/relabel-v4-colab-resumable.ipynb`). Any plan
  that needs an 8 h training run is gated on that quota; inference-only moves
  (fold ensembling) are the cheap ones and should be spent first.

## 2026-08-19 — re-aiming: we have been optimising the wrong conditions

Two measurements taken today change the plan.

### 1. Rank the conditions by the *image model*, not by label quality

Per-condition `report_auc` from `artifacts/train_v3_f{0,1}/v3_meta.json`, worst first:

| condition | n_pos | fold 0 | fold 1 | mean |
|---|---|---|---|---|
| **MCL** | 133 | 0.8289 | 0.8436 | **0.8362** |
| **Lateral Meniscus** | 117 | 0.8225 | 0.8545 | **0.8385** |
| **Baker's** | 272 | 0.8706 | 0.8469 | **0.8588** |
| **PF OA** | 260 | 0.8648 | 0.8657 | **0.8653** |
| Contusion | 178 | 0.8916 | 0.8521 | 0.8719 |
| Lateral OA | 85 | 0.8693 | 0.8920 | 0.8806 |
| Fracture | 64 | 0.8674 | 0.8961 | 0.8818 |
| Medial Meniscus | 318 | 0.8743 | 0.8895 | 0.8819 |
| Synovitis | 108 | 0.8768 | 0.8876 | 0.8822 |
| Effusion | 432 | 0.8810 | 0.8926 | 0.8868 |
| ACL | 132 | 0.9043 | 0.8922 | 0.8982 |
| Medial OA | 219 | 0.8807 | 0.9230 | 0.9019 |
| **macro** | | | | **0.8737** |

The v4 relabel was scoped by *label* AUC on the 58 gold studies, which named
Synovitis, Contusion, Effusion and PF OA. Three of those four are among the
image model's **best** conditions. Only PF OA appears on both lists. v4 is not
wrong, it is just aimed at ground we already hold — which is consistent with the
2-fold ensemble stalling at 0.905.

Since macro AUC is the plain mean of 12 *independent* per-condition AUCs, the
score lives entirely in the bottom of that table. Lifting MCL, Lateral Meniscus,
Baker's and PF OA from ~0.85 to ~0.92 is worth **+0.023 macro on its own**, and
nothing whatsoever is gained by improving Medial OA or ACL.

### 2. We are throwing away the sequences those four conditions are read on

`notebooks/cache_volumes_24` takes **one series per plane, fluid-sensitive
preferred**. Measured over `data/train_series.csv`:

- the cache uses **13,221 of 24,371 series (54%)**
- **97% of what the model sees is fluid-sensitive** — only 416 non-fluid series
  reach it, out of 10,361 that exist

The discarded non-fluid series are the PD/T1 *anatomy* sequences, and they are
plentiful: **4,266 studies have a non-fluid sagittal (97%)** and **3,406 have a
non-fluid coronal (77%)**. Non-fluid axial is rare (857, 19%) — do not plan on it.

Clinically that is exactly backwards for the four weak conditions. Meniscal tears
are read on sagittal PD/T1, not on fat-suppressed T2; the collateral ligaments are
read on coronal. Fluid-sensitive is the right default for Effusion, Synovitis and
Contusion — and those are precisely the three the model already does well.
**The contrast we always keep matches the conditions we are already good at, and
the contrast we always discard matches the conditions we are bad at.**

### The move

Train a second, **anatomy** model on non-fluid sagittal + coronal volumes and
combine it with the existing fluid model **per condition**. Because the metric
scores each condition independently, per-condition selection between two models
is free and cannot hurt the conditions it is not applied to — unlike the flat
3-checkpoint average of 08-14, which cost 0.005 by diluting every column at once.

Cost: one cache rebuild (~1 h CPU) plus one 8 h training run — it fits a single
Kaggle session and does not need the v4 retrain to happen first.

Select per condition on OOF `report_auc` over all 4,407 studies, never on the 58
gold (13 studies per fold cannot rank anything).

### Caveat on the target

Top public LB is now **0.952** with 1,936 teams; we are 601st at 0.905. This plan
is a credible route to roughly 0.92–0.93. It is not, by itself, a route to 0.95.

### 3. The bigger finding: we were mirroring the laterality targets away

Re-reading the code after the table above, the medial/lateral asymmetry has a
much simpler cause than sequence choice. The pipeline destroys laterality in
**three** places:

| where | code | effect |
|---|---|---|
| `train_v4_f0` augmentation | `x = x[:, :, :, ::-1]` at p=0.5 | half of every epoch is trained with medial and lateral swapped |
| `infer_v4_ens` TTA | `torch.flip(xb, dims=[4])`, averaged | the mirrored prediction is averaged into the real one |
| `cache_volumes_24` | no canonicalisation at all | left and right knees are mirror images of each other |

Axis 4 of `(B, P, D, H, W)` is W, which on the axial and coronal planes *is* the
medial/lateral axis. This is the exact failure this document predicted in bold —
"the single easiest way to destroy this model while every loss curve still looks
healthy" — and it shipped in every run to date.

The per-condition table is the fingerprint. Under laterality ambiguity a model
falls back on the prior, and medial findings are ~2.7x more prevalent than
lateral ones, so the medial member of each pair survives and the lateral member
collapses — Meniscus 0.882/0.839, OA 0.902/0.881, and MCL (a *medial* structure
with no lateral counterpart in the label set to anchor it) worst of all twelve.

**Medial is not always the horizontal axis**, which is why "flip the left knees"
is not the fix:

* axial and coronal — image columns run left-right, so medial/lateral is **W**;
* sagittal — image columns run anterior-posterior, so medial/lateral is the
  **slice axis**, and a horizontal flip there only corrupts anterior/posterior.

`cache_volumes_lat` resolves the axis from `ImageOrientationPatient` rather than
from the plane name, takes the knee side from `Laterality` with the sign of
`ImagePositionPatient[0]` as fallback, and flips whichever axis carries the
x-component so that **increasing index always points medial**. The decision table
is unit-tested across plane x side before it ever reaches Kaggle.

### Revised order of work

1. **`cache_volumes_lat`** — CPU-only, no GPU quota, ~1 h. *(pushed 2026-08-19)*
2. **`train_v5_f0`** — v4 labels, canonical cache, horizontal flip deleted,
   everything else identical to v4 so the delta is attributable. 8 h.
3. **Re-infer with `TTA_FLIP = False`.** Non-negotiable — leaving it on hands the
   ambiguity straight back at test time and would mask the gain.
4. *Then* the non-fluid PD/T1 anatomy channels from §2, which is a separate cache
   and a separate run.

Laterality comes first because it is a deletion rather than an addition: it costs
no extra compute, and it plausibly explains the four weakest conditions on its
own. The sequence-selection work in §2 is still real, but it is additive on top
of a model that can tell left from right.

### Laterality canonicalisation — built and shipped 2026-08-19

Three cache versions, each rejected by its own gate until the numbers held.

| version | side rule | agreement vs `Laterality` tag | outcome |
|---|---|---|---|
| v1 | per-**series**, sign of one unsorted slice's `ipp[0]` | — | 413/4,407 studies (9.4%) had series *disagreeing* on which knee |
| v2 | per-**study**, `sign(mean_x)` | 0.901 | rejected by the 0.98 gate |
| v3 | tag > `SeriesDescription` > report > mean-x @ -68 mm | **0.980** | shipped |

Two real bugs surfaced, both mine, both caught before any GPU was spent:

1. **Side is a property of the study, not the series.** A sagittal stack *spans*
   the medial-lateral axis, so `ImagePositionPatient[0]` changes sign across
   slices and one arbitrary slice's sign is noise.
2. **The mean-x cut is not zero.** Right knees centre at x = -139 and left at
   +39, so zero cuts through the right-hand cluster. Moving the cut to -68 mm
   took text-silent studies from 0.883 to **0.974 on a held-out half**.

A third fix was to the laterality vocabulary: Turkish `sag` (right) also
abbreviates *sagittal* and matched nearly every series description. Dropping it
for the dotted `sağ` took `SeriesDescription` from 0.902 to 0.985.

Final cascade coverage over 4,407 studies: tag 2,196 / report 664 / desc 179 /
mean-x 1,368. Sides come out 2,329 R and 2,078 L — and **status quo consistency
was 0.529**, so before this the pipeline was mirroring 47% of studies relative to
the rest. The fix takes that to 2%, a **23x** reduction.

Cache v3 verification: flip rate is exactly 1.0 for one side and 0.0 for the
other on every plane, off-axis flips are <0.1%, and **0 studies** contain
contradictory sides (v1 had 413).

**Inference calibrates the threshold at run time.** -68 mm is a property of this
site mix and the test distribution differs by design, so `infer_v5` refits the cut
on the ~50% of *test* studies carrying the `Laterality` tag — free ground truth in
the test distribution — falling back to -68 only under 50 calibration studies.
Test has no reports, so that rung of the cascade is absent there; the divergence
is pinned by assertions in `src/tools/check_decode_parity.py`.

Gates now enforced by `make check-lat`: 13 laterality assertions (including a
regression test reproducing the per-slice sign flip) and 24 parity checks between
the training cache and the inference decoder.

**Launched 2026-08-19:** `train_v5_f0` and `train_v5_f1`, 8 h each, ~16 h of the
30 h weekly quota. Read on completion: **MCL** and **Lateral Meniscus** (the two
weakest conditions, both laterality-dependent) should move most, and the
medial/lateral gap within each pair should narrow. If they do not move, the
laterality hypothesis is wrong and the non-fluid PD/T1 anatomy channels become
the next lever.

### v5 landed 2026-08-19 — the laterality fix did not lift the model, but it built a second one

Both folds finished clean (8.23 h / 8.22 h, 4,400 steps each, same schedule as v3).
`gold_auc` is unusable — 13 and 11 gold studies per fold — and the printed
`report_auc` is scored against **v4** teacher labels while v3's was scored against
**v3**, so neither headline number is comparable.

The comparable read: v3 and v5 use different fold splits, but **357 studies are
held out by both**. Scoring both models' saved val predictions on exactly those
rows, against one common label set, gives a paired comparison. Rank-normalised
per fold, then pooled:

| | macro AUC on the 357 shared held-out studies |
|---|---|
| v3 (flip aug, mirrored cache) | **0.8692** |
| v5 (no flip, laterality-canonical cache) | **0.8594** |
| 0.6·v3 + 0.4·v5 | **0.8769** |

**v5 alone is worse:** −0.0098, bootstrap 95% CI [−0.0215, +0.0011], P(v5>v3)=0.043.
The laterality hypothesis as stated — that canonicalising sides lifts the weak
laterality-dependent conditions — is **not supported**.

The prediction was half right, which is the interesting part. The two conditions
named in advance are the only weak ones that moved *up* (MCL +0.014, Lateral
Meniscus +0.011) while nine of twelve moved down, and both medial/lateral gaps
narrowed (Meniscus +0.104 → +0.082, OA +0.036 → +0.029). So the fix did what it
was supposed to do locally and lost more than that elsewhere.

**Confound, stated plainly:** v5 changed three things at once — laterality-canonical
cache, flip aug swapped for roll + gain/bias, and v3 → v4 margins. Since PLAN
already established that v4 relabelling targeted conditions the image model had
already solved, part of the −0.01 is plausibly the labels rather than the cache.
Disentangling costs another 8 h fold and is not worth it, because:

**The blend is the actual result.** v5's per-condition rank correlation with v3 is
**0.813** — decorrelated enough to be worth ensembling despite being weaker.
0.6·v3 + 0.4·v5 beats v3 alone by **+0.0077**, CI [+0.0028, +0.0126],
P(blend>v3) = **0.999**, and the gain lands where the score lives: Lateral
Meniscus +0.024, MCL +0.017, Fracture +0.016, PF OA +0.008. Weight is flat
between 0.3 and 0.5; 0.4 is the peak.

Caveat: ρ=0.81 means much of that +0.008 is the generic two-model ensemble effect,
not evidence for laterality. And n=357 is small — the fold-split change is what
cost us the other 1,400 held-out rows.

**Next move is inference-only, no GPU quota.** `infer_v5` currently ships
`CKPT_NAMES = ["v5_fold0.pt"]` — a single fold, deliberately, to read the
laterality delta cleanly. That read is now in, so the submission model should be
all four checkpoints (`v3_fold{0,1}`, `v5_fold{0,1}`) at 0.6/0.4 group weight with
`TTA_FLIP` still False. Feeding the canonical cache to the v3 checkpoints is safe
precisely because they were trained with flip aug at p=0.5 and are therefore
mirror-invariant — the property that made them mediocre is what makes them
portable.

After that, the non-fluid PD/T1 anatomy channels remain the next real lever, and
they are still untouched.

### infer_v5 rewired to the four-checkpoint blend — pushed 2026-08-19

`CKPT_NAMES = ["v5_fold0.pt"]` → weighted blend of `v3_fold{0,1}` (0.30 each) and
`v5_fold{0,1}` (0.20 each), `MIN_CKPTS = 4`, `TTA_FLIP` still False.

Three decisions worth recording:

**Rank blend, not probability average.** v3 trained against v3 margins and v5
against v4 margins, so their sigmoid outputs sit on different scales; a
probability mean would re-weight them by calibration rather than by the weights
fitted offline. The notebook now keeps per-model outputs (`PM`, ~240 KB), maps
each column to its within-test percentile rank, and combines at the group
weights. Replaying that exact code path on the 357 shared val rows reproduces
**0.8769 vs 0.8692, +0.0076** — matching the offline figure, so the shipped code
is the thing that was validated rather than a re-derivation of it.

**No timeout risk added.** 4 checkpoints x no TTA = 4 forwards/study. The banked
0.905 submission was 2 checkpoints x flip TTA — the same 4 forwards on the same
CPU kernel — and it finished inside the 9 h cap. Decode is per study and shared
across the four. So GPU inference was not needed and the kernel stays
`enable_gpu: false`.

**v3 checkpoints on the canonical cache is safe**, and for the opposite reason to
v5: they were trained with a horizontal flip at p=0.5, so they are
mirror-invariant. The property that made them mediocre is what makes them
portable onto the new cache.

Two new assertions guard the blend: both generations must be mounted (losing one
turns a fitted blend into a single-generation run at a weight never fitted for
it), and each checkpoint is checked for constant columns individually — a
collapsed model would otherwise vanish into the blend without tripping the
existing whole-submission constant-column check. The run also prints mean
pairwise rank correlation; near 1.0 would mean the mounts are wrong and the
ensemble is buying nothing.

`make check-lat` passes with these edits (13 laterality + 24 parity checks).

### LB result: 0.906 — new best, 2026-08-20

The four-checkpoint blend scored **0.906**, beating the banked 0.905. Two reads:

**Runtime is settled.** The run succeeded, so 4 forwards/study on the CPU kernel
fits the 9 h cap at real test-set scale. The preview's 18 s/study (→ ~6.2 h
extrapolated) was warmup-dominated, as suspected; treat single-batch preview
timings as an upper bound only.

**The gain is +0.001, against +0.0077 measured offline** on 357 shared val rows
(CI [+0.0028, +0.0126]). The public LB is its own sample so this isn't a
contradiction, but it lands at the bottom of the interval. Working rule going
forward: an offline delta is an optimistic upper bound on the LB delta, and
anything under ~0.005 offline is not worth a submission slot.

The remaining untouched lever is still the non-fluid PD/T1 anatomy series — the
cache feeds ~97% fluid-sensitive and discards the anatomy series entirely. That
is a data-coverage change, not another ensembling step, and it is the only
candidate left with a plausible delta above the ~0.005 threshold.

### 2026-08-20 — recon, and the fold bug underneath everything

**Public recon.** Pulled the top-voted public kernels with the Kaggle CLI. The
0.943-0.952 leaders are forks of one shared pipeline —
`mattiaangeli/bend-the-knee-to-dinov3-ensembled` (0.917), a **35-checkpoint**
ensemble assembled from ~10 contributors' published weights (20 DINOv2-small,
5 DINOv3-small, 10 RadImageNet attention heads on a shared ResNet-50). Its own
author warns it is overfit to the public 30% and that 0.001-0.003 moves are noise.
The best *independent* public imaging model — five fine-tuned DINOv2 ViT-B/14 at
336 px — scores **0.883**, below our 0.906.

**Component comparison.** Labels: ours 0.8801 vs their 0.881 on the 58 gold —
level. Single model: we are ahead. Resolution: we run 224 px = 0.67 mm/px over a
150 mm FOV, against a published argument for 336 px = 0.45 mm/px. Ensemble
breadth: 4 checkpoints, 1 architecture, 1 author vs 35/3/~10. **The gap is
breadth**, not any component we built.

**Both our measuring instruments are too blunt for the effects we chase.**
The 58-gold label benchmark has a bootstrap 95% CI of [0.8403, 0.9127] — +/-0.036.
The public LB is ~370 studies, roughly +/-0.01-0.015. We have been deciding 0.008
effects with instruments that cannot see them. That is the real story of v3/v4/v5,
and it explains v4 exactly: gold label AUC rose 0.874 -> 0.880 while the LB fell
0.905 -> 0.900.

**Root cause found: the fold split moves when the labels move.** `make_folds`
stratifies on abnormality burden computed from `labels`, and `StratifiedGroupKFold`
assigns groups greedily. Measured: v3 -> v4 labels changed burden on **6.6%** of
studies and reassigned **74.6%** of folds. So every label improvement silently
destroyed comparability with every earlier run — which is why comparing v3 to v5
left only **357** shared held-out rows instead of ~880, and why a +0.0077
measured there (+/-0.005) returned +0.001 on the board.

**Fixed, at no GPU cost.** `src/labels/folds.py` gains `freeze_folds` /
`load_folds`, and the split is committed to `data/folds.csv`. `freeze_folds`
refuses to overwrite an existing file. Verified: 4,257 report groups, **0 spanning
more than one fold** (including the 37-study group), fold sizes 882/881/882/881/881,
per-condition positive-rate spread <= 0.06.

Once all five folds are trained against the frozen split, comparisons run on
**4,407 OOF rows instead of 357** — SE shrinks **3.5x**, turning a +/-0.005 paired
delta into ~+/-0.0015. That makes 0.005 effects decidable offline, for free,
without spending a submission. Existing v3/v5 checkpoints remain usable for
inference and blending, but are **not** valid for OOF comparison — their val rows
predate the freeze.

**Submissions are not the scarce resource.** Five submissions in twelve days,
against several per day allowed and 63 days left. Kernel wall-clock is the
constraint; inference is CPU-only and bypasses GPU quota.

**Order of work from here.** Frozen folds first (done) -> 336 px cache rebuild ->
retrain 5 folds on the frozen split to get a real OOF instrument -> then judge
336 px, architecture and breadth changes against it -> blend last.
