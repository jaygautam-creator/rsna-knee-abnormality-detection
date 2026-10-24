# [499th / Bronze Medal] 0.958 Public / 0.943 Private — 4-Family Rank Ensemble, LLM Report Mining & Honest Split-Half Validation on the 58 Gold Studies

First, thank you to RSNA and Kaggle for hosting a clinically realistic MRI competition, and to the community contributors whose public work raised the bar across the board (**nartaa**, **heliosli**, **goodpjw2008**, **dreaddevelopment**, and **varshithakolupuri**).

Below is a summary of our **499th place / Bronze Medal (`0.958` Public / `0.943` Private)** solution, including what actually moved the score across 112 submissions, how we mined multilingual radiology reports with LLM yes/no logit probing (`+0.22` AUC over regex), and how we audited contamination on the 58 expert-labelled studies using a 600-split held-out protocol.

---

## 1. Executive Summary

| Metric / Choice | Value |
| :--- | :--- |
| **Final Standing** | **499 / 6,425 (Top 7.8% — Bronze Medal)** |
| **Best Public / Private Score** | **`0.958` Public / `0.943` Private** |
| **Selected Submissions** | **`v90`** (`W_S6 = 0.20`, neutral Stage 3 anchor) & **`v96-a`** (`W_S6 = 0.35`, R6-lean Stage 3) — both scored **`0.943` Private** |
| **Total Runtime (Hidden Test)** | ~8.2–8.5 hours on 1x T4 GPU (guarded by per-stage wall-clock deadlines at `8.40 h` / `8.55 h`) |

---

## 2. Final Inference Architecture

Because the evaluation metric is **12-class macro ROC-AUC**, only within-column ordering matters. All multi-stage blending operates strictly on **percentile ranks** (`rank(method='average', pct=True)`):

1. **Raptor Half (`CoAtNet` Dual-Resolution)**:
   - `Raptor-384` (96 slices, SWA checkpoint) + `Raptor-224` (`0.8` anatomical crop):
   $$R_{\text{raptor}, c} = 0.65 \cdot \text{pct}(\text{Raptor384}_c) + 0.35 \cdot \text{pct}(\text{Raptor224}_c)$$
2. **R6 4-Fold Compact `ConvNeXt-small` + Flip TTA + Stackers (`ours_sub`)**:
   - Trained on 4,344 non-gold studies (`gold_in_training = 0`), making it completely clean on the 58 expert-labelled studies.
3. **Stage 3 Per-Finding Rank Blend**:
   - Combines the Raptor arm and the R6 arm per finding:
   $$\text{Stage3}_c = w_{\text{raptor}, c} \cdot R_{\text{raptor}, c} + (1 - w_{\text{raptor}, c}) \cdot \text{pct}(\text{R6}_c)$$
4. **Part B — 2.5D `ConvNeXt` MIL Reader Injection (`0.3% – 3.5%`)**:
   - Adds small finding-specific rank weights (`3.5%` on Baker's cyst, `3.0%` on Contusion, `2.5%` on Medial OA / ACL) to form the **`v74` anchor**.
5. **Part C — All-Data `S6` (`ConvNeXt-small`, `E_prior_s42_fall.pt`) Rank Mix**:
   - Runs last behind a strict `8.40 h` soft stop / `8.55 h` hard kill guard and blends into the `v74` anchor at weight $W_{\text{S6}} \in \{0.20, 0.35\}$:
   $$\text{Final}_c = (1 - W_{\text{S6}}) \cdot \text{pct}(v74_c) + W_{\text{S6}} \cdot \text{pct}(\text{S6}_c)$$

---

## 3. Four Findings That Actually Moved the Needle

### A. Multilingual Report Mining via LLM Yes/No Logit Probing (`+0.22` AUC over Regex)
Out of 4,407 training studies, **only 58 carry radiologist ground-truth labels**, while 61% of the free-text radiology reports are non-English.
- Rule-based regex extraction scored **0.629 AUC** on the 58 gold studies (near chance on non-English reports).
- Replacing regex with **single-forward-pass logit probing on `Qwen3-4B`** (`P("Yes") / (P("Yes") + P("No"))` per finding question) jumped report-label quality to **0.845 AUC** (`+0.216` AUC) and powered our in-house `0.906` Public LB baseline.
- **Deduplication lesson**: 4,407 studies share only 4,257 unique reports (one boilerplate normal report appears 37 times). Grouping CV folds by normalized report hash is essential to prevent cross-fold leakage.

### B. Leakage Audit on the 58 Gold Studies
To evaluate ensemble weights before the deadline, we ran the full inference pipeline over **104 training studies** (the 58 expert-labelled studies + 46 random unlabelled training studies so percentile ranking stayed in >100-row production mode). Before tuning a single weight, we audited checkpoint provenance:

| Component | Saw Gold-58 in Training? | Gold-58 Macro AUC | Implication for Blending |
| :--- | :--- | :---: | :--- |
| **R6 (`ConvNeXt-small` x4)** | **No** (`gold_in_training = 0`) | **0.9366** | Unbiased held-out measurement |
| **S6 (`ConvNeXt-small` all-data)** | **No** (`expect_holdout = 0`) | **0.9314** | Unbiased held-out measurement |
| **Raptor-384 / Raptor-224** | Not trained on them, but **checkpoint selected on Gold-58** | 0.9228 / 0.9150 | **Selection-biased upward on Gold-58** |
| **2.5D MIL Reader** | No | 0.9153 | Clean complementary reader |

Because Raptor's checkpoint was selected on the 58 gold studies, any naive grid search on Gold-58 over-weights Raptor. Conversely, shifting weight *away* from Raptor toward **R6** and **S6** works against selection bias.

### C. Honest 600-Split Half-Sample Validation (`split_half_cv.py`)
In-sample optimization on 58 studies easily overfits noise. Instead of trusting in-sample Gold-58 gains, we evaluated *selection rules* across **600 random 29/29 splits** (pick weights on Half A, score macro AUC on unseen Half B):
- **Raising $W_{\text{S6}}$ from `0.10` (`v89`) to `0.20` (`v90`) and `0.35` (`v96-a`)**: Improved Gold-58 by `+0.0010` to `+0.0014` (`P(better) = 0.76–0.87`), climbed **87 ranks** inside the `0.958` public tie, and improved **Private LB from `0.942` (`v89`) to `0.943` (`v90`, `v96-a`)**.
- **Correcting Stage 3 Synovitis (`0.85 → 0.50/0.35`) & PF OA (`0.65 → 0.50/0.35`)**: The inherited Stage 3 weights gave Raptor 85% of Synovitis and 65% of PF OA, even though clean R6 beat the Raptor stack by `+0.035` on Synovitis and `+0.032` on PF OA. Our conservative split-half rule ("keep `v90` or revert to `0.50`") yielded **`+0.0014` held-out macro AUC** (`helps 83% / hurts 9%` of splits).
- **What failed the split-half test**: Free 12-finding grid searches (`hurts 24–34%`), per-finding S6 weights (`-0.0006` held-out mean, `hurts 65%`), label co-occurrence nudges (`hurts 15–48%`), and aggressive R6-max weights (`0.15` Raptor weight in `v96-c`, which dropped to `0.955` Public / `0.941` Private).

### D. Final-Day Public vs. Private Leaderboard Alignment

| Variant | Summary of Changes vs `v90` | Gold-58 (In-Sample) | Public LB | **Private LB** |
| :--- | :--- | :---: | :---: | :---: |
| **`v90` (Selected)** | Base `v74` anchor + `W_S6 = 0.20` | 0.9386 | **0.958** | **0.943** |
| **`v96-a` (Selected)** | R6-lean Stage 3 (`LatOA 0.0, ACL/PF/Syn 0.35`) + `W_S6 = 0.35` | 0.9438 | **0.958** | **0.943** |
| `v96-b` | Same R6-lean Stage 3 + `W_S6 = 0.45` | 0.9440 | 0.957 | **0.943** |
| `v95` | Stage 3 `PF/Syn 0.35, Frac 0.75` + 2-tier S6 (`0.35/0.20`) | 0.9442 | 0.957 | **0.943** |
| `v94` | Stage 3 `PF/Syn/Baker's 0.50` + `W_S6 = 0.35` | 0.9419 | 0.957 | 0.942 |
| `v89` | Base `v74` anchor + `W_S6 = 0.10` | 0.9375 | **0.958** | 0.942 |
| `v96-c` | Aggressive R6-max (`ACL/PF/Syn 0.15`) + `W_S6 = 0.45` | 0.9446 | 0.955 | 0.941 |

---

## 4. Engineering & Reliability Guards for 9-Hour Code Competitions

1. **Strict Commit Run, Forgiving Hidden Run**: On the 3-study commit run, any missing checkpoint, failed study, or parity mismatch raises immediately (`parity_max_abs_dp == 0.0` verified against reference outputs). On the hidden test run, every stage is wrapped in fallback guards so a single corrupt DICOM series never aborts `submission.csv`.
2. **Jupyter `nbconvert` Pitfall**: Programmatically modifying `.ipynb` JSON and leaving `"outputs": null` on a **markdown** cell causes Kaggle's final `nbconvert` step to mark a completed GPU run as `ERROR`. Stripping `"outputs"` from markdown cells in our builder script eliminated this failure mode.
3. **Deterministic Execution**: Four separate reruns of `v90` produced identical Public and Private scores (`0.958` / `0.943`), confirming zero run-to-run lottery variance.
