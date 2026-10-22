# Evaluation on the 58 expert-labelled studies

Only 58 of 4,407 training studies have expert labels. Every other label is mined from a radiology report. The 58 are
the only yardstick that matches the hidden test's labels, so they decide weights, but they are small and
positive-enriched (Fracture 4.4x, Synovitis 3.4x, ACL 2.9x the corpus rate). This page is the protocol we used to
avoid fooling ourselves with them.

## The evaluation run

`pipeline/eval/gold58-eval.ipynb` runs the full v90 notebook on **58 labelled + 46 random other training studies**
(104 rows) and dumps every component's predictions: `raptor_raw.csv`, `raptor224_raw.csv`, `ours_sub.csv` (R6),
`_own.csv` (reader), `s6_raw.csv`, and the stage outputs.

* Ranks are taken over all 104 rows, as in a real run, and AUC is measured on the 58.
* Staying above 100 rows keeps the notebook in hidden-run mode, so the commit-run-only checks don't fire.
* From these dumps, `knee.blend.full_chain` rebuilds any variant offline. For v90 it gives 0.9386, the same score as
  the notebook's own `submission.csv`.

## Leakage audit

| Component | Saw the 58 in training? | Consequence |
|---|---|---|
| R6 | No (`gold_in_training: 0`, pool of 4,344 non-gold studies) | Its gold-58 score is a held-out measurement |
| S6 all-data | No (4,263 studies, gold excluded, `expect_holdout 0`) | Held-out |
| Raptor-384 / 224 | Not trained on them, but the **checkpoint was selected on gold-58 AUC** | Its gold-58 scores are inflated, so any blend fitted on the 58 leans too far toward Raptor |

So a weight change *toward* R6 or S6 is conservative (it works against the bias), while a change *toward* Raptor
(for example Fracture 0.75) is suspect.

## Single components (gold-58 macro AUC)

| Raptor-384 | Raptor-224 | R6 | Reader | S6 |
|---|---|---|---|---|
| 0.9228 | 0.9150 | 0.9366 | 0.9153 | 0.9314 |

## Variants (gold-58 AUC, in-sample)

| Finding | v90 | v94 | v95 | v96-a | v96-b | v96-c |
|---|---|---|---|---|---|---|
| ACL | 0.9890 | 0.9902 | 0.9902 | 0.9926 | 0.9926 | 0.9939 |
| MCL | 0.9705 | 0.9705 | 0.9705 | 0.9705 | 0.9751 | 0.9751 |
| Medial Meniscus | 0.9736 | 0.9712 | 0.9736 | 0.9712 | 0.9712 | 0.9712 |
| Lateral Meniscus | 0.9255 | 0.9354 | 0.9354 | 0.9354 | 0.9342 | 0.9342 |
| Medial OA | 0.9876 | 0.9891 | 0.9891 | 0.9891 | 0.9891 | 0.9891 |
| Lateral OA | 0.8549 | 0.8549 | 0.8549 | 0.8627 | 0.8627 | 0.8627 |
| PF OA | 0.8893 | 0.8996 | 0.9073 | 0.9073 | 0.9086 | 0.9138 |
| Effusion | 0.9839 | 0.9801 | 0.9839 | 0.9801 | 0.9789 | 0.9789 |
| Synovitis | 0.8256 | 0.8483 | 0.8530 | 0.8530 | 0.8542 | 0.8554 |
| Baker's | 0.9783 | 0.9801 | 0.9783 | 0.9801 | 0.9819 | 0.9819 |
| Contusion | 0.9568 | 0.9568 | 0.9568 | 0.9568 | 0.9541 | 0.9541 |
| Fracture | 0.9278 | 0.9264 | 0.9375 | 0.9264 | 0.9250 | 0.9250 |
| **Macro** | **0.9386** | **0.9419** | **0.9442** | **0.9438** | **0.9440** | **0.9446** |

S6 weight on the v90 Stage 3: 0.00 → 0.9372, 0.10 → 0.9375, 0.20 → 0.9386, 0.30-0.45 → 0.9395-0.9396 (flat),
0.60 → 0.9375. The public leaderboard agreed on direction (0.10 → 0.20 moved us up 87 places).

These numbers are **in-sample**: the weights were chosen after looking at the same 58 studies. They overstate the
real gain. The split-half test below estimates it honestly.

## Split-half test

A *rule* for choosing weights is applied to a random half (29 studies) and scored on the other half, over 600 random
splits (`knee.metrics.split_half`, `scripts/split_half_cv.py`). We trusted a rule only if it helped the held-out half
in at least 80 % of splits and hurt in at most 10 %.

| Rule (on top of v90 Stage 3, W_S6 0.35) | Held-out mean | Helps | Hurts | Trusted |
|---|---|---|---|---|
| Per finding: keep the v90 weight or revert to the neutral 0.5 (**v94**) | +0.0014 | 83 % | 9 % | yes |
| Per finding: free choice from {0.15 … 0.85} | +0.0012 to +0.0021 | 66-76 % | 24-34 % | no |
| v95's full search (Stage 3 grid x per-finding S6 {0.20, 0.35}) | +0.0014 | 74 % | 26 % | no |
| Per-finding S6 weight {0.20, 0.35} alone | -0.0006 | 22 % | 65 % | no |
| Co-occurrence nudges (e.g. Synovitis from Effusion) | -0.0007 to +0.0009 | ≤ 70 % | ≥ 15 % | no |
| Raptor-224 share per finding | -0.0001 | 38 % | 39 % | no |

Reading: the one robust lever was moving Synovitis and PF OA off their Raptor-heavy weights. Everything finer than
that fits noise. The v96 variants go further in the same direction (toward R6 and S6). The split-half gain for that
extra step is small (+0.0001 to +0.0006); the case for it rests on the Raptor selection bias above.

## Bootstrap (in-sample, vs v90)

| v92 (S6 0.35) | v94 | v95 | v96-a | v96-b | v96-c |
|---|---|---|---|---|---|
| +0.0010 (P 0.76) | +0.0033 (P 0.95) | +0.0057 (P 0.99) | +0.0053 (P 0.98) | +0.0054 (P 0.98) | +0.0061 (P 0.97) |

These P values are optimistic for the same reason as the in-sample AUCs. The honest expectation for the final-day
variants is about +0.002 to +0.003 macro AUC over v90.
