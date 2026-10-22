# Architecture of the final inference notebook

The submitted notebook is `pipeline/base/infer-v90.ipynb` (v90). Every final-day variant changes only constants in
cells 8 and 10, generated from `pipeline/variants/*.json` by `src/knee/variants.py`.

## Cells

| Cell | Stage | What it does | Origin |
|---|---|---|---|
| 0-1 | | Title, credits and the required OAI acknowledgement | |
| 2-3 | Stage 1, Raptor-384 | CoAtNet-2 (`coatnet_rmlp_2_rw_384`) SWA checkpoint, 96-slice volumes, 384 px. Writes raw probabilities and ranks. Checkpoint verified by SHA-256 | nartaa's public notebook, two marked `[blend]` edits |
| 4-5 | Stage 1b, Raptor-224 | Second checkpoint of the same weights: 0.8 centre crop resized to 224 px, one view | nartaa's public notebook |
| 6-7 | Stage 2, R6 | ConvNeXt-small compact model, 4 fold models, mean logit; flip TTA with medial/lateral heads swapped back; its own Effusion fluid stacker and Baker's state stacker. Runs as a controller with 2 GPU workers and a deadline | heliosli's bundle `fe055406fad6a12f` |
| 8 | Stage 3 | Per-finding rank blend: `w * RaptorHalf + (1 - w) * rank(R6)`, RaptorHalf = `0.65 rank(384) + 0.35 rank(224)`. `w` is `W_RAPTOR_C` (default 0.5) | goodpjw2008's public recipe; final-day variants change `W_RAPTOR_C` |
| 9 | Part B | 2.5D ConvNeXt MIL reader, 3 folds, rank-blended at 0.3-3.5 % per finding, then re-ranked | goodpjw2008 (Apache 2.0) |
| 10 | Part C | S6 all-data ConvNeXt (`E_prior_s42_fall.pt`, 0.950 public alone) at its trained geometry (320/288 px, v4 views), blended at `W_S6` on the studies it scored | Our integration of nartaa's model |

`knee.blend` re-implements cells 8-10 as plain functions, and the gold-58 evaluation reproduces the notebook's
own output from the component dumps (`scripts/evaluate_gold58.py`, v90 = 0.9386 both ways).

## Part C in detail

```
R74 = rank(v74 output),  S = studies S6 scored before its deadline
final = rank( R74 + W_S6 * (rank_S(S6) - rank_S(R74)) )   on S,   R74 elsewhere
```

At full coverage this is exactly `rank((1 - W_S6) R74 + W_S6 R_S6)`. If the deadline stops S6 early, only the scored
studies move, by a nudge centred so that unscored studies keep their v74 rank. Exact ties follow v74's order.
`W_S6` may be a scalar or a per-finding dict.

## Runtime budget (Kaggle T4 x2, 9-hour hard limit)

Marginal cost per test study, measured on a 104-study timing probe:

| Stage | s / study |
|---|---|
| Raptor-384 | 1.2 |
| Raptor-224 | 0.3 |
| R6 (controller, 2 workers) | 18-19 |
| Part B reader | 7.7 |
| S6 all-data | 3.6 |

For a hidden test of ~1,000-1,230 studies the v74 chain ends around 6-7.9 h. S6 runs last, so a deadline miss only
dilutes the newest component:

* `S6_STOP_H` (8.40 h): no new S6 study starts after this.
* `S6_KILL_H` (8.55 h): hard kill of the S6 process group.
* `S6_MIN_LEFT_H` (0.50 h): S6 is skipped entirely if less time than this remains.
* `S6_MIN_COVER` (25 %): fewer scored studies keeps the v74 file.

CPU inference is not an option: Raptor-384 alone takes about 223 s per study on 4 vCPUs (about 76 h for the test).

## Safety guards

* **Commit run vs hidden run.** On Kaggle's commit run (3 public test rows) every failure raises, so a broken
  notebook can never be submitted. On the hidden run any Part C failure keeps the v74 file byte for byte.
* **Parity check.** On the commit run, S6's probabilities for the 3 public test studies must match the reference
  output within 0.02 (they match to 0.0). The reference rows are competition data and are stripped from the public
  copy (see the README).
* **Rank sanity.** In the hidden run, S6 must correlate with v74 (mean Spearman >= 0.5, every finding >= 0.3),
  which guards against a garbage render.
* **Checkpoints** are verified by SHA-256 before loading.
* **Notebook structure.** The builder refuses markdown cells with an `outputs` key: `outputs: null` crashed Kaggle's
  final render, and the platform marked a fully successful run as ERROR.

## Validated before every submission

`scripts/validate_run.py` checks a finished commit run: S6 status `blended`, parity 0.0, the Stage 3 and S6
weights equal the config, no model failures, and `submission.csv` has the right columns and finite values in [0, 1].
