# Lessons

What worked, what didn't, and what we would do differently. The chronological detail is in
[research-log.md](research-log.md).

## Labels

* **The reports are the label source, and an LLM reads them far better than regex.** Logit-scored yes/no questions
  (Qwen3-4B, one forward pass per question, `P(Yes) / (P(Yes) + P(No))`) reached 0.845 AUC on the 58 against 0.629
  for regex. Regex was at chance on non-English reports, which are 61 % of the corpus.
* **The 58 are a curated, abnormality-rich sample.** Never read prevalence or priors off them.
* **Duplicate reports leak across folds.** 4,407 studies share 4,257 reports (one report covers 37 studies). Folds
  must group on a hash of the normalised report.
* **Freeze the fold split.** A label update reshuffled 75 % of the folds through stratification and silently broke
  every comparison between runs. The split is now computed once and committed.
* **Rank conditions by image-model headroom, not label quality.** Our v4 relabel improved findings the image model
  already handled. The real bottlenecks were Synovitis, PF OA, Lateral OA and Lateral Meniscus.

## Images

* **Laterality was destroyed in three ways at once:** flip augmentation, flip TTA, and no canonicalisation. On
  sagittal series the medial-lateral axis is the slice axis, so a horizontal flip does not swap medial and lateral.
  A later "fix" that swapped heads under flip TTA cost -0.008 on the leaderboard.
* **The cache fed 97 % fluid-sensitive series** and discarded the PD/T1 anatomy series that OA findings are read on.
* **A 448-pixel model turned out to be a 2x2 montage of 224-pixel tiles.** Check what a "high-res" checkpoint
  actually sees before blending it.

## Ensembling

* **Only new, strong models moved the leaderboard.** Reweighting, probit/quantile fusion, power sharpening and
  agreement gates never beat their base by a visible margin.
* **A third model needs to be strong and different:** roughly >= 0.94 alone at rank correlation <= 0.70 with the
  blend. Weak but "diverse" members (DINOv2 at 0.665, V-JEPA at 0.54, a specialist at chance) only dilute.
  An offline gain of +0.035 from a 0.65-correlated in-house arm never transferred to the leaderboard.
* **The top of the public leaderboard was one shared ensemble.** About 1,500 teams tied at 0.958, ordered by unrounded
  score and then submission time. Small, genuine improvements still moved us hundreds of places inside the tie.
* **Weights inherited from a public notebook encode someone else's validation.** Re-check them against the labels
  you trust. Here the Raptor-heavy Synovitis (0.85) and PF OA (0.65) weights were wrong on the 58.

## Evaluation

* **In-sample bootstrap probabilities on 58 studies are optimistic.** Use a split-half test of the selection rule.
  Free per-finding tuning helped the held-out half only about 2 times in 3. The simple "keep or revert to neutral"
  rule helped 83 % of the time.
* **On the final day the public leaderboard predicted private better than the 58 did.** Every variant tuned on the
  58 scored higher than v90 there, yet none beat v90 on private, and the two the public board marked down were
  also the weakest on private. A 0.001 public drop was a signal, not just noise.
* **Know which components saw the evaluation set.** Raptor was checkpoint-selected on the 58, so the 58 flatter it.
  R6 and S6 never saw them.
* **Gate every training run with a smoke test.** One 6-hour GPU run trained at chance (AUC ~0.53) before anyone
  noticed.

## Kaggle engineering

* **A markdown cell with `"outputs": null` crashes Kaggle's final render.** A fully successful run is then marked
  ERROR. The builder now strips `outputs` from markdown cells.
* **P100 cannot run the image's torch build.** Pin the T4 machine shape explicitly.
* **`/kaggle/input` layout is not guessable.** Discover mounts by a marker file.
* **Make the commit run strict and the hidden run forgiving.** Every failure raises on the 3-row commit run, while
  the hidden run falls back to the last good file. A parity check on the 3 public test studies proves the new
  component renders exactly as it was trained.
* **Runtime is a hard budget.** The pipeline spends about 8 of its 9 hours on the hidden test, so a new component
  must run last behind a deadline, and CPU inference is out (223 s per study for one model).
* **Reruns of a deterministic pipeline score identically.** Four reruns of v90 all gave the same score, so they were
  not a variance lottery.

## What separated top-5 gold solutions from our stack

Comparing our pipeline (`0.958` public / `0.943` private, 499th) against the [5th-place write-up](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/5th-place-solution) highlights four concrete gaps that post-hoc rank blending could not close:

1. **Bone-segmentation-anchored anatomical ROI crops (2x effective resolution):** Our models saw full-field volumes or a fixed `0.8` center crop (`Raptor-224`). The 5th-place team ran a bone-segmentation model to anchor high-resolution crops on the **patellofemoral joint**, **medial/lateral compartments**, and **central knee joint**, directly addressing our weakest compartments (`PF OA`, `Lateral OA`, `Lateral Meniscus`).
2. **Tri-modal soft labels + iterative OOF pseudo-labeling:** We mined reports with a text LLM (`Qwen3-4B`) once. The 5th-place team combined (a) LLM report extraction, (b) vision-model interpretations of selected MRI slices, and (c) iterative out-of-fold teacher pseudo-labels from earlier image models.
3. **Using the 58 gold studies to calibrate training targets (not just final blend weights):** Rather than only tuning post-hoc ensemble weights on the 58 expert-labelled studies, they used the 58 studies upstream to calibrate soft-label probabilities before training their CNN families (`EfficientNetV2-M`, `ConvNeXt-Base`, `ConvNeXt-Tiny`).
4. **Compartment-anchored specialist models:** Our unanchored specialist experiments failed to transfer (`~0.494` in-pipeline on the 58), whereas bone-segmented ROI crops allowed the 5th-place team to train dedicated region specialists (`M4`) and four fine-tuned `Lateral Meniscus` specialists (`M5`).
