# Results

Public leaderboard = 30 % of the hidden test. Final standings use the private 70 %. The full list of 112 submissions
is in [submission-history.csv](submission-history.csv).

## Final result

**Private 0.943, rank 499 of 6,425 teams (bronze medal).** The final picks were v90 (57479867) and v96-a
(57527153). Both scored 0.943 on private, the best of all 112 submissions, so no other pick would have done better at
the leaderboard's 3-decimal precision. The public fork we started from (v74) scored 0.942 private; the S6 blend was
worth about +0.001.

## Milestones (public score)

| Date | Public | Submission |
|---|---|---|
| 2026-08-08 | 0.826 | First submission |
| 2026-08-19 | 0.906 | In-house 4-checkpoint ConvNeXt blend with laterality canonicalisation (best in-house result) |
| 2026-08-20 | 0.918 | Reproduction of the public ensemble |
| 2026-09-08 | 0.935 | v8: CoAtNet-2 + DINOv2 + in-house ConvNeXt |
| 2026-09-17 | 0.941 | v11: 4-view Raptor CoAtNet + specialist routing |
| 2026-10-07 | 0.945 | v39: per-finding power routing on the community stack |
| 2026-10-09 | 0.954 | heliosli's S6 compact model + CoAtNet-384 |
| 2026-10-18 | **0.958** | v74: Raptor-384 + Raptor-224 + R6 flip TTA + 2.5D MIL (fork of the public 0.958 recipe) |
| 2026-10-20 | 0.958 | v89: v74 + 0.10 S6 all-data rank mix (rank 1,568 → 425 inside the 0.958 tie) |
| 2026-10-21 | 0.958 | v90: S6 weight 0.20 (rank 455 → 368). Four reruns scored identically: the pipeline is deterministic |

Between v74 and v89, eleven submissions (v75-v88) stayed at or below 0.957: DINOv2 as a fourth arm, probit fusion,
amplified reader weights, a 448-pixel montage model (-0.002), R7 instead of R6, and a laterality "fix" that swapped
medial/lateral heads (-0.008). See [lessons.md](lessons.md).

## Final day (2026-10-22)

All five final-day submissions are v90 with different constants. Configs are in `pipeline/variants/`; gold-58
numbers are in [evaluation.md](evaluation.md).

| Variant | Change vs v90 | Gold-58 (in-sample) | Submission | Public | Private |
|---|---|---|---|---|---|
| v94 | Stage 3 PF OA / Synovitis / Baker's → 0.5; S6 0.35 | 0.9419 | 57522189 | 0.957 | 0.942 |
| v95 | Stage 3 PF OA / Synovitis 0.35, Fracture 0.75; per-finding S6; longer S6 window | 0.9442 | 57524258 | 0.957 | 0.943 |
| v96-a | R6-lean Stage 3 (Lateral OA 0, ACL / PF OA / Synovitis 0.35); S6 0.35 | 0.9438 | 57527153 | 0.958 | 0.943 |
| v96-b | Same Stage 3; S6 0.45 | 0.9440 | 57527149 | 0.957 | 0.943 |
| v96-c | Strongest R6 lean (ACL / PF OA / Synovitis 0.15); S6 0.45 | 0.9446 | 57527151 | 0.955 | 0.941 |
| v90 (reference) | | 0.9386 | 57479867 | 0.958 | 0.943 |

Expected gain over v90 from the split-half test: about +0.002 to +0.003 macro AUC.

Public scores came in flat to slightly lower (0.955-0.958). Final selection: v90 (57479867) and v96-a
(57527153), the only variant that held 0.958 on public while also scoring higher than v90 on the gold-58.

On private, the public leaderboard turned out to be the better guide. The two variants it marked down (v94 at 0.957,
v96-c at 0.955) were also the weakest on private (0.942, 0.941). The extra gold-58 tuning did not beat v90 on private.
