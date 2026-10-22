"""Generate Gold-Calibrated Continuous Posterior Labels for RSNA Knee Abnormality Detection.

Calculates genuine empirical P(Gold=1 | Report State) using the TRUE 58 Gold studies
from data/train.csv, preserving continuous margin rankings and avoiding binary collapse.
"""

import numpy as np
import pandas as pd

from knee.paths import REPO_ROOT as ROOT

LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture'
]

def main():
    print("=" * 75)
    print("BUILDING GENUINE GOLD-CALIBRATED POSTERIOR LABELS (TRUE GROUND TRUTH)")
    print("=" * 75)

    train_path = ROOT / 'data' / 'train.csv'
    v4_path = ROOT / 'artifacts' / 'train_labels_v4.csv'

    train_df = pd.read_csv(train_path)
    v4_df = pd.read_csv(v4_path)

    # Extract genuine gold studies (exact binary annotations 0.0 or 1.0)
    gold_df = train_df.dropna(subset=LABELS).copy()
    assert len(gold_df) == 58, f"Expected 58 gold studies, found {len(gold_df)}"

    gold_sids = set(gold_df['StudyInstanceUID'].astype(str))
    print(f"Verified {len(gold_df)} genuine Gold studies with binary ground truth.")
    print(f"Total training cohort: {len(v4_df)} studies.")

    # Calculate empirical posterior P(True Gold=1 | Rep=1) and P(True Gold=1 | Rep=0) with Laplace smoothing
    posterior_pos = {}
    posterior_neg = {}

    gold_lookup = gold_df.set_index('StudyInstanceUID')
    v4_lookup = v4_df.set_index('StudyInstanceUID')

    print("\n--- Empirical Gold Posteriors on the 58 Ground-Truth Studies ---")
    for c in LABELS:
        g = np.array([gold_lookup.loc[sid, c] for sid in gold_sids], dtype=np.float64)
        rep = np.array([v4_lookup.loc[sid, c] for sid in gold_sids], dtype=np.float64)

        pos_mask = (rep >= 0.5)
        neg_mask = (rep < 0.5)

        n_pos = pos_mask.sum()
        n_pos_gold = (g[pos_mask] == 1.0).sum()

        n_neg = neg_mask.sum()
        n_neg_gold = (g[neg_mask] == 1.0).sum()

        # Laplace smoothed posterior: (k + 1) / (n + 2)
        p_pos = (n_pos_gold + 1.0) / (n_pos + 2.0)
        p_neg = (n_neg_gold + 1.0) / (n_neg + 2.0)

        # Ensure sensible bounds
        p_pos = float(np.clip(p_pos, 0.70, 0.98))
        p_neg = float(np.clip(p_neg, 0.02, 0.25))

        posterior_pos[c] = p_pos
        posterior_neg[c] = p_neg
        print(f"  {c:<18}: N_pos={n_pos:>2} (Gold+={n_pos_gold:>2}) -> P(G=1|Rep=1)={p_pos:.3f} | "
              f"N_neg={n_neg:>2} (Gold+={n_neg_gold:>2}) -> P(G=1|Rep=0)={p_neg:.3f}")

    out_df = pd.DataFrame({'StudyInstanceUID': v4_df['StudyInstanceUID']})
    out_df['is_gold'] = out_df['StudyInstanceUID'].isin(gold_sids).astype(int)

    for c in LABELS:
        targets = []
        weights = []
        p_pos = posterior_pos[c]
        p_neg = posterior_neg[c]

        for sid in v4_df['StudyInstanceUID']:
            if sid in gold_sids:
                # Exact ground truth binary label with high loss weight
                exact_g = float(gold_lookup.loc[sid, c])
                targets.append(exact_g)
                weights.append(2.5)  # 2.5x weight for true human expert labels
            else:
                # Preserve continuous ranking margin via linear interpolation between p_neg and p_pos
                margin = float(v4_lookup.loc[sid, c])
                margin = float(np.clip(margin, 0.0, 1.0))
                continuous_target = (1.0 - margin) * p_neg + margin * p_pos
                targets.append(continuous_target)
                # Confidence weight based on how definitive the report margin is
                conf_weight = 1.0 if (margin > 0.8 or margin < 0.2) else 0.8
                weights.append(conf_weight)

        out_df[c] = targets
        out_df[f'{c}__weight'] = weights

    out_path = ROOT / 'artifacts' / 'train_labels_posterior_calibrated.csv'
    out_df.to_csv(out_path, index=False)
    print(f"\nSuccessfully generated {out_path} ({len(out_df)} studies)")

if __name__ == '__main__':
    main()
