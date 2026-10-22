#!/usr/bin/env python3
"""
Build Gold-Calibrated 3-State Labels for RSNA Knee Abnormality Detection
Solves the systematic under-calling (false negative bias) on unmentioned findings.

Gold ground-truth prevalence (from 58 expert radiologist annotations):
- ACL: 0.414 (vs 0.258 in report labels -> 38% false negatives)
- Lateral Meniscus: 0.397 (vs 0.249 in report labels -> 37% false negatives)
- Fracture: 0.310 (vs 0.187 in report labels -> 40% false negatives)
- Synovitis: 0.466 (vs 0.259 in report labels -> 44% false negatives)
- Effusion: 0.603 (vs 0.475 in report labels -> 21% false negatives)
- Lateral OA: 0.190 (vs 0.293 in report labels -> 54% over-called)

Calibrated formulation:
y_cal = c * y_v5 + (1 - c) * p_gold
where c is the LLM extraction confidence.
"""

from pathlib import Path

import pandas as pd

from knee.paths import REPO_ROOT as ROOT

LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture'
]

GOLD_PRIORS = {
    'ACL': 0.414,
    'MCL': 0.155,
    'Medial Meniscus': 0.448,
    'Lateral Meniscus': 0.397,
    'Medial OA': 0.259,
    'Lateral OA': 0.190,
    'PF OA': 0.362,
    'Effusion': 0.603,
    'Synovitis': 0.466,
    "Baker's": 0.207,
    'Contusion': 0.328,
    'Fracture': 0.310,
}

def main():
    train_path = ROOT / 'data' / 'train.csv'
    v5_path = Path('/tmp/labels_4m/report_labels_v5.csv')
    if not v5_path.is_file():
        v5_path = ROOT / 'artifacts' / 'report_labels_v5.csv'

    print(f'Loading train data from {train_path}...')
    train_df = pd.read_csv(train_path)
    print(f'Loading v5 report labels from {v5_path}...')
    v5_df = pd.read_csv(v5_path).set_index('StudyInstanceUID')

    gold_mask = train_df['ACL'].notna()
    gold_df = train_df[gold_mask].copy().set_index('StudyInstanceUID')
    gold_uids = set(gold_df.index)
    print(f'Identified {len(gold_uids)} Gold studies and {len(train_df) - len(gold_uids)} non-gold studies.')

    out_rows = []
    for _, row in train_df.iterrows():
        uid = str(row['StudyInstanceUID'])
        res = {'StudyInstanceUID': uid, 'is_gold': int(uid in gold_uids)}

        if uid in gold_uids:
            # 1. Exact ground truth for Gold studies
            for cond in LABELS:
                gt_val = float(gold_df.loc[uid, cond])
                res[cond] = gt_val
                res[cond + '__conf'] = 1.0
                res[cond + '__weight'] = 1.0
                res[cond + '__state'] = int(gt_val > 0.5)
        else:
            # 2. Gold-Calibrated 3-State formulation for non-gold studies
            v5_row = v5_df.loc[uid] if uid in v5_df.index else None

            for cond in LABELS:
                p_gold = GOLD_PRIORS[cond]
                p_raw = float(v5_row[cond]) if v5_row is not None else p_gold
                conf = float(v5_row[cond + '__conf']) if (v5_row is not None and cond + '__conf' in v5_row) else 0.5

                # 3-State categorization:
                # State 1: Confident Positive
                # State 0: Confident Negative
                # State -1: Unmentioned / Ambiguous
                if conf >= 0.65 and p_raw >= 0.60:
                    state = 1
                    y_cal = p_raw
                    w = conf
                elif conf >= 0.70 and p_raw <= 0.08:
                    state = 0
                    y_cal = p_raw
                    w = conf
                else:
                    state = -1
                    # Unmentioned: Anchor to empirical gold base rate rather than forcing to 0.0!
                    y_cal = conf * p_raw + (1.0 - conf) * p_gold
                    w = 0.50 + 0.50 * conf  # Bounded weight

                res[cond] = y_cal
                res[cond + '__conf'] = conf
                res[cond + '__weight'] = w
                res[cond + '__state'] = state

            # Clinical Effusion -> Synovitis prior adjustment
            # If high effusion present but synovitis unmentioned, rescue synovitis probability
            if res['Effusion'] >= 0.70 and res['Synovitis__state'] == -1:
                res['Synovitis'] = max(res['Synovitis'], 0.45 * res['Effusion'])
                res['Synovitis__weight'] = 0.75

            # Lateral OA de-biasing
            # If lateral OA is unmentioned / low confidence, curb the over-calling
            if res['Lateral OA__state'] == -1:
                res['Lateral OA'] = min(res['Lateral OA'], 0.22)

        out_rows.append(res)

    out_df = pd.DataFrame(out_rows)
    out_dir = ROOT / 'artifacts'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'train_labels_3state_calibrated.csv'
    out_df.to_csv(out_csv, index=False)
    print(f'Successfully wrote {out_csv} ({len(out_df)} studies).')

    # Distribution comparison
    non_gold_df = out_df[out_df['is_gold'] == 0]
    print('\nCalibrated Distribution on 4,349 Unannotated Studies:')
    print(f'{"Finding":18s} | {"Old v5 Mean":11s} | {"Calibrated Mean":15s} | {"Gold Target":11s}')
    print('-'*65)
    for cond in LABELS:
        old_m = v5_df.loc[~v5_df.index.isin(gold_uids), cond].mean() if v5_df is not None else float('nan')
        new_m = non_gold_df[cond].mean()
        gt_m = GOLD_PRIORS[cond]
        print(f'{cond:18s} | {old_m:11.3f} | {new_m:15.3f} | {gt_m:11.3f}')

if __name__ == '__main__':
    main()
