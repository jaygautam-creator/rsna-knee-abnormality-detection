# Part C (v90 = v89 with W_S6 0.10 -> 0.20): nartaa's S6 all-data ConvNeXt (E_prior_s42_fall.pt, 0.950 public alone), run AFTER the complete v74 chain, at the
# geometry it was trained and scored with (nartaa/rsna-knee-s6fall-solo k1/B/argv_part0.json: size 320, crop 288, v4, 3 slices,
# 3.5 mm, eval stride 1, fp16; no side model, no state spec, no flip TTA, no stackers).  Mixed into the final ranks at one fixed
# weight for every finding, chosen before any run and not fitted:
#   R74 = pct(v74 final), S = studies S6 scored,
#   final = pct(R74 + W_S6 * (pct_S(S6) - pct_S(R74)))   on S;  R74 elsewhere.
# Full coverage: exactly pct(0.80 * R74 + 0.20 * R_S6).  Partial coverage (deadline): a centred nudge inside S, the rest keep R74.
# Commit run (< 100 test rows): any failure, or a parity miss against nartaa's own scored B output on the 3 public test studies,
# raises.  Hidden run: any failure keeps the v74 file byte for byte.
import csv as _c_csv, signal as _c_signal
import numpy as np
import pandas as pd

W_S6 = 0.20
S6_SLUG, S6_FILE = 'rsna-knee-s6fall-weights', 'E_prior_s42_fall.pt'
S6_SHA = '0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12'
S6_ARGS = ['--backbone', 'convnext_small', '--size', '320', '--crop', '288', '--views', 'v4', '--slices', '3',
           '--phys-mm', '3.5', '--eval-stride', '1', '--chunk', '192', '--item-timeout', '900',
           '--anchor-weights', str(CODE_SRC / 'anatomy.pt')]
S6_STOP_H = 8.40          # compact_infer --deadline: no new study starts after this (hours from NB_T0)
S6_KILL_H = 8.55          # hard kill of the whole S6 process group
S6_MIN_LEFT_H = 0.50      # do not start S6 with less than this before S6_STOP_H
S6_MIN_COVER = 0.25       # fewer scored studies than this fraction: keep v74
S6_MIN_RHO_MEAN, S6_MIN_RHO_FINDING = 0.50, 0.30   # hidden run: Spearman(S6, v74) floor on the scored rows (garbage-render guard)
S6_PARITY_TOL = 0.02      # max |p - p_ref| on the 3 public test studies (same docker image, T4, fp16)
S6_PARITY = {}            # [public copy] the 3 public test-study reference rows are competition data and are not
                          # distributed; knee.variants re-inserts them from pipeline/local/s6_parity.json
S6_LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis',
             "Baker's", 'Contusion', 'Fracture']
assert 0.0 < W_S6 < 0.5 and S6_STOP_H < S6_KILL_H < 8.75


def _c_pct(v):
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values


def s6_twin_blend(v74_df, got, w):
    """v74_df: the v74 submission (StudyInstanceUID + labels, any row order); got: {uid: [12 S6 probabilities]}.
    Returns (blended frame in v74's row / column order, info)."""
    labels = [c for c in v74_df.columns if c != 'StudyInstanceUID']
    ids = v74_df['StudyInstanceUID'].astype(str).tolist()
    ok = np.array([u in got for u in ids])
    out = v74_df.copy()
    rho = {}
    for j, c in enumerate(labels):
        r74 = _c_pct(v74_df[c].values)
        b = r74.copy()
        if ok.any():
            s6c = np.array([got[u][S6_LABELS.index(c)] for u, k in zip(ids, ok) if k], dtype=np.float64)
            # = (1 - w) R74 + w R_S6 at full coverage (pct_S(R74) == R74 there, so the last term is exactly 0)
            b[ok] = (1.0 - w) * r74[ok] + w * _c_pct(s6c) + w * (r74[ok] - _c_pct(r74[ok]))
            if ok.sum() >= 3:
                rho[c] = float(pd.Series(s6c).corr(pd.Series(r74[ok]), method='spearman'))
        out[c] = _c_pct(b + 1e-9 * r74).astype(np.float32)   # exact ties of the mix (frequent: k/n grid) follow v74's order
    return out, {'scored': int(ok.sum()), 'rows': len(ids), 'rho_s6_vs_v74': rho}


SUB = WORK / 'submission.csv'
V74_COPY = WORK / 'submission_v74.csv'
shutil.copyfile(SUB, V74_COPY)
_c_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
COMMIT_RUN = len(_c_ids) < 100
s6 = {'w_s6': W_S6, 'status': 'not_run', 'commit_run': COMMIT_RUN}
_c_t0 = time.time()
try:
    left_h = S6_STOP_H - (time.time() - NB_T0) / 3600
    s6['start_h'] = round((time.time() - NB_T0) / 3600, 3)
    s6['r6_seconds'] = receipt.get('seconds') if isinstance(receipt, dict) else None
    if left_h < S6_MIN_LEFT_H:
        raise TimeoutError('only %.2f h before the S6 stop; v74 kept' % left_h)
    S6_SRC = find_mount(S6_SLUG, S6_FILE)
    S6_BUNDLE = TMP / 'bundle_s6f'
    S6_BUNDLE.mkdir(parents=True, exist_ok=True)
    assert hashlib.sha256((S6_SRC / S6_FILE).read_bytes()).hexdigest() == S6_SHA, 'S6 sha256 mismatch'
    shutil.copy(S6_SRC / S6_FILE, S6_BUNDLE / S6_FILE)
    assert sorted(p.name for p in S6_BUNDLE.glob('E_prior_s*_f*.pt')) == [S6_FILE]
    PRED_S6 = TMP / 'prediction_s6f'
    cmd = [sys.executable, '-u', str(SRC / 'probe_dual.py'), '--data', str(DATA), '--split', 'test', '--code', str(CODE),
           '--bundle', str(S6_BUNDLE), '--arm', 'E_prior', '--output', str(PRED_S6), '--amp', 'fp16', '--loader-workers', '2',
           '--deadline', '%.0f' % (NB_T0 + S6_STOP_H * 3600), '--worker', 'compact_infer.py',
           '--worker-args', json.dumps(S6_ARGS), '--max-fallback-fraction', '1.0']
    print('S6_START', json.dumps({'left_h': round(left_h, 2), 'r6_seconds': s6['r6_seconds']}), flush=True)
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        s6['rc'] = proc.wait(timeout=max(60.0, NB_T0 + S6_KILL_H * 3600 - time.time()))
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, _c_signal.SIGKILL)
        proc.wait()
        s6['rc'] = 'killed'
    for log in sorted(PRED_S6.glob('worker*/log.txt')):
        shutil.copyfile(log, WORK / ('s6f_' + log.parent.name + '_log.txt'))
    got = {}
    for f in PRED_S6.glob('worker*/predictions.csv'):
        with open(f, encoding='utf-8') as fh:
            for r in _c_csv.DictReader(fh):
                try:
                    v = [float(r[L]) for L in S6_LABELS]
                except (KeyError, ValueError, TypeError):
                    continue
                if all(np.isfinite(x) and 0.0 <= x <= 1.0 for x in v):
                    got[r['StudyInstanceUID']] = v
    par = [u for u in S6_PARITY if u in got]
    if par:
        s6['parity_max_abs_dp'] = round(max(abs(a - b) for u in par for a, b in zip(got[u], S6_PARITY[u])), 6)
        s6['parity_studies'] = len(par)
        if s6['parity_max_abs_dp'] > S6_PARITY_TOL:
            raise RuntimeError('S6 parity miss vs nartaa scored output: max |dp| %.4f' % s6['parity_max_abs_dp'])
    elif COMMIT_RUN:
        raise RuntimeError('S6 scored none of the 3 public parity studies')
    v74_df = pd.read_csv(V74_COPY, dtype={'StudyInstanceUID': str})
    assert sorted(v74_df['StudyInstanceUID']) == sorted(_c_ids), 'v74 rows differ from test.csv'
    assert sorted(c for c in v74_df.columns if c != 'StudyInstanceUID') == sorted(S6_LABELS), 'v74 columns differ'
    if len(got) < max(3, S6_MIN_COVER * len(_c_ids)):
        raise RuntimeError('S6 scored %d of %d studies (< %.0f %%)' % (len(got), len(_c_ids), 100 * S6_MIN_COVER))
    blended, info = s6_twin_blend(v74_df, got, W_S6)
    s6.update(info)
    if not COMMIT_RUN:
        rv = list(info['rho_s6_vs_v74'].values())
        if not rv or np.mean(rv) < S6_MIN_RHO_MEAN or min(rv) < S6_MIN_RHO_FINDING:
            raise RuntimeError('S6 vs v74 rank correlation too low: %s' % info['rho_s6_vs_v74'])
    L = [c for c in blended.columns if c != 'StudyInstanceUID']
    assert blended['StudyInstanceUID'].tolist() == v74_df['StudyInstanceUID'].tolist()
    assert np.isfinite(blended[L].values).all() and ((blended[L].values >= 0) & (blended[L].values <= 1)).all()
    blended.to_csv(WORK / '_s6_twin.csv', index=False)
    os.replace(WORK / '_s6_twin.csv', SUB)
    s6['status'] = 'blended'
except Exception as _c_e:
    shutil.copyfile(V74_COPY, SUB)
    s6['status'] = 'v74_kept'
    s6['reason'] = repr(_c_e)[:500]
    if COMMIT_RUN:
        (WORK / 's6_receipt.json').write_text(json.dumps(s6, indent=1))
        raise
s6['seconds'] = round(time.time() - _c_t0, 1)
s6['total_hours'] = round((time.time() - NB_T0) / 3600, 3)
(WORK / 's6_receipt.json').write_text(json.dumps(s6, indent=1))
print('S6_TWIN', json.dumps(s6), flush=True)
