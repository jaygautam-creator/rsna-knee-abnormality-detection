# =====================================================================
# STAGE 7: IN-HOUSE 5-FOLD TRI-PLANE 2.5D CALIBRATED ENSEMBLE
# =====================================================================
import os, sys, time, gc, json, re, glob
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

cv2.setNumThreads(0)

print("\n" + "=" * 70)
print("STAGE 7: IN-HOUSE 5-FOLD TRI-PLANE 2.5D ENSEMBLE (Target: 0.960+)")
print("=" * 70)

_SUB_PATH = Path('/kaggle/working/submission.csv')
_BACKUP_PATH = Path('/kaggle/working/submission_anchor_0942.csv')

LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture'
]

MENISCI_CONDS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus']
OA_BONE_CONDS = ['Medial OA', 'Lateral OA', 'Fracture']
FLUID_CONDS = ['PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion']

order = MENISCI_CONDS + OA_BONE_CONDS + FLUID_CONDS
inv_perm = [order.index(c) for c in LABELS]

class AnatomicalTriPlane25DNet(nn.Module):
    def __init__(self, in_slices=16, num_classes=12, drop=0.25):
        super().__init__()
        base = models.resnet34(weights=None)
        old_w = base.conv1.weight.data
        self.conv1 = nn.Conv2d(in_slices, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            for c in range(in_slices):
                self.conv1.weight[:, c, :, :] = old_w[:, c % 3, :, :] * (3.0 / in_slices)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        feat_dim = 512
        
        self.head_menisci = nn.Sequential(
            nn.Dropout(drop), nn.Linear(feat_dim * 2, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 4)
        )
        self.head_oa_bone = nn.Sequential(
            nn.Dropout(drop), nn.Linear(feat_dim * 2, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 3)
        )
        self.head_fluid = nn.Sequential(
            nn.Dropout(drop), nn.Linear(feat_dim * 2, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 5)
        )
        self.global_proj = nn.Sequential(
            nn.Dropout(drop), nn.Linear(feat_dim * 3, 12)
        )
        self.register_buffer('inv_perm', torch.tensor(inv_perm, dtype=torch.long))

    def forward_plane(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        B, P, S, H, W = x.shape
        x_flat = x.reshape(B * P, S, H, W)
        f_all = self.forward_plane(x_flat).reshape(B, P, -1)
        f_ax = f_all[:, 0]
        f_cor = f_all[:, 1]
        f_sag = f_all[:, 2]
        
        feat_menisci = torch.cat([f_sag, f_cor], dim=1)
        feat_oa_bone = torch.cat([f_cor, f_sag], dim=1)
        feat_fluid = torch.cat([f_ax, f_cor], dim=1)
        f_cat = torch.cat([f_ax, f_cor, f_sag], dim=1)
        
        p_menisci = self.head_menisci(feat_menisci)
        p_oa_bone = self.head_oa_bone(feat_oa_bone)
        p_fluid = self.head_fluid(feat_fluid)
        
        all_preds = torch.cat([p_menisci, p_oa_bone, p_fluid], dim=1)
        logits = all_preds[:, self.inv_perm] + self.global_proj(f_cat)
        return logits

def find_file(name, root='/kaggle/input'):
    for r, _, fs in os.walk(root):
        if name in fs:
            return Path(r) / name
    return None

try:
    if not _SUB_PATH.exists():
        raise FileNotFoundError("Base submission.csv absent; cannot apply Stage 7")
        
    _anchor_df = pd.read_csv(_SUB_PATH, dtype={'StudyInstanceUID': str})
    _anchor_df.to_csv(_BACKUP_PATH, index=False)
    print(f"Backed up 0.942 anchor submission to {_BACKUP_PATH} ({len(_anchor_df)} studies)")
    
    ckpt_paths = [find_file(f'triplane_fold{k}.pt') for k in range(5)]
    mounted_ckpts = [p for p in ckpt_paths if p is not None and p.exists()]
    print(f"Mounted Tri-Plane checkpoints: {len(mounted_ckpts)}/5 ({[p.name for p in mounted_ckpts]})")
    if len(mounted_ckpts) < 1:
        raise FileNotFoundError("No Tri-Plane checkpoints found! Retaining pure 0.942 anchor submission.")
        
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tri_models = []
    for p in mounted_ckpts:
        m = AnatomicalTriPlane25DNet().to(dev)
        st = torch.load(p, map_location='cpu', weights_only=True)
        m.load_state_dict(st, strict=True)
        m.eval()
        tri_models.append(m)
    print(f"Successfully loaded {len(tri_models)} Tri-Plane models on {dev}")
    
    # Competition metadata discovery with multiple robust fallbacks
    comp_file = find_file('test_series.csv') or find_file('sample_submission.csv') or find_file('test.csv')
    if comp_file is not None:
        comp_root = comp_file.parent
    elif 'COMP' in globals():
        comp_root = Path(globals()['COMP'])
    else:
        raise FileNotFoundError("Could not find competition root directory!")
        
    test_series_csv = comp_root / 'test_series.csv'
    if test_series_csv.exists():
        series_df = pd.read_csv(test_series_csv, dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    elif 'ser_csv' in globals():
        series_df = globals()['ser_csv'].copy()
    else:
        raise FileNotFoundError(f"test_series.csv not found at {test_series_csv}")

    test_sids = _anchor_df['StudyInstanceUID'].astype(str).tolist()
    print(f"Test cohort: {len(test_sids)} studies, {len(series_df)} series")
    
    PLANES = ["Axial", "Coronal", "Sagittal"]
    D_SLICES = 24
    PICK_SLICES = [2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 20, 21]
    SIZE = 336
    
    if 'SERIES_ROOT' in globals() and Path(globals()['SERIES_ROOT']).is_dir():
        series_dir_root = Path(globals()['SERIES_ROOT'])
    else:
        series_dir_root = comp_root / "test_series"
        if not series_dir_root.is_dir():
            series_dir_root = comp_root / "test_images"
        
    sel = series_df.copy()
    sel["Fluid_Sensitive"] = sel["Fluid_Sensitive"].fillna(False).astype(bool)
    n_slices = {}
    for r in sel.itertuples():
        d = series_dir_root / r.StudyInstanceUID / r.SeriesInstanceUID
        n_slices[r.SeriesInstanceUID] = len(list(d.glob("*.dcm"))) if d.exists() else 0
    sel["n"] = sel.SeriesInstanceUID.map(n_slices)
    sel = sel[sel.n > 0]
    sel = (sel.sort_values(["Fluid_Sensitive", "n"], ascending=[False, False])
              .groupby(["StudyInstanceUID", "Anatomical_Plane"], as_index=False)
              .first())
    sel = sel[sel.Anatomical_Plane.isin(PLANES)]
    plan = {sid: g.set_index("Anatomical_Plane").SeriesInstanceUID.to_dict()
            for sid, g in sel.groupby("StudyInstanceUID")}
            
    AXIS_TOL = 0.5
    SIDE_THR_TRAIN = -68.0
    MIN_CALIB = 50
    RIGHT_RE = re.compile(r"\bright\b|\brt\b|\bderech\w+|\bdcha\b|\bdcho\b|\brechts\b|\brechte[sn]?\b|\bsağ\b|\bправ\w+|\bdroit\w*|\bdestr\w+", re.I | re.U)
    LEFT_RE = re.compile(r"\bleft\b|\blt\b|\bizquierd\w+|\bizq\b|\blinks\b|\blinke[sn]?\b|\bsol\b|\bлев\w+|\bgauche\b|\bsinistr\w+", re.I | re.U)

    def _geom(ds):
        iop = getattr(ds, "ImageOrientationPatient", None)
        if iop is None or len(iop) != 6: return None, None
        row = np.array(iop[:3], float)
        col = np.array(iop[3:], float)
        return row, np.cross(row, col)

    def _tag_side(ds):
        for tag in ("Laterality", "ImageLaterality", "FrameLaterality"):
            v = str(getattr(ds, tag, "") or "").strip().upper()
            if v.startswith("R"): return +1
            if v.startswith("L"): return -1
        return None

    def _ipp_x(ds):
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) == 3: return float(ipp[0])
        return None

    def text_side(text):
        if not isinstance(text, str): return 0
        nr, nl = len(RIGHT_RE.findall(text)), len(LEFT_RE.findall(text))
        return 1 if nr > nl else (-1 if nl > nr else 0)

    def study_signals(head_sets):
        votes = [s for hs in head_sets for _, ds in hs if (s := _tag_side(ds)) is not None]
        tag = (+1 if sum(votes) > 0 else -1) if votes else None
        desc = " ".join(str(getattr(ds, "SeriesDescription", "") or "") for hs in head_sets for _, ds in hs[:1])
        d = text_side(desc)
        xs = [x for hs in head_sets for _, ds in hs if (x := _ipp_x(ds)) is not None]
        mean_x = float(np.mean(xs)) if xs else None
        return tag, (d or None), mean_x

    def read_heads(sdir):
        heads = []
        if sdir.exists():
            for f in sdir.glob("*.dcm"):
                try: heads.append((f, pydicom.dcmread(f, stop_before_pixels=True)))
                except Exception: pass
        return heads

    def _position_from(ds, normal):
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) == 3:
            if normal is not None: return float(np.dot(normal, np.array(ipp, float)))
            return float(ipp[2])
        return float(getattr(ds, "InstanceNumber", 0) or 0)

    def _pixels(ds):
        a = ds.pixel_array.astype(np.float32)
        a = a * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
        if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
            a = a.max() - a
        return a

    def load_volume(heads, side):
        zeros = np.zeros((D_SLICES, SIZE, SIZE), np.uint8)
        if not heads: return zeros
        row, normal = _geom(heads[0][1])
        heads = sorted(heads, key=lambda fh: _position_from(fh[1], normal))
        files = [f for f, _ in heads]
        flip_w = flip_d = False
        if row is not None and abs(row[0]) > AXIS_TOL: flip_w = (np.sign(row[0]) != side)
        if normal is not None and abs(normal[0]) > AXIS_TOL: flip_d = (np.sign(normal[0]) != side)
        idx = np.linspace(0, len(files) - 1, D_SLICES).round().astype(int)
        if flip_d: idx = idx[::-1]
        out = np.zeros((D_SLICES, SIZE, SIZE), np.float32)
        for j, i in enumerate(idx):
            try: a = _pixels(pydicom.dcmread(files[i]))
            except Exception: continue
            out[j] = cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        if flip_w: out = out[:, :, ::-1]
        lo, hi = np.percentile(out, [1, 99])
        if hi <= lo: return zeros
        return np.ascontiguousarray((np.clip((out - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8))

    def _prepass(sids):
        rows = {}
        for sid in sids:
            head_sets = []
            for plane in PLANES:
                suid = plan.get(sid, {}).get(plane)
                if suid:
                    hs = read_heads(series_dir_root / sid / suid)
                    if hs: head_sets.append(hs)
            rows[sid] = study_signals(head_sets)
        return rows

    def _fit_threshold(sig):
        pts = [(mx, tag) for tag, _, mx in sig.values() if tag is not None and mx is not None]
        if len(pts) < MIN_CALIB: return SIDE_THR_TRAIN
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        grid = np.arange(-220.0, 60.0, 1.0)
        accs = np.array([(np.where(xs < t, 1, -1) == ys).mean() for t in grid])
        return float(grid[int(accs.argmax())])

    t_pre = time.time()
    SIGNALS = _prepass(test_sids)
    THR = _fit_threshold(SIGNALS)
    SIDE_OF = {}
    for sid, (tag, desc, mx) in SIGNALS.items():
        if tag is not None: SIDE_OF[sid] = tag
        elif desc is not None: SIDE_OF[sid] = desc
        elif mx is not None: SIDE_OF[sid] = (1 if mx < THR else -1)
        else: SIDE_OF[sid] = 1

    class TriPlaneTestDS(Dataset):
        def __init__(self, sids):
            self.sids = sids
        def __len__(self):
            return len(self.sids)
        def __getitem__(self, i):
            sid = self.sids[i]
            vol_planes = np.zeros((3, len(PICK_SLICES), SIZE, SIZE), np.float32)
            side = SIDE_OF.get(sid, 1)
            for p, plane in enumerate(PLANES):
                suid = plan.get(sid, {}).get(plane)
                if suid:
                    hs = read_heads(series_dir_root / sid / suid)
                    if hs:
                        raw_plane = load_volume(hs, side)
                        sub_plane = raw_plane[PICK_SLICES].astype(np.float32)
                        vol_planes[p] = (sub_plane / 255.0 - 0.45) / 0.22
            return torch.from_numpy(vol_planes).float(), i

    test_loader = DataLoader(TriPlaneTestDS(test_sids), batch_size=8, shuffle=False, num_workers=2)
    P_triplane = np.zeros((len(test_sids), len(LABELS)), np.float32)
    t_inf = time.time()
    print("Running Tri-Plane 5-fold inference across test cohort...")
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for step, (xb, idx_batch) in enumerate(test_loader):
            xb = xb.to(dev, non_blocking=True)
            with torch.amp.autocast(device_type, enabled=torch.cuda.is_available()):
                probs = sum(torch.sigmoid(m(xb)) for m in tri_models) / float(len(tri_models))
            P_triplane[idx_batch.numpy()] = probs.cpu().numpy()
            if (step + 1) % 25 == 0 or (step + 1) == len(test_loader):
                print(f"  [{step+1}/{len(test_loader)}] elapsed: {(time.time()-t_inf):.1f}s", flush=True)

    if not np.isfinite(P_triplane).all():
        print("[WARNING] Non-finite values detected in Tri-Plane predictions; clipping to valid range")
        P_triplane = np.nan_to_num(P_triplane, nan=0.5, posinf=1.0, neginf=0.0)

    print(f"Tri-Plane inference complete in {(time.time()-t_inf)/60:.1f} minutes!")

    # Headroom weighting: heavily boosts weak findings (Lateral Meniscus, Synovitis, PF OA, Lateral OA, Fracture)
    FUSE_WEIGHTS = {
        'Lateral Meniscus': 0.40,
        'Synovitis': 0.40,
        'Medial Meniscus': 0.35,
        'PF OA': 0.35,
        'Lateral OA': 0.35,
        'Fracture': 0.35,
        'ACL': 0.30,
        'MCL': 0.20,
        'Contusion': 0.20,
        'Medial OA': 0.15,
        'Effusion': 0.15,
        "Baker's": 0.15,
    }

    df_tri = pd.DataFrame(P_triplane, columns=LABELS, index=test_sids)
    df_anchor_aligned = _anchor_df.set_index('StudyInstanceUID').reindex(test_sids)

    rank_anchor = df_anchor_aligned[LABELS].rank(method='average', pct=True)
    rank_triplane = df_tri[LABELS].rank(method='average', pct=True)

    mean_corr = np.mean([rank_anchor[c].corr(rank_triplane[c], method='spearman') for c in LABELS])
    print(f"\nMean Spearman Rank Correlation (0.942 Anchor vs Tri-Plane): {mean_corr:.3f}")

    df_fused = pd.DataFrame(index=test_sids)
    for c in LABELS:
        w = float(FUSE_WEIGHTS[c])
        df_fused[c] = (1.0 - w) * rank_anchor[c] + w * rank_triplane[c]

    df_final = df_fused.rank(method='average', pct=True).reset_index()
    df_final = df_final.rename(columns={'index': 'StudyInstanceUID'})

    assert len(df_final) == len(test_sids), "Length mismatch"
    assert (df_final['StudyInstanceUID'].values == test_sids).all(), "UID drift"
    assert list(df_final.columns) == ['StudyInstanceUID'] + LABELS, "Column drift"
    assert np.isfinite(df_final[LABELS].to_numpy()).all(), "Non-finite values detected"
    assert (df_final[LABELS].to_numpy() >= 0).all() and (df_final[LABELS].to_numpy() <= 1).all()

    tmp_out = _SUB_PATH.with_suffix('.csv.tmp')
    df_final.to_csv(tmp_out, index=False)
    os.replace(tmp_out, _SUB_PATH)
    print(f"\n>>> BREAKTHROUGH SUBMISSION PUBLISHED SUCCESSFULLY: {_SUB_PATH} ({len(df_final)} studies) <<<")

    receipt = {
        'status': 'SUCCESS',
        'anchor_version': 'infer-v16 (0.942)',
        'triplane_checkpoints': len(mounted_ckpts),
        'mean_rank_correlation': round(float(mean_corr), 4),
        'fusion_weights': FUSE_WEIGHTS,
        'studies': len(df_final)
    }
    Path('/kaggle/working/triplane_blend_receipt.json').write_text(json.dumps(receipt, indent=2))

except Exception as exc:
    print(f"\n[STAGE 7 WARNING] Exception during Tri-Plane integration: {type(exc).__name__}: {exc}")
    import traceback
    traceback.print_exc()
    if _BACKUP_PATH.exists():
        _anchor_df = pd.read_csv(_BACKUP_PATH)
        _anchor_df.to_csv(_SUB_PATH, index=False)
        print("[SAFETY FALLBACK] Banked 0.942 anchor submission.csv safely restored!")
    else:
        print("[SAFETY FALLBACK] Existing submission.csv kept untouched.")
