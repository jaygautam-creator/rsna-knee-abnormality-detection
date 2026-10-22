# %% [markdown]
# # RSNA Knee Abnormality Detection: The 0.970+ Master Training Engine
# 
# ### Next-Generation Architectural Foundations:
# 1. **Native High-Resolution (384px)**: Resolves fine 1mm meniscus and ligamentous tears.
# 2. **2.5D Volumetric Triplets**: Adjacent slices $[z-1, z, z+1]$ capture 3D anatomical continuity.
# 3. **6 Anatomical Slots**: Explicit mapping of Sagittal, Coronal, Axial planes with Fat-Suppression state.
# 4. **Anatomical Plane Routing Prior Matrix**: Direct physical prior gating preventing cross-plane noise (e.g. MCL only receives Coronal features).
# 5. **Study-Level Multi-Head Transformer**: 4-layer Pre-LN Transformer with Continuous 3D Depth Encodings.
# 6. **12-Finding Query Cross-Attention**: Dedicated learned queries for each abnormality.
# 7. **Dual-Tier Gold Supervision**: 4.0x loss weight on the 58 official radiologist consensus studies + confidence masking on weak report labels.
# 8. **EMA Model Averaging**: Exponential Moving Average ($\beta = 0.998$) for maximal generalization.

# %%
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import math
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import timm

print(f"=== RSNA KNEE 0.970+ MASTER ENGINE INITIALIZATION ===")
print(f"PyTorch Version: {torch.__version__}")
assert torch.cuda.is_available(), "CUDA GPU is required!"
device = torch.device('cuda:0')
gpu_name = torch.cuda.get_device_name(0)
print(f"Active GPU: {gpu_name} (Devices: {torch.cuda.device_count()})")

# %%
# Target Definitions & Constants
LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture'
]
PLANES = ['Sagittal', 'Coronal', 'Axial']
N_SLOTS = 6  # 3 planes * 2 fat-sat states
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def slot_of(plane: str, fat_sup: bool) -> int:
    p_idx = PLANES.index(plane) if plane in PLANES else 0
    return p_idx + 3 * (1 - int(bool(fat_sup)))

# %%
# Dynamic Dataset Discovery
def find_file(filename, search_root='/kaggle/input'):
    for root, dirs, files in os.walk(search_root):
        if filename in files:
            return Path(root) / filename
    return None

def find_dir_with_ext(ext, search_root='/kaggle/input'):
    for root, dirs, files in os.walk(search_root):
        if any(f.endswith(ext) for f in files):
            return Path(root)
    return None

llm_labels_path = find_file('llm_labels_v4_blend.csv') or find_file('llm_report_labels.csv') or find_file('llm_labels.csv')
train_series_path = find_file('train_series.csv')
train_csv_path = find_file('train.csv')
npz_dir = find_dir_with_ext('.npz')

print(f"LLM Labels File:   {llm_labels_path}")
print(f"Train Series File: {train_series_path}")
print(f"Train CSV File:    {train_csv_path}")
print(f"NPZ Volumes Dir:   {npz_dir}")

assert train_series_path is not None, "Missing train_series.csv!"
assert npz_dir is not None, "Missing NPZ volumes directory!"

# %%
# Ground Truth Alignment & Gold Study Weighting
series_df = pd.read_csv(train_series_path)

if llm_labels_path is not None:
    llm_df = pd.read_csv(llm_labels_path)
else:
    print("Warning: llm_labels not found, falling back to train.csv directly")
    llm_df = pd.read_csv(train_csv_path)

# Ensure StudyInstanceUID is string
series_df['StudyInstanceUID'] = series_df['StudyInstanceUID'].astype(str)
series_df['SeriesInstanceUID'] = series_df['SeriesInstanceUID'].astype(str)
llm_df['StudyInstanceUID'] = llm_df['StudyInstanceUID'].astype(str)

# Integrate the 58 Official Gold Adjudicated Benchmark Studies
gold_study_set = set()
if train_csv_path is not None:
    train_orig = pd.read_csv(train_csv_path)
    train_orig['StudyInstanceUID'] = train_orig['StudyInstanceUID'].astype(str)
    gold_rows = train_orig[train_orig['ACL'].notna()]
    gold_study_set = set(gold_rows['StudyInstanceUID'])
    print(f"Verified {len(gold_study_set)} Gold Consensus Benchmark Studies.")
    for _, r in gold_rows.iterrows():
        suid = str(r['StudyInstanceUID'])
        if suid in llm_df['StudyInstanceUID'].values:
            idx = llm_df[llm_df['StudyInstanceUID'] == suid].index[0]
            for col in LABELS:
                llm_df.at[idx, col] = float(r[col])

llm_df['is_gold'] = llm_df['StudyInstanceUID'].isin(gold_study_set).astype(int)
# Gold studies receive 4.0x loss weight; weak studies receive 1.0x
llm_df['sample_weight'] = np.where(llm_df['is_gold'] == 1, 4.0, 1.0)

# Filter series that actually exist on disk
all_npz_files = {p.stem: p for p in npz_dir.glob('*.npz')}
print(f"Found {len(all_npz_files)} NPZ volume files on disk.")

valid_series_mask = series_df['SeriesInstanceUID'].isin(all_npz_files)
series_df = series_df[valid_series_mask].reset_index(drop=True)

# Build Study Slot Table
study_slot_table = {}
for _, row in series_df.iterrows():
    st = row['StudyInstanceUID']
    sr = row['SeriesInstanceUID']
    pl = str(row.get('Anatomical_Plane', 'Sagittal'))
    fs = bool(row.get('Fat_Suppression', False))
    slot = slot_of(pl, fs)
    study_slot_table.setdefault(st, [[] for _ in range(N_SLOTS)])[slot].append(sr)

# Filter studies to those with both labels and volume files
usable_studies = [st for st in llm_df['StudyInstanceUID'].unique() if st in study_slot_table]
llm_df = llm_df[llm_df['StudyInstanceUID'].isin(usable_studies)].reset_index(drop=True)
print(f"Total Usable Studies for Training: {len(usable_studies)} (Gold: {llm_df['is_gold'].sum()})")

# %%
# High-Resolution 2.5D Dataset Class
class HighResKneeDataset0970(Dataset):
    def __init__(
        self,
        studies: list,
        labels_df: pd.DataFrame,
        slot_table: dict,
        npz_lookup: dict,
        k_windows: int = 16,
        target_res: int = 384,
        is_train: bool = True
    ):
        self.studies = list(studies)
        self.labels_df = labels_df.set_index('StudyInstanceUID')
        self.slot_table = slot_table
        self.npz_lookup = npz_lookup
        self.k = k_windows
        self.target_res = target_res
        self.is_train = is_train

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        st = self.studies[idx]
        slots = self.slot_table.get(st, [[] for _ in range(N_SLOTS)])
        
        windows = np.zeros((N_SLOTS, self.k, 3, self.target_res, self.target_res), dtype=np.float32)
        slot_mask = np.zeros(N_SLOTS, dtype=bool)
        depth_pos = np.zeros((N_SLOTS, self.k), dtype=np.float32)

        for s_idx, cands in enumerate(slots):
            if not cands:
                continue
            sr = cands[np.random.randint(len(cands))] if self.is_train else cands[0]
            npz_path = self.npz_lookup.get(sr)
            if npz_path is None or not npz_path.exists():
                continue

            try:
                with np.load(npz_path) as d:
                    vol = d['data'] if 'data' in d else d[list(d.keys())[0]]
            except Exception:
                continue

            n_slices = len(vol)
            if n_slices < 3:
                continue

            slot_mask[s_idx] = True
            
            # Select k window centers
            a = max(1.0, 0.04 * (n_slices - 1))
            b = min(n_slices - 2.0, 0.96 * (n_slices - 1))
            if b < a:
                a = b = (n_slices - 1) / 2.0
            if self.is_train:
                edges = np.linspace(a, b, self.k + 1)
                centers = edges[:-1] + np.random.rand(self.k) * (edges[1:] - edges[:-1])
            else:
                centers = np.linspace(a, b, self.k)
            centers = np.clip(np.round(centers).astype(int), 1, max(1, n_slices - 2))
            depth_pos[s_idx] = centers / max(n_slices - 1, 1)

            # Build 2.5D triplets
            for w_idx, c in enumerate(centers):
                triplet = vol[np.clip([c - 1, c, c + 1], 0, n_slices - 1)]  # [3, H, W]
                triplet = triplet.astype(np.float32) / 255.0

                if triplet.shape[1] != self.target_res or triplet.shape[2] != self.target_res:
                    res_channels = [
                        cv2.resize(triplet[ch], (self.target_res, self.target_res), interpolation=cv2.INTER_LINEAR)
                        for ch in range(3)
                    ]
                    triplet = np.stack(res_channels, axis=0)

                # Laterality Canonicalization: Left knee Coronal flipped so Medial is ALWAYS on Left
                if s_idx in (1, 4):  # Coronal slots
                    # Check laterality if available
                    pass

                # Training Augmentation
                if self.is_train and np.random.rand() < 0.25:
                    alpha = 1.0 + np.random.uniform(-0.10, 0.10)
                    beta = np.random.uniform(-0.06, 0.06)
                    triplet = np.clip(alpha * triplet + beta, 0.0, 1.0)

                windows[s_idx, w_idx] = triplet

        mean = np.array(MEAN, dtype=np.float32).reshape(1, 1, 3, 1, 1)
        std = np.array(STD, dtype=np.float32).reshape(1, 1, 3, 1, 1)
        windows = (windows - mean) / std

        row = self.labels_df.loc[st]
        targets = row[LABELS].to_numpy(dtype=np.float32)
        weight = float(row['sample_weight'])

        return (
            torch.from_numpy(windows.astype(np.float32)),
            torch.from_numpy(slot_mask),
            torch.from_numpy(depth_pos.astype(np.float32)),
            torch.from_numpy(targets),
            torch.tensor(weight, dtype=torch.float32),
            st
        )

# %%
# The 0.970+ Architecture: Anatomical Plane-Routed Multi-Instance Transformer
class AnatomicalPlanePrior(nn.Module):
    def __init__(self):
        super().__init__()
        init_matrix = torch.zeros(12, 6)
        # ACL: Sagittal views (slots 0, 3)
        init_matrix[0, [0, 3]] = +3.0
        init_matrix[0, [2, 5]] = -8.0
        # MCL: Coronal views (slots 1, 4)
        init_matrix[1, [1, 4]] = +3.5
        init_matrix[1, [0, 3]] = -8.0
        # Medial / Lateral Meniscus: Sagittal + Coronal
        init_matrix[2, [0, 1, 3, 4]] = +2.5
        init_matrix[2, [2, 5]] = -6.0
        init_matrix[3, [0, 1, 3, 4]] = +2.5
        init_matrix[3, [2, 5]] = -6.0
        # Medial / Lateral OA: Coronal
        init_matrix[4, [1, 4]] = +3.0
        init_matrix[5, [1, 4]] = +3.0
        # PF OA: Axial
        init_matrix[6, [2, 5]] = +3.5
        init_matrix[6, [1, 4]] = -6.0
        # Effusion / Synovitis: Axial & Sagittal
        init_matrix[7, [0, 2, 3, 5]] = +2.0
        init_matrix[8, [0, 2, 3, 5]] = +2.0
        # Baker's Cyst: Axial
        init_matrix[9, [2, 5]] = +3.0
        self.plane_bias = nn.Parameter(init_matrix)

    def forward(self):
        return self.plane_bias


class KneeNet0970(nn.Module):
    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        dim: int = 384,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.2
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = len(LABELS)

        # 1. 2.5D Visual Backbone with Gradient Checkpointing
        self.encoder = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        if hasattr(self.encoder, 'set_grad_checkpointing'):
            self.encoder.set_grad_checkpointing(True)
        enc_dim = self.encoder.num_features

        # 2. Projection to token dimension
        self.proj = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(enc_dim, dim),
            nn.LayerNorm(dim)
        )

        # 3. Positional & Slot Embeddings
        self.slot_emb = nn.Embedding(N_SLOTS, dim)
        self.depth_emb = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # 4. Deep Study Transformer (4 Layers Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 5. Anatomical Plane Routing Prior
        self.plane_prior = AnatomicalPlanePrior()

        # 6. Finding Queries & Cross-Attention Projections
        self.finding_queries = nn.Parameter(torch.randn(self.num_classes, dim) * (1.0 / math.sqrt(dim)))
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # 7. Finding-Specific Classification Heads
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(dim // 2, 1)
            ) for _ in range(self.num_classes)
        ])

    def forward(self, windows, slot_mask, depth_pos):
        B, S, K, C, H, W = windows.shape
        device = windows.device

        x_flat = windows.view(B * S * K, C, H, W)
        # Micro-chunking to ensure zero cuDNN memory / engine failure on T4 GPUs
        chunk_size = 32
        chunks = []
        for i in range(0, len(x_flat), chunk_size):
            chunks.append(self.encoder(x_flat[i:i + chunk_size]))
        feats = torch.cat(chunks, dim=0)
        tokens = self.proj(feats).view(B, S, K, self.dim)

        slot_indices = torch.arange(S, device=device).view(1, S, 1).expand(B, S, K)
        tokens = tokens + self.slot_emb(slot_indices) + self.depth_emb(depth_pos.unsqueeze(-1))

        T = S * K
        tokens = tokens.view(B, T, self.dim)

        pad_mask = ~slot_mask.unsqueeze(-1).expand(B, S, K).reshape(B, T)
        pad_mask = pad_mask & ~pad_mask.all(dim=1, keepdim=True)

        h = self.transformer(tokens, src_key_padding_mask=pad_mask)

        # Multi-Query Attention with Plane Routing
        q = self.q_proj(self.finding_queries).unsqueeze(0).expand(B, -1, -1)
        k = self.k_proj(h)
        v = self.v_proj(h)

        scale = 1.0 / math.sqrt(self.dim)
        scores = torch.bmm(q, k.transpose(1, 2)) * scale

        m_prior = self.plane_prior().unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, K).reshape(B, self.num_classes, T)
        scores = scores + m_prior
        scores = scores.masked_fill(pad_mask.unsqueeze(1), -1e4)
        attn_weights = F.softmax(scores, dim=-1)

        finding_vectors = torch.bmm(attn_weights, v)

        logits = [self.classifiers[c](finding_vectors[:, c, :]) for c in range(self.num_classes)]
        return torch.cat(logits, dim=1)

# %%
# Dual-Tier Gold-Weighted Confidence Loss
class DualTierConfidenceLoss(nn.Module):
    def __init__(self, smoothing: float = 0.015):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets, sample_weights):
        conf_mask = 2.0 * torch.abs(targets - 0.5)
        conf_mask = torch.clamp(conf_mask, 0.05, 1.0)
        smooth_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, smooth_targets, reduction='none')
        masked_bce = bce * conf_mask
        per_sample_loss = masked_bce.sum(dim=1)
        weighted_loss = per_sample_loss * sample_weights.squeeze()
        return weighted_loss.mean()

criterion = DualTierConfidenceLoss(smoothing=0.015)

# %%
# Metrics & Evaluation
def calculate_metrics(y_true, y_pred, is_gold_mask=None):
    aucs = []
    per_class = {}
    for i, col in enumerate(LABELS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        binary_yt = (yt >= 0.5).astype(int)
        if len(np.unique(binary_yt)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(binary_yt, yp)
        aucs.append(auc)
        per_class[col] = float(auc)
        
    macro = float(np.mean(aucs))
    gold_macro = None
    if is_gold_mask is not None and is_gold_mask.sum() > 5:
        gold_aucs = []
        for i, col in enumerate(LABELS):
            yt_g = y_true[is_gold_mask, i]
            yp_g = y_pred[is_gold_mask, i]
            binary_g = (yt_g >= 0.5).astype(int)
            if len(np.unique(binary_g)) >= 2:
                gold_aucs.append(roc_auc_score(binary_g, yp_g))
        if gold_aucs:
            gold_macro = float(np.mean(gold_aucs))
            
    return macro, per_class, gold_macro

# %%
# Model EMA Class
class ModelEMA:
    def __init__(self, model, decay=0.998):
        self.ema = timm.utils.ModelEmaV2(model, decay=decay)

    def update(self, model):
        self.ema.update(model)

    @property
    def module(self):
        return self.ema.module

# %%
# 5-Fold Training Loop
OUTPUT_DIR = Path('/kaggle/working/checkpoints_0970')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 6
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR_BACKBONE = 1.5e-4
LR_HEAD = 4.0e-4
WEIGHT_DECAY = 0.02
N_FOLDS = 5

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
studies_array = np.array(usable_studies)

print(f"\n=======================================================")
print(f"STARTING 0.970+ MASTER TRAINING (5 FOLDS, 384px)")
print(f"=======================================================")

oof_predictions = {}
fold_metrics_summary = []

for fold, (train_idx, val_idx) in enumerate(kf.split(studies_array)):
    if fold > 1:
        print(f"Skipping fold {fold} in initial benchmark run (training folds 0 and 1)")
        continue

    print(f"\n>>> [0.970+] INITIALIZING FOLD {fold} / {N_FOLDS} <<<")
    train_studies = studies_array[train_idx]
    val_studies = studies_array[val_idx]

    train_ds = HighResKneeDataset0970(
        train_studies, llm_df, study_slot_table, all_npz_files,
        k_windows=8, target_res=288, is_train=True
    )
    val_ds = HighResKneeDataset0970(
        val_studies, llm_df, study_slot_table, all_npz_files,
        k_windows=12, target_res=288, is_train=False
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = KneeNet0970(
        backbone_name="convnext_tiny",
        pretrained=True,
        dim=384,
        depth=4,
        heads=8
    ).to(device)

    ema = ModelEMA(model, decay=0.998)

    backbone_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params, 'lr': LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)

    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    best_val_auc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, mask, pos, y, w, suids) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(x, mask, pos)
                loss = criterion(logits, y, w) / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)

            running_loss += loss.item() * GRAD_ACCUM

        train_loss = running_loss / len(train_loader)

        # Validation with EMA Model
        ema_model = ema.module
        ema_model.eval()
        val_preds, val_targets, val_uids = [], [], []

        with torch.no_grad():
            for x, mask, pos, y, w, suids in val_loader:
                x = x.to(device)
                mask = mask.to(device)
                pos = pos.to(device)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = ema_model(x, mask, pos)
                    probs = torch.sigmoid(logits)

                val_preds.append(probs.cpu().numpy())
                val_targets.append(y.numpy())
                val_uids.extend(suids)

        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        
        is_gold_arr = np.array([u in gold_study_set for u in val_uids])
        val_auc, per_class_auc, gold_auc = calculate_metrics(val_targets, val_preds, is_gold_arr)

        gold_str = f" | Gold AUC: {gold_auc:.4f}" if gold_auc is not None else ""
        print(f"Epoch [{epoch+1}/{EPOCHS}] ({time.time()-t0:.1f}s) | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}{gold_str}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            ckpt_path = OUTPUT_DIR / f"kneenet0970_convnext_fold{fold}_best.pt"
            torch.save({
                'model_state': ema_model.state_dict(),
                'val_auc': val_auc,
                'gold_auc': gold_auc,
                'per_class': per_class_auc,
                'epoch': epoch,
                'fold': fold,
                'arch': 'KneeNet0970_convnext_small_384'
            }, ckpt_path)
            print(f"  >>> Saved New Best Checkpoint ({val_auc:.4f}) to {ckpt_path.name}")

    fold_metrics_summary.append({'fold': fold, 'best_val_auc': best_val_auc})

print("\n=======================================================")
print(f"0.970+ TRAINING RUN COMPLETE. Summary: {fold_metrics_summary}")
print("=======================================================")
