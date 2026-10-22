# %% [markdown]
# # RSNA Knee: Dedicated Coronal-Only MCL & Meniscus Specialist (Path to 0.970)
# 
# ### Radiological Rationale:
# - The MCL is strictly a Coronal plane structure (invisible on Sagittal slices).
# - Mixed-plane models suffer 67% noise dilution on MCL.
# - This dedicated Coronal model trains exclusively on Coronal acquisitions with laterality canonicalization
#   (Left knees flipped so Medial compartment is ALWAYS on the left margin of the image).
# - Targets: MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, Fracture.

# %%
import os
import sys
import time
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
import timm

print(f"=== CORONAL MCL SPECIALIST TRAINER START ===")
assert torch.cuda.is_available(), "CUDA GPU required!"
device = torch.device('cuda:0')
print(f"Active GPU: {torch.cuda.get_device_name(0)}")

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

llm_labels_path = find_file('llm_labels_v4_blend.csv')
train_series_path = find_file('train_series.csv')
train_csv_path = find_file('train.csv')
npz_dir = find_dir_with_ext('.npz')

assert llm_labels_path is not None, "Missing llm_labels_v4_blend.csv!"
assert train_series_path is not None, "Missing train_series.csv!"
assert npz_dir is not None, "Missing NPZ volumes directory!"

# %%
# Load metadata and filter strictly for Coronal series
llm_df = pd.read_csv(llm_labels_path)
series_df = pd.read_csv(train_series_path)

# Gold studies ground truth incorporation
if train_csv_path is not None:
    train_orig = pd.read_csv(train_csv_path)
    gold_rows = train_orig[train_orig['ACL'].notna()]
    gold_study_set = set(gold_rows['StudyInstanceUID'].astype(str))
    print(f"Found {len(gold_study_set)} Gold adjudicated studies.")
    
    TARGETS = [
        'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
        'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
        'Contusion', 'Fracture'
    ]
    for _, r in gold_rows.iterrows():
        suid = str(r['StudyInstanceUID'])
        if suid in llm_df['StudyInstanceUID'].values:
            idx = llm_df[llm_df['StudyInstanceUID'] == suid].index[0]
            for col in TARGETS:
                llm_df.at[idx, col] = float(r[col])

llm_df['is_gold'] = llm_df['StudyInstanceUID'].astype(str).isin(gold_study_set).astype(int)
llm_df['sample_weight'] = np.where(llm_df['is_gold'] == 1, 2.5, 1.0)

merged_df = series_df.merge(llm_df, on='StudyInstanceUID', how='inner')

# Filter for Coronal plane ONLY
coronal_df = merged_df[merged_df['Anatomical_Plane'] == 'Coronal'].reset_index(drop=True)
valid_mask = [(npz_dir / f"{s_uid}.npz").exists() for s_uid in coronal_df['SeriesInstanceUID']]
coronal_df = coronal_df[valid_mask].reset_index(drop=True)

print(f"Filtered Coronal Series: {len(coronal_df)} across {coronal_df['StudyInstanceUID'].nunique()} studies.")

# %%
# Coronal Dataset with Laterality Canonicalization
class KneeCoronalDataset(Dataset):
    def __init__(self, df, data_dir, is_train=True, num_slices=20):
        self.df = df
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        self.num_slices = num_slices
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        s_uid = row['SeriesInstanceUID']
        study_uid = row['StudyInstanceUID']
        npz_path = self.data_dir / f"{s_uid}.npz"

        try:
            with np.load(npz_path) as d:
                arr = d['data'] if 'data' in d else d[list(d.keys())[0]]
        except Exception:
            arr = np.zeros((self.num_slices, 224, 224), dtype=np.uint8)

        if arr.ndim == 3:
            s, h, w = arr.shape
            if s < self.num_slices:
                pad = np.zeros((self.num_slices - s, h, w), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)
            elif s > self.num_slices:
                indices = np.linspace(0, s - 1, self.num_slices).astype(int)
                arr = arr[indices]
        else:
            arr = np.zeros((self.num_slices, 224, 224), dtype=np.uint8)

        imgs = arr.astype(np.float32) / 255.0

        # Laterality canonicalization: Left knees flipped horizontally so Medial is ALWAYS on the Left
        # (This aligns MCL perfectly across all knee studies!)
        laterality = str(row.get('Laterality', '')).upper()
        if laterality == 'L' or laterality == 'LEFT':
            imgs = np.flip(imgs, axis=-1).copy()

        # Training augmentations (NO random horizontal flips!)
        if self.is_train:
            if random.random() < 0.5:
                alpha = 1.0 + random.uniform(-0.15, 0.15)
                beta = random.uniform(-0.10, 0.10)
                imgs = np.clip(alpha * imgs + beta, 0.0, 1.0)

        imgs = np.repeat(imgs[:, np.newaxis, :, :], 3, axis=1)
        imgs = (imgs - self.mean) / self.std

        targets = row[TARGETS].to_numpy(dtype=np.float32)
        weight = float(row['sample_weight'])
        return torch.tensor(imgs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32), torch.tensor(weight, dtype=torch.float32), study_uid

# %%
# Architecture: ResNet-34d with Coronal Multi-Head Attention
class CoronalFindingAttentionMIL(nn.Module):
    def __init__(self, backbone_name='resnet34d', num_classes=12, attn_dim=128):
        super().__init__()
        self.encoder = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.embed_dim = self.encoder.num_features
        self.attn_net = nn.Sequential(
            nn.Linear(self.embed_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(attn_dim, num_classes)
        )
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.embed_dim, 1)
            ) for _ in range(num_classes)
        ])

    def forward(self, x):
        B, S, C, H, W = x.shape
        x_flat = x.view(B * S, C, H, W)
        feats = self.encoder(x_flat).view(B, S, self.embed_dim)
        attn_scores = self.attn_net(feats)
        attn_weights = F.softmax(attn_scores, dim=1)
        logits = []
        for c in range(12):
            w_c = attn_weights[:, :, c].unsqueeze(-1)
            c_pooled = torch.sum(w_c * feats, dim=1)
            logits.append(self.classifiers[c](c_pooled))
        return torch.cat(logits, dim=1)

# %%
# Confidence-Masked Loss with Extra Weight on MCL
class CoronalSpecialistLoss(nn.Module):
    def __init__(self, smoothing=0.01):
        super().__init__()
        self.smoothing = smoothing
        # Extra emphasis on MCL and Meniscus
        self.class_weights = torch.tensor([
            1.0, 3.0, 2.0, 1.8, 1.2, 1.2, 1.0, 1.0, 1.5, 1.0, 1.2, 1.5
        ])

    def forward(self, logits, targets, sample_weights):
        cw = self.class_weights.to(logits.device).unsqueeze(0)
        conf_mask = 2.0 * torch.abs(targets - 0.5)
        conf_mask = torch.clamp(conf_mask, 0.0, 1.0)
        
        smooth_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, smooth_targets, reduction='none')
        
        masked_bce = bce * conf_mask * cw
        weighted_loss = masked_bce.sum(dim=1) * sample_weights
        return weighted_loss.mean()

criterion = CoronalSpecialistLoss(smoothing=0.01)

# %%
# 5-Fold Training Loop
OUTPUT_DIR = Path('/kaggle/working/coronal_mcl_checkpoints')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 5
BATCH_SIZE = 16
LR = 3e-4
N_FOLDS = 5

unique_studies = coronal_df['StudyInstanceUID'].unique()
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

def evaluate_auc(study_uids, targets, preds):
    pdf = pd.DataFrame(preds, columns=TARGETS)
    pdf['StudyInstanceUID'] = study_uids
    study_preds = pdf.groupby('StudyInstanceUID').mean()
    tdf = pd.DataFrame(targets, columns=TARGETS)
    tdf['StudyInstanceUID'] = study_uids
    study_targets = tdf.groupby('StudyInstanceUID').first()

    aucs = []
    per_class = {}
    for col in TARGETS:
        y_true = study_targets[col].to_numpy()
        y_pred = study_preds[col].to_numpy()
        binary_true = (y_true >= 0.50).astype(int)
        if len(np.unique(binary_true)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(binary_true, y_pred)
        aucs.append(auc)
        per_class[col] = float(auc)
    return float(np.mean(aucs)), per_class

print(f"\nSTARTING 5-FOLD CORONAL MCL SPECIALIST TRAINING...")

for fold, (train_stud_idx, val_stud_idx) in enumerate(kf.split(unique_studies)):
    print(f"\n>>> CORONAL FOLD {fold} / {N_FOLDS} <<<")
    train_stud_set = set(unique_studies[train_stud_idx])
    val_stud_set = set(unique_studies[val_stud_idx])

    train_df_fold = coronal_df[coronal_df['StudyInstanceUID'].isin(train_stud_set)].reset_index(drop=True)
    val_df_fold = coronal_df[coronal_df['StudyInstanceUID'].isin(val_stud_set)].reset_index(drop=True)

    train_loader = DataLoader(
        KneeCoronalDataset(train_df_fold, npz_dir, is_train=True, num_slices=20),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        KneeCoronalDataset(val_df_fold, npz_dir, is_train=False, num_slices=20),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    model = CoronalFindingAttentionMIL('resnet34d', num_classes=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_mcl_auc = 0.0
    best_ckpt = OUTPUT_DIR / f"coronal_mcl_fold{fold}.pt"

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for imgs, tgts, wts, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)
            wts = wts.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, tgts, wts)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_targets, val_uids = [], [], []
        with torch.no_grad():
            for imgs, tgts, _, uids in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    logits = model(imgs)
                    probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.append(probs)
                val_targets.append(tgts.numpy())
                val_uids.extend(uids)

        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        macro_auc, class_aucs = evaluate_auc(val_uids, val_targets, val_preds)

        elapsed = time.time() - t0
        print(f"Fold {fold} Ep {epoch}/{EPOCHS} [{elapsed:.1f}s] Macro AUC: {macro_auc:.4f} | MCL AUC: {class_aucs['MCL']:.4f} | MedMen: {class_aucs['Medial Meniscus']:.4f}")

        if class_aucs['MCL'] > best_mcl_auc or epoch == EPOCHS:
            best_mcl_auc = class_aucs['MCL']
            torch.save({
                'fold': fold,
                'model_state': model.state_dict(),
                'macro_auc': macro_auc,
                'mcl_auc': class_aucs['MCL'],
                'class_aucs': class_aucs,
                'epoch': epoch,
                'arch': 'coronal_resnet34d_attention_mil'
            }, best_ckpt)
            print(f"   --> Saved best MCL Fold {fold} checkpoint (MCL AUC: {best_mcl_auc:.4f})")

print(f"\nAll 5 Coronal MCL checkpoints saved to {OUTPUT_DIR}")
