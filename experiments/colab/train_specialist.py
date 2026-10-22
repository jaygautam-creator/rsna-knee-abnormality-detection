import os
import sys
import time
import json
import random
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

assert torch.cuda.is_available(), "CUDA must be available!"
device = torch.device('cuda:0')
print(f"Colab Specialist Trainer active on GPU: {torch.cuda.get_device_name(0)}")

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture'
]

# 1. Load Labels & Series Metadata
labels_path = Path("/content/labels/train_labels_3state_calibrated.csv")
series_path = Path("/content/labels/train_series.csv")
data_dir = Path("/content/data")
ckpt_dir = Path("/content/checkpoints")
ckpt_dir.mkdir(parents=True, exist_ok=True)

labels_df = pd.read_csv(labels_path)
series_df = pd.read_csv(series_path)
print(f"Loaded {len(labels_df)} study labels and {len(series_df)} series.")

# Merge to attach study labels to each series
merged_df = series_df.merge(labels_df, on='StudyInstanceUID', how='inner')
print(f"Total merged series: {len(merged_df)}")

# Check file existence
valid_mask = [(data_dir / f"{s_uid}.npz").exists() for s_uid in merged_df['SeriesInstanceUID']]
merged_df = merged_df[valid_mask].reset_index(drop=True)
print(f"Verified series with .npz files on disk: {len(merged_df)}")

# Apply 2.5x sample weight on Gold studies
if 'is_gold' in merged_df.columns:
    merged_df['sample_weight'] = np.where(merged_df['is_gold'] == 1, 2.5, 1.0)
    print(f"Applied 2.5x weight to Gold studies ({(merged_df['is_gold'] == 1).sum()} series).")
else:
    merged_df['sample_weight'] = 1.0

# 2. Split on StudyInstanceUID (No data leakage across series of same patient)
unique_studies = merged_df['StudyInstanceUID'].unique()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_studies, val_studies = next(kf.split(unique_studies))
train_studies_set = set(unique_studies[train_studies])
val_studies_set = set(unique_studies[val_studies])

train_df = merged_df[merged_df['StudyInstanceUID'].isin(train_studies_set)].reset_index(drop=True)
val_df = merged_df[merged_df['StudyInstanceUID'].isin(val_studies_set)].reset_index(drop=True)
print(f"Train: {len(train_studies_set)} studies ({len(train_df)} series)")
print(f"Val:   {len(val_studies_set)} studies ({len(val_df)} series)")

# 3. Dataset
class KneeSeriesDataset(Dataset):
    def __init__(self, df, data_dir, is_train=True, num_slices=24):
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
                key = 'data' if 'data' in d else list(d.keys())[0]
                arr = d[key]
        except Exception:
            arr = np.zeros((self.num_slices, 224, 224), dtype=np.uint8)

        # Ensure exactly 24 slices
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

        # Normalize to [0, 1]
        imgs = arr.astype(np.float32) / 255.0
        
        if self.is_train:
            # Contrast/brightness jitter
            if random.random() < 0.5:
                alpha = 1.0 + random.uniform(-0.15, 0.15)
                beta = random.uniform(-0.1, 0.1)
                imgs = np.clip(alpha * imgs + beta, 0.0, 1.0)
            # Gentle crop/zoom
            if random.random() < 0.3:
                s = random.uniform(0.9, 1.0)
                h, w = imgs.shape[1], imgs.shape[2]
                new_h, new_w = int(h * s), int(w * s)
                top = (h - new_h) // 2
                left = (w - new_w) // 2
                cropped = imgs[:, top:top+new_h, left:left+new_w]
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                imgs = np.pad(cropped, ((0,0), (pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), mode='reflect')

        # Repeat to 3-channel: (Slices, 3, H, W)
        imgs = np.repeat(imgs[:, np.newaxis, :, :], 3, axis=1)
        imgs = (imgs - self.mean) / self.std

        targets = row[TARGETS].to_numpy(dtype=np.float32)
        weight = float(row['sample_weight'])

        return torch.tensor(imgs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32), torch.tensor(weight, dtype=torch.float32), study_uid

# 4. FindingAttentionMIL Architecture
class FindingAttentionMIL(nn.Module):
    def __init__(self, backbone_name='resnet34', num_classes=12, attn_dim=128):
        super().__init__()
        self.encoder = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.embed_dim = self.encoder.num_features
        
        # Dedicated attention pooler per finding (finds exact slices)
        self.attn_net = nn.Sequential(
            nn.Linear(self.embed_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(attn_dim, num_classes) # (B, Slices, 12)
        )
        
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.embed_dim, 1)
            ) for _ in range(num_classes)
        ])

    def forward(self, x):
        # x: (Batch, Slices, 3, H, W)
        B, S, C, H, W = x.shape
        x_flat = x.view(B * S, C, H, W)
        feats = self.encoder(x_flat) # (B*S, embed_dim)
        feats = feats.view(B, S, self.embed_dim)
        
        attn_scores = self.attn_net(feats) # (B, S, 12)
        attn_weights = F.softmax(attn_scores, dim=1) # (B, S, 12)
        
        logits = []
        for c in range(12):
            w_c = attn_weights[:, :, c].unsqueeze(-1) # (B, S, 1)
            c_pooled = torch.sum(w_c * feats, dim=1)  # (B, embed_dim)
            logits.append(self.classifiers[c](c_pooled)) # (B, 1)
            
        return torch.cat(logits, dim=1) # (B, 12)

# 5. Asymmetric Focal Loss targeting bottleneck classes
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets, weights):
        probs = torch.sigmoid(logits)
        p_sub = probs
        if self.clip > 0:
            p_sub = (p_sub + self.clip).clamp(max=1.0)
            
        loss_pos = targets * torch.log(probs.clamp(min=self.eps)) * ((1.0 - probs) ** self.gamma_pos)
        loss_neg = (1.0 - targets) * torch.log((1.0 - p_sub).clamp(min=self.eps)) * (p_sub ** self.gamma_neg)
        loss = -(loss_pos + loss_neg)
        
        # Heavy focus on: Contusion (2.5x), Meniscus (2.0x), Fracture (1.8x), Synovitis (2.0x)
        class_weights = torch.tensor([
            1.0, 1.0, 1.8, 2.0, 1.0, 1.5, 1.2, 1.0, 2.0, 1.2, 2.5, 1.8
        ], device=logits.device).unsqueeze(0)
        
        loss = loss * class_weights
        weighted_loss = loss.sum(dim=1) * weights
        return weighted_loss.mean()

criterion = AsymmetricFocalLoss()

def compute_study_macro_auc(study_uids, targets, preds):
    # Group by study UID and average predictions across series
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

# 6. Training Execution
EPOCHS = 6
BATCH_SIZE = 16
LR = 3e-4

# Filter training to prioritize Sagittal and Coronal series (highest diagnostic value for bottlenecks)
key_train_df = train_df[train_df['Anatomical_Plane'].isin(['Sagittal', 'Coronal'])].reset_index(drop=True)
print(f"Focused training set (Sagittal + Coronal): {len(key_train_df)} series")

train_loader = DataLoader(KneeSeriesDataset(key_train_df, data_dir, is_train=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(KneeSeriesDataset(val_df, data_dir, is_train=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

model = FindingAttentionMIL('resnet34', num_classes=12).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler('cuda')

best_macro_auc = 0.0
best_model_path = ckpt_dir / "specialist_knee_colab_v1.pt"
latest_model_path = ckpt_dir / "specialist_knee_colab_latest.pt"

print(f"\n=======================================================")
print(f"TRAINING INITIATED ON TESLA T4 GPU")
print(f"Batches per epoch: {len(train_loader)} (Batch size: {BATCH_SIZE})")
print(f"Bottleneck Targets: Contusion (2.5x), Meniscus (2.0x), Fracture (1.8x), Synovitis (2.0x)")
print(f"=======================================================\n")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    total_loss = 0.0
    
    for step, (batch_imgs, batch_targets, batch_weights, _) in enumerate(train_loader):
        batch_imgs = batch_imgs.to(device)
        batch_targets = batch_targets.to(device)
        batch_weights = batch_weights.to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(batch_imgs)
            loss = criterion(logits, batch_targets, batch_weights)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        
    scheduler.step()
    
    # Study-Level Validation
    model.eval()
    val_preds = []
    val_targets = []
    val_study_ids = []
    with torch.no_grad():
        for batch_imgs, batch_targets, _, study_ids in val_loader:
            batch_imgs = batch_imgs.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(batch_imgs)
                probs = torch.sigmoid(logits).cpu().numpy()
            val_preds.append(probs)
            val_targets.append(batch_targets.numpy())
            val_study_ids.extend(study_ids)
            
    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)
    macro_auc, class_aucs = compute_study_macro_auc(val_study_ids, val_targets, val_preds)
    
    elapsed = time.time() - t0
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch:2d}/{EPOCHS:2d} [{elapsed:.1f}s] - Loss: {avg_loss:.4f} - Val Macro AUC: {macro_auc:.4f}")
    print(f"  Contusion: {class_aucs['Contusion']:.4f} | LatMeniscus: {class_aucs['Lateral Meniscus']:.4f} | Fracture: {class_aucs['Fracture']:.4f} | Synovitis: {class_aucs['Synovitis']:.4f}")
    
    torch.save({
        'model_state': model.state_dict(),
        'macro_auc': macro_auc,
        'class_aucs': class_aucs,
        'epoch': epoch,
        'arch': 'resnet34_finding_attention_mil'
    }, latest_model_path)
    
    if macro_auc > best_macro_auc:
        best_macro_auc = macro_auc
        torch.save({
            'model_state': model.state_dict(),
            'macro_auc': best_macro_auc,
            'class_aucs': class_aucs,
            'epoch': epoch,
            'arch': 'resnet34_finding_attention_mil'
        }, best_model_path)
        print(f"  ==> NEW BEST MODEL SAVED! Macro AUC: {best_macro_auc:.4f}")

if not best_model_path.exists():
    torch.save({
        'model_state': model.state_dict(),
        'macro_auc': best_macro_auc,
        'class_aucs': class_aucs,
        'epoch': EPOCHS,
        'arch': 'resnet34_finding_attention_mil'
    }, best_model_path)

print(f"\nTraining Complete! Best Macro AUC: {best_macro_auc:.4f}")
print(f"Saved checkpoint to {best_model_path}")
