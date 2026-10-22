import os
import sys
import time
import json
import random
import shutil
import subprocess
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

print(f"=== RSNA KNEE SPECIALIST PIPELINE START ===")
print(f"CUDA Available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA GPU required!"
device = torch.device('cuda:0')
print(f"Active GPU: {torch.cuda.get_device_name(0)}")

# -------------------------------------------------------------
# 1. SETUP KAGGLE CREDENTIALS
# -------------------------------------------------------------
KAGGLE_USERNAME = "jaygautam007"
kdir = Path.home() / ".kaggle"
kdir.mkdir(parents=True, exist_ok=True)
token_src = Path("/content/access_token")
if token_src.exists():
    shutil.copy(token_src, kdir / "access_token")
    os.chmod(kdir / "access_token", 0o600)
    token = token_src.read_text().strip()
    os.environ["KAGGLE_API_TOKEN"] = token
    print("Kaggle credentials configured in ~/.kaggle/access_token")

res = subprocess.run(["kaggle", "datasets", "list", "-m"], capture_output=True, text=True)
print("Kaggle Auth Status:", "OK" if res.returncode == 0 else f"FAILED: {res.stderr}")

# -------------------------------------------------------------
# 2. DOWNLOAD LABELS & VOLUMES
# -------------------------------------------------------------
labels_dir = Path("/content/labels")
data_dir = Path("/content/data")
ckpt_dir = Path("/content/checkpoints")
pkg_dir = Path("/content/kaggle_export")

for d in [labels_dir, data_dir, ckpt_dir, pkg_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Copy train_series.csv
if Path("/content/train_series.csv").exists():
    shutil.copy("/content/train_series.csv", labels_dir / "train_series.csv")

# Download calibrated 3-state labels
if not (labels_dir / "train_labels_3state_calibrated.csv").exists():
    print("Downloading calibrated labels dataset...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", f"{KAGGLE_USERNAME}/knee-labels-3state-calibrated",
        "-p", str(labels_dir),
        "--unzip", "-q"
    ], check=True)

existing_npz = list(data_dir.rglob("*.npz"))
if len(existing_npz) < 1000:
    print("Downloading 3D volumes from barun2104 (~17GB)...")
    t0 = time.time()
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "barun2104/rsna-knee-mri-processed-3d-volumes",
        "-p", str(data_dir),
        "--unzip", "-q"
    ], check=True)
    print(f"Volumes downloaded & unzipped in {time.time() - t0:.1f}s")

actual_data_dir = list(data_dir.rglob("*.npz"))[0].parent
print(f"Data directory verified: {actual_data_dir} ({len(list(data_dir.rglob('*.npz')))} volumes)")

# -------------------------------------------------------------
# 3. MERGE METADATA & DATA SPLIT
# -------------------------------------------------------------
TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture'
]

labels_df = pd.read_csv(labels_dir / "train_labels_3state_calibrated.csv")
series_df = pd.read_csv(labels_dir / "train_series.csv")
merged_df = series_df.merge(labels_df, on='StudyInstanceUID', how='inner')

# Filter for files existing on disk
valid_mask = [(actual_data_dir / f"{s_uid}.npz").exists() for s_uid in merged_df['SeriesInstanceUID']]
merged_df = merged_df[valid_mask].reset_index(drop=True)
print(f"Total merged verified series: {len(merged_df)}")

# 2.5x sample weights on Gold studies
if 'is_gold' in merged_df.columns:
    merged_df['sample_weight'] = np.where(merged_df['is_gold'] == 1, 2.5, 1.0)
    print(f"Applied 2.5x weight on Gold adjudicated studies ({(merged_df['is_gold'] == 1).sum()} series).")
else:
    merged_df['sample_weight'] = 1.0

# Split by StudyInstanceUID
unique_studies = merged_df['StudyInstanceUID'].unique()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_stud_idx, val_stud_idx = next(kf.split(unique_studies))
train_stud_set = set(unique_studies[train_stud_idx])
val_stud_set = set(unique_studies[val_stud_idx])

# Fast validation subset (all Gold val studies + sample of others = ~150 studies for fast 20s validation)
val_all_df = merged_df[merged_df['StudyInstanceUID'].isin(val_stud_set)].reset_index(drop=True)
val_gold_studies = val_all_df[val_all_df['is_gold'] == 1]['StudyInstanceUID'].unique()
val_other_studies = val_all_df[val_all_df['is_gold'] == 0]['StudyInstanceUID'].unique()
np.random.seed(42)
selected_val_studies = set(val_gold_studies).union(set(np.random.choice(val_other_studies, size=min(120, len(val_other_studies)), replace=False)))
val_fast_df = val_all_df[val_all_df['StudyInstanceUID'].isin(selected_val_studies)].reset_index(drop=True)

# High-yield training series (Sagittal + Coronal)
key_train_df = merged_df[(merged_df['StudyInstanceUID'].isin(train_stud_set)) & (merged_df['Anatomical_Plane'].isin(['Sagittal', 'Coronal']))].reset_index(drop=True)
print(f"Focused Train Set: {len(key_train_df)} series across {len(train_stud_set)} studies")
print(f"Fast Val Set:    {len(val_fast_df)} series across {len(selected_val_studies)} studies")

# -------------------------------------------------------------
# 4. DATASET & ARCHITECTURE
# -------------------------------------------------------------
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
        if self.is_train:
            if random.random() < 0.5:
                alpha = 1.0 + random.uniform(-0.15, 0.15)
                beta = random.uniform(-0.1, 0.1)
                imgs = np.clip(alpha * imgs + beta, 0.0, 1.0)
            if random.random() < 0.3:
                scale = random.uniform(0.9, 1.0)
                h, w = imgs.shape[1], imgs.shape[2]
                nh, nw = int(h * scale), int(w * scale)
                top = (h - nh) // 2
                left = (w - nw) // 2
                cropped = imgs[:, top:top+nh, left:left+nw]
                pad_h = (h - nh) // 2
                pad_w = (w - nw) // 2
                imgs = np.pad(cropped, ((0,0), (pad_h, h-nh-pad_h), (pad_w, w-nw-pad_w)), mode='reflect')

        imgs = np.repeat(imgs[:, np.newaxis, :, :], 3, axis=1)
        imgs = (imgs - self.mean) / self.std

        targets = row[TARGETS].to_numpy(dtype=np.float32)
        weight = float(row['sample_weight'])
        return torch.tensor(imgs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32), torch.tensor(weight, dtype=torch.float32), study_uid

class FindingAttentionMIL(nn.Module):
    def __init__(self, backbone_name='resnet34', num_classes=12, attn_dim=128):
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

class SmoothBCEWithClassWeights(nn.Module):
    def __init__(self, label_smoothing=0.02):
        super().__init__()
        self.smoothing = label_smoothing

    def forward(self, logits, targets, weights):
        smooth_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, smooth_targets, reduction='none')
        class_weights = torch.tensor([
            1.0, 1.0, 1.8, 2.0, 1.0, 1.5, 1.2, 1.0, 2.0, 1.2, 2.5, 1.8
        ], device=logits.device).unsqueeze(0)
        return ((bce * class_weights).sum(dim=1) * weights).mean()

criterion = SmoothBCEWithClassWeights(label_smoothing=0.02)

def compute_study_macro_auc(study_uids, targets, preds):
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

def export_checkpoint_to_kaggle(ckpt_path):
    print(f"\n[AUTO-EXPORT] Packaging and uploading weights to Kaggle...")
    dest_file = pkg_dir / "specialist_knee_colab_v1.pt"
    shutil.copy(ckpt_path, dest_file)
    meta = {
        "title": "Knee Colab Specialist Weights",
        "id": f"{KAGGLE_USERNAME}/knee-colab-specialist-weights",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(pkg_dir / "dataset-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    # Try version first (since dataset already exists), fallback to create
    res = subprocess.run([
        "kaggle", "datasets", "version", "-p", str(pkg_dir),
        "-m", f"Auto-export from Colab at {time.strftime('%Y-%m-%d %H:%M')}",
        "-r", "tar"
    ], capture_output=True, text=True)
    if res.returncode != 0:
        res = subprocess.run([
            "kaggle", "datasets", "create", "-p", str(pkg_dir), "-r", "tar"
        ], capture_output=True, text=True)
    print("[AUTO-EXPORT RESULT]:", res.stdout or res.stderr)

# -------------------------------------------------------------
# 5. TRAINING LOOP
# -------------------------------------------------------------
EPOCHS = 6
BATCH_SIZE = 16
LR = 3e-4

train_loader = DataLoader(KneeSeriesDataset(key_train_df, actual_data_dir, is_train=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(KneeSeriesDataset(val_fast_df, actual_data_dir, is_train=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

model = FindingAttentionMIL('resnet34', num_classes=12).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler('cuda')

best_macro_auc = 0.0
best_ckpt_path = ckpt_dir / "specialist_knee_colab_v1.pt"

print(f"\n--- STARTING TRAINING: {EPOCHS} EPOCHS ({len(train_loader)} batches/epoch) ---")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    total_loss = 0.0
    
    for batch_imgs, batch_targets, batch_weights, _ in train_loader:
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
    
    # Fast Validation
    model.eval()
    val_preds, val_targets, val_study_ids = [], [], []
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
    print(f"Epoch {epoch}/{EPOCHS} [{elapsed:.1f}s] - Loss: {avg_loss:.4f} - Val Macro AUC: {macro_auc:.4f}")
    print(f"  Contusion: {class_aucs['Contusion']:.4f} | LatMeniscus: {class_aucs['Lateral Meniscus']:.4f} | Fracture: {class_aucs['Fracture']:.4f} | Synovitis: {class_aucs['Synovitis']:.4f}")
    
    if macro_auc > best_macro_auc or epoch >= 2:
        if macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
        torch.save({
            'model_state': model.state_dict(),
            'macro_auc': macro_auc,
            'class_aucs': class_aucs,
            'epoch': epoch,
            'arch': 'resnet34_finding_attention_mil'
        }, best_ckpt_path)
        print(f"  ==> Saved checkpoint (Macro AUC: {macro_auc:.4f})")
        export_checkpoint_to_kaggle(best_ckpt_path)

print(f"\n=== PIPELINE FINISHED! Best Macro AUC: {best_macro_auc:.4f} ===")
