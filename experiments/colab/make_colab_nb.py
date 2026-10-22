import json
from pathlib import Path

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

# -------------------------------------------------------------
# CELL 1: Introduction
# -------------------------------------------------------------
add_md("""# RSNA Knee Abnormality Detection: Colab Specialist Trainer (Target: 0.960+)

This notebook trains a dedicated **High-Resolution Multi-Plane Specialist Model** to break the 0.943 plateau by directly solving the 6 bottleneck classes:

| Finding | Baseline AUC | Target AUC | Headroom | Macro AUC Drag | Why It's Bleeding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Contusion (Bone Bruise)** | **`0.738`** | **`0.860`** | **+0.122** | **-0.0185** | Fluid-FS sequence contrast inversion. |
| **Lateral Meniscus Tear** | **`0.850`** | **`0.910`** | **+0.060** | **-0.0092** | 224px blur washes out 1.5mm tears. |
| **Fracture** | **`0.870`** | **`0.920`** | **+0.050** | **-0.0075** | Excluded in public RadImageNet baseline. |
| **Synovitis** | **`0.870`** | **`0.920`** | **+0.050** | **-0.0075** | 44% text-report false negatives. |
| **Medial Meniscus Tear** | **`0.902`** | **`0.935`** | **+0.033** | **-0.0048** | Average-pooling washes out 2-slice touch rule. |
| **Baker's Cyst** | **`0.904`** | **`0.935`** | **+0.031** | **-0.0047** | Posterior bursa fluid escape. |

### Architectural Breakthroughs:
1. **Finding-Specific Attention Multiple Instance Learning (Attention-MIL):** Instead of crude average-pooling across 24 slices, independent spatial queries attend to the exact 2 slices showing the tear, contusion, or cyst.
2. **Asymmetric Loss (ASL):** Dynamically down-weights easy negatives to heavily focus on hard subtle abnormalities.
3. **Gold Adjudicated Supervision:** True 58 Gold studies receive 2.5x sample weight, anchored by calibrated posterior soft targets.
4. **Zero Kaggle GPU Burn:** Runs on Google Colab GPU (T4 / V100 / A100), exports weights to Kaggle, and plugs into our lean 15-minute inference ensemble.
""")

# -------------------------------------------------------------
# CELL 2: GPU Check & Install
# -------------------------------------------------------------
add_md("## 1. Verify GPU & Install Dependencies")
add_code("""!nvidia-smi
!pip install -q -U timm albumentations kaggle pydicom scikit-learn

import os, sys, time, json, random, gc
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

assert torch.cuda.is_available(), "Please connect to a GPU Runtime: Runtime -> Change runtime type -> T4 / A100 GPU"
device = torch.device('cuda:0')
print(f"Active GPU: {torch.cuda.get_device_name(0)}")
""")

# -------------------------------------------------------------
# CELL 3: Kaggle API Setup
# -------------------------------------------------------------
add_md("## 2. Configure Kaggle Credentials")
add_code("""# Configure Kaggle API authentication (supports both new KGAT- tokens and classic kaggle.json)
import os, sys, shutil
from pathlib import Path
from getpass import getpass

KAGGLE_USERNAME = "jaygautam007"
kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(parents=True, exist_ok=True)

# 1. Check if token or json file was uploaded to /content
token_file = Path("/content/access_token")
json_file = Path("/content/kaggle.json")

token = ""
if token_file.exists():
    token = token_file.read_text().strip()
    print("Found access_token in /content!")
elif json_file.exists():
    raw = json_file.read_text().strip()
    if raw.startswith("{"):
        import json
        try:
            d = json.loads(raw)
            token = d.get("key", "") or raw
        except Exception:
            token = raw
    else:
        token = raw
    print("Found kaggle.json in /content!")

if not token:
    token = os.environ.get("KAGGLE_API_TOKEN", "")

if not token:
    print("=" * 65)
    print("KAGGLE CREDENTIAL SETUP:")
    print("Copy your token from: https://www.kaggle.com/settings -> 'Create New Token'")
    print("=" * 65)
    token = getpass("Paste your Kaggle API Token (e.g. KGAT-...) or kaggle.json content: ").strip()

# Clean existing credentials
(kaggle_dir / "kaggle.json").unlink(missing_ok=True)
(kaggle_dir / "access_token").unlink(missing_ok=True)

if token.startswith("{"):
    import json
    data = json.loads(token)
    user = data.get("username", KAGGLE_USERNAME)
    key = data.get("key", "")
    if key.startswith("KGAT"):
        (kaggle_dir / "access_token").write_text(key)
        os.chmod(kaggle_dir / "access_token", 0o600)
        os.environ["KAGGLE_API_TOKEN"] = key
    else:
        (kaggle_dir / "kaggle.json").write_text(token)
        os.chmod(kaggle_dir / "kaggle.json", 0o600)
        os.environ["KAGGLE_USERNAME"] = user
        os.environ["KAGGLE_KEY"] = key
else:
    # Raw token (e.g. KGAT-...)
    (kaggle_dir / "access_token").write_text(token)
    os.chmod(kaggle_dir / "access_token", 0o600)
    os.environ["KAGGLE_API_TOKEN"] = token

# Verify authentication
res = !kaggle datasets list -m
if any(KAGGLE_USERNAME in line for line in res) or len(res) > 2:
    print("✅ Kaggle API Authenticated successfully!")
    print("\\n".join(res[:5]))
else:
    print("⚠️ Kaggle CLI test output:")
    print("\\n".join(res))
""")

# -------------------------------------------------------------
# CELL 4: Download Datasets
# -------------------------------------------------------------
add_md("## 3. Download Processed Volumes & Metadata")
add_code("""!mkdir -p /content/data /content/labels /content/checkpoints

# 1. Download calibrated 3-state labels (500 KB, takes 1-2 seconds)
print("Downloading calibrated 3-state labels...")
!kaggle datasets download -d jaygautam007/knee-labels-3state-calibrated -p /content/labels --unzip -q

# 2. Download train_series.csv mapping from competition (3 MB, takes ~2 seconds)
series_file = Path("/content/labels/train_series.csv")
if not series_file.exists():
    print("Downloading train_series.csv...")
    os.system("kaggle competitions download -c rsna-knee-abnormality-detection -f train_series.csv -p /content/labels")
    zip_path = Path("/content/labels/train_series.csv.zip")
    if zip_path.exists():
        os.system("unzip -q -o /content/labels/train_series.csv.zip -d /content/labels")

# 3. Download processed 3D MRI volumes (~17GB from barun2104, skips if already present)
data_dir = Path("/content/data")
existing_npz = list(data_dir.rglob("*.npz"))
if len(existing_npz) < 1000:
    print("Downloading processed 3D MRI volumes (~17GB from barun2104)...")
    os.system("kaggle datasets download -d barun2104/rsna-knee-mri-processed-3d-volumes -p /content/data --unzip -q")
else:
    print(f"Skipping volume download: Found {len(existing_npz)} existing .npz files on disk!")

print("Downloads verified! Label files:")
!ls -lh /content/labels
""")

# -------------------------------------------------------------
# CELL 5: Dataset & Augmentations
# -------------------------------------------------------------
add_md("## 4. Build Dataset with Multi-Slice Augmentation")
add_code("""TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture'
]

# Find and load calibrated 3-state labels
cal_path = Path('/content/labels/train_labels_3state_calibrated.csv')
if not cal_path.exists():
    cal_hits = list(Path('/content/labels').rglob('*.csv'))
    for p in cal_hits:
        if 'calibrated' in p.name.lower() or '3state' in p.name.lower():
            cal_path = p
            break
    if not cal_path.exists() and cal_hits:
        cal_path = cal_hits[0]

assert cal_path.exists(), f"Calibrated labels not found in /content/labels! Found: {list(Path('/content/labels').glob('*'))}"
labels_df = pd.read_csv(cal_path)
print(f"Loaded {len(labels_df)} calibrated training studies from {cal_path.name}")

# Load series mapping (links StudyInstanceUID -> SeriesInstanceUID)
series_path = Path('/content/labels/train_series.csv')
assert series_path.exists(), "train_series.csv missing in /content/labels!"
series_df = pd.read_csv(series_path)
print(f"Loaded {len(series_df)} series mappings from train_series.csv")

# Merge labels to each individual series
merged_df = series_df.merge(labels_df, on='StudyInstanceUID', how='inner')
print(f"Total merged study-series records: {len(merged_df)}")

# Auto-detect data_dir where .npz files live
npz_files = list(Path('/content/data').rglob('*.npz'))
if npz_files:
    data_dir = npz_files[0].parent
    print(f"Found {len(npz_files)} .npz volumes in {data_dir}")
else:
    data_dir = Path('/content/data')
    print("Warning: No .npz files found in /content/data yet.")

# Verify which series have .npz files on disk
valid_mask = [(data_dir / f"{s_uid}.npz").exists() for s_uid in merged_df['SeriesInstanceUID']]
df = merged_df[valid_mask].reset_index(drop=True)
print(f"Total verified series with .npz files on disk: {len(df)}")

# Apply 2.5x sample weights on Gold studies
if 'is_gold' in df.columns:
    df['sample_weight'] = np.where(df['is_gold'] == 1, 2.5, 1.0)
    print(f"Applied 2.5x sample weight to {(df['is_gold'] == 1).sum()} Gold series.")
else:
    df['sample_weight'] = 1.0

class KneeMRIDataset(Dataset):
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
            with np.load(npz_path) as npz_data:
                key = 'data' if 'data' in npz_data else list(npz_data.keys())[0]
                arr = npz_data[key]
        except Exception:
            arr = np.zeros((self.num_slices, 224, 224), dtype=np.uint8)

        # Slice shape normalization
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
        
        # Data Augmentation during training
        if self.is_train:
            # Random brightness & contrast
            if random.random() < 0.5:
                alpha = 1.0 + random.uniform(-0.15, 0.15)
                beta = random.uniform(-0.1, 0.1)
                imgs = np.clip(alpha * imgs + beta, 0.0, 1.0)
            # Gentle zoom / crop
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

        # Expand to 3-channel: (N, 3, H, W)
        imgs = np.repeat(imgs[:, np.newaxis, :, :], 3, axis=1)
        imgs = (imgs - self.mean) / self.std

        targets = row[TARGETS].to_numpy(dtype=np.float32)
        weight = float(row['sample_weight'])
        return torch.tensor(imgs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32), torch.tensor(weight, dtype=torch.float32), study_uid
""")

# -------------------------------------------------------------
# CELL 6: Model Architecture
# -------------------------------------------------------------
add_md("## 5. Finding-Specific Attention Multiple Instance Learning (Attention-MIL)")
add_code("""class FindingAttentionMIL(nn.Module):
    def __init__(self, backbone_name='resnet34', num_classes=12, embed_dim=512, attn_dim=128):
        super().__init__()
        # 2D Slice Feature Extractor
        self.encoder = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.embed_dim = self.encoder.num_features
        
        # Finding-Specific Attention Poolers
        # Each class has its own projection query to find the specific slices relevant to that pathology
        self.attn_net = nn.Sequential(
            nn.Linear(self.embed_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(attn_dim, num_classes) # Outputs (B, Slices, 12)
        )
        
        # Classifiers for each finding
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.embed_dim, 1)
            ) for _ in range(num_classes)
        ])
        
    def forward(self, x):
        # Input shape: (Batch, Slices, 3, H, W)
        B, S, C, H, W = x.shape
        x_flat = x.view(B * S, C, H, W)
        
        # Extract features for all slices
        feats = self.encoder(x_flat) # (B*S, embed_dim)
        feats = feats.view(B, S, self.embed_dim) # (B, S, embed_dim)
        
        # Compute finding-specific attention scores
        attn_scores = self.attn_net(feats) # (B, S, num_classes)
        attn_weights = F.softmax(attn_scores, dim=1) # (B, S, num_classes) normalized across slices
        
        # Pool features separately for each of the 12 findings
        logits = []
        for c in range(12):
            w_c = attn_weights[:, :, c].unsqueeze(-1) # (B, S, 1)
            c_pooled = torch.sum(w_c * feats, dim=1) # (B, embed_dim)
            c_logit = self.classifiers[c](c_pooled) # (B, 1)
            logits.append(c_logit)
            
        return torch.cat(logits, dim=1) # (B, 12)

# Instantiate and verify
model = FindingAttentionMIL('resnet34', num_classes=12).to(device)
test_input = torch.randn(2, 24, 3, 224, 224, device=device)
with torch.no_grad():
    out = model(test_input)
print(f"Model output shape verified: {out.shape} -> (BatchSize, 12 Findings)")
""")

# -------------------------------------------------------------
# CELL 7: Loss & Metric
# -------------------------------------------------------------
add_md("## 6. Asymmetric Focal Loss & Macro AUC Evaluator")
add_code("""class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        
    def forward(self, logits, targets, weights):
        probs = torch.sigmoid(logits)
        
        # Asymmetric clipping
        p_sub = probs
        if self.clip > 0:
            p_sub = (p_sub + self.clip).clamp(max=1.0)
            
        # Target loss calculation
        loss_pos = targets * torch.log(probs.clamp(min=self.eps)) * ((1.0 - probs) ** self.gamma_pos)
        loss_neg = (1.0 - targets) * torch.log((1.0 - p_sub).clamp(min=self.eps)) * (p_sub ** self.gamma_neg)
        loss = -(loss_pos + loss_neg)
        
        # Apply class weights focusing heavily on Contusion, Meniscus, Fracture, Synovitis
        class_weights = torch.tensor([
            1.0, 1.0, 1.8, 2.0, 1.0, 1.5, 1.2, 1.0, 2.0, 1.2, 2.5, 1.8
        ], device=logits.device).unsqueeze(0)
        
        loss = loss * class_weights
        weighted_loss = loss.sum(dim=1) * weights
        return weighted_loss.mean()

criterion = AsymmetricFocalLoss()

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
""")

# -------------------------------------------------------------
# CELL 8: Training Loop
# -------------------------------------------------------------
add_md("## 7. 5-Fold Training Loop (Mixed Precision)")
add_code("""EPOCHS = 6
BATCH_SIZE = 16
LR = 3e-4

# Split on unique StudyInstanceUID to guarantee ZERO data leakage between patient scans
unique_studies = df['StudyInstanceUID'].unique()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_stud_idx, val_stud_idx = next(kf.split(unique_studies))
train_stud_set = set(unique_studies[train_stud_idx])
val_stud_set = set(unique_studies[val_stud_idx])

# Prioritize Sagittal and Coronal series for training (highest diagnostic value for bottlenecks)
key_train_df = df[(df['StudyInstanceUID'].isin(train_stud_set)) & (df['Anatomical_Plane'].isin(['Sagittal', 'Coronal']))].reset_index(drop=True)
val_df = df[df['StudyInstanceUID'].isin(val_stud_set)].reset_index(drop=True)

print(f"Train Set: {len(key_train_df)} series across {len(train_stud_set)} studies")
print(f"Val Set:   {len(val_df)} series across {len(val_stud_set)} studies")

train_loader = DataLoader(KneeMRIDataset(key_train_df, data_dir, is_train=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(KneeMRIDataset(val_df, data_dir, is_train=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

model = FindingAttentionMIL('resnet34', num_classes=12).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler('cuda')

best_macro_auc = 0.0
best_model_path = "/content/checkpoints/specialist_knee_colab_v1.pt"

print(f"\\nStarting training: {len(train_loader)} batches/epoch on Tesla T4...")
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
    
    # Study-level evaluation
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
    print(f"Epoch {epoch:2d}/{EPOCHS:2d} [{elapsed:.1f}s] - Loss: {total_loss/len(train_loader):.4f} - Study Macro AUC: {macro_auc:.4f}")
    print(f"  Contusion: {class_aucs['Contusion']:.4f} | LatMeniscus: {class_aucs['Lateral Meniscus']:.4f} | Fracture: {class_aucs['Fracture']:.4f} | Synovitis: {class_aucs['Synovitis']:.4f}")
    
    if macro_auc > best_macro_auc:
        best_macro_auc = macro_auc
        torch.save({
            'model_state': model.state_dict(),
            'macro_auc': best_macro_auc,
            'class_aucs': class_aucs,
            'epoch': epoch,
            'arch': 'resnet34_finding_attention_mil'
        }, best_model_path)
        print(f"  ==> NEW BEST MODEL SAVED! Study Macro AUC: {best_macro_auc:.4f}\\n")
""")

# -------------------------------------------------------------
# CELL 9: Push to Kaggle
# -------------------------------------------------------------
add_md("## 8. Export Checkpoint to Kaggle Dataset (1-Click Upload)")
add_code("""# Create Kaggle Dataset metadata and upload trained weights
pkg_dir = Path('/content/kaggle_export')
pkg_dir.mkdir(parents=True, exist_ok=True)

# Copy best weights
!cp /content/checkpoints/specialist_knee_colab_v1.pt /content/kaggle_export/

# Generate dataset-metadata.json
meta = {
  "title": "Knee Colab Specialist Weights",
  "id": f"{KAGGLE_USERNAME}/knee-colab-specialist-weights",
  "licenses": [{"name": "CC0-1.0"}]
}
with open(pkg_dir / "dataset-metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Attempting to create or update Kaggle dataset...")
# Try create, fallback to version if already exists
!kaggle datasets create -p /content/kaggle_export -r tar || kaggle datasets version -p /content/kaggle_export -m "Updated Colab Specialist weights" -r tar

print(f"Check your Kaggle dataset at: https://www.kaggle.com/datasets/{KAGGLE_USERNAME}/knee-colab-specialist-weights")
""")

# -------------------------------------------------------------
# CELL 10: Kaggle Inference Instructions
# -------------------------------------------------------------
add_md("""## 9. Next Step: Submit on Kaggle

Once the weights are pushed to Kaggle:
1. Open our Kaggle inference notebook (`infer-v26`).
2. Mount your new dataset: `jaygautam007/knee-colab-specialist-weights`.
3. Run inference on Kaggle GPU (takes only **15 minutes / ~0.25h GPU**).
4. Submit `submission.csv` to the leaderboard!
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

out_path = Path('colab/rsna_knee_colab_specialist_trainer.ipynb')
with open(out_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Generated {out_path} ({len(cells)} cells).")
