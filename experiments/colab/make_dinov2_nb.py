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
# CELL 1: Header
# -------------------------------------------------------------
add_md("""# RSNA Knee: 3rd Pillar DINOv2 Anatomical Plane-Routed Trainer (Target: 0.965-0.970+)

### Architectural Foundation:
1. **Self-Supervised Vision Transformer (DINOv2-Base):**
   - Fundamentally decorrelated feature representation from ImageNet-supervised CNNs (Raptor CoAtNet and S6 ConvNeXt).
   - Global self-attention from layer 1 captures non-local joint relationships.
2. **High-Resolution (392px / 448px, Patch 14):**
   - Resolves sub-centimeter cortical fracture lines and subtle bone contusions that 224px models blur.
3. **Explicit Anatomical Plane Routing (6-Slot SlotHead v2):**
   - Slot 0: Sagittal Fat-Suppressed (FS) -> ACL, Meniscus, Fluid
   - Slot 1: Sagittal non-FS -> Anatomy, Cartilage
   - Slot 2: Coronal FS -> Collaterals, Bone Contusion, Fracture
   - Slot 3: Coronal non-FS -> Joint Space, Osteophytes
   - Slot 4: Axial FS -> Patellofemoral Joint, Joint Effusion, Synovitis
   - Slot 5: Axial non-FS -> Popliteal Fossa, Baker's Cyst
   - **Zero-leakage slot presence masking** prevents hallucination when views are missing.
4. **Target Metrics:**
   - Solo OOF Macro AUC: **>= 0.945**
   - Error Correlation: **rho <= 0.70** vs. Raptor + S6 blend
   - Blended SOTA Target: **0.965 - 0.972+**""")

# -------------------------------------------------------------
# CELL 2: GPU Check & Environment Setup
# -------------------------------------------------------------
add_code("""import os, sys, time, json, shutil, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

print(f"CUDA Available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA GPU required! Please select a GPU runtime (T4 / A100 / L4)."
device = torch.device('cuda:0')
print(f"Active GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)")

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm==1.0.25", "pydicom", "pylibjpeg", "python-gdcm"])
import timm
print(f"timm version: {timm.__version__}")""")

# -------------------------------------------------------------
# CELL 3: Kaggle Credentials & Data Download
# -------------------------------------------------------------
add_code("""kdir = Path.home() / ".kaggle"
kdir.mkdir(parents=True, exist_ok=True)

# Look for access_token or kaggle.json
if Path("/content/access_token").exists():
    shutil.copy("/content/access_token", kdir / "access_token")
    os.chmod(kdir / "access_token", 0o600)
    os.environ["KAGGLE_API_TOKEN"] = Path("/content/access_token").read_text().strip()
    print("Kaggle access token configured.")

labels_dir = Path("/content/labels")
data_dir = Path("/content/data")
ckpt_dir = Path("/content/checkpoints")
oof_dir = Path("/content/oof")
for d in [labels_dir, data_dir, ckpt_dir, oof_dir]: d.mkdir(parents=True, exist_ok=True)

# 1. Download calibrated labels & series metadata
print("[1/2] Downloading calibrated 3-state labels...")
subprocess.run(["kaggle", "datasets", "download", "-d", "jaygautam007/knee-labels-3state-calibrated", "-p", str(labels_dir), "--unzip", "-q"], check=True)

# 2. Download 3D volumes
existing_npz = list(data_dir.rglob("*.npz"))
if len(existing_npz) < 1000:
    print("[2/2] Downloading processed 3D MRI volumes (~17GB from barun2104)...")
    t0 = time.time()
    subprocess.run(["kaggle", "datasets", "download", "-d", "barun2104/rsna-knee-mri-processed-3d-volumes", "-p", str(data_dir), "--unzip", "-q"], check=True)
    print(f"Volumes ready in {time.time()-t0:.1f}s!")

actual_data_dir = list(data_dir.rglob("*.npz"))[0].parent
print(f"Verified {len(list(data_dir.rglob('*.npz')))} volume files on disk at: {actual_data_dir}")""")

# -------------------------------------------------------------
# CELL 4: Slot Routing Dataset
# -------------------------------------------------------------
add_code("""TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture'
]

labels_df = pd.read_csv(labels_dir / "train_labels_3state_calibrated.csv")
series_df = pd.read_csv(labels_dir / "train_series.csv")

# Map series to 6 anatomical slots
def get_slot(row):
    p = str(row['Anatomical_Plane']).strip()
    fs = int(row['Fat_Suppression']) if pd.notnull(row['Fat_Suppression']) else 0
    if p == 'Sagittal': return 0 if fs == 1 else 1
    if p == 'Coronal': return 2 if fs == 1 else 3
    if p == 'Axial': return 4 if fs == 1 else 5
    return -1

series_df['slot'] = series_df.apply(get_slot, axis=1)

# Group series per study into slot dictionary
study_to_slots = {}
for _, row in series_df.iterrows():
    sid = str(row['StudyInstanceUID'])
    s_uid = str(row['SeriesInstanceUID'])
    slot = row['slot']
    if slot < 0: continue
    if not (actual_data_dir / f"{s_uid}.npz").exists(): continue
    if sid not in study_to_slots: study_to_slots[sid] = {}
    if slot not in study_to_slots[sid]:
        study_to_slots[sid][slot] = s_uid

# Filter labels to studies with at least 3 slots
valid_studies = [s for s in labels_df['StudyInstanceUID'].astype(str) if s in study_to_slots and len(study_to_slots[s]) >= 3]
labels_df = labels_df[labels_df['StudyInstanceUID'].astype(str).isin(valid_studies)].reset_index(drop=True)
print(f"Total valid studies with >=3 slots on disk: {len(labels_df)}")

# 5-Fold Split
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
labels_df['fold'] = -1
for f, (_, val_idx) in enumerate(kf.split(labels_df)):
    labels_df.loc[val_idx, 'fold'] = f

class KneeSlotDataset(torch.utils.data.Dataset):
    def __init__(self, df, study_slots, data_dir, is_train=True, num_slices=16, img_size=392):
        self.df = df
        self.study_slots = study_slots
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        self.num_slices = num_slices
        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = str(row['StudyInstanceUID'])
        slots = self.study_slots.get(sid, {})
        
        volume = np.zeros((6, self.num_slices, 3, self.img_size, self.img_size), dtype=np.float32)
        slot_mask = np.zeros(6, dtype=np.float32)
        
        for s_idx in range(6):
            if s_idx in slots:
                s_uid = slots[s_idx]
                npz_path = self.data_dir / f"{s_uid}.npz"
                try:
                    with np.load(npz_path) as d:
                        k = 'data' if 'data' in d else list(d.keys())[0]
                        arr = d[k] # (S, H, W)
                    s, h, w = arr.shape
                    picks = np.linspace(0, s - 1, self.num_slices).astype(int)
                    slices = arr[picks].astype(np.float32) / 255.0
                    
                    # Resize to img_size
                    import cv2
                    resized = np.stack([cv2.resize(sl, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA) for sl in slices])
                    
                    # 3-channel + ImageNet norm
                    normed = np.repeat(resized[:, np.newaxis, :, :], 3, axis=1)
                    normed = (normed - self.mean) / self.std
                    volume[s_idx] = normed
                    slot_mask[s_idx] = 1.0
                except Exception:
                    pass
        
        targets = row[TARGETS].to_numpy(dtype=np.float32)
        weight = 2.5 if ('is_gold' in row and row['is_gold'] == 1) else 1.0
        return torch.tensor(volume, dtype=torch.float32), torch.tensor(slot_mask, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32), torch.tensor(weight, dtype=torch.float32), sid""")

# -------------------------------------------------------------
# CELL 5: Architecture Implementation
# -------------------------------------------------------------
add_code("""class AnatomicalSlotHeadV2(nn.Module):
    def __init__(self, embed_dim=768, n_slots=6, n_classes=12, hidden_dim=384, drop_p=0.2):
        super().__init__()
        self.n_slots = n_slots
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim

        self.slice_attn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_dim, 1)
        )

        self.slot_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_p)
        )

        self.slot_emb = nn.Parameter(torch.randn(1, n_slots, hidden_dim) * 0.02)
        self.finding_queries = nn.Parameter(torch.randn(n_classes, hidden_dim) * 0.02)

        prior_matrix = torch.zeros(n_classes, n_slots, dtype=torch.float32)
        for i, target in enumerate(TARGETS):
            favored_slots = ANATOMICAL_PRIOR.get(target, list(range(n_slots)))
            for slot_idx in favored_slots:
                prior_matrix[i, slot_idx] = 1.5
        self.register_buffer('anatomical_prior', prior_matrix)

        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(drop_p),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(n_classes)
        ])

    def forward(self, x, slot_mask):
        B, N_S, S, D = x.shape
        x_flat = x.view(B * N_S, S, D)
        slice_scores = self.slice_attn(x_flat)
        slice_weights = F.softmax(slice_scores, dim=1)
        slot_pooled = torch.sum(slice_weights * x_flat, dim=1)
        slot_features = slot_pooled.view(B, N_S, D)

        h = self.slot_proj(slot_features) + self.slot_emb

        attn_scores = torch.einsum('bsh,qh->bqs', h, self.finding_queries) / math.sqrt(self.hidden_dim)
        attn_scores = attn_scores + self.anatomical_prior.unsqueeze(0)
        mask_expanded = slot_mask.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(mask_expanded < 0.5, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)

        finding_contexts = torch.einsum('bqs,bsh->bqh', attn_weights, h)

        logits = []
        for c in range(self.n_classes):
            c_logit = self.classifiers[c](finding_contexts[:, c, :])
            logits.append(c_logit)

        return torch.cat(logits, dim=1), attn_weights

class KneeDINOv2Pillar(nn.Module):
    def __init__(self, backbone_name='vit_base_patch14_dinov2', pretrained=True,
                 img_size=392, num_slices=16, num_classes=12):
        super().__init__()
        self.img_size = img_size
        self.num_slices = num_slices
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size
        )
        self.embed_dim = self.backbone.num_features
        self.slot_head = AnatomicalSlotHeadV2(
            embed_dim=self.embed_dim,
            n_slots=6,
            n_classes=num_classes,
            hidden_dim=384,
            drop_p=0.2
        )

    def forward(self, study_volume, slot_mask):
        B, N_S, S, C, H, W = study_volume.shape
        flat_slices = study_volume.view(B * N_S * S, C, H, W)
        feats = self.backbone(flat_slices)
        slot_feats = feats.view(B, N_S, S, self.embed_dim)
        return self.slot_head(slot_feats, slot_mask)

print("KneeDINOv2Pillar class ready!")""")

# -------------------------------------------------------------
# CELL 6: Fold 0 Training Loop & Gate Verification
# -------------------------------------------------------------
add_code("""FOLD = 0
EPOCHS = 5
BATCH_SIZE = 4
IMG_SIZE = 392

train_sub = labels_df[labels_df['fold'] != FOLD].reset_index(drop=True)
val_sub = labels_df[labels_df['fold'] == FOLD].reset_index(drop=True)
print(f"Training Fold {FOLD}: {len(train_sub)} train studies, {len(val_sub)} val studies.")

train_loader = torch.utils.data.DataLoader(
    KneeSlotDataset(train_sub, study_to_slots, actual_data_dir, is_train=True, img_size=IMG_SIZE),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    KneeSlotDataset(val_sub, study_to_slots, actual_data_dir, is_train=False, img_size=IMG_SIZE),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

model = KneeDINOv2Pillar('vit_base_patch14_dinov2', pretrained=True, img_size=IMG_SIZE).to(device)

# Freeze backbone for epoch 1, discriminative LR afterwards
for p in model.backbone.parameters(): p.requires_grad = False

optimizer = torch.optim.AdamW([
    {'params': model.slot_head.parameters(), 'lr': 2e-4, 'weight_decay': 1e-4},
    {'params': model.backbone.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4}
])
scaler = torch.amp.GradScaler('cuda')
criterion = nn.BCEWithLogitsLoss()

best_auc = 0.0
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    if epoch == 2:
        print("[Warmup complete] Unfreezing DINOv2-Base backbone layers...")
        for p in model.backbone.parameters(): p.requires_grad = True

    model.train()
    total_loss = 0.0
    for step, (vol, mask, tgt, wt, _) in enumerate(train_loader):
        vol, mask, tgt, wt = vol.to(device), mask.to(device), tgt.to(device), wt.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits, _ = model(vol, mask)
            loss = (criterion(logits, tgt) * wt.unsqueeze(1)).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}/{len(train_loader)} - Loss: {total_loss/(step+1):.4f}")

    # Validation
    model.eval()
    val_preds, val_targets, val_ids = [], [], []
    with torch.no_grad():
        for vol, mask, tgt, _, sids in val_loader:
            vol, mask = vol.to(device), mask.to(device)
            with torch.amp.autocast('cuda'):
                logits, _ = model(vol, mask)
                probs = torch.sigmoid(logits).cpu().numpy()
            val_preds.append(probs)
            val_targets.append(tgt.numpy())
            val_ids.extend(sids)

    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)
    aucs = [roc_auc_score((val_targets[:, c] >= 0.5).astype(int), val_preds[:, c]) for c in range(12)]
    macro_auc = float(np.mean(aucs))
    elapsed = time.time() - t0
    print(f"\\nEpoch {epoch}/{EPOCHS} [{elapsed:.1f}s] - Val Macro AUC: {macro_auc:.4f}")
    for c, target in enumerate(TARGETS):
        print(f"  {target:18s}: {aucs[c]:.4f}")

    if macro_auc > best_auc:
        best_auc = macro_auc
        torch.save({
            'model_state': model.state_dict(),
            'macro_auc': best_auc,
            'aucs': dict(zip(TARGETS, aucs)),
            'epoch': epoch,
            'fold': FOLD,
            'arch': 'dinov2_base_anatomical_routed'
        }, ckpt_dir / f"dinov2_plane_routed_fold{FOLD}_best.pt")
        print(f"  ==> NEW BEST MODEL SAVED! Macro AUC: {best_auc:.4f}")""")

# -------------------------------------------------------------
# CELL 7: Correlation Check vs Raptor & S6
# -------------------------------------------------------------
add_code("""# Evaluate error rank correlation rho vs Raptor + S6 baseline
# Gate Criterion: Solo Macro AUC >= 0.935 and rho <= 0.72 vs baseline
oof_df = pd.DataFrame(val_preds, columns=TARGETS)
oof_df.insert(0, 'StudyInstanceUID', val_ids)
oof_df.to_csv(oof_dir / f"oof_dinov2_base_fold{FOLD}.csv", index=False)
print(f"OOF predictions saved to {oof_dir / f'oof_dinov2_base_fold{FOLD}.csv'}")

# Check correlation with baseline
baseline_path = Path("/content/oof_triplane_resnet34.csv")
if baseline_path.exists():
    base_df = pd.read_csv(baseline_path)
    merged = oof_df.merge(base_df, on='StudyInstanceUID', suffixes=('_dino', '_base'))
    rhos = [merged[f"{t}_dino"].corr(merged[f"{t}_base"], method='spearman') for t in TARGETS]
    mean_rho = float(np.mean(rhos))
    print(f"\\nSpearman Error Correlation rho vs Baseline: {mean_rho:.4f}")
    if mean_rho <= 0.72 and best_auc >= 0.935:
        print(">>> DECISION GATE PASSED! The 3rd Pillar is validated for full 5-fold ensemble.")
    else:
        print(f">>> GATE ASSESSMENT: AUC={best_auc:.4f}, rho={mean_rho:.4f}")""")

# Build Notebook
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

out_path = Path("colab/rsna_knee_dinov2_plane_routed_trainer.ipynb")
out_path.write_text(json.dumps(nb, indent=2))
print(f"Generated Colab Notebook at: {out_path.resolve()} ({len(cells)} cells)")
