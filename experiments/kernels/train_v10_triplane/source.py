# %% [markdown]
# # Tri-Plane 2.5D End-to-End Trainer on Gold-Calibrated 3-State Labels
#
# ## Purpose & Architecture
# Addresses the root bottleneck identified in RSNA Knee Abnormality Detection:
# **Systematic under-calling (false negative bias)** in public report-derived labels.
#
# ### Key Innovations:
# 1. **Gold-Calibrated 3-State Labels (`knee-labels-3state-calibrated`):**
#    - Ground-truth Gold studies (58 studies) retain exact radiologist image annotations (2.0x weight).
#    - Confident Positive (state 1): soft targets ~0.80-0.95 with extraction confidence weight.
#    - Confident Negative (state 0): soft targets ~0.02-0.08 with extraction confidence weight.
#    - Unmentioned (state -1): Anchored to empirical Gold prior (e.g. 0.41 ACL, 0.47 Synovitis)
#      rather than forced to 0.0 false negatives!
#    - Clinical priors: Effusion->Synovitis coupling and Lateral OA de-biasing.
#
# 2. **Tri-Plane 2.5D Feature Routing:**
#    - Multi-plane input: Axial (16 slices), Coronal (16 slices), Sagittal (16 slices).
#    - Shared deep ResNet-34 backbone with depth-slice channel adaptation.
#    - Anatomically routed finding heads:
#      * Sagittal (Cruciates & Menisci): ACL, MCL, Medial Meniscus, Lateral Meniscus
#      * Coronal (Cartilage OA & Bones): Medial OA, Lateral OA, Fracture
#      * Axial (Fluids & Patellofemoral): PF OA, Effusion, Synovitis, Baker's, Contusion
#      * Global cross-plane residual projection across all 12 findings.
#
# 3. **Rock-Solid Frozen 5-Fold Validation:**
#    - Uses `data/folds.csv` covering all 4,407 studies.
#    - Produces full 4,407-study Out-of-Fold (OOF) predictions and genuine macro ROC-AUC.
#    - Decorrelated member to blend with the banked 0.942 ensemble (`infer-v16`).

# %%
import os, sys, time, json, math, random, gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# %%
# --- Configuration & Constants ---
LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture'
]

# Anatomical routing groups
MENISCI_CONDS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus']
OA_BONE_CONDS = ['Medial OA', 'Lateral OA', 'Fracture']
FLUID_CONDS = ['PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion']

MENISCI_IDX = [LABELS.index(c) for c in MENISCI_CONDS]
OA_BONE_IDX = [LABELS.index(c) for c in OA_BONE_CONDS]
FLUID_IDX = [LABELS.index(c) for c in FLUID_CONDS]

# Headroom priority loss weights
COND_WEIGHTS = {
    'Lateral Meniscus': 1.5,
    'Synovitis': 1.5,
    'Medial Meniscus': 1.3,
    'PF OA': 1.3,
    'Lateral OA': 1.3,
    'Fracture': 1.2,
    'ACL': 1.2,
    'MCL': 1.0,
    'Medial OA': 1.0,
    'Effusion': 1.0,
    "Baker's": 1.0,
    'Contusion': 1.0
}
COND_W_TENSOR = torch.tensor([COND_WEIGHTS[c] for c in LABELS], dtype=torch.float32)

IN_SLICES = 16
# Slices sampled evenly from the 24 depth slices
PICK_SLICES = [2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 20, 21]

BATCH_SIZE = 8
ACCUM_STEPS = 2
EPOCHS = 8
LR = 4e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
SEED = 42

OUT_DIR = Path('/kaggle/working')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

seed_everything(SEED)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Active compute device: {dev}')
if dev == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}, Device count: {torch.cuda.device_count()}')

# %%
# --- Find Files and Mounts ---
def find_dirs(*markers, max_depth=4):
    out, frontier = [], [(Path('/kaggle/input'), 0)]
    while frontier:
        d, depth = frontier.pop(0)
        if all((d / m).exists() for m in markers):
            out.append(d)
            continue
        if depth >= max_depth:
            continue
        try:
            children = sorted(p for p in d.iterdir() if p.is_dir())
        except PermissionError:
            continue
        for c in children:
            if c.name in ('train_series', 'test_series'):
                continue
            frontier.append((c, depth + 1))
    if not out:
        raise FileNotFoundError(f'Could not find directories with markers: {markers}')
    return out

def find_file(name, max_depth=4):
    for root, _, files in os.walk('/kaggle/input'):
        if name in files:
            return Path(root) / name
    raise FileNotFoundError(f'File {name} not found in /kaggle/input')

# %%
# --- Load 336px Volume Cache ---
print('Discovering 336px volume cache shards...')
SHARDS = sorted(
    find_dirs('cache_meta.json', 'volumes.npy', 'index.csv'),
    key=lambda d: json.loads((d / 'cache_meta.json').read_text())['row_lo']
)
metas = [json.loads((d / 'cache_meta.json').read_text()) for d in SHARDS]
cache_meta = metas[0]
D_SLICES, SIZE = cache_meta['d_slices'], cache_meta['size']
N_ALL = cache_meta['n_studies_all']

print(f'Found {len(SHARDS)} shards covering {N_ALL} studies at {SIZE}x{SIZE} resolution.')
edge = 0
for m in metas:
    assert m['size'] == SIZE and m['d_slices'] == D_SLICES
    assert m['row_lo'] == edge, f'Shard range mismatch: expected {edge}, got {m["row_lo"]}'
    edge = m['row_hi']
assert edge == N_ALL, f'Shards cover {edge} of {N_ALL} studies'

_vm = np.load(SHARDS[0] / 'volumes.npy', mmap_mode='r')
VOL_DTYPE, VOL_OFFSET = _vm.dtype, _vm.offset
ITEM_SHAPE = _vm.shape[1:]  # (3, 24, 336, 336)
ITEM_BYTES = int(np.prod(ITEM_SHAPE)) * VOL_DTYPE.itemsize
del _vm

SHARD_FD = [os.open(str(d / 'volumes.npy'), os.O_RDONLY) for d in SHARDS]
SHARD_LO = np.array([m['row_lo'] for m in metas])

def read_vol(r):
    s = int(np.searchsorted(SHARD_LO, r, side='right')) - 1
    off = VOL_OFFSET + (r - SHARD_LO[s]) * ITEM_BYTES
    buf = os.pread(SHARD_FD[s], ITEM_BYTES, off)
    return np.frombuffer(buf, dtype=VOL_DTYPE).reshape(ITEM_SHAPE)

# %%
# --- Load Calibrated Labels and Folds ---
labels_path = find_file('train_labels_3state_calibrated.csv')
folds_path = find_file('folds.csv')
print(f'Loading labels from: {labels_path}')
print(f'Loading folds from: {folds_path}')

labels_df = pd.read_csv(labels_path)
folds_df = pd.read_csv(folds_path)

# Verify alignment with cache index
index_df = pd.concat([pd.read_csv(d / 'index.csv') for d in SHARDS], ignore_index=True)
index_df = index_df.sort_values('row').reset_index(drop=True)
assert len(index_df) == N_ALL, 'Index study count mismatch'

# Merge everything onto index order [0..N_ALL-1]
full_df = index_df[['row', 'StudyInstanceUID']].merge(labels_df, on='StudyInstanceUID', how='left')
full_df = full_df.merge(folds_df, on='StudyInstanceUID', how='left')
assert len(full_df) == N_ALL, 'Merged study count mismatch'
assert full_df['fold'].notna().all(), 'Missing fold assignments'

# Prepare label matrices in index order
Y_PROB = full_df[LABELS].to_numpy(np.float32)
Y_CONF = np.column_stack([full_df[f'{c}__conf'].to_numpy(np.float32) for c in LABELS])
Y_WEIGHT = np.column_stack([full_df[f'{c}__weight'].to_numpy(np.float32) for c in LABELS])
IS_GOLD = full_df['is_gold'].to_numpy(np.int64)
FOLDS = full_df['fold'].to_numpy(np.int64)

print(f'Dataset prepared: {N_ALL} total studies | {IS_GOLD.sum()} Gold ground-truth studies.')

# %%
# --- Dataset Definition ---
class TriPlaneKneeDataset(Dataset):
    def __init__(self, indices, is_train=True):
        self.indices = indices
        self.is_train = is_train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row_id = self.indices[i]
        raw_vol = read_vol(row_id) # (3, 24, 336, 336) uint8
        
        # Sub-sample 16 slices per plane
        vol = raw_vol[:, PICK_SLICES, :, :].copy() # (3, 16, 336, 336)
        
        if self.is_train:
            # Anatomy-safe in-plane translation (no horizontal flip!)
            roll_y = np.random.randint(-6, 7)
            roll_x = np.random.randint(-6, 7)
            vol = np.roll(vol, roll_y, axis=2)
            vol = np.roll(vol, roll_x, axis=3)
            
            # Subtle intensity scaling
            gain = np.random.uniform(0.92, 1.08)
            bias = np.random.uniform(-8.0, 8.0)
            vol_float = np.clip(vol.astype(np.float32) * gain + bias, 0.0, 255.0)
        else:
            vol_float = vol.astype(np.float32)
            
        # Scale to [0, 1] and standardize
        vol_norm = (vol_float / 255.0 - 0.45) / 0.22
        
        x_tensor = torch.from_numpy(vol_norm).float()
        y_prob = torch.from_numpy(Y_PROB[row_id])
        y_weight = torch.from_numpy(Y_WEIGHT[row_id])
        is_gold = IS_GOLD[row_id]
        
        return x_tensor, y_prob, y_weight, is_gold

# %%
# --- Model Architecture ---
class AnatomicalTriPlane25DNet(nn.Module):
    def __init__(self, in_slices=IN_SLICES, num_classes=len(LABELS), drop=0.25):
        super().__init__()
        # Pretrained ResNet-34 backbone
        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        old_w = base.conv1.weight.data # (64, 3, 7, 7)
        
        # Expand conv1 to in_slices channels with variance-preserving weight repeat
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
        
        # Anatomically routed heads
        self.head_menisci = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(feat_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(256, len(MENISCI_IDX))
        )
        self.head_oa_bone = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(feat_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(256, len(OA_BONE_IDX))
        )
        self.head_fluid = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(feat_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(256, len(FLUID_IDX))
        )
        self.global_proj = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(feat_dim * 3, num_classes)
        )
        
        order = MENISCI_CONDS + OA_BONE_CONDS + FLUID_CONDS
        inv_perm = [order.index(c) for c in LABELS]
        self.register_buffer('inv_perm', torch.tensor(inv_perm, dtype=torch.long))

    def forward_plane(self, x):
        # x: (B*P, S, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1) # (B*P, 512)

    def forward(self, x):
        # x: (B, P=3, S=16, H, W)
        B, P, S, H, W = x.shape
        x_flat = x.reshape(B * P, S, H, W)
        
        f_all = self.forward_plane(x_flat).reshape(B, P, -1)
        f_ax = f_all[:, 0]   # (B, 512)
        f_cor = f_all[:, 1]  # (B, 512)
        f_sag = f_all[:, 2]  # (B, 512)
        
        # Clinical feature combinations
        feat_menisci = torch.cat([f_sag, f_cor], dim=1) # (B, 1024)
        feat_oa_bone = torch.cat([f_cor, f_sag], dim=1) # (B, 1024)
        feat_fluid = torch.cat([f_ax, f_cor], dim=1)    # (B, 1024)
        f_cat = torch.cat([f_ax, f_cor, f_sag], dim=1)  # (B, 1536)
        
        p_menisci = self.head_menisci(feat_menisci)
        p_oa_bone = self.head_oa_bone(feat_oa_bone)
        p_fluid = self.head_fluid(feat_fluid)
        
        all_preds = torch.cat([p_menisci, p_oa_bone, p_fluid], dim=1)
        logits = all_preds[:, self.inv_perm] + self.global_proj(f_cat)
        
        return logits

# %%
# --- Metric Calculation ---
def evaluate_predictions(y_true, y_pred):
    aucs = []
    for j in range(12):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        # Binarize if continuous for AUC calculation
        yt_bin = (yt >= 0.5).astype(int)
        if len(np.unique(yt_bin)) > 1:
            try:
                a = roc_auc_score(yt_bin, yp)
                aucs.append(a)
            except Exception:
                aucs.append(0.5)
        else:
            aucs.append(0.5)
    return np.mean(aucs), aucs

# %%
# --- 5-Fold Cross Validation Training Loop ---
COND_W_DEV = COND_W_TENSOR.to(dev)

oof_probabilities = np.full((N_ALL, 12), np.nan, dtype=np.float32)
fold_results = []

print('\n' + '='*70)
print('STARTING 5-FOLD TRI-PLANE 2.5D TRAINING')
print('='*70)

total_t0 = time.time()

for fold in range(5):
    print(f'\n--- [FOLD {fold}/5] ---')
    train_idx = np.where(FOLDS != fold)[0]
    val_idx = np.where(FOLDS == fold)[0]
    
    val_gold_mask = IS_GOLD[val_idx] == 1
    n_val_gold = val_gold_mask.sum()
    print(f'Train studies: {len(train_idx)} | Val studies: {len(val_idx)} (including {n_val_gold} Gold ground-truth)')
    
    train_ds = TriPlaneKneeDataset(train_idx, is_train=True)
    val_ds = TriPlaneKneeDataset(val_idx, is_train=False)
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    model = AnatomicalTriPlane25DNet().to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LR)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_auc = -1.0
    best_gold_auc = -1.0
    best_val_preds = None
    best_model_state = None
    
    for epoch in range(1, EPOCHS + 1):
        ep_t0 = time.time()
        model.train()
        train_loss = 0.0
        n_train_batches = 0
        optimizer.zero_grad(set_to_none=True)
        
        for step, (xb, yb, wb, is_g) in enumerate(train_loader):
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            wb = wb.to(dev, non_blocking=True)
            is_g = is_g.to(dev, non_blocking=True)
            
            # Loss weights: confidence * condition_headroom * gold_boost (2.0x)
            gold_boost = torch.where(is_g.unsqueeze(1) == 1, 2.0, 1.0)
            sample_w = wb * COND_W_DEV * gold_boost
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb, weight=sample_w) / ACCUM_STEPS
                
            scaler.scale(loss).backward()
            train_loss += loss.item() * ACCUM_STEPS
            n_train_batches += 1
            
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                
        train_loss /= max(1, n_train_batches)
        
        # Validation
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for xb, _, _, _ in val_loader:
                xb = xb.to(dev, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(xb)
                    probs = torch.sigmoid(logits)
                val_preds_list.append(probs.cpu().numpy())
                
        val_preds = np.concatenate(val_preds_list, axis=0)
        
        # Macro ROC-AUC on all validation studies
        val_macro_auc, _ = evaluate_predictions(Y_PROB[val_idx], val_preds)
        
        # Macro ROC-AUC on Gold studies in this fold
        gold_macro_auc = 0.5
        if n_val_gold >= 4:
            gold_val_preds = val_preds[val_gold_mask]
            gold_val_true = Y_PROB[val_idx][val_gold_mask]
            gold_macro_auc, _ = evaluate_predictions(gold_val_true, gold_val_preds)
            
        ep_sec = time.time() - ep_t0
        print(f'Epoch {epoch:02d}/{EPOCHS:02d} [{ep_sec:4.1f}s] - Loss: {train_loss:.4f} | Val AUC: {val_macro_auc:.4f} | Gold AUC: {gold_macro_auc:.4f}')
        
        if val_macro_auc > best_val_auc:
            best_val_auc = val_macro_auc
            best_gold_auc = gold_macro_auc
            best_val_preds = val_preds.copy()
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    print(f'Fold {fold} Complete! Best Val AUC: {best_val_auc:.4f} (Gold AUC: {best_gold_auc:.4f})')
    
    # Save checkpoint
    ckpt_path = OUT_DIR / f'triplane_fold{fold}.pt'
    torch.save(best_model_state, ckpt_path)
    print(f'Saved checkpoint to: {ckpt_path}')
    
    # Save OOF predictions
    oof_probabilities[val_idx] = best_val_preds
    fold_results.append({
        'fold': fold,
        'val_macro_auc': float(best_val_auc),
        'gold_macro_auc': float(best_gold_auc),
        'checkpoint': str(ckpt_path.name)
    })
    
    del model, optimizer, scheduler, scaler
    gc.collect()
    if dev == 'cuda':
        torch.cuda.empty_cache()

# %%
# --- Overall OOF Evaluation ---
total_min = (time.time() - total_t0) / 60.0
print('\n' + '='*70)
print(f'ALL 5 FOLDS COMPLETED in {total_min:.1f} minutes!')
print('='*70)

# Check complete OOF coverage
assert not np.isnan(oof_probabilities).any(), 'OOF probabilities contain NaNs!'

# Compute Full 4,407-Study OOF AUC
oof_macro_auc, oof_per_cond = evaluate_predictions(Y_PROB, oof_probabilities)

# Compute 58-Study Gold OOF AUC
gold_indices = np.where(IS_GOLD == 1)[0]
gold_macro_auc, gold_per_cond = evaluate_predictions(Y_PROB[gold_indices], oof_probabilities[gold_indices])

print(f'\n>>> FULL 4,407-STUDY OOF MACRO ROC-AUC: {oof_macro_auc:.4f} <<<')
print(f'>>> 58-STUDY GOLD OOF MACRO ROC-AUC     : {gold_macro_auc:.4f} <<<\n')

print(f'{"Finding":18s} | {"OOF Macro AUC":14s} | {"Gold AUC":10s}')
print('-'*50)
for j, cond in enumerate(LABELS):
    print(f'{cond:18s} | {oof_per_cond[j]:14.4f} | {gold_per_cond[j]:10.4f}')

# %%
# --- Save OOF Predictions and Summary ---
oof_df = pd.DataFrame(oof_probabilities, columns=LABELS)
oof_df.insert(0, 'StudyInstanceUID', full_df['StudyInstanceUID'])
oof_df.insert(1, 'is_gold', IS_GOLD)
oof_df.insert(2, 'fold', FOLDS)
oof_csv = OUT_DIR / 'oof_triplane_resnet34.csv'
oof_df.to_csv(oof_csv, index=False)
print(f'\nSaved OOF predictions to {oof_csv}')

summary = {
    'architecture': 'AnatomicalTriPlane25DNet (ResNet-34)',
    'in_slices': IN_SLICES,
    'epochs_per_fold': EPOCHS,
    'training_time_minutes': round(total_min, 2),
    'oof_macro_auc': round(float(oof_macro_auc), 5),
    'gold_macro_auc': round(float(gold_macro_auc), 5),
    'per_condition_oof_auc': {c: round(float(oof_per_cond[j]), 4) for j, c in enumerate(LABELS)},
    'per_condition_gold_auc': {c: round(float(gold_per_cond[j]), 4) for j, c in enumerate(LABELS)},
    'fold_results': fold_results
}
summary_path = OUT_DIR / 'triplane_training_summary.json'
summary_path.write_text(json.dumps(summary, indent=2))
print(f'Saved training summary to {summary_path}')
print('\n[COMPLETE] Tri-Plane 2.5D Model is ready for Kaggle submission ensemble!')
