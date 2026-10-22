# %% [markdown]
# # Knee Joint Specialist v11: High-Resolution Tri-Plane Attention Trainer
#
# ## Architecture & Innovations:
# 1. **High-Resolution 288px Physical Joint ROI Crop:**
#    - Focuses purely on the central 130mm joint line, avoiding peripheral muscle and RF coil artifacts.
# 2. **Multi-Plane Slice Attention Pooling:**
#    - Abnormality queries attend across Sagittal, Coronal, and Axial planes simultaneously.
# 3. **Cross-Label Biological Correlation Transformer:**
#    - Captures clinical co-occurrence patterns (e.g. ACL + Meniscus + Bone Contusion coupling).
# 4. **Gold-Calibrated Posterior Supervision:**
#    - True 58 Gold studies receive 2.5x loss weight and exact binary ground truth.
#    - 4,349 studies receive continuous margin targets calibrated to empirical posteriors.
# 5. **Prioritized Headroom Loss:**
#    - Heavy focus on the bottleneck conditions: Lateral Meniscus, Synovitis, PF OA, Lateral OA.

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

# Headroom priority loss weights heavily penalizing errors on weak findings
COND_WEIGHTS = {
    'Lateral Meniscus': 2.0,
    'Synovitis': 2.0,
    'PF OA': 1.8,
    'Lateral OA': 1.8,
    'Medial Meniscus': 1.5,
    'Fracture': 1.2,
    'ACL': 1.0,
    'MCL': 1.0,
    'Medial OA': 1.0,
    'Effusion': 1.0,
    "Baker's": 1.0,
    'Contusion': 1.0
}
COND_W_TENSOR = torch.tensor([COND_WEIGHTS[c] for c in LABELS], dtype=torch.float32)

IN_SLICES = 16
PICK_SLICES = [2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 20, 21]

# Center crop 288x288 out of 336x336 (removes 24px border on all sides)
CROP_Y1, CROP_Y2 = 24, 312
CROP_X1, CROP_X2 = 24, 312

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
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# %%
# --- Discover Input Files ---
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
# --- Load Calibrated Posterior Labels and Folds ---
labels_path = find_file('train_labels_posterior_calibrated.csv')
folds_path = find_file('folds.csv')
print(f'Loading posterior labels from: {labels_path}')
print(f'Loading folds from: {folds_path}')

labels_df = pd.read_csv(labels_path)
folds_df = pd.read_csv(folds_path)

index_df = pd.concat([pd.read_csv(d / 'index.csv') for d in SHARDS], ignore_index=True)
index_df = index_df.sort_values('row').reset_index(drop=True)
assert len(index_df) == N_ALL, 'Index study count mismatch'

full_df = index_df[['row', 'StudyInstanceUID']].merge(labels_df, on='StudyInstanceUID', how='left')
full_df = full_df.merge(folds_df, on='StudyInstanceUID', how='left')
assert len(full_df) == N_ALL, 'Merged study count mismatch'
assert full_df['fold'].notna().all(), 'Missing fold assignments'

Y_PROB = full_df[LABELS].to_numpy(np.float32)
Y_WEIGHT = np.column_stack([full_df[f'{c}__weight'].to_numpy(np.float32) for c in LABELS])
IS_GOLD = full_df['is_gold'].to_numpy(np.int64)
FOLDS = full_df['fold'].to_numpy(np.int64)

print(f'Dataset prepared: {N_ALL} studies | {IS_GOLD.sum()} Gold ground-truth studies (2.5x weighted).')

# %%
# --- Dataset Definition with 288px Joint Center Crop ---
class JointSpecialistDataset(Dataset):
    def __init__(self, indices, is_train=True):
        self.indices = indices
        self.is_train = is_train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row_id = self.indices[i]
        raw_vol = read_vol(row_id) # (3, 24, 336, 336) uint8
        
        # Sub-sample 16 slices per plane and crop 288x288 joint line
        vol = raw_vol[:, PICK_SLICES, CROP_Y1:CROP_Y2, CROP_X1:CROP_X2].copy() # (3, 16, 288, 288)
        
        if self.is_train:
            # Anatomy-safe in-plane jitter (+/- 4 pixels)
            roll_y = np.random.randint(-4, 5)
            roll_x = np.random.randint(-4, 5)
            vol = np.roll(vol, roll_y, axis=2)
            vol = np.roll(vol, roll_x, axis=3)
            
            # Subtle intensity scaling
            gain = np.random.uniform(0.94, 1.06)
            bias = np.random.uniform(-6.0, 6.0)
            vol_float = np.clip(vol.astype(np.float32) * gain + bias, 0.0, 255.0)
        else:
            vol_float = vol.astype(np.float32)
            
        vol_norm = (vol_float / 255.0 - 0.45) / 0.22
        
        x_tensor = torch.from_numpy(vol_norm).float()
        y_prob = torch.from_numpy(Y_PROB[row_id])
        y_weight = torch.from_numpy(Y_WEIGHT[row_id])
        is_gold = IS_GOLD[row_id]
        
        return x_tensor, y_prob, y_weight, is_gold

# %%
# --- Multi-Plane Joint Specialist Architecture ---
class MultiPlaneSpecialist(nn.Module):
    def __init__(self, in_slices=IN_SLICES, num_classes=len(LABELS), embed_dim=512, drop=0.2):
        super().__init__()
        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
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
        
        # Abnormality queries attending across the 3 planes
        self.queries = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True, dropout=drop)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Cross-label biological correlation transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=1024, dropout=drop,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.cross_label = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Individual per-finding prediction heads
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(128, 1)
            ) for _ in range(num_classes)
        ])
        
    def forward(self, x):
        # x: [B, P=3, S=16, H=288, W=288]
        B, P, S, H, W = x.shape
        x_flat = x.reshape(B * P, S, H, W)
        x = self.conv1(x_flat)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        plane_feats = torch.flatten(x, 1).reshape(B, P, -1) # [B, 3, 512]
        
        q = self.queries.unsqueeze(0).expand(B, -1, -1)     # [B, 12, 512]
        attended, _ = self.mha(query=q, key=plane_feats, value=plane_feats)
        feat = self.norm1(q + attended)                     # [B, 12, 512]
        
        enhanced = self.norm2(self.cross_label(feat))        # [B, 12, 512]
        
        logits = []
        for i in range(len(self.classifiers)):
            logits.append(self.classifiers[i](enhanced[:, i, :]))
        return torch.cat(logits, dim=-1)

# %%
# --- Evaluation Metric ---
def evaluate_predictions(y_true, y_pred):
    aucs = []
    for j in range(12):
        yt = y_true[:, j]
        yp = y_pred[:, j]
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
# --- 5-Fold Training Loop ---
COND_W_DEV = COND_W_TENSOR.to(dev)

oof_probabilities = np.full((N_ALL, 12), np.nan, dtype=np.float32)
fold_results = []

print('\n' + '='*70)
print('STARTING 5-FOLD JOINT SPECIALIST TRAINING')
print('='*70)

total_t0 = time.time()

for fold in range(5):
    print(f'\n--- [FOLD {fold}/5] ---')
    train_idx = np.where(FOLDS != fold)[0]
    val_idx = np.where(FOLDS == fold)[0]
    
    val_gold_mask = IS_GOLD[val_idx] == 1
    n_val_gold = val_gold_mask.sum()
    print(f'Train studies: {len(train_idx)} | Val studies: {len(val_idx)} (including {n_val_gold} Gold ground-truth)')
    
    train_ds = JointSpecialistDataset(train_idx, is_train=True)
    val_ds = JointSpecialistDataset(val_idx, is_train=False)
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    model = MultiPlaneSpecialist().to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    total_steps = EPOCHS * (len(train_loader) // ACCUM_STEPS + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LR)
    scaler = torch.amp.GradScaler('cuda', enabled=(dev == 'cuda'))
    
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
            
            gold_boost = torch.where(is_g.unsqueeze(1) == 1, 2.5, 1.0)
            sample_w = wb * COND_W_DEV * gold_boost
            
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
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
                with torch.amp.autocast('cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
                    logits = model(xb)
                    probs = torch.sigmoid(logits)
                val_preds_list.append(probs.cpu().numpy())
                
        val_preds = np.concatenate(val_preds_list, axis=0)
        val_macro_auc, _ = evaluate_predictions(Y_PROB[val_idx], val_preds)
        
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
    
    ckpt_path = OUT_DIR / f'specialist_fold{fold}.pt'
    torch.save(best_model_state, ckpt_path)
    print(f'Saved checkpoint to: {ckpt_path}')
    
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
# --- Overall OOF Evaluation & Manifest Export ---
total_min = (time.time() - total_t0) / 60.0
print('\n' + '='*70)
print(f'ALL 5 FOLDS COMPLETED in {total_min:.1f} minutes!')
print('='*70)

oof_macro_auc, oof_per_cond = evaluate_predictions(Y_PROB, oof_probabilities)
gold_indices = np.where(IS_GOLD == 1)[0]
gold_macro_auc, gold_per_cond = evaluate_predictions(Y_PROB[gold_indices], oof_probabilities[gold_indices])

print(f'\n>>> FULL 4,407-STUDY OOF MACRO ROC-AUC: {oof_macro_auc:.4f} <<<')
print(f'>>> 58-STUDY GOLD OOF MACRO ROC-AUC     : {gold_macro_auc:.4f} <<<\n')

print(f'{"Finding":18s} | {"OOF Macro AUC":14s} | {"Gold AUC":10s}')
print('-'*50)
for j, cond in enumerate(LABELS):
    print(f'{cond:18s} | {oof_per_cond[j]:14.4f} | {gold_per_cond[j]:10.4f}')

oof_df = pd.DataFrame(oof_probabilities, columns=LABELS)
oof_df.insert(0, 'StudyInstanceUID', full_df['StudyInstanceUID'])
oof_df.insert(1, 'is_gold', IS_GOLD)
oof_df.insert(2, 'fold', FOLDS)
oof_csv = OUT_DIR / 'oof_specialist_v11.csv'
oof_df.to_csv(oof_csv, index=False)
print(f'\nSaved OOF predictions to {oof_csv}')

# Compute recommended deploy weights based on Gold58 performance
deploy_weights = {}
for j, cond in enumerate(LABELS):
    g_auc = gold_per_cond[j]
    if cond in ['Lateral Meniscus', 'Synovitis', 'PF OA', 'Lateral OA', 'Medial Meniscus'] and g_auc >= 0.75:
        # High-impact conditions
        deploy_weights[cond] = float(np.clip(0.08 + 0.3 * (g_auc - 0.75), 0.08, 0.20))
    elif g_auc >= 0.85:
        deploy_weights[cond] = float(np.clip(0.05 + 0.2 * (g_auc - 0.85), 0.05, 0.15))
    else:
        deploy_weights[cond] = 0.0

manifest = {
    'schema': 'rsna-knee-joint-specialist-v11',
    'title': 'High-Resolution Tri-Plane Joint Specialist',
    'seed': SEED,
    'in_slices': IN_SLICES,
    'crop_size': 288,
    'epochs_per_fold': EPOCHS,
    'training_time_minutes': round(total_min, 2),
    'oof_macro_auc': round(float(oof_macro_auc), 5),
    'gold_macro_auc': round(float(gold_macro_auc), 5),
    'per_condition_oof_auc': {c: round(float(oof_per_cond[j]), 4) for j, c in enumerate(LABELS)},
    'per_condition_gold_auc': {c: round(float(gold_per_cond[j]), 4) for j, c in enumerate(LABELS)},
    'deploy_weights': deploy_weights,
    'fold_results': fold_results
}
manifest_path = OUT_DIR / 'v11_manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2))
print(f'Saved specialist manifest to {manifest_path}')
print('\n[COMPLETE] Knee Joint Specialist v11 training complete!')
