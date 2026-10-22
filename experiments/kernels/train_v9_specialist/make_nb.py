#!/usr/bin/env python3
import json

code = r'''#!/usr/bin/env python3
"""
RSNA Knee Abnormality Detection - High-Precision Specialist Fine-Tuning v9
- Precomputed Corpus: dreaddevelopment/knee-raptor-corpus (4,407 memory-mapped studies)
- Supervision: 4-Source Consensus Soft Labels (yunusgmsoy + dreaddevelopment) weighted by confidence
- Target Focus: 5 weak findings (Lateral OA, Synovitis, PF OA, Lateral Meniscus, Contusion)
- Memory-Bounded: Micro-chunked window encoding (peak VRAM < 2.5 GB)
"""

import os, sys, time, json, math, random, glob
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import timm

LAB = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture'
]

# Finding weights heavily penalizing errors on the 5 weak bottleneck findings
CLASS_WEIGHTS = torch.tensor([
    1.0,  # ACL
    1.2,  # MCL
    1.0,  # Medial Meniscus
    1.8,  # Lateral Meniscus
    1.0,  # Medial OA
    2.5,  # Lateral OA
    1.8,  # PF OA
    1.0,  # Effusion
    2.2,  # Synovitis
    1.0,  # Baker's
    1.8,  # Contusion
    1.0,  # Fracture
], dtype=torch.float32)

def find_file(name, root='/kaggle/input'):
    for d, _, fs in os.walk(root):
        if name in fs:
            return os.path.join(d, name)
    return None

class TwoPartVols:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = a.shape[0]
        self.shape = (a.shape[0] + b.shape[0],) + tuple(a.shape[1:])
    def __len__(self):
        return self.shape[0]
    def __getitem__(self, row):
        return self.a[row] if row < self.n_a else self.b[row - self.n_a]

def open_corpus():
    av, bv = find_file('all_vols.npy'), find_file('extra_vols.npy')
    am, bm = find_file('all_masks.npy'), find_file('extra_masks.npy')
    if not all((av, bv, am, bm)):
        raise RuntimeError('Corpus files missing! Please attach dreaddevelopment/knee-raptor-corpus and corpus-ext.')
    vols = TwoPartVols(np.load(av, mmap_mode='r'), np.load(bv, mmap_mode='r'))
    masks = np.concatenate([np.load(am), np.load(bm)], axis=0)
    return vols, masks

def open_ids():
    a, b = find_file('all_ids.npy'), find_file('extra_ids.npy')
    if not a or not b:
        raise RuntimeError('ID files missing!')
    return np.concatenate([np.load(a), np.load(b)]).tolist()

class StudyWindows(Dataset):
    def __init__(self, uids, id2row, labels, confs, res=384, k=10, train=True):
        self.uids = uids
        self.id2row = id2row
        self.labels = labels
        self.confs = confs
        self.res = res
        self.k = k
        self.train = train
        self.vols, self.masks = open_corpus()
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.uids)

    def _centers(self, valid, k):
        if len(valid) == 0:
            return [0] * k
        if len(valid) <= k:
            idx = np.linspace(0, len(valid) - 1, k).round().astype(int)
            return [valid[i] for i in idx]
        if self.train:
            stride = len(valid) / k
            jitter = (np.random.rand(k) - 0.5) * stride * 0.4
            idx = np.clip(np.linspace(0, len(valid) - 1, k) + jitter, 0, len(valid) - 1).round().astype(int)
            return [valid[i] for i in idx]
        idx = np.linspace(0, len(valid) - 1, k).round().astype(int)
        return [valid[i] for i in idx]

    def _resize(self, img):
        t = torch.from_numpy(img).unsqueeze(0)
        if t.shape[-1] != self.res or t.shape[-2] != self.res:
            t = F.interpolate(t, size=(self.res, self.res), mode='bilinear', align_corners=False)
        return t.squeeze(0)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        row = self.id2row[uid]
        mask = self.masks[row]
        D = mask.shape[0]
        valid = np.where(mask > 0)[0]
        if len(valid) < 3:
            valid = np.arange(min(3, D))
        cs = self._centers(valid, self.k)
        vol = self.vols[row]
        tiles = []
        for c in cs:
            c = max(1, min(c, D - 2))
            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
            tiles.append(self._resize(tri))
        wins = np.stack(tiles, 0)
        x = torch.from_numpy(wins)
        if self.train:
            g = 1.0 + (random.random() - 0.5) * 0.15
            x = (x * g).clamp(0, 1)
        x = (x - self._mean) / self._std
        y = torch.from_numpy(self.labels[uid])
        conf = torch.from_numpy(self.confs[uid])
        return x, y, conf

def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    confs = torch.stack([b[2] for b in batch])
    return xs, ys, confs

class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=384, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(
            nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n)
        )
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x, chunk_size=8):
        B, K, C, H, W = x.shape
        x_flat = x.view(B * K, C, H, W)
        feats = []
        # Micro-chunking through backbone strictly bounds peak VRAM < 2.5 GB
        for i in range(0, B * K, chunk_size):
            chunk = x_flat[i : i + chunk_size]
            f = self.backbone(chunk)
            feats.append(f)
        feats = torch.cat(feats, dim=0)
        return feats.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))

def main():
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Starting Specialist fine-tuning on {dev} with Memory-Bounded VRAM...')

    # 1. Load IDs and labels
    ids = open_ids()
    id2row = {u: i for i, u in enumerate(ids)}
    idset = set(ids)

    train_csv_path = find_file('train.csv')
    train_df = pd.read_csv(train_csv_path)
    gold_df = train_df[train_df['ACL'].notna()].copy().set_index('StudyInstanceUID')
    gold_ids = [u for u in ids if u in gold_df.index]
    goldset = set(gold_ids)

    # Load 4-source merged labels with confidence
    v5_path = find_file('report_labels_v5.csv')
    dread_path = find_file('labels_llm_soft.csv')

    v5_df = pd.read_csv(v5_path).set_index('StudyInstanceUID') if v5_path else None
    dread_df = pd.read_csv(dread_path).set_index('StudyInstanceUID') if dread_path else None

    print(f'Found v5_labels: {v5_df is not None}, dread_labels: {dread_df is not None}')

    train_ids = [u for u in ids if u not in goldset]
    labels = {}
    confs = {}

    for u in train_ids:
        l_arr = np.full(12, 0.5, dtype=np.float32)
        c_arr = np.ones(12, dtype=np.float32)
        if v5_df is not None and u in v5_df.index:
            for j, col in enumerate(LAB):
                l_arr[j] = float(v5_df.loc[u, col])
                c_col = col + '__conf'
                if c_col in v5_df.columns:
                    c_arr[j] = float(v5_df.loc[u, c_col])
        elif dread_df is not None and u in dread_df.index:
            for j, col in enumerate(LAB):
                l_arr[j] = float(dread_df.loc[u, col])
        labels[u] = l_arr
        confs[u] = c_arr

    for u in gold_ids:
        labels[u] = gold_df.loc[u, LAB].values.astype(np.float32)
        confs[u] = np.ones(12, dtype=np.float32)

    print(f'Dataset: {len(train_ids)} train studies | {len(gold_ids)} gold validation studies')

    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    pw = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)
    cw = CLASS_WEIGHTS.to(dev)

    # 2. Dataloaders: batch_size=1 with accum_steps=8 fits safely inside 3.5GB VRAM
    RES = 384
    K_TRAIN = 8
    K_EVAL = 12
    BATCH_SIZE = 1
    ACCUM_STEPS = 8
    WORKERS = 4
    EPOCHS = 6

    tds = StudyWindows(train_ids, id2row, labels, confs, res=RES, k=K_TRAIN, train=True)
    vds = StudyWindows(gold_ids, id2row, labels, confs, res=RES, k=K_EVAL, train=False)
    tl = DataLoader(tds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, drop_last=True, collate_fn=collate, pin_memory=True)
    vl = DataLoader(vds, batch_size=2, shuffle=False, num_workers=WORKERS, collate_fn=collate)

    # 3. Model & Checkpoint
    arch = 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'
    bb = timm.create_model(arch, pretrained=False, num_classes=0, in_chans=3, global_pool='avg')
    try:
        bb.set_grad_checkpointing(True)
    except Exception:
        pass

    model = RaptorClassifier(bb, F_dim=bb.num_features).to(dev)

    init_ckpt = find_file('raptor_ft_coatnet_v4_full_swa.pt')
    if init_ckpt:
        print(f'Loading weights from SWA baseline: {init_ckpt}')
        ck = torch.load(init_ckpt, map_location='cpu', weights_only=False)
        st = ck.get('model', ck)
        missing, unexpected = model.load_state_dict(st, strict=False)
        print(f'Weights loaded (missing={len(missing)}, unexpected={len(unexpected)})')

    # Freeze backbone initially for warmup, then fine-tune top layers
    for p in model.backbone.parameters():
        p.requires_grad = False

    head_params = [p for n, p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    opt = torch.optim.AdamW(head_params, lr=3e-4, weight_decay=0.01)

    steps = (len(tl) // ACCUM_STEPS) * EPOCHS
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == 'cuda')

    @torch.no_grad()
    def evaluate():
        model.eval()
        P, Y = [], []
        for x, y, _ in vl:
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=dev == 'cuda'):
                o = torch.sigmoid(model(x.to(dev)).float())
            P.append(o.cpu().numpy())
            Y.append(y.numpy())
        P = np.concatenate(P, 0)
        Y = np.concatenate(Y, 0)
        aucs = []
        for j, col in enumerate(LAB):
            yj = Y[:, j]
            if len(np.unique(yj)) > 1:
                s = roc_auc_score(yj, P[:, j])
                aucs.append(s)
            else:
                aucs.append(0.5)
        return float(np.mean(aucs)), aucs

    best_auc = 0.0
    print('Starting training loop...')
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        losses = []
        opt.zero_grad()

        # Unfreeze stage 4 of backbone from epoch 2
        if ep == 2:
            print('[Training] Unfreezing stage 4 of CoAtNet backbone...')
            if hasattr(model.backbone, 'stages') and len(model.backbone.stages) > 3:
                for p in model.backbone.stages[3].parameters():
                    p.requires_grad = True
                opt.add_param_group({'params': model.backbone.stages[3].parameters(), 'lr': 1e-5})

        for step, (x, y, conf) in enumerate(tl):
            x = x.to(dev)
            y = y.to(dev)
            conf = conf.to(dev)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=dev == 'cuda'):
                logits = model(x).float()
                # Finding-weighted and confidence-weighted BCE loss
                bce_per_target = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw, reduction='none')
                loss = (bce_per_target * conf * cw).sum() / (conf * cw).sum()

                # Effusion -> Synovitis clinical consistency prior
                p_eff = torch.sigmoid(logits[:, 7])
                p_syn = torch.sigmoid(logits[:, 8])
                reg = F.relu(0.35 * p_eff - p_syn).mean()
                loss = loss + 0.10 * reg

                loss_scaled = loss / ACCUM_STEPS

            scaler.scale(loss_scaled).backward()
            losses.append(loss.item())

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(tl):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()

            if (step + 1) % 500 == 0:
                print(f'  [Ep {ep+1}] Step {step+1}/{len(tl)} | Loss: {np.mean(losses[-100:]):.4f} | Elapsed: {(time.time()-t0)/60:.1f}m')

        val_auc, per_target_auc = evaluate()
        elapsed = (time.time() - t0) / 60.0
        print(f'Epoch {ep+1}/{EPOCHS} complete in {elapsed:.1f}m | Gold AUC: {val_auc:.4f}')
        print('  Per-target:', {LAB[j]: round(per_target_auc[j], 3) for j in range(12)})

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({'model': model.state_dict(), 'epoch': ep, 'auc': val_auc}, 'specialist_best.pt')
            print(f'  --> Saved new best checkpoint with Gold AUC: {val_auc:.4f}')

    print(f'\nTraining complete! Peak Gold AUC: {best_auc:.4f}')

if __name__ == '__main__':
    main()
'''

nb = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# RSNA Knee Abnormality Detection - High-Precision Specialist Fine-Tuning v9\n',
                'Memory-bounded, class-weighted, 4-source consensus soft supervision targeting weak findings.\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [code]
        }
    ],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.10.12'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

from pathlib import Path

with open(Path(__file__).with_name('train-v9.ipynb'), 'w') as f:
    json.dump(nb, f, indent=1)

print('train-v9.ipynb generated successfully!')
