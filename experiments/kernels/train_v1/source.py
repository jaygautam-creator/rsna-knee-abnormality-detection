# %% [markdown]
# # Image model v0 — 2.5D backbone distilled from report soft labels
#
# First model that can actually produce a leaderboard number. Deliberately lean:
# the Efficiency Prize maths in PLAN.md says runtime beats accuracy by an order
# of magnitude, so v0 is a single small backbone, no TTA, no ensemble.
#
# **Targets.** Report margins, not the banked probabilities. The labeller's
# margins span ±25 logits, so `sigmoid(margin)` is 2e-9 at one end and 1.0 at the
# other — the "soft" labels are hard labels with extra steps. Dividing by a
# per-condition temperature restores the gradient information that motivated
# using a text model in the first place.
#
# **Validation.** Grouped by *report*, not study: 4,407 studies share only 4,257
# reports and one report covers 37 studies, so a random split would let the model
# score on rows it memorised. We report two numbers — AUC against held-out report
# labels (n≈880, noisy targets) and AUC against the expert 58 (clean targets,
# tiny n). Neither alone is trustworthy; agreement between them is the signal.

# %%
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

BACKBONE = "resnet18"
# Per-condition pooling. The v0->attn ablation moved focal and diffuse findings
# in opposite directions and they cancelled in the mean, so one pooling for all
# twelve was the wrong question. Groups are taken straight from that ablation's
# per-condition deltas rather than from anatomical intuition.
ATTN_CONDS = ["Medial Meniscus", "Lateral Meniscus", "MCL", "Fracture", "Baker's"]
POOL = "percond"
EPOCHS = 16            # v0 was still climbing at ep 8
BATCH = 8
LR = 3e-4
VAL_FOLD = 0
N_SPLITS = 5
SEED = 42
TEMP_DIV = 3.0        # T_c = std(margin_c) / TEMP_DIV

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "| gpus:", torch.cuda.device_count())


def find_dir(*markers: str, max_depth: int = 4) -> Path:
    """Find the mount containing every `markers` file.

    Mount paths are not guessable — the competition lands at
    /kaggle/input/competitions/<slug> and attached kernel output does not land at
    /kaggle/input/<slug>. Search for the files we need rather than assume.
    """
    frontier = [(Path("/kaggle/input"), 0)]
    while frontier:
        d, depth = frontier.pop(0)
        if all((d / m).exists() for m in markers):
            return d
        if depth >= max_depth:
            continue
        try:
            children = sorted(p for p in d.iterdir() if p.is_dir())
        except PermissionError:
            continue
        for child in children:
            if child.name in ("train_series", "test_series"):
                continue          # huge DICOM trees, never the anchor
            frontier.append((child, depth + 1))

    print(f"--- could not find {markers}; /kaggle/input tree (depth 3) ---")
    for p in sorted(Path("/kaggle/input").rglob("*")):
        rel = p.relative_to("/kaggle/input")
        if len(rel.parts) <= 3:
            print(" ", rel, "/" if p.is_dir() else "")
    raise FileNotFoundError(f"no mount containing {markers}")


COMP = find_dir("train.csv", "train_series.csv")
CACHE = find_dir("cache_meta.json", "volumes.npy", "index.csv")
LAB = find_dir("train_margins_v2.csv")
print("comp: ", COMP)
print("cache:", CACHE)
print("labels:", LAB)
meta = json.loads((CACHE / "cache_meta.json").read_text())
print(json.dumps(meta, indent=2))

vols = np.load(CACHE / "volumes.npy", mmap_mode="r")   # (N,3,D,H,W) uint8
index = pd.read_csv(CACHE / "index.csv")
train = pd.read_csv(COMP / "train.csv")
margins = pd.read_csv(LAB / "train_margins_v2.csv")
# Expert ground truth lives in train.csv itself — 58 of 4,407 rows carry all
# 12 labels as binary 0/1. artifacts/val_labelled_58.csv is NOT this: it holds
# the *labeller's predictions* on those studies, and feeding it here silently
# compares the image model against another model instead of against truth.
gold = train.loc[train[LABELS].notna().all(1), ["StudyInstanceUID"] + LABELS].copy()
assert len(gold) == 58, f"expected 58 expert-labelled studies, got {len(gold)}"
assert set(np.unique(gold[LABELS].to_numpy())) <= {0.0, 1.0}, "gold must be binary"
print("volumes:", vols.shape, "| index:", len(index))

# %% [markdown]
# ## Align, build soft targets, split by report

# %%
# The cache was written in train.csv row order, but align on UID explicitly —
# a silent row shift would pair every volume with the wrong label and still train.
assert (index.StudyInstanceUID.values == train.StudyInstanceUID.values).all(), \
    "cache index does not match train.csv order"

m = train[["StudyInstanceUID"]].merge(margins, on="StudyInstanceUID", how="left")
assert m[LABELS].notna().all().all(), "missing margins for some studies"
M = m[LABELS].to_numpy(np.float32)

T = M.std(0) / TEMP_DIV
Y = 1.0 / (1.0 + np.exp(-M / T))          # soft targets
print("\nper-condition temperature and resulting target spread:")
for i, c in enumerate(LABELS):
    q = np.percentile(Y[:, i], [10, 50, 90])
    print(f"  {c:18} T={T[i]:6.2f}  p10/50/90 = {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}")

# %%
import hashlib
import re
from sklearn.model_selection import StratifiedGroupKFold

_WS = re.compile(r"\s+")
norm = train.Report.fillna("").str.lower().str.replace(_WS, " ", regex=True).str.strip()
groups = norm.map(lambda s: hashlib.md5(s.encode()).hexdigest())
burden = (Y > 0.5).sum(1)
strata = np.digitize(burden, [1, 3, 5, 7])

fold = np.full(len(train), -1)
for k, (_, va) in enumerate(
        StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED)
        .split(train, strata, groups)):
    fold[va] = k
assert (fold >= 0).all()

tr_idx = np.where(fold != VAL_FOLD)[0]
va_idx = np.where(fold == VAL_FOLD)[0]
print(f"\ntrain {len(tr_idx)}  val {len(va_idx)}  "
      f"unique reports {groups.nunique()}")

gold_pos = train.StudyInstanceUID.isin(gold.StudyInstanceUID).to_numpy().nonzero()[0]
gold_va = np.intersect1d(gold_pos, va_idx)
print(f"gold studies: {len(gold_pos)} total, {len(gold_va)} in the held-out fold")
if len(gold_va) < 20:
    print(f"  WARNING: gold_auc rests on {len(gold_va)} studies — indicative only.\n"
          f"  The trustworthy number needs a 5-fold OOF run over all 58.")

# %% [markdown]
# ## Data
#
# Volumes are mmapped, so `__getitem__` touches only the 2.4 MB it needs and the
# 10.6 GB array never enters RAM.

# %%
class KneeDS(Dataset):
    def __init__(self, idx, targets, train_mode: bool):
        self.idx, self.y, self.train = idx, targets, train_mode

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r = self.idx[i]
        x = np.asarray(vols[r], dtype=np.float32) / 255.0   # (3,D,H,W)
        if self.train:
            if np.random.rand() < 0.5:
                x = x[:, :, :, ::-1]                        # L/R flip
            k = np.random.randint(-8, 9)
            x = np.roll(x, k, axis=3)                       # small translate
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(self.y[r])


dl_tr = DataLoader(KneeDS(tr_idx, Y, True), batch_size=BATCH, shuffle=True,
                   num_workers=2, pin_memory=True, drop_last=True)
dl_va = DataLoader(KneeDS(va_idx, Y, False), batch_size=BATCH, shuffle=False,
                   num_workers=2, pin_memory=True)

# %% [markdown]
# ## Model
#
# One 2D backbone shared across all 3 planes x D slices; mean-pool over slices,
# concatenate the three planes, one linear head. Sharing the backbone keeps the
# parameter count at single-model size while still seeing every plane — the
# cheapest thing that can use all the data.

# %%
class Net(nn.Module):
    """Shared 2D backbone over 3 planes x D slices, pooled across slices.

    POOL is the thing under test. `mean` (v0) averages a focal finding with
    every normal slice in the series, which is the obvious suspect for the weak
    focal conditions — Lateral Meniscus 0.778, MCL 0.786. `attn` lets the model
    weight the slices where the finding actually is; `max` is the crude version
    of the same idea and a useful control — if `max` matches `attn`, the gain is
    from not-averaging rather than from anything learned.
    """

    def __init__(self, name=BACKBONE, n_planes=3):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=True, in_chans=1, num_classes=0)
        f = self.backbone.num_features
        self.att = nn.Sequential(nn.Linear(f, 128), nn.Tanh(), nn.Linear(128, 1))
        self.attn_idx = [LABELS.index(c) for c in ATTN_CONDS]
        self.mean_idx = [i for i in range(len(LABELS)) if i not in self.attn_idx]
        self.head_attn = nn.Sequential(nn.Dropout(0.3),
                                       nn.Linear(f * n_planes, len(self.attn_idx)))
        self.head_mean = nn.Sequential(nn.Dropout(0.3),
                                       nn.Linear(f * n_planes, len(self.mean_idx)))
        # Reassemble the two heads by permutation rather than index assignment.
        # `out[:, idx] = head(...)` raises under AMP when the destination is
        # float32 and the head returns half, which killed v2 run 1. cat+gather
        # has no destination tensor, so no dtype to mismatch.
        order = self.attn_idx + self.mean_idx
        inv = [0] * len(LABELS)
        for pos, lab in enumerate(order):
            inv[lab] = pos
        self.register_buffer("inv_perm", torch.tensor(inv), persistent=False)

    def forward(self, x):                      # (B,P,D,H,W)
        B, P, D, H, W = x.shape
        f = self.backbone(x.reshape(B * P * D, 1, H, W)).reshape(B, P, D, -1)
        g_mean = f.mean(2).reshape(B, -1)
        w = torch.softmax(self.att(f), dim=2)
        g_attn = (f * w).sum(2).reshape(B, -1)

        return torch.cat([self.head_attn(g_attn),
                          self.head_mean(g_mean)], dim=1)[:, self.inv_perm]


model = Net().to(dev)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * len(dl_tr))
scaler = torch.amp.GradScaler()


def evaluate():
    model.eval()
    P = []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for xb, _ in dl_va:
            P.append(torch.sigmoid(model(xb.to(dev, non_blocking=True))).float().cpu())
    P = torch.cat(P).numpy()

    # vs held-out report labels (noisy targets, n~880)
    hard = (Y[va_idx] > 0.5).astype(int)
    aucs = [roc_auc_score(hard[:, i], P[:, i])
            for i in range(len(LABELS)) if hard[:, i].sum() not in (0, len(hard))]
    report_auc = float(np.mean(aucs))

    # vs the expert 58, restricted to the held-out fold (clean targets, tiny n).
    # v0 silently reported nan here because only 9 gold studies landed in the
    # fold and the guard was >=10 — the one clean number we have, lost to a
    # threshold. Compute it whenever it is computable and label its reliability.
    gold_auc = float("nan")
    if len(gold_va) >= 5:
        pos = {r: j for j, r in enumerate(va_idx)}
        rows = [pos[r] for r in gold_va]
        g = (gold.set_index("StudyInstanceUID")
                 .loc[train.StudyInstanceUID.values[gold_va], LABELS].to_numpy())
        ga = [roc_auc_score(g[:, i], P[rows, i])
              for i in range(len(LABELS)) if 0 < g[:, i].sum() < len(g)]
        if ga:
            gold_auc = float(np.mean(ga))
    return report_auc, gold_auc, P


# %%
best, t0 = -1, time.time()
for ep in range(EPOCHS):
    model.train()
    tot = 0.0
    for xb, yb in dl_tr:
        xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
        with torch.amp.autocast("cuda"):
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        tot += loss.item()

    r_auc, g_auc, _ = evaluate()
    print(f"ep {ep+1}/{EPOCHS}  loss {tot/len(dl_tr):.4f}  "
          f"report_auc {r_auc:.4f}  gold_auc {g_auc:.4f}  "
          f"[{(time.time()-t0)/60:.1f} min]", flush=True)

    if r_auc > best:
        best = r_auc
        sd = (model.module if hasattr(model, "module") else model).state_dict()
        torch.save(sd, f"/kaggle/working/v0_{POOL}_fold{VAL_FOLD}.pt")

print(f"\nbest report_auc {best:.4f} in {(time.time()-t0)/60:.1f} min")

# %%
r_auc, g_auc, P = evaluate()
per = []
hard = (Y[va_idx] > 0.5).astype(int)
for i, c in enumerate(LABELS):
    a = (roc_auc_score(hard[:, i], P[:, i])
         if hard[:, i].sum() not in (0, len(hard)) else float("nan"))
    per.append({"condition": c, "n_pos": int(hard[:, i].sum()), "auc": round(a, 4)})
print(pd.DataFrame(per).to_string(index=False))

json.dump({"backbone": BACKBONE, "pool": POOL, "epochs": EPOCHS,
           "val_fold": VAL_FOLD, "n_gold_val": int(len(gold_va)),
           "report_auc": r_auc, "gold_auc": g_auc, "per_condition": per,
           "minutes": round((time.time() - t0) / 60, 1)},
          open(f"/kaggle/working/v0_{POOL}_meta.json", "w"), indent=2)
print(f"\nsaved v0_{POOL}_fold{VAL_FOLD}.pt + v0_{POOL}_meta.json")
