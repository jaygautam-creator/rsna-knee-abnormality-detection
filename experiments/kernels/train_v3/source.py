# %% [markdown]
# # v3 — train in chunked subprocesses so the leak stops mattering
#
# Six runs failed to find what grows this process by ~38 MB per step. It
# survived removing the mmap, the dataloader workers, the pinned buffers, a 4x
# cut in host allocation, and `malloc_trim`. At that rate a single process gets
# ~485 steps before it has to stop — about 1.1 epochs — and every v2 result so
# far is a model that never finished learning.
#
# So stop trying to find it. A process that exits returns everything it leaked,
# including whatever this is. Training therefore runs as a sequence of
# subprocesses, each doing a bounded number of steps and handing model,
# optimizer, scheduler and scaler state to the next through a file.
#
# 11 chunks x 400 steps is 4,400 steps — roughly 10 epochs, 8.2 h at the
# measured 6.7 s/step, against a 10.5 h budget. Restart overhead is ~40 s a
# chunk, ~7 min total, which is a rounding error against what it buys.
#
# `OneCycleLR` is constructed with the *global* total and its state is restored
# each chunk, so the LR schedule spans the whole run rather than restarting.
# Run 6 established that a schedule which completes matters; this makes the
# schedule long enough to be worth completing.
#
# **On reading the output.** `report_auc` scores against the teacher's own
# labels, so it is only comparable between runs that used the same label
# version — v0's 0.8365 was measured against v1 labels and is *not* a valid
# baseline for a v3-label run. `gold_auc` is the comparable metric, but fold 0
# covers only 13 of the 58 expert-labelled studies. Neither number settles
# anything on its own; the leaderboard is the real test.

# %%
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path("/kaggle/working")
FOLD = 0                    # one fold per session; folds 1-2 are reruns of this
CHUNK_STEPS = 400           # ~15 GB of leak per chunk, against ~27 GB usable
TOTAL_STEPS = 4400          # ~10 epochs
BUDGET_S = 10.0 * 3600      # leave headroom under the 12 h kill
STATE = OUT / "state.pt"

print("python:", sys.version.split()[0])

# %% [markdown]
# ## The worker
#
# Self-contained on purpose: it re-discovers its own inputs and rebuilds the
# model from scratch each chunk. Duplicating the setup costs ~40 s per restart
# and removes any dependency on notebook state surviving across processes.

# %%
CHUNK_SRC = r'''
import argparse, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
          "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
ATTN_CONDS = ["Medial Meniscus", "Lateral Meniscus", "MCL", "Fracture", "Baker's"]
BACKBONE = "convnext_nano"
BATCH = 8
LR = 3e-4
N_SPLITS = 5
SEED = 42
TEMP_DIV = 3.0

ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--steps", type=int, default=0)
ap.add_argument("--total-steps", type=int, required=True)
ap.add_argument("--state", type=str, required=True)
ap.add_argument("--mode", choices=["train", "validate"], default="train")
ap.add_argument("--out", type=str, default="/kaggle/working")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"


def rss_g():
    return int(Path("/proc/self/statm").read_text().split()[1]) * 4096 / 2**30


def find_dir(*markers, max_depth=4):
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
        for c in children:
            if c.name in ("train_series", "test_series"):
                continue
            frontier.append((c, depth + 1))
    raise FileNotFoundError(str(markers))


COMP = find_dir("train.csv", "train_series.csv")
CACHE = find_dir("cache_meta.json", "volumes.npy", "index.csv")
LAB = find_dir("train_margins_v3.csv")
cache_meta = json.loads((CACHE / "cache_meta.json").read_text())
D_SLICES, SIZE = cache_meta["d_slices"], cache_meta["size"]

_vm = np.load(CACHE / "volumes.npy", mmap_mode="r")
VOL_DTYPE, VOL_OFFSET = _vm.dtype, _vm.offset
ITEM_SHAPE = _vm.shape[1:]
ITEM_BYTES = int(np.prod(ITEM_SHAPE)) * VOL_DTYPE.itemsize
del _vm
VOL_FD = os.open(str(CACHE / "volumes.npy"), os.O_RDONLY)

train = pd.read_csv(COMP / "train.csv")
index = pd.read_csv(CACHE / "index.csv")
assert (index.StudyInstanceUID.values == train.StudyInstanceUID.values).all()
margins = pd.read_csv(LAB / "train_margins_v3.csv")
m = train[["StudyInstanceUID"]].merge(margins, on="StudyInstanceUID", how="left")
assert m[LABELS].notna().all().all(), "missing margins"
M = m[LABELS].to_numpy(np.float32)
Y = 1.0 / (1.0 + np.exp(-M / (M.std(0) / TEMP_DIV)))

import hashlib, re
from sklearn.model_selection import StratifiedGroupKFold
_WS = re.compile(r"\s+")
norm = train.Report.fillna("").str.lower().str.replace(_WS, " ", regex=True).str.strip()
groups = norm.map(lambda s: hashlib.md5(s.encode()).hexdigest())
strata = np.digitize((Y > 0.5).sum(1), [1, 3, 5, 7])
fold = np.full(len(train), -1)
for k, (_, va) in enumerate(StratifiedGroupKFold(
        N_SPLITS, shuffle=True, random_state=SEED).split(train, strata, groups)):
    fold[va] = k
assert (fold >= 0).all()


def read_vol(r):
    buf = os.pread(VOL_FD, ITEM_BYTES, VOL_OFFSET + r * ITEM_BYTES)
    return np.frombuffer(buf, dtype=VOL_DTYPE).reshape(ITEM_SHAPE)


class KneeDS(Dataset):
    def __init__(self, idx, train_mode):
        self.idx, self.train = idx, train_mode

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r = self.idx[i]
        x = read_vol(r)
        if self.train:
            if np.random.rand() < 0.5:
                x = x[:, :, :, ::-1]
            x = np.roll(x, np.random.randint(-8, 9), axis=3)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(Y[r])


class Net(nn.Module):
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
        order = self.attn_idx + self.mean_idx
        inv = [0] * len(LABELS)
        for pos, lab in enumerate(order):
            inv[lab] = pos
        self.register_buffer("inv_perm", torch.tensor(inv), persistent=False)

    def forward(self, x):
        B, P, D, H, W = x.shape
        f = self.backbone(x.reshape(B * P * D, 1, H, W)).reshape(B, P, D, -1)
        g_mean = f.mean(2).reshape(B, -1)
        w = torch.softmax(self.att(f), dim=2)
        g_attn = (f * w).sum(2).reshape(B, -1)
        return torch.cat([self.head_attn(g_attn), self.head_mean(g_mean)],
                         dim=1)[:, self.inv_perm]


tr_idx = np.where(fold != a.fold)[0]
va_idx = np.where(fold == a.fold)[0]

model = Net().to(dev)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=a.total_steps)
scaler = torch.amp.GradScaler()

gstep = 0
sp = Path(a.state)
if sp.exists():
    st = torch.load(sp, map_location="cpu")
    (model.module if hasattr(model, "module") else model).load_state_dict(st["model"])
    opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    scaler.load_state_dict(st["scaler"])
    gstep = st["gstep"]
    print("resumed at gstep", gstep, flush=True)

if a.mode == "validate":
    dl = DataLoader(KneeDS(va_idx, False), batch_size=BATCH, shuffle=False, num_workers=0)
    model.eval()
    P = []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for xb, _ in dl:
            xb = xb.to(dev, non_blocking=True).float().div_(255.0)
            P.append(torch.sigmoid(model(xb)).float().cpu())
    P = torch.cat(P).numpy()
    np.save(Path(a.out) / ("val_fold%d.npy" % a.fold), P)
    np.save(Path(a.out) / ("val_idx_fold%d.npy" % a.fold), va_idx)
    print("validated", P.shape, "rss %.1fG" % rss_g(), flush=True)
    raise SystemExit(0)

# Reseed per chunk so restarts do not replay the same shuffle and augmentations.
np.random.seed(SEED + gstep)
torch.manual_seed(SEED + gstep)
dl = DataLoader(KneeDS(tr_idx, True), batch_size=BATCH, shuffle=True,
                num_workers=0, drop_last=True)

target = min(gstep + a.steps, a.total_steps)
model.train()
t0, tot, nb = time.time(), 0.0, 0
while gstep < target:
    for xb, yb in dl:
        if gstep >= target:
            break
        xb = xb.to(dev, non_blocking=True).float().div_(255.0)
        yb = yb.to(dev, non_blocking=True)
        with torch.amp.autocast("cuda"):
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        gstep += 1
        tot += loss.item()
        nb += 1
        if gstep % 100 == 0:
            print("  gstep %d/%d loss %.4f lr %.2e rss %.1fG"
                  % (gstep, a.total_steps, tot / max(nb, 1),
                     sched.get_last_lr()[0], rss_g()), flush=True)

torch.save({"model": (model.module if hasattr(model, "module") else model).state_dict(),
            "opt": opt.state_dict(), "sched": sched.state_dict(),
            "scaler": scaler.state_dict(), "gstep": gstep}, sp)
print("chunk done: gstep %d loss %.4f  %.1f min  rss %.1fG"
      % (gstep, tot / max(nb, 1), (time.time() - t0) / 60, rss_g()), flush=True)
'''

SCRIPT = OUT / "train_chunk.py"
SCRIPT.write_text(CHUNK_SRC)
print("wrote", SCRIPT, len(CHUNK_SRC), "bytes")

# %% [markdown]
# ## Chunk loop
#
# Each chunk is a fresh process. A non-zero exit stops the loop rather than
# looping on a broken state file — the previous chunk's state is still on disk,
# so the run is resumable rather than lost.

# %%
t_start = time.time()
chunk = 0
gstep = 0
while gstep < TOTAL_STEPS:
    spent = time.time() - t_start
    if spent > BUDGET_S:
        print(f"\nstopping: {spent/3600:.2f} h spent of {BUDGET_S/3600:.1f} h")
        break
    chunk += 1
    print(f"\n=== chunk {chunk}  gstep {gstep}/{TOTAL_STEPS}  "
          f"elapsed {spent/3600:.2f} h", flush=True)
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--fold", str(FOLD),
         "--steps", str(CHUNK_STEPS), "--total-steps", str(TOTAL_STEPS),
         "--state", str(STATE)],
        capture_output=True, text=True)
    print(r.stdout[-4000:], flush=True)
    if r.returncode != 0:
        print("chunk failed, stderr tail:\n", r.stderr[-4000:], flush=True)
        break
    # Read the step count back from the state file rather than assuming the
    # chunk did exactly CHUNK_STEPS.
    prev, gstep = gstep, torch.load(STATE, map_location="cpu")["gstep"]
    if gstep <= prev:
        print("chunk made no progress — stopping rather than looping", flush=True)
        break

print(f"\ntrained {gstep} steps in {(time.time()-t_start)/3600:.2f} h")

# %% [markdown]
# ## Validate in its own process too

# %%
r = subprocess.run(
    [sys.executable, str(SCRIPT), "--fold", str(FOLD), "--total-steps", str(TOTAL_STEPS),
     "--state", str(STATE), "--mode", "validate"],
    capture_output=True, text=True)
print(r.stdout[-2000:])
if r.returncode != 0:
    print("validation failed:\n", r.stderr[-3000:])

# %%
from sklearn.metrics import roc_auc_score

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
          "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]


def find_dir(*markers, max_depth=4):
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
        for c in children:
            if c.name in ("train_series", "test_series"):
                continue
            frontier.append((c, depth + 1))
    raise FileNotFoundError(str(markers))


COMP = find_dir("train.csv", "train_series.csv")
LAB = find_dir("train_margins_v3.csv")
train = pd.read_csv(COMP / "train.csv")
margins = pd.read_csv(LAB / "train_margins_v3.csv")
m = train[["StudyInstanceUID"]].merge(margins, on="StudyInstanceUID", how="left")
M = m[LABELS].to_numpy(np.float32)
Y = 1.0 / (1.0 + np.exp(-M / (M.std(0) / 3.0)))

P = np.load(OUT / f"val_fold{FOLD}.npy")
va_idx = np.load(OUT / f"val_idx_fold{FOLD}.npy")
hard = (Y[va_idx] > 0.5).astype(int)

gold_mask = train[LABELS].notna().all(1).to_numpy()
gm = np.isin(va_idx, np.where(gold_mask)[0])
g_true = train.loc[va_idx[gm], LABELS].to_numpy()

rows = []
for i, c in enumerate(LABELS):
    r_auc = (roc_auc_score(hard[:, i], P[:, i])
             if 0 < hard[:, i].sum() < len(hard) else np.nan)
    g_auc = (roc_auc_score(g_true[:, i], P[gm][:, i])
             if 0 < g_true[:, i].sum() < len(g_true) else np.nan)
    rows.append({"condition": c, "n_pos_report": int(hard[:, i].sum()),
                 "report_auc": round(r_auc, 4), "gold_auc": round(g_auc, 4)})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
print(f"\nreport_auc {df.report_auc.mean():.4f}   gold_auc {df.gold_auc.mean():.4f} "
      f"(on {gm.sum()} gold studies)")
print("report_auc is scored against v3 teacher labels — NOT comparable to v0's "
      "0.8365, which was scored against v1 labels.")

json.dump({"fold": FOLD, "gstep": int(gstep), "total_steps": TOTAL_STEPS,
           "chunks": chunk, "report_auc": float(df.report_auc.mean()),
           "gold_auc": float(df.gold_auc.mean()), "n_gold": int(gm.sum()),
           "hours": round((time.time() - t_start) / 3600, 2),
           "per_condition": rows},
          open(OUT / "v3_meta.json", "w"), indent=2)

if STATE.exists():
    torch.save(torch.load(STATE, map_location="cpu")["model"], OUT / f"v3_fold{FOLD}.pt")
    STATE.unlink()   # optimizer state is not worth keeping as kernel output
print(f"saved v3_fold{FOLD}.pt + v3_meta.json")
