# %% [markdown]
# # v6 — 336 px, three shards, frozen folds
#
# Two changes from v5, both mechanical, both prerequisites for any offline
# number meaning anything again.
#
# ## 1. 224 -> 336 px
#
# The cache measured native in-plane resolution while building: median shortest
# native edge is 512 px, and **82% of studies are natively >=336**, 99.5% >=224.
# Downsampling to 224 was discarding real detail on four studies in five. The
# targets that need it are the ones we are worst at — meniscal tears and
# cartilage/OA grading are fine-structure calls.
#
# The cost is that 336 px is 2.25x the pixels and this backbone is pixel-bound,
# so a 10 h session buys ~2,400 steps instead of 4,400. See `TOTAL_STEPS`.
#
# ## 2. The folds are read, not recomputed
#
# v5 rebuilt its split inline with `StratifiedGroupKFold`. That stratifies on
# label-derived abnormality burden and assigns groups greedily, so when the
# labels moved v3 -> v4 the burden changed on 6.6% of studies and **74.6% of
# folds were reassigned**. Two runs on "fold 0" were scoring on different rows,
# leaving 357 shared held-out studies where there should have been ~880. Every
# offline comparison made across that boundary was noise.
#
# `data/folds.csv` is the frozen split, now shipped inside `knee-labels-v4`.
# This run reads it and asserts it covers `train.csv`. Nothing recomputes a
# split again.
#
# ## Attaching the cache
#
# The 336 cache is 35.8 GB, over the 20 GB single-kernel-output ceiling, so it
# ships as three contiguous row-range shards. **All three must be attached.**
# The loader sorts them by `row_lo` and asserts the ranges tile
# `[0, 4407)` exactly — a forgotten shard would otherwise train happily on
# two-thirds of the data and report a plausible AUC.
#
# ## Reading the output
#
# `report_auc` scores against the v4 teacher labels and is comparable only
# across runs sharing a label version *and* a fold split — v6 is the first run
# on the frozen split, so it is comparable to future runs, not past ones.
# `gold_auc` lands on ~13 gold studies per fold. Neither settles anything; the
# leaderboard does.

# %%
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

OUT = Path("/kaggle/working")
FOLD = 2                    # one fold per session; folds 1-4 are reruns of this
# 336 px is 2.25x the pixels of 224, and this backbone is pixel-bound: it runs
# BATCH*3*24 = 576 single-channel images per step. v5 measured 6.7 s/step at
# 224, so budget ~15 s/step here. 10 h of compute is therefore ~2,400 steps
# (~5.5 epochs), not v5's 4,400 (~10). OneCycleLR is built from TOTAL_STEPS and
# run 6 established that a schedule which *completes* beats a longer one that
# gets truncated, so the step count is cut to fit the budget rather than left
# at 4,400 to be killed two-thirds of the way through the anneal.
CHUNK_STEPS = 140           # 336 volumes leak faster per step than 224 did
# Run 1 of this kernel died of a host-RAM OOM after ~0.9 h having reached at
# least step 100, so the true rate is 18-24 s/step, not the 15 s the 2400
# figure assumed. 2400 could not have finished the anneal inside 10 h even if
# it had not been killed. 1800 completes at 20 s/step with the budget to spare,
# and a schedule that completes beats a longer one that is truncated.
TOTAL_STEPS = 550          # ~4 epochs; see the s/step print in chunk 1
BUDGET_S = 8.2 * 3600      # leave headroom under the 12 h kill
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
# Effective batch stays 8 so the LR and schedule carry over from v5 unchanged.
# It is reached by accumulation, because a step feeds BATCH*3*24 single-channel
# images and at 336 those activations are 2.25x what BATCH=8 was tuned for at
# 224 — on a 15 GiB T4 that is an OOM ten minutes into a ten-hour session.
# MICRO=2 is deliberately conservative: the cost of being wrong upward is the
# whole session, and the cost of being wrong downward is a few percent of
# throughput on kernel-launch overhead.
BATCH = 8
MICRO = 4
assert BATCH % MICRO == 0
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
if dev == "cuda":
    torch.backends.cudnn.benchmark = True


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


def find_dirs(*markers, max_depth=4):
    """Every directory carrying all markers. The 336 cache is 35.8 GB and ships
    as three row-range shards, so there is no longer a single cache dir."""
    out, frontier = [], [(Path("/kaggle/input"), 0)]
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
            if c.name in ("train_series", "test_series"):
                continue
            frontier.append((c, depth + 1))
    if not out:
        raise FileNotFoundError(str(markers))
    return out


COMP = find_dir("train.csv", "train_series.csv")
LAB = find_dir("train_margins_v4.csv", "folds.csv")

# --- the 336 cache, reassembled from its shards -----------------------------
# Each shard holds a contiguous global row range [row_lo, row_hi) in train.csv
# order. Sort by row_lo, then assert the ranges tile [0, n_studies_all) exactly:
# a missing or double-attached shard is otherwise a silent 1/3-of-the-data bug
# that still trains and still reports a plausible AUC.
SHARDS = sorted(find_dirs("cache_meta.json", "volumes.npy", "index.csv"),
                key=lambda d: json.loads((d / "cache_meta.json").read_text())["row_lo"])
metas = [json.loads((d / "cache_meta.json").read_text()) for d in SHARDS]
cache_meta = metas[0]
D_SLICES, SIZE = cache_meta["d_slices"], cache_meta["size"]
N_ALL = cache_meta["n_studies_all"]

assert SIZE == 336, f"expected the 336 cache, got size={SIZE}"
assert len(SHARDS) == cache_meta["n_shards"], \
    f"attached {len(SHARDS)} shard(s), cache declares {cache_meta['n_shards']}"
edge = 0
for m in metas:
    assert m["size"] == SIZE and m["d_slices"] == D_SLICES, "shards disagree on geometry"
    assert m["n_studies_all"] == N_ALL, "shards disagree on the study count"
    assert m.get("laterality_canonical"), "shard is not laterality-canonical"
    assert m["row_lo"] == edge, f"row gap/overlap at {edge}: shard starts {m['row_lo']}"
    edge = m["row_hi"]
assert edge == N_ALL, f"shards cover {edge} of {N_ALL} studies"

_vm = np.load(SHARDS[0] / "volumes.npy", mmap_mode="r")
VOL_DTYPE, VOL_OFFSET = _vm.dtype, _vm.offset
ITEM_SHAPE = _vm.shape[1:]
ITEM_BYTES = int(np.prod(ITEM_SHAPE)) * VOL_DTYPE.itemsize
del _vm
# Global row -> (fd, local row). Opened once; os.pread is thread- and
# fork-safe on a shared fd because it carries its own offset.
SHARD_FD = [os.open(str(d / "volumes.npy"), os.O_RDONLY) for d in SHARDS]
SHARD_LO = np.array([m["row_lo"] for m in metas])

train = pd.read_csv(COMP / "train.csv")
index = pd.concat([pd.read_csv(d / "index.csv") for d in SHARDS], ignore_index=True)
index = index.sort_values("row").reset_index(drop=True)
assert (index.row.values == np.arange(N_ALL)).all(), "global row ids are not 0..N-1"
assert (index.StudyInstanceUID.values == train.StudyInstanceUID.values).all(), \
    "cache row order does not match train.csv"
margins = pd.read_csv(LAB / "train_margins_v4.csv")
m = train[["StudyInstanceUID"]].merge(margins, on="StudyInstanceUID", how="left")
assert m[LABELS].notna().all().all(), "missing margins"
M = m[LABELS].to_numpy(np.float32)
Y = 1.0 / (1.0 + np.exp(-M / (M.std(0) / TEMP_DIV)))

# The split is read, never recomputed. StratifiedGroupKFold stratifies on
# label-derived burden and assigns groups greedily, so moving v3 -> v4 labels
# reassigned 74.6% of folds and left only 357 shared held-out rows to compare
# two runs on. data/folds.csv freezes it; every run from here reads that file.
folds = pd.read_csv(LAB / "folds.csv")
assert set(folds.StudyInstanceUID) == set(train.StudyInstanceUID), \
    "frozen folds do not cover train.csv"
fold = train[["StudyInstanceUID"]].merge(folds, on="StudyInstanceUID",
                                         how="left").fold.to_numpy()
assert (fold >= 0).all() and len(np.unique(fold)) == N_SPLITS


def read_vol(r):
    """`r` is a global row; pick its shard and offset within that shard's file."""
    s = int(np.searchsorted(SHARD_LO, r, side="right")) - 1
    off = VOL_OFFSET + (r - SHARD_LO[s]) * ITEM_BYTES
    buf = os.pread(SHARD_FD[s], ITEM_BYTES, off)
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
            # No horizontal flip. Axis 3 is W, the medial/lateral axis on the
            # axial and coronal planes, and mirroring it swaps the medial and
            # lateral targets while every loss curve still looks healthy.
            # A roll shifts the anatomy without reflecting it, so it is safe.
            x = np.roll(x, np.random.randint(-8, 9), axis=3)
            x = np.roll(x, np.random.randint(-8, 9), axis=2)
            # Per-volume gain/bias in place of the flip's regularisation. MRI has
            # no fixed intensity scale, so this is a plausible nuisance transform
            # rather than a geometric lie.
            g = np.float32(np.random.uniform(0.9, 1.1))
            b = np.float32(np.random.uniform(-12, 12))
            x = np.clip(x.astype(np.float32) * g + b, 0, 255).astype(np.uint8)
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
        with torch.amp.autocast("cuda", dtype=torch.float16):
            B, P, D, H, W = x.shape
            x_4d = x.reshape(B * P * D, 1, H, W).contiguous(memory_format=torch.channels_last)
            f = self.backbone(x_4d).reshape(B, P, D, -1)
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
scaler = torch.amp.GradScaler('cuda')

# The container has ~29 GB. Chunk 1 of run 1 was at 22.8 GB by step 100 and was
# SIGKILLed before step 250 — no traceback, just a dead process. A ceiling well
# under the kill turns that into a clean save-and-restart: the next chunk is a
# fresh process with a fresh heap, so the loop makes progress either way.
RSS_LIMIT_G = 18.0

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
    dl = DataLoader(KneeDS(va_idx, False), batch_size=MICRO, shuffle=False, num_workers=0)
    model.eval()
    P = []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for xb, _ in dl:
            xb = xb.to(dev, non_blocking=True).float().div_(255.0)
            P.append(torch.sigmoid(model(xb)).float().cpu())
            if len(P) % 50 == 0 and rss_g() > RSS_LIMIT_G:
                raise MemoryError("validate rss %.1fG over ceiling" % rss_g())
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
        opt.zero_grad(set_to_none=True)
        # Accumulate over micro-batches. Each micro-loss is divided by the
        # number of micro-batches so the gradient matches a single BATCH pass
        # rather than being n_micro times too large.
        n_micro = (xb.shape[0] + MICRO - 1) // MICRO
        loss_v = 0.0
        for mb in range(n_micro):
            xs = xb[mb * MICRO:(mb + 1) * MICRO]
            ys = yb[mb * MICRO:(mb + 1) * MICRO]
            if xs.shape[0] == 0:
                continue
            with torch.amp.autocast("cuda"):
                loss = F.binary_cross_entropy_with_logits(model(xs), ys) / n_micro
            scaler.scale(loss).backward()
            loss_v += loss.item()
        scaler.step(opt)
        scaler.update()
        sched.step()
        gstep += 1
        tot += loss_v
        nb += 1
        if gstep % 25 == 0:
            print("  gstep %d/%d loss %.4f lr %.2e rss %.1fG"
                  % (gstep, a.total_steps, tot / max(nb, 1),
                     sched.get_last_lr()[0], rss_g()), flush=True)
        if gstep % 10 == 0 and rss_g() > RSS_LIMIT_G:
            print("  rss %.1fG over the %.1fG ceiling — saving and restarting"
                  % (rss_g(), RSS_LIMIT_G), flush=True)
            break
    else:
        continue
    break

torch.save({"model": (model.module if hasattr(model, "module") else model).state_dict(),
            "opt": opt.state_dict(), "sched": sched.state_dict(),
            "scaler": scaler.state_dict(), "gstep": gstep}, sp)
el = time.time() - t0
print("chunk done: gstep %d loss %.4f  %.1f min  %.2f s/step  rss %.1fG"
      % (gstep, tot / max(nb, 1), el / 60, el / max(nb, 1), rss_g()), flush=True)
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
# glibc raises its own mmap threshold to the size of the first large block it
# sees freed, after which every 8.1 MB volume comes off the brk heap and
# fragments it instead of being returned to the OS. Pinning the threshold keeps
# the volumes and their float32 augmentation temporaries on mmap, where a free
# is a munmap. This is the suspected cause of run 1's growth; the RSS ceiling in
# the chunk is the guard that does not depend on the diagnosis being right.
CHUNK_ENV = dict(os.environ,
                 MALLOC_MMAP_THRESHOLD_="131072",
                 MALLOC_TRIM_THRESHOLD_="131072")

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
        capture_output=True, text=True, env=CHUNK_ENV)
    print(r.stdout[-4000:], flush=True)
    if r.returncode != 0:
        # A negative code is a signal, not an exception: -9 is the OOM killer,
        # and stderr will hold nothing but warnings. Print it either way.
        print("chunk failed rc=%d, stderr tail:\n" % r.returncode,
              r.stderr[-4000:], flush=True)
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
# Run 1 trained zero steps, then validated and saved v7_fold0.pt anyway — a
# checkpoint of an untrained backbone reporting AUC 0.50, which is exactly the
# kind of artefact that gets picked up later and blended. No steps, no output.
assert gstep > 0, "no steps trained — refusing to validate or save a checkpoint"

r = subprocess.run(
    [sys.executable, str(SCRIPT), "--fold", str(FOLD), "--total-steps", str(TOTAL_STEPS),
     "--state", str(STATE), "--mode", "validate"],
    capture_output=True, text=True, env=CHUNK_ENV)
print(r.stdout[-2000:])
if r.returncode != 0:
    print("validation failed rc=%d:\n" % r.returncode, r.stderr[-3000:])

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
LAB = find_dir("train_margins_v4.csv", "folds.csv")
SIZE_REPORT = json.loads(
    (find_dir("cache_meta.json", "volumes.npy") / "cache_meta.json").read_text())["size"]
train = pd.read_csv(COMP / "train.csv")
margins = pd.read_csv(LAB / "train_margins_v4.csv")
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
print("report_auc is scored against v4 teacher labels — NOT comparable across\n"
      "label versions. gold_auc is the comparable metric, but it lands on only\n"
      "~13 gold studies per fold, so the leaderboard is the real verdict.")

json.dump({"fold": FOLD, "size": SIZE_REPORT, "gstep": int(gstep), "total_steps": TOTAL_STEPS,
           "chunks": chunk, "report_auc": float(df.report_auc.mean()),
           "gold_auc": float(df.gold_auc.mean()), "n_gold": int(gm.sum()),
           "hours": round((time.time() - t_start) / 3600, 2),
           "per_condition": rows},
          open(OUT / "v7_meta.json", "w"), indent=2)

if STATE.exists():
    torch.save(torch.load(STATE, map_location="cpu")["model"], OUT / f"v7_fold{FOLD}.pt")
    STATE.unlink()   # optimizer state is not worth keeping as kernel output
print(f"saved v7_fold{FOLD}.pt + v7_meta.json")
