# %% [markdown]
# # v2 — everything in one run
#
# Combines the changes that v0/v1 measured or motivated, in a single session:
#
# * **24 slices** (was 16) — focal findings were the weak spot
# * **Per-condition pooling** — the attn ablation moved focal and diffuse
#   conditions in opposite directions and they cancelled in the mean, so
#   attention is applied only to the conditions that actually gained from it
# * **Stronger backbone** — resnet18 was a placeholder
# * **One fold per session, split geometry still 5-way.** Folds 1 and 2 are this
#   same notebook with `FOLD_LIST` changed, run separately.
#
# **The binding constraint is memory, not the 12 h session kill.** Something in
# this loop grows the process by ~32 MB per step and five runs have failed to
# find it: it survived removing the mmap (run 3), the dataloader workers and
# pinned buffers (run 4), and a 4x cut in host allocation plus `malloc_trim`
# (run 5). What each run *did* establish is the rate, and a measured rate is
# enough to plan around even when the cause is not known.
#
# So the run sizes itself twice. It times real steps to get throughput, and it
# measures RSS growth across those same steps to get the leak rate. From the
# two it derives a step budget that fits in the memory available, and hands
# that budget to OneCycleLR as `total_steps`.
#
# That last part is the fix for run 5's real failure. Run 5 stopped 640 steps
# into a 1,760-step schedule — 36% through, just past the LR peak at step 528 —
# so the saved weights had never been annealed. It scored 0.7788 against v0's
# 0.8365 and lost on all 12 conditions, which looked like the architecture
# failing and was actually a schedule cut off at its worst possible moment.
# A short schedule that finishes beats a long one that gets truncated.
#
# Because several changes land together, a single averaged number cannot say
# which one worked. The per-condition table at the end is what makes the run
# still interpretable — compare it against v0's, not just the mean.

# %%
import ctypes
import json
import os
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
# From the v0 -> attn per-condition deltas, not anatomical intuition.
ATTN_CONDS = ["Medial Meniscus", "Lateral Meniscus", "MCL", "Fracture", "Baker's"]

# Backbone candidates in preference order; the first that builds is used.
# convnext_tiny was the first choice but measures ~3.8 h/fold at 24 slices —
# 19 h for 5 folds against a 10.5 h budget, i.e. 2 folds and a wasted run.
BACKBONES = ["convnext_nano", "efficientnet_b0", "resnet34"]
BATCH_TRY = [8, 4, 2]   # 24 slices x 3 planes means 8 -> 576 images/step
# Floor was 8, which silently overrode the budget: run 1 measured 35.85 h of
# work against a 10.5 h budget and started anyway, because np.clip raised the
# 2.15 epochs the budget allowed up to the floor. The floor is now 3 — low
# enough that the budget, not the clip, decides.
EPOCH_RANGE = (3, 18)
LR = 3e-4
N_SPLITS = 5            # split geometry: keep 5-way so fold 0 stays comparable to v0
# One fold per session. The leak does not reset when a fold ends — run 5 came
# out of fold 0 at rss 26.5G and had to stop — so a second fold in the same
# session gets a fraction of the steps the first one did. Separate sessions each
# start at 3G. Folds 1 and 2 are the same notebook with this list changed.
FOLD_LIST = [0]
MEM_FLOOR_G = 8.0       # abandon the epoch below this; leaves room to validate
MEM_RESERVE_G = 9.0     # headroom the step budget must leave for validation
SEED = 42
TEMP_DIV = 3.0
BUDGET_S = 10.5 * 3600      # leave headroom under the 12 h kill
PROBE_STEPS = 12            # training steps timed to size the run
OUT = Path("/kaggle/working")

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "| gpus:", torch.cuda.device_count())

# Run 4's leak survived removing the workers, the mmap and the pinned buffers,
# and nothing in the loop retains a Python reference — the remaining explanation
# is glibc holding freed large blocks. When a big mmap'd block is freed, glibc
# raises its mmap threshold, after which same-size allocations come off the sbrk
# heap and fragment it; RSS then climbs with no leak visible from Python.
# malloc_trim(0) hands the free heap back to the OS. If the theory is wrong this
# is a no-op costing microseconds, so it is worth trying either way.
try:
    _libc = ctypes.CDLL("libc.so.6")
    _libc.mallopt(-3, 131072)   # M_MMAP_THRESHOLD: stop the dynamic raise
except OSError:
    _libc = None
    print("libc unavailable — heap trimming disabled")


def trim_heap() -> None:
    if _libc is not None:
        _libc.malloc_trim(0)


def rss_g() -> float:
    return int(Path("/proc/self/statm").read_text().split()[1]) * 4096 / 2**30


def mem_avail_g() -> float:
    for ln in Path("/proc/meminfo").read_text().splitlines():
        p = ln.split()
        if p[0] == "MemAvailable:":
            return int(p[1]) / 2**20
    return float("inf")


def mem() -> str:
    """RSS + container-visible free memory.

    Run 1 was SIGKILLed at 2.63 h with no Python traceback — the signature of a
    host-RAM OOM, but nothing logged memory. One line per 100 steps is what
    turned run 3 from a mystery into a measured 43 MB/step.
    """
    info = dict(
        (p[0].rstrip(":"), int(p[1]) / 2**20)
        for p in (ln.split() for ln in Path("/proc/meminfo").read_text().splitlines())
    )
    return (f"rss {rss_g():.1f}G | avail {info.get('MemAvailable', 0):.1f}G "
            f"| cached {info.get('Cached', 0):.1f}G")


def find_dir(*markers: str, max_depth: int = 4) -> Path:
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
                continue
            frontier.append((child, depth + 1))
    print(f"--- could not find {markers}; /kaggle/input tree (depth 3) ---")
    for p in sorted(Path("/kaggle/input").rglob("*")):
        rel = p.relative_to("/kaggle/input")
        if len(rel.parts) <= 3:
            print(" ", rel, "/" if p.is_dir() else "")
    raise FileNotFoundError(f"no mount containing {markers}")


COMP = find_dir("train.csv", "train_series.csv")
CACHE = find_dir("cache_meta.json", "volumes.npy", "index.csv")
# v3 labels, not v2: the 8B relabel measures 0.8738 against the 58 expert-labelled
# studies where v2 measures 0.8548, and the gain is concentrated in v2's three
# weakest conditions (Effusion +0.070, Contusion +0.074, PF OA +0.055). Better
# targets are the cheapest lever we have. Note this changes two things at once
# against run 4 — labels and the memory fix — so a good result here cannot be
# attributed to either alone.
LAB = find_dir("train_margins_v3.csv")
cache_meta = json.loads((CACHE / "cache_meta.json").read_text())
print("cache:", CACHE, "|", cache_meta["shape"], cache_meta["gb"], "GB")

# Run 2 died in host RAM: rss went 4.8G -> 20.7G across one shuffled epoch and
# the kernel was killed with no traceback. A shuffled epoch touches every study,
# so the whole 15.92 GB mapping became resident and was charged to the container.
# Read the same file positionally instead. Page cache still fills, but it is
# clean and unmapped, so the kernel can drop it under pressure; rss stays flat.
# os.pread is offset-free, so the fd is safe to share across forked workers.
_vm = np.load(CACHE / "volumes.npy", mmap_mode="r")
VOL_SHAPE, VOL_DTYPE, VOL_OFFSET = _vm.shape, _vm.dtype, _vm.offset
ITEM_SHAPE = VOL_SHAPE[1:]
ITEM_BYTES = int(np.prod(ITEM_SHAPE)) * VOL_DTYPE.itemsize
del _vm
VOL_FD = os.open(str(CACHE / "volumes.npy"), os.O_RDONLY)


def read_vol(r: int) -> np.ndarray:
    buf = os.pread(VOL_FD, ITEM_BYTES, VOL_OFFSET + r * ITEM_BYTES)
    assert len(buf) == ITEM_BYTES, f"short read at row {r}"
    return np.frombuffer(buf, dtype=VOL_DTYPE).reshape(ITEM_SHAPE)


index = pd.read_csv(CACHE / "index.csv")
train = pd.read_csv(COMP / "train.csv")
margins = pd.read_csv(LAB / "train_margins_v3.csv")

assert (index.StudyInstanceUID.values == train.StudyInstanceUID.values).all(), \
    "cache index does not match train.csv order"

# Expert truth is in train.csv itself — 58 rows with all 12 labels, binary.
gold = train.loc[train[LABELS].notna().all(1), ["StudyInstanceUID"] + LABELS].copy()
assert len(gold) == 58, f"expected 58 expert-labelled studies, got {len(gold)}"
assert set(np.unique(gold[LABELS].to_numpy())) <= {0.0, 1.0}, "gold must be binary"

m = train[["StudyInstanceUID"]].merge(margins, on="StudyInstanceUID", how="left")
assert m[LABELS].notna().all().all(), "missing margins"
M = m[LABELS].to_numpy(np.float32)
T = M.std(0) / TEMP_DIV
Y = 1.0 / (1.0 + np.exp(-M / T))
D_SLICES = cache_meta["d_slices"]
SIZE = cache_meta["size"]
print(f"volumes {VOL_SHAPE}  slices/plane {D_SLICES}  size {SIZE}")

# %%
import hashlib
import re
from sklearn.model_selection import StratifiedGroupKFold

_WS = re.compile(r"\s+")
norm = train.Report.fillna("").str.lower().str.replace(_WS, " ", regex=True).str.strip()
groups = norm.map(lambda s: hashlib.md5(s.encode()).hexdigest())
strata = np.digitize((Y > 0.5).sum(1), [1, 3, 5, 7])
fold = np.full(len(train), -1)
for k, (_, va) in enumerate(
        StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED)
        .split(train, strata, groups)):
    fold[va] = k
assert (fold >= 0).all()
print("fold sizes:", np.bincount(fold).tolist())

# %%
class KneeDS(Dataset):
    def __init__(self, idx, train_mode: bool):
        self.idx, self.train = idx, train_mode

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        # Stay in uint8 all the way to the GPU. The flip and the roll are both
        # dtype-agnostic, so the float32 conversion bought nothing on the host
        # and cost 4x the allocation: a batch is 29 MB as uint8 against 115 MB
        # as float32. Run 4 leaked ~43 MB/step with no workers, no mmap and no
        # pinned buffers, which points at host allocation churn rather than a
        # retained Python object, so cutting the churn is the direct attack.
        r = self.idx[i]
        x = read_vol(r)
        if self.train:
            if np.random.rand() < 0.5:
                x = x[:, :, :, ::-1]
            x = np.roll(x, np.random.randint(-8, 9), axis=3)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(Y[r])


class Net(nn.Module):
    def __init__(self, name, n_planes=3):
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

    def forward(self, x):
        B, P, D, H, W = x.shape
        f = self.backbone(x.reshape(B * P * D, 1, H, W)).reshape(B, P, D, -1)
        g_mean = f.mean(2).reshape(B, -1)
        w = torch.softmax(self.att(f), dim=2)
        g_attn = (f * w).sum(2).reshape(B, -1)
        return torch.cat([self.head_attn(g_attn),
                          self.head_mean(g_mean)], dim=1)[:, self.inv_perm]


def predict(model, idx):
    dl = DataLoader(KneeDS(idx, False), batch_size=BATCH, shuffle=False,
                    num_workers=0)
    model.eval()
    P = []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for j, (xb, _) in enumerate(dl):
            xb = xb.to(dev, non_blocking=True).float().div_(255.0)
            P.append(torch.sigmoid(model(xb)).float().cpu())
            if j % 50 == 0:
                trim_heap()
    return torch.cat(P).numpy()


def mean_auc(y_true_bin, P):
    a = [roc_auc_score(y_true_bin[:, i], P[:, i])
         for i in range(len(LABELS))
         if 0 < y_true_bin[:, i].sum() < len(y_true_bin)]
    return float(np.mean(a)) if a else float("nan")


# %% [markdown]
# ## Size the run to the hardware, do not guess
#
# The first draft of this notebook hardcoded convnext_tiny / batch 8 / 14 epochs.
# Measured against v0's throughput that is ~3.8 h per fold at 24 slices — 19 h
# for 5 folds against a 10.5 h budget, so it would have produced 2 folds and no
# usable OOF matrix. Batch 8 x 3 planes x 24 slices is also 576 images per step,
# 1.5x v0's, at roughly 4x the activation memory: an OOM waiting to happen.
#
# So: pick the first backbone that builds, find the largest batch that fits by
# actually trying it, time real training steps, and derive the epoch count from
# the measurement. The run adapts to whatever GPU Kaggle hands us.

# %%
BACKBONE = None
for name in BACKBONES:
    try:
        _m = timm.create_model(name, pretrained=True, in_chans=1, num_classes=0)
        BACKBONE = name
        del _m
        break
    except Exception as e:
        print(f"  {name} unavailable ({type(e).__name__}) — trying next")
assert BACKBONE, f"none of {BACKBONES} could be built"
print("backbone:", BACKBONE)

_probe_idx = np.where(fold != 0)[0]


def _try_batch(bs: int) -> bool:
    """True if one fwd+bwd at this batch size fits in memory."""
    try:
        mdl = Net(BACKBONE).to(dev)
        if torch.cuda.device_count() > 1:
            mdl = nn.DataParallel(mdl)
        sc = torch.amp.GradScaler()
        op = torch.optim.AdamW(mdl.parameters(), lr=LR)
        xb = torch.zeros(bs, 3, D_SLICES, SIZE, SIZE)
        yb = torch.zeros(bs, len(LABELS))
        with torch.amp.autocast("cuda"):
            loss = F.binary_cross_entropy_with_logits(
                mdl(xb.to(dev)), yb.to(dev))
        sc.scale(loss).backward()
        op.zero_grad(set_to_none=True)
        del mdl, sc, op, xb, yb, loss
        torch.cuda.empty_cache()
        return True
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # DataParallel can surface OOM as a plain RuntimeError; anything else
        # is a real bug and must not be silently treated as "batch too big".
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        return False


BATCH = None
for bs in BATCH_TRY:
    if _try_batch(bs):
        BATCH = bs
        print(f"batch {bs}: fits ({bs * 3 * D_SLICES} images/step)")
        break
    print(f"batch {bs}: OOM")
assert BATCH, "even the smallest batch OOMs"

# Time real steps at the chosen batch, then solve for epochs.
# Must match the fold loop's loader exactly, or EPOCHS is solved from a
# throughput the real loop never achieves.
_dl = DataLoader(KneeDS(_probe_idx, True), batch_size=BATCH, shuffle=True,
                 num_workers=0, drop_last=True)
_model = Net(BACKBONE).to(dev)
if torch.cuda.device_count() > 1:
    _model = nn.DataParallel(_model)
_opt = torch.optim.AdamW(_model.parameters(), lr=LR)
_scaler = torch.amp.GradScaler()
_model.train()

_t = None
_rss0 = None
for i, (xb, yb) in enumerate(_dl):
    if i == 2:                     # skip warmup/compile steps
        torch.cuda.synchronize()
        _t = time.time()
        _rss0 = rss_g()
    if i >= 2 + PROBE_STEPS:
        break
    with torch.amp.autocast("cuda"):
        loss = F.binary_cross_entropy_with_logits(
            _model(xb.to(dev, non_blocking=True).float().div_(255.0)),
            yb.to(dev, non_blocking=True))
    _opt.zero_grad(set_to_none=True)
    _scaler.scale(loss).backward()
    _scaler.step(_opt)
    _scaler.update()
torch.cuda.synchronize()
s_per_step = (time.time() - _t) / PROBE_STEPS
leak_per_step_g = max(0.0, (rss_g() - _rss0) / PROBE_STEPS)
del _model, _opt, _scaler, _dl
torch.cuda.empty_cache()
trim_heap()

steps_per_epoch = len(_probe_idx) // BATCH
epoch_s = s_per_step * steps_per_epoch
budget_per_fold = BUDGET_S / len(FOLD_LIST) * 0.92
FOLD_CAP_S = budget_per_fold * 1.15

# The step budget, not the epoch count, is the real unit here.
#
# Run 5's memory bail landed 640 steps into a 1,760-step OneCycle schedule —
# 36% through, just past the step-528 LR peak. The saved weights were therefore
# a snapshot taken at maximum learning rate with no annealing whatsoever, which
# is why fold 0 scored 0.7788 against v0's 0.8365 on every single condition.
# That number says nothing about 24 slices, the two-head pooling or the v3
# labels; it measures an interrupted schedule.
#
# So: measure the leak during the probe, work out how many steps actually fit
# in the memory we have, and hand *that* to OneCycleLR as total_steps. A short
# schedule that completes beats a long one that gets cut off.
steps_time = int(budget_per_fold / s_per_step)
if leak_per_step_g > 1e-4:
    steps_mem = int(max(0.0, (mem_avail_g() - MEM_RESERVE_G) / leak_per_step_g))
else:
    steps_mem = steps_time
STEP_BUDGET = int(min(steps_time, steps_mem, EPOCH_RANGE[1] * steps_per_epoch))
EPOCHS = max(1, -(-STEP_BUDGET // steps_per_epoch))    # ceil, for the epoch loop

print(f"\nmeasured {s_per_step:.2f} s/step, {BATCH*3*D_SLICES/s_per_step:.0f} img/s")
print(f"  {epoch_s/60:.1f} min/epoch  x  {steps_per_epoch} steps")
print(f"  leak {leak_per_step_g*1024:.0f} MB/step -> {steps_mem} steps before the "
      f"{MEM_RESERVE_G:.0f}G reserve; time allows {steps_time}")
print(f"  STEP_BUDGET={STEP_BUDGET} ({STEP_BUDGET/steps_per_epoch:.2f} epochs, "
      f"{STEP_BUDGET*s_per_step/3600:.2f} h/fold) — LR anneals over exactly these")
assert STEP_BUDGET >= 100, (
    f"only {STEP_BUDGET} steps fit before the memory reserve — the leak "
    f"({leak_per_step_g*1024:.0f} MB/step) makes this configuration untrainable")

# %% [markdown]
# ## Fold loop under a wall-clock budget

# %%
OOF = np.full((len(train), len(LABELS)), np.nan, np.float32)
t_start = time.time()
fold_rows = []

print("mem at fold-loop start:", mem(), flush=True)

for k in FOLD_LIST:
    tr_idx = np.where(fold != k)[0]
    va_idx = np.where(fold == k)[0]
    # num_workers=0: the leak is in *this* process (statm is per-process), so it
    # was never the workers. Removing them cost ~5% throughput and ruled the
    # dataloader out as a suspect, which was worth the 5%.
    dl_tr = DataLoader(KneeDS(tr_idx, True), batch_size=BATCH, shuffle=True,
                       num_workers=0, drop_last=True)

    model = Net(BACKBONE).to(dev)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # total_steps is the measured budget, so the cosine actually reaches its
    # floor. Run 5 annealed over 1,760 planned steps and got 640 of them.
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEP_BUDGET)
    scaler = torch.amp.GradScaler()

    t0 = time.time()
    starved = False
    gstep = 0
    for ep in range(EPOCHS):
        if gstep >= STEP_BUDGET or time.time() - t0 > FOLD_CAP_S:
            break
        model.train()
        tot, nb = 0.0, 0
        for step, (xb, yb) in enumerate(dl_tr):
            if gstep >= STEP_BUDGET:
                print(f"    f{k}: step budget {STEP_BUDGET} reached — "
                      f"LR schedule complete", flush=True)
                break
            if gstep % 50 == 0:
                trim_heap()
            if gstep % 100 == 0:
                print(f"    f{k} ep{ep+1} gstep {gstep}/{STEP_BUDGET} "
                      f"lr {sched.get_last_lr()[0]:.2e} {mem()}", flush=True)
                # The budget is derived from a probe-measured leak rate, so this
                # should not fire. It is the backstop for the rate changing
                # mid-run, and it keeps the failure recoverable rather than fatal.
                if mem_avail_g() < MEM_FLOOR_G:
                    print(f"    f{k}: only {mem_avail_g():.1f}G available — "
                          f"abandoning epoch {ep+1} to save and validate",
                          flush=True)
                    starved = True
                    break
            gstep += 1
            xb = xb.to(dev, non_blocking=True).float().div_(255.0)
            yb = yb.to(dev, non_blocking=True)
            with torch.amp.autocast("cuda"):
                loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            nb += 1
        print(f"  fold {k} ep {ep+1}/{EPOCHS} loss {tot/max(nb, 1):.4f} "
              f"[{(time.time()-t0)/60:.1f} min] {mem()}", flush=True)
        if starved:
            break

    # Checkpoint before validating, not after. Validation is another ~110 steps
    # of whatever is consuming memory, so it is the most likely place to die —
    # and dying there used to throw away a fold that had already trained.
    sd = (model.module if hasattr(model, "module") else model).state_dict()
    torch.save(sd, OUT / f"v2_fold{k}.pt")

    P = predict(model, va_idx)
    OOF[va_idx] = P
    r_auc = mean_auc((Y[va_idx] > 0.5).astype(int), P)
    fold_rows.append({"fold": k, "report_auc": r_auc,
                      "minutes": round((time.time() - t0) / 60, 1)})
    print(f"fold {k}: report_auc {r_auc:.4f}  ({(time.time()-t0)/60:.1f} min)  "
          f"elapsed {(time.time()-t_start)/3600:.1f} h", flush=True)

    # Flush OOF after every fold. Run 1 saved only at the very end, so a crash
    # 2.6 h in returned literally nothing; now a late crash still yields the
    # folds that finished.
    np.save(OUT / "oof.npy", OOF)
    del model
    torch.cuda.empty_cache()
    if starved:
        # Whatever grew does not come back on fold teardown, so the next fold
        # would just hit the floor sooner. Keep what we have.
        print(f"stopping after fold {k}: {mem()}", flush=True)
        break

n_done = len(fold_rows)
print(f"\ntrained {n_done}/{len(FOLD_LIST)} folds in {(time.time()-t_start)/3600:.2f} h")

# %% [markdown]
# ## The number that matters: OOF vs the expert labels it covers
#
# Every study in a trained fold is predicted by a model that never saw it. With
# 3 of 5 folds that is ~35 of the 58 gold studies — a clean read, on a subset.
# `gold studies covered` below is the honest denominator; do not quote the AUC
# without it.

# %%
done = ~np.isnan(OOF[:, 0])
oof_report = mean_auc((Y[done] > 0.5).astype(int), OOF[done])

gold_mask = train[LABELS].notna().all(1).to_numpy() & done
g_true = train.loc[gold_mask, LABELS].to_numpy()
gold_auc = mean_auc(g_true, OOF[gold_mask])
print(f"studies with OOF preds: {done.sum()}/{len(train)}")
print(f"gold studies covered:   {gold_mask.sum()}/58")
print(f"\nOOF report_auc {oof_report:.4f}   |   OOF gold_auc {gold_auc:.4f}")
print("v0 baseline: report_auc 0.8365 (fold 0)  ->  LB 0.826")

# %%
per = []
hard = (Y[done] > 0.5).astype(int)
V0 = {"ACL": .858, "MCL": .7914, "Medial Meniscus": .8802, "Lateral Meniscus": .7894,
      "Medial OA": .8689, "Lateral OA": .7715, "PF OA": .798, "Effusion": .8475,
      "Synovitis": .8783, "Baker's": .8433, "Contusion": .8173, "Fracture": .8946}
for i, c in enumerate(LABELS):
    a = roc_auc_score(hard[:, i], OOF[done][:, i]) if 0 < hard[:, i].sum() < len(hard) else np.nan
    g = (roc_auc_score(g_true[:, i], OOF[gold_mask][:, i])
         if 0 < g_true[:, i].sum() < len(g_true) else np.nan)
    per.append({"condition": c, "pool": "attn" if c in ATTN_CONDS else "mean",
                "oof_auc": round(a, 4), "v0_auc": V0[c],
                "delta": round(a - V0[c], 4), "gold_auc": round(g, 4)})
print(pd.DataFrame(per).to_string(index=False))

np.save(OUT / "oof.npy", OOF)
json.dump({"backbone": BACKBONE, "epochs": EPOCHS, "d_slices": D_SLICES,
           "attn_conds": ATTN_CONDS, "folds_done": n_done, "fold_list": FOLD_LIST,
           "step_budget": STEP_BUDGET, "leak_mb_per_step": round(leak_per_step_g*1024, 1),
           "oof_report_auc": oof_report, "oof_gold_auc": gold_auc,
           "folds": fold_rows, "per_condition": per,
           "hours": round((time.time() - t_start) / 3600, 2)},
          open(OUT / "v2_meta.json", "w"), indent=2)
print(f"\nsaved oof.npy + v2_meta.json + {n_done} checkpoints")
