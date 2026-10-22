# %% [markdown]
# # 336 loader probe — is the 30 s/step I/O or compute?
#
# v6 fold 0 ran at a flat **30 s/step** across all ten chunks and grew RSS by a
# perfectly linear **122 MB/step**, resetting on every chunk restart. Pinning
# `MALLOC_MMAP_THRESHOLD_` changed neither number, so the glibc-fragmentation
# story was wrong and both symptoms are still unexplained.
#
# The standing suspicion is `num_workers=0`: `train_chunk.py` does a synchronous
# 8.1 MB `os.pread` per item and then three float32 passes of augmentation, all
# on the main process between GPU steps. That would explain the step time *and*
# put the allocation churn somewhere a worker restart would clean up.
#
# That is a guess, and guessing is what cost the last two sessions. This probe
# measures the split directly instead:
#
# | | what it isolates |
# |---|---|
# | A cold pread | raw random-read bandwidth off `/kaggle/input` |
# | B warm pread | the same rows again — separates bandwidth from per-call cost |
# | C full `__getitem__` | A plus the numpy augmentation passes |
# | D synthetic fwd/bwd | pure GPU compute, no data pipeline at all |
# | E/F/G real loop | `num_workers` 0, 2, 4 — the actual answer |
#
# **D is the floor.** No loader change can beat it. If E is already close to D
# the pipeline is compute-bound, 336 costs what it costs, and the only lever
# left is a smaller backbone or fewer slices. If E is far above D and G lands
# near it, workers are the fix and a full 1800-step anneal fits in a session.
#
# Each variant also prints the RSS slope, so we find out whether the 122 MB/step
# follows the workers out of the main process or stays put.
#
# Nothing here trains anything or writes a checkpoint. It is pure measurement.

# %%
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
from torch.utils.data import DataLoader, Dataset

# Kept byte-identical to train_chunk.py. A probe that measures a different
# geometry than the run it is diagnosing measures nothing.
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
          "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
ATTN_CONDS = ["Medial Meniscus", "Lateral Meniscus", "MCL", "Fracture", "Baker's"]
BACKBONE = "convnext_nano"
BATCH = 8
MICRO = 2
LR = 3e-4
SEED = 42
TEMP_DIV = 3.0

N_IO = 64          # rows for the pread / getitem timings
N_COMPUTE = 20     # synthetic fwd/bwd steps
N_LOOP = 20        # real steps per num_workers variant
WORKER_VARIANTS = [0, 2, 4]
DEADLINE_MIN = 50  # skip whatever is left rather than lose the session

T_START = time.time()
dev = "cuda" if torch.cuda.is_available() else "cpu"


def elapsed_min():
    return (time.time() - T_START) / 60


def rss_g():
    return int(Path("/proc/self/statm").read_text().split()[1]) * 4096 / 2**30


def cgroup_g():
    """Whole-container usage. num_workers>0 could move allocation out of the
    main process without reducing it, and only this number would show that."""
    for p in ("/sys/fs/cgroup/memory.current",
              "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            return int(Path(p).read_text().strip()) / 2**30
        except (OSError, ValueError):
            continue
    return float("nan")


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
    assert m["row_lo"] == edge, f"row gap/overlap at {edge}: shard starts {m['row_lo']}"
    edge = m["row_hi"]
assert edge == N_ALL, f"shards cover {edge} of {N_ALL} studies"

_vm = np.load(SHARDS[0] / "volumes.npy", mmap_mode="r")
VOL_DTYPE, VOL_OFFSET = _vm.dtype, _vm.offset
ITEM_SHAPE = _vm.shape[1:]
ITEM_BYTES = int(np.prod(ITEM_SHAPE)) * VOL_DTYPE.itemsize
del _vm
SHARD_FD = [os.open(str(d / "volumes.npy"), os.O_RDONLY) for d in SHARDS]
SHARD_LO = np.array([m["row_lo"] for m in metas])

print("shards      ", [d.name for d in SHARDS])
print("item shape  ", ITEM_SHAPE, VOL_DTYPE, "= %.2f MB" % (ITEM_BYTES / 2**20))
print("batch bytes  %.1f MB" % (BATCH * ITEM_BYTES / 2**20))
print("devices     ", torch.cuda.device_count(), "gpu")
print("rss %.1fG  cgroup %.1fG" % (rss_g(), cgroup_g()), flush=True)

# %%
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

folds = pd.read_csv(LAB / "folds.csv")
fold = train[["StudyInstanceUID"]].merge(folds, on="StudyInstanceUID",
                                         how="left").fold.to_numpy()
tr_idx = np.where(fold != 0)[0]
print("train rows", len(tr_idx), flush=True)


def read_vol(r):
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
            x = np.roll(x, np.random.randint(-8, 9), axis=3)
            x = np.roll(x, np.random.randint(-8, 9), axis=2)
            g = np.float32(np.random.uniform(0.9, 1.1))
            b = np.float32(np.random.uniform(-12, 12))
            x = np.clip(x.astype(np.float32) * g + b, 0, 255).astype(np.uint8)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(Y[r])


RESULTS = {}

# %% [markdown]
# ## A/B — raw read bandwidth
#
# A reads `N_IO` random rows the loop has never touched; B re-reads the same
# rows. If B is much faster the file is being served from page cache and the
# real run's cost is cold bandwidth. If A and B are equally slow, the cost is
# per-call and no amount of prefetching depth will hide it — only fewer, larger,
# more sequential reads would.

# %%
rng = np.random.default_rng(SEED)
probe_rows = rng.choice(tr_idx, size=N_IO, replace=False)

t0 = time.time()
for r in probe_rows:
    read_vol(int(r))
cold = time.time() - t0

t0 = time.time()
for r in probe_rows:
    read_vol(int(r))
warm = time.time() - t0

RESULTS["A_cold_read_s_per_item"] = cold / N_IO
RESULTS["B_warm_read_s_per_item"] = warm / N_IO
print("A cold pread  %.3f s/item  %6.1f MB/s  -> %.1f s per batch of %d"
      % (cold / N_IO, N_IO * ITEM_BYTES / 2**20 / cold,
         BATCH * cold / N_IO, BATCH))
print("B warm pread  %.3f s/item  %6.1f MB/s  -> %.1f s per batch of %d"
      % (warm / N_IO, N_IO * ITEM_BYTES / 2**20 / warm,
         BATCH * warm / N_IO, BATCH), flush=True)

# %% [markdown]
# ## C — the full `__getitem__`
#
# C minus A is what the two `np.roll` copies and the float32 gain/bias pass cost.
# Those run on the main process too, so if the augmentation is a large share the
# fix is cheaper augmentation (uint8 lookup table, roll via slicing) and not
# only workers.

# %%
ds = KneeDS(tr_idx, True)
np.random.seed(SEED)
pos = rng.choice(len(tr_idx), size=N_IO, replace=False)
t0 = time.time()
for p in pos:
    ds[int(p)]
getitem = time.time() - t0

RESULTS["C_getitem_s_per_item"] = getitem / N_IO
print("C getitem     %.3f s/item  -> %.1f s per batch of %d"
      % (getitem / N_IO, BATCH * getitem / N_IO, BATCH))
print("  of which read %.3f, augmentation %.3f"
      % (cold / N_IO, max(getitem - cold, 0) / N_IO), flush=True)

# %%
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
        for p_, lab in enumerate(order):
            inv[lab] = p_
        self.register_buffer("inv_perm", torch.tensor(inv), persistent=False)

    def forward(self, x):
        B, P, D, H, W = x.shape
        f = self.backbone(x.reshape(B * P * D, 1, H, W)).reshape(B, P, D, -1)
        g_mean = f.mean(2).reshape(B, -1)
        w = torch.softmax(self.att(f), dim=2)
        g_attn = (f * w).sum(2).reshape(B, -1)
        return torch.cat([self.head_attn(g_attn), self.head_mean(g_mean)],
                         dim=1)[:, self.inv_perm]


def build():
    model = Net().to(dev)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()
    model.train()
    return model, opt, scaler


def one_step(model, opt, scaler, xb, yb):
    """Byte-for-byte the accumulation the real run does, so the compute floor
    is the real floor and not an idealised one."""
    opt.zero_grad(set_to_none=True)
    n_micro = (xb.shape[0] + MICRO - 1) // MICRO
    for mb in range(n_micro):
        xs, ys = xb[mb * MICRO:(mb + 1) * MICRO], yb[mb * MICRO:(mb + 1) * MICRO]
        if xs.shape[0] == 0:
            continue
        with torch.amp.autocast("cuda"):
            loss = F.binary_cross_entropy_with_logits(model(xs), ys) / n_micro
        scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()


# %% [markdown]
# ## D — the compute floor
#
# One batch generated on the GPU and reused, so the data pipeline contributes
# exactly nothing. Five warm-up steps are discarded: the first steps pay cuDNN
# autotuning and allocator growth, and folding those into the average would
# flatter the floor by understating how much headroom the loader has.

# %%
model, opt, scaler = build()
xb = torch.randint(0, 255, (BATCH, 3, D_SLICES, SIZE, SIZE),
                   dtype=torch.uint8, device=dev).float().div_(255.0)
yb = torch.rand(BATCH, len(LABELS), device=dev)

for _ in range(5):
    one_step(model, opt, scaler, xb, yb)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N_COMPUTE):
    one_step(model, opt, scaler, xb, yb)
torch.cuda.synchronize()
compute = (time.time() - t0) / N_COMPUTE

RESULTS["D_compute_s_per_step"] = compute
print("D compute floor  %.2f s/step  (no data pipeline at all)" % compute)
print("   elapsed %.1f min" % elapsed_min(), flush=True)

del model, opt, scaler, xb, yb
torch.cuda.empty_cache()

# %% [markdown]
# ## E/F/G — the real loop at each `num_workers`
#
# The full pipeline, `N_LOOP` steps per setting, timed after one warm-up step so
# worker start-up does not land in the average. RSS is sampled at the first and
# last step: `rss_slope` is the main-process growth per step, `cg_slope` the
# whole container's. If workers fix the memory, the first drops and the second
# does not — that is the difference between the leak being gone and the leak
# merely having moved.

# %%
def run_loop(nw):
    model, opt, scaler = build()
    dl = DataLoader(KneeDS(tr_idx, True), batch_size=BATCH, shuffle=True,
                    num_workers=nw, drop_last=True,
                    persistent_workers=False,
                    prefetch_factor=(4 if nw else None))
    it = iter(dl)
    xb, yb = next(it)                       # warm-up step, not timed
    one_step(model, opt, scaler, xb.to(dev).float().div_(255.0), yb.to(dev))
    torch.cuda.synchronize()

    r0, c0 = rss_g(), cgroup_g()
    t0, n = time.time(), 0
    while n < N_LOOP:
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(dl)
            continue
        one_step(model, opt, scaler, xb.to(dev).float().div_(255.0), yb.to(dev))
        n += 1
    torch.cuda.synchronize()
    sec = (time.time() - t0) / n
    r1, c1 = rss_g(), cgroup_g()

    del it, dl, model, opt, scaler
    torch.cuda.empty_cache()
    return {"s_per_step": sec,
            "rss_slope_mb": (r1 - r0) * 1024 / n,
            "cg_slope_mb": (c1 - c0) * 1024 / n,
            "rss_end_g": r1}


for nw in WORKER_VARIANTS:
    if elapsed_min() > DEADLINE_MIN:
        print("skipping num_workers=%d — %.0f min elapsed, past the %d min deadline"
              % (nw, elapsed_min(), DEADLINE_MIN), flush=True)
        continue
    try:
        res = run_loop(nw)
    except Exception as e:                  # a worker config that cannot run is
        print("num_workers=%d failed: %r" % (nw, e), flush=True)   # itself a result
        RESULTS["loop_nw%d" % nw] = {"error": repr(e)}
        continue
    RESULTS["loop_nw%d" % nw] = res
    print("num_workers=%d  %5.2f s/step   rss %+6.0f MB/step  cgroup %+6.0f MB/step"
          "   rss_end %.1fG   [%.1f min]"
          % (nw, res["s_per_step"], res["rss_slope_mb"], res["cg_slope_mb"],
             res["rss_end_g"], elapsed_min()), flush=True)

# %% [markdown]
# ## Verdict
#
# `budget_1800_h` is what a full annealed 1800-step run would cost at each
# setting. Anything at or under ~9 h fits a session with room for validation;
# anything above it truncates the anneal again, which is exactly what made the
# v6 checkpoint uninterpretable.

# %%
compute = RESULTS["D_compute_s_per_step"]
print("compute floor            %5.2f s/step  -> %4.1f h for 1800 steps"
      % (compute, compute * 1800 / 3600))
print()
best_nw, best_s = None, float("inf")
for nw in WORKER_VARIANTS:
    r = RESULTS.get("loop_nw%d" % nw)
    if not r or "error" in r:
        continue
    s = r["s_per_step"]
    print("num_workers=%-2d           %5.2f s/step  -> %4.1f h for 1800 steps"
          "   overhead over floor %+5.2f s" % (nw, s, s * 1800 / 3600, s - compute))
    if s < best_s:
        best_nw, best_s = nw, s

print()
if best_nw is None:
    print("no loop variant completed — nothing is settled, do not change the loader")
elif best_s <= compute * 1.25:
    print("VERDICT: num_workers=%d is within 25%% of the compute floor. The loader "
          "is no longer the bottleneck; 336 costs %.1f h for 1800 steps and any "
          "further gain has to come from the model or the slice count."
          % (best_nw, best_s * 1800 / 3600))
else:
    print("VERDICT: even the best setting (num_workers=%d, %.2f s/step) sits %.2f s "
          "above the %.2f s floor, so %.0f%% of every step is still the data path. "
          "Workers alone are not enough — look at C above to see whether that "
          "remainder is read bandwidth or augmentation."
          % (best_nw, best_s, best_s - compute, compute,
             100 * (best_s - compute) / best_s))

Path("/kaggle/working/io_probe_336.json").write_text(json.dumps(
    {"item_bytes": ITEM_BYTES, "batch": BATCH, "micro": MICRO,
     "d_slices": D_SLICES, "size": SIZE,
     "n_gpu": torch.cuda.device_count(), "results": RESULTS}, indent=2))
print("\nwrote io_probe_336.json   total %.1f min" % elapsed_min())
