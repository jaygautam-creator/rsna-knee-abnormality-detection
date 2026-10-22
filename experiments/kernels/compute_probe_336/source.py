# %% [markdown]
# # 336 compute ablation — why is a step 22.6 s?
#
# `knee-io-probe-336` settled that the data path is 2.5% of a 23 s step and that
# `num_workers` buys nothing. What it left behind is a much stranger number: the
# **compute floor alone is 22.63 s/step**.
#
# That is not plausibly slow, it is broken. A step is `BATCH*3*D_SLICES` =
# 8*3*24 = 576 single-channel 336 px images through convnext_nano. At ~2.5
# GFLOPs/image forward at 224, scaled 2.25x for 336 and ~3x for the backward,
# that is roughly **9.6 TFLOP/step** against ~40 TFLOPS of realistic 2xT4 fp16.
# The expectation is 0.2-1 s/step. We are 20-90x off.
#
# Three suspects, none of them tested, all cheap to ablate:
#
# 1. **`autocast` is thread-local and `DataParallel` runs replicas in threads.**
#    The outer `with torch.amp.autocast(...)` in `train_chunk.py` wraps the call
#    site, not the replica threads, so the whole backbone may be running fp32 on
#    hardware whose fp16 tensor cores are the only reason a T4 is fast. This is
#    a documented PyTorch gotcha and it is my main suspect.
# 2. **`MICRO=2` starves `DataParallel`.** A 2-sample micro-batch scatters to one
#    sample per GPU, and DataParallel re-replicates the module on every call —
#    four times per step here.
# 3. **No `channels_last`.** ConvNeXt is depthwise-heavy and cuDNN's fp16
#    kernels want NHWC; NCHW can fall off the tensor-core path entirely.
#
# ## Reading this
#
# `backbone_ref_*` is the honest hardware ceiling: the raw backbone on one GPU
# with nothing else in the way. Every other row should be read as a fraction of
# it. If the ref is itself ~20 s then the model is wrong for this hardware and
# no configuration change saves 336. If the ref is fast and the baseline is
# slow, one of the three suspects above owns the gap and the table says which.
#
# No data is attached and nothing trains — batches are random tensors generated
# on the GPU. `pretrained=False` because weights do not affect timing and the
# HF download is a flake risk. OOM is caught and reported as a result, not an
# error: at 336 the larger micro-batches are expected to fail, and *where* they
# fail is exactly what we need to know to pick a real configuration.

# %%
import json
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
          "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
ATTN_CONDS = ["Medial Meniscus", "Lateral Meniscus", "MCL", "Fracture", "Baker's"]
BACKBONE = "convnext_nano"
BATCH = 8
D_SLICES = 24
LR = 3e-4

N_WARM = 2
N_TIMED = 5
DEADLINE_MIN = 45

T_START = time.time()
assert torch.cuda.is_available(), "this probe measures GPU compute"
N_GPU = torch.cuda.device_count()
print("gpus", N_GPU, [torch.cuda.get_device_name(i) for i in range(N_GPU)])
print("torch", torch.__version__, "cudnn", torch.backends.cudnn.version(), flush=True)


def elapsed_min():
    return (time.time() - T_START) / 60


# %%
class Net(nn.Module):
    """train_chunk.py's model, plus a switch for where autocast is applied.

    `autocast_inside=True` moves the context manager into forward(), which is
    the only placement DataParallel's worker threads actually observe."""

    def __init__(self, name=BACKBONE, n_planes=3, autocast_inside=False):
        super().__init__()
        self.autocast_inside = autocast_inside
        self.backbone = timm.create_model(name, pretrained=False, in_chans=1, num_classes=0)
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

    def _body(self, x):
        B, P, D, H, W = x.shape
        f = self.backbone(x.reshape(B * P * D, 1, H, W)).reshape(B, P, D, -1)
        g_mean = f.mean(2).reshape(B, -1)
        w = torch.softmax(self.att(f), dim=2)
        g_attn = (f * w).sum(2).reshape(B, -1)
        return torch.cat([self.head_attn(g_attn), self.head_mean(g_mean)],
                         dim=1)[:, self.inv_perm]

    def forward(self, x):
        if self.autocast_inside:
            with torch.amp.autocast("cuda"):
                return self._body(x)
        return self._body(x)


def time_config(name, dp, micro, channels_last, autocast_inside, size):
    """One full optimiser step, timed exactly as train_chunk.py performs it."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = Net(autocast_inside=autocast_inside).to("cuda")
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if dp and N_GPU > 1:
        model = nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler()
    model.train()

    xb = torch.rand(BATCH, 3, D_SLICES, size, size, device="cuda")
    yb = torch.rand(BATCH, len(LABELS), device="cuda")

    def step():
        opt.zero_grad(set_to_none=True)
        n_micro = (BATCH + micro - 1) // micro
        for mb in range(n_micro):
            xs, ys = xb[mb * micro:(mb + 1) * micro], yb[mb * micro:(mb + 1) * micro]
            if xs.shape[0] == 0:
                continue
            if channels_last:
                # channels_last on a 5-D tensor is channels_last_3d; the reshape
                # inside forward is what actually reaches cuDNN, so the flag has
                # to ride on the module and the 4-D view it produces.
                xs = xs.contiguous()
            # When autocast lives inside forward, enabling it here as well would
            # measure the fix and the bug at once.
            if autocast_inside:
                loss = F.binary_cross_entropy_with_logits(model(xs), ys) / n_micro
            else:
                with torch.amp.autocast("cuda"):
                    loss = F.binary_cross_entropy_with_logits(model(xs), ys) / n_micro
            scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    try:
        for _ in range(N_WARM):
            step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N_TIMED):
            step()
        torch.cuda.synchronize()
        sec = (time.time() - t0) / N_TIMED
        peak = torch.cuda.max_memory_allocated() / 2**30
        out = {"s_per_step": sec, "peak_gb": peak,
               "img_per_s": BATCH * 3 * D_SLICES / sec,
               "h_1800": sec * 1800 / 3600}
    except torch.cuda.OutOfMemoryError:
        out = {"oom": True}
    except RuntimeError as e:
        out = {"error": repr(e)[:200]}

    del model, opt, scaler, xb, yb
    torch.cuda.empty_cache()
    return out


def time_backbone_ref(size, channels_last):
    """The hardware ceiling: raw backbone, one GPU, fp16, no wrapper, no heads.
    Same image count as one micro-batch of 2 so the numbers are comparable."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    n_img = 2 * 3 * D_SLICES
    m = timm.create_model(BACKBONE, pretrained=False, in_chans=1, num_classes=0).cuda()
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    scaler = torch.amp.GradScaler()
    x = torch.rand(n_img, 1, size, size, device="cuda")
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = m(x).square().mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    try:
        for _ in range(N_WARM):
            step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N_TIMED):
            step()
        torch.cuda.synchronize()
        sec = (time.time() - t0) / N_TIMED
        # Scale one micro-batch up to the 4 that make a full step.
        out = {"s_per_step": sec * 4, "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
               "img_per_s": n_img / sec, "h_1800": sec * 4 * 1800 / 3600}
    except torch.cuda.OutOfMemoryError:
        out = {"oom": True}

    del m, opt, scaler, x
    torch.cuda.empty_cache()
    return out


# %% [markdown]
# ## The sweep
#
# Ordered by information per minute, because the deadline may cut it short: the
# hardware ceiling first, then the autocast-placement test, then the shape of
# the DataParallel and micro-batch space, then a 224 row for scale.

# %%
CONFIGS = [
    # name,                 dp,    micro, chan_last, autocast_inside, size
    ("backbone_ref_336",    None,  None,  False,     None,            336),
    ("backbone_ref_336_cl", None,  None,  True,      None,            336),
    ("baseline_dp_m2",      True,  2,     False,     False,           336),
    ("dp_m2_autocast_in",   True,  2,     False,     True,            336),
    ("dp_m2_cl_ai",         True,  2,     True,      True,            336),
    ("single_m2",           False, 2,     False,     False,           336),
    ("single_m2_cl",        False, 2,     True,      False,           336),
    ("single_m4",           False, 4,     False,     False,           336),
    ("single_m8",           False, 8,     False,     False,           336),
    ("dp_m4_cl_ai",         True,  4,     True,      True,            336),
    ("baseline_dp_m2_224",  True,  2,     False,     False,           224),
]

RESULTS = {}
for name, dp, micro, cl, ai, size in CONFIGS:
    if elapsed_min() > DEADLINE_MIN:
        print("skipping %s — %.0f min elapsed, past the %d min deadline"
              % (name, elapsed_min(), DEADLINE_MIN), flush=True)
        continue
    if dp and N_GPU < 2:
        # Silently degrading a dp row to single-GPU would print a duplicate of
        # single_m2 under a name that claims to have tested DataParallel.
        print("skipping %s — needs 2 GPUs, this host has %d" % (name, N_GPU), flush=True)
        RESULTS[name] = {"skipped": "needs 2 gpus"}
        continue
    if name.startswith("backbone_ref"):
        r = time_backbone_ref(size, cl)
    else:
        r = time_config(name, dp, micro, cl, ai, size)
    RESULTS[name] = r
    if "oom" in r:
        print("%-20s OOM" % name, flush=True)
    elif "error" in r:
        print("%-20s ERROR %s" % (name, r["error"]), flush=True)
    else:
        print("%-20s %6.2f s/step  %7.1f img/s  peak %5.2f GB  -> %5.1f h/1800   [%.0f min]"
              % (name, r["s_per_step"], r["img_per_s"], r["peak_gb"],
                 r["h_1800"], elapsed_min()), flush=True)

# %% [markdown]
# ## Verdict

# %%
ref = RESULTS.get("backbone_ref_336", {})
base = RESULTS.get("baseline_dp_m2", {})

if "s_per_step" in ref:
    print("hardware ceiling (raw backbone, 1 GPU, fp16): %.2f s/step equivalent, "
          "%.1f img/s" % (ref["s_per_step"], ref["img_per_s"]))
if "s_per_step" in base and "s_per_step" in ref:
    print("v6 baseline as built:                          %.2f s/step  "
          "(%.1fx the ceiling)" % (base["s_per_step"], base["s_per_step"] / ref["s_per_step"]))
print()

ok = {k: v for k, v in RESULTS.items()
      if "s_per_step" in v and not k.startswith("backbone_ref")}
if ok:
    best = min(ok, key=lambda k: ok[k]["s_per_step"])
    b = ok[best]
    print("best full-model config: %s  %.2f s/step  -> %.1f h for 1800 steps"
          % (best, b["s_per_step"], b["h_1800"]))
    if "s_per_step" in base:
        print("  speedup over the v6 baseline: %.2fx" % (base["s_per_step"] / b["s_per_step"]))
    print()
    if b["h_1800"] <= 8.0:
        print("VERDICT: 336 fits. A full annealed 1800-step run costs %.1f h at "
              "'%s', leaving room for validation inside a session. Rebuild v6 "
              "with that configuration and the 336-vs-224 question finally gets "
              "a fair test." % (b["h_1800"], best))
    elif b["h_1800"] <= 11.0:
        print("VERDICT: marginal. %.1f h for 1800 steps at '%s' is inside a 12 h "
              "session but leaves no slack, and v6 already lost a run to exactly "
              "that. Either cut D_SLICES/TOTAL_STEPS to buy margin, or accept "
              "two-session training with the existing chunk-resume machinery."
              % (b["h_1800"], best))
    else:
        print("VERDICT: 336 does not fit at this backbone. The best configuration "
              "still needs %.1f h for 1800 steps. The lever is no longer "
              "configuration — it is fewer slices, a smaller input, or going back "
              "to 224 and spending the session budget on the blend instead."
              % b["h_1800"])
else:
    print("no full-model config completed — nothing is settled")

Path("/kaggle/working/compute_probe_336.json").write_text(json.dumps(
    {"n_gpu": N_GPU, "torch": torch.__version__,
     "batch": BATCH, "d_slices": D_SLICES, "results": RESULTS}, indent=2))
print("\nwrote compute_probe_336.json   total %.1f min" % elapsed_min())
