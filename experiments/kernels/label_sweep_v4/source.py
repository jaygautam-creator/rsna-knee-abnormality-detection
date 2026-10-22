# %% [markdown]
# # v4 — question sweep on the four weak conditions
#
# The v3 labels score 0.874 mean AUC against the 58 expert-labelled studies, but
# that average hides the real problem:
#
# | condition | label AUC vs gold |
# |---|---|
# | Synovitis | 0.747 |
# | Contusion | 0.810 |
# | Effusion  | 0.812 |
# | PF OA     | 0.853 |
# | ACL / MCL / Medial OA | 0.943 – 0.947 |
#
# The image model has already caught up to its teacher (0.869 vs labels), so
# these four are the binding constraint on everything downstream.
#
# **Why they fail is not prompt sloppiness.** Checked against the 58 gold
# studies, the reports frequently do not contain the finding at all:
#
# - Synovitis: 16 of 27 gold-positive reports never mention synovium/inflammation.
# - PF OA: only 6 of 21 gold positives mention the patellofemoral joint, while
#   13 of 37 gold *negatives* do — the obvious keyword is anti-correlated.
# - Effusion: gold says 0 for a report reading "small joint effusion", so the
#   expert threshold sits above the radiologist's phrasing.
#
# The experts labelled from images; the report is a lossy proxy. So the winning
# question is often **not** the literal one. For Synovitis the report's
# observable correlates (effusion, Hoffa fat pad impingement, inflammatory
# arthropathy) may rank gold better than the word "synovitis" does. This
# notebook tests that directly: 5 phrasings per condition, scored on the 58,
# ranked by AUC against expert truth.
#
# 58 studies x 4 conditions x 5 variants = 1,160 prompts, roughly 8 minutes.
# Cheap enough to iterate on, which the 6 h full relabel is not.

# %%
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MODELS = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-4B-Instruct-2507",
]
MODEL = os.environ.get("KNEE_MODEL") or None
BATCH = 16
OUT = "/kaggle/working"

print("GPUs:", torch.cuda.device_count(), torch.cuda.get_device_name(0))


def find_comp(max_depth: int = 3) -> Path:
    """Locate the competition mount by marker file rather than by slug — Kaggle
    does not guarantee the directory name or its depth."""
    roots = [Path("/kaggle/input")]
    seen = []
    for root in roots:
        if not root.exists():
            continue
        stack = [(root, 0)]
        while stack:
            d, depth = stack.pop()
            if depth > max_depth:
                continue
            if (d / "train.csv").exists() and (d / "train_series.csv").exists():
                return d
            seen.append(str(d))
            try:
                for c in d.iterdir():
                    if c.is_dir() and not c.name.startswith("."):
                        stack.append((c, depth + 1))
            except PermissionError:
                pass
    raise FileNotFoundError(f"competition data not found. searched: {seen[:40]}")


COMP = find_comp()
print("competition mount:", COMP)

train = pd.read_csv(f"{COMP}/train.csv")
ALL_LABELS = [c for c in train.columns if c not in ("StudyInstanceUID", "Report")]
labelled = train[train[ALL_LABELS[0]].notna()].reset_index(drop=True)
print(f"studies {len(train)}  labelled {len(labelled)}")

WEAK = ["Synovitis", "Contusion", "Effusion", "PF OA"]

# %% [markdown]
# ## The variants
#
# `v0` is always v3's current wording, so every run re-measures the baseline on
# the same 58 studies rather than trusting a number from a previous session.
# The others move along one axis each: literal → broader → proxy-by-correlate,
# and for Effusion, threshold up and down. Keeping the axis explicit is what
# makes a win interpretable instead of a lucky phrasing.

# %%
VARIANTS = {
    "Synovitis": {
        "v0": "Does this report describe synovitis — inflammation, thickening, "
              "hypertrophy or proliferation of the synovium or joint lining, synovial "
              "enhancement, or pannus?",
        "v1": "Does this report describe synovitis or any related synovial abnormality "
              "— synovial thickening, hypertrophy, proliferation or enhancement, pannus, "
              "synovial plica syndrome, or an inflammatory arthropathy?",
        "v2": "Based on this report, would a radiologist reviewing the images likely see "
              "synovial inflammation or thickening? Consider that synovitis usually "
              "accompanies a joint effusion, Hoffa's fat pad impingement or oedema, "
              "inflammatory arthropathy, or advanced osteoarthritis, even when the report "
              "does not use the word synovitis.",
        "v3": "Does this report describe any intra-articular inflammatory change — "
              "synovitis, synovial thickening or enhancement, pannus, Hoffa's fat pad "
              "impingement or oedema, plica syndrome, or a joint effusion with "
              "inflammatory features?",
        "v4": "Does this report describe an effusion, a synovial abnormality, or a fat "
              "pad abnormality of any kind?",
    },
    "Contusion": {
        "v0": "Does this report describe a bone contusion, bone bruise, or bone marrow "
              "oedema pattern from injury? Answer No if marrow signal is normal, and No if "
              "the marrow oedema is purely degenerative or reactive to arthritis.",
        "v1": "Does this report describe abnormal bone marrow signal of any cause — bone "
              "contusion, bone bruise, marrow oedema, oedema-like signal, or a stress "
              "reaction? Answer No only if marrow signal is described as normal or is not "
              "mentioned.",
        "v2": "Does this report describe a bone contusion or bone bruise, including "
              "subchondral marrow oedema adjacent to a ligament tear, meniscal tear, "
              "impaction, or other injury?",
        "v3": "Does this report describe any bone marrow signal abnormality — oedema, "
              "bruise, contusion, stress reaction, or oedema-like change?",
        "v4": "Does this report describe an acute traumatic bone injury pattern — bone "
              "bruise, marrow oedema, impaction, or pivot-shift injury?",
    },
    "Effusion": {
        "v0": "Does this report describe a joint effusion or excess joint fluid that is "
              "MORE THAN a trace or physiologic amount? Answer No for 'trace', 'minimal', "
              "'physiologic', or no effusion.",
        "v1": "Does this report describe a MODERATE or LARGE joint effusion? Answer No for "
              "'small', 'trace', 'minimal', or 'physiologic' effusion, and No if there is "
              "no effusion.",
        "v2": "Does this report describe a joint effusion or any excess joint fluid, of "
              "any size?",
        "v3": "Does this report describe a joint effusion whose volume is clearly "
              "abnormal — enough to distend the suprapatellar recess or bursa? Answer No "
              "for small, trace, minimal or physiologic amounts.",
        "v4": "Does this report describe joint fluid that is a significant finding rather "
              "than an incidental one?",
    },
    "PF OA": {
        "v0": "Does this report describe osteoarthritis or degenerative change in the "
              "PATELLOFEMORAL compartment? Count chondromalacia patellae, cartilage loss, "
              "thinning or ulceration of the patellar or trochlear cartilage, and "
              "patellofemoral osteophytes.",
        "v1": "Does this report describe ANY cartilage abnormality of the patella or the "
              "trochlea, even mild or early — chondromalacia of any grade including grade "
              "1, chondropathy, chondrosis, softening, fissuring, thinning or full "
              "thickness loss?",
        "v2": "Based on this report, is there degenerative change affecting the "
              "patellofemoral joint? Count patellar or trochlear cartilage change and "
              "patellofemoral osteophytes, and also count generalised, global or "
              "tricompartmental osteoarthritis, which by definition includes the "
              "patellofemoral compartment.",
        "v3": "Does this report describe degenerative change involving the patella or "
              "trochlea — retropatellar chondropathy or chondromalacia of any grade, "
              "patellofemoral cartilage loss, patellofemoral osteophytes — OR "
              "tricompartmental or global osteoarthritis?",
        "v4": "Based on this report, would a radiologist reviewing the images likely see "
              "cartilage wear behind the kneecap? Consider that patellofemoral cartilage "
              "change is common alongside osteophytes, meniscal degeneration and "
              "tibiofemoral osteoarthritis, even when the report does not mention the "
              "patella.",
    },
}
assert set(VARIANTS) == set(WEAK)
PAIRS = [(c, v) for c in WEAK for v in VARIANTS[c]]
print(f"{len(PAIRS)} (condition, variant) pairs -> {len(PAIRS) * len(labelled)} prompts")

SYSTEM = (
    "You are an expert musculoskeletal radiologist. You read knee MRI reports written "
    "in any language (English, Spanish, German, Turkish, Russian, Dutch, French, "
    "Italian and others) and answer questions about their findings. Base your answer "
    "only on what the report states or clearly implies. Answer with a single word."
)

# %%
from transformers import AutoConfig, AutoTokenizer

if not MODEL:
    for _cand in MODELS:
        try:
            AutoConfig.from_pretrained(_cand, trust_remote_code=True)
            MODEL = _cand
            break
        except Exception as e:
            print(f"  {_cand} unavailable ({type(e).__name__}: {str(e)[:80]}) — next")
assert MODEL, f"none of {MODELS} resolved on the Hub"
print("teacher candidate:", MODEL)

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def build_prompt(report: str, question: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"Knee MRI report:\n<report>\n{report.strip()}\n</report>\n\n"
            f"Question: {question}\n\nAnswer Yes or No."},
    ]
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


# %%
from transformers import AutoModelForCausalLM


def _param_billions(name: str) -> float:
    import re as _re
    m = _re.search(r"(\d+(?:\.\d+)?)B", name)
    return float(m.group(1)) if m else 4.0


def _load(name):
    b = _param_billions(name)
    dm = {"": 0} if b * 2 < 13 else "auto"
    print(f"loading {name} (~{b:.0f}B, ~{b*2:.0f} GB fp16) device_map={dm}")
    return AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map=dm, trust_remote_code=True), dm


def _probe_forward(mdl) -> None:
    """Loading is not proof of fitting — weights can occupy the card and leave
    nothing for activations. Run a real forward before committing."""
    ids = tok("probe " * 512, return_tensors="pt", truncation=True, max_length=2048)
    dev0 = next(mdl.parameters()).device
    with torch.no_grad():
        mdl(input_ids=ids.input_ids.to(dev0),
            attention_mask=ids.attention_mask.to(dev0))
    torch.cuda.empty_cache()


model = None
for cand in [MODEL] + [m for m in MODELS if m != MODEL]:
    try:
        model, DEVICE_MAP = _load(cand)
        _probe_forward(model)
        print(f"  {cand}: forward probe OK")
        if cand != MODEL:
            MODEL = cand
            tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
        break
    except Exception as e:
        print(f"  {cand} failed ({type(e).__name__}: {str(e)[:140]}) — next")
        model = None
        gc.collect()
        torch.cuda.empty_cache()
assert model is not None, "no candidate teacher loaded"
model.eval()
print("teacher:", MODEL)


# %%
def _first_token_ids(words) -> list:
    ids = set()
    for w in words:
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize(), w.upper()):
            enc = tok.encode(variant, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
    return sorted(ids)


def _logit_kw() -> dict:
    import inspect
    params = inspect.signature(model.forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return {name: 1}
    return {}


LOGIT_KW = _logit_kw()
YES_IDS = _first_token_ids(["yes"])
NO_IDS = _first_token_ids(["no"])
assert YES_IDS and NO_IDS and not (set(YES_IDS) & set(NO_IDS))


@torch.no_grad()
def score_prompts(prompts: list) -> np.ndarray:
    """Logit margin log P(Yes) - log P(No) at the first generated position.

    Margin, not sigmoid: in float32 sigmoid saturates to exactly 1.0 above ~17,
    and AUC scores ties at half credit, which silently capped earlier runs."""
    global BATCH
    out = np.zeros(len(prompts), dtype=np.float32)
    order = np.argsort([len(p) for p in prompts])
    dev = next(model.parameters()).device
    t0, i = time.time(), 0
    while i < len(order):
        idx = order[i: i + BATCH]
        batch = [prompts[j] for j in idx]
        try:
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(dev)
            logits = model(**enc, **LOGIT_KW).logits[:, -1, :].float()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if BATCH == 1:
                raise
            BATCH = max(1, BATCH // 2)
            print(f"  OOM -> batch {BATCH}")
            continue
        y = torch.logsumexp(logits[:, YES_IDS], dim=-1)
        n = torch.logsumexp(logits[:, NO_IDS], dim=-1)
        out[idx] = (y - n).cpu().numpy()
        i += len(idx)
        if i % (BATCH * 10) < BATCH:
            el = time.time() - t0
            print(f"  {i}/{len(order)}  {i/el:.1f} prompt/s  "
                  f"eta {(len(order)-i)/max(i/el,1e-9)/60:.1f} min", flush=True)
    return out


# %%
prompts, index = [], []
for sid, rep in zip(labelled.StudyInstanceUID, labelled.Report):
    rep = rep if isinstance(rep, str) else ""
    for cond, var in PAIRS:
        prompts.append(build_prompt(rep, VARIANTS[cond][var]))
        index.append((sid, cond, var))

t0 = time.time()
margins = score_prompts(prompts)
print(f"scored {len(prompts)} prompts in {(time.time()-t0)/60:.1f} min")

wide = {}
for (sid, cond, var), m in zip(index, margins):
    wide.setdefault(sid, {})[f"{cond}|{var}"] = float(m)
mar = (pd.DataFrame.from_dict(wide, orient="index")
         .rename_axis("StudyInstanceUID").reset_index())
mar.to_csv(f"{OUT}/sweep_v4_margins.csv", index=False)

# %% [markdown]
# ## Results
#
# AUC against the 58 expert labels, per variant. `v0` is v3's live wording, so
# the column to read is `delta` — a variant only earns a relabel if it beats the
# thing already in production.
#
# n=58 is small: the standard error on an AUC here is roughly ±0.06, so a
# variant winning by 0.01 has not shown anything. The bootstrap column gives the
# fraction of resamples where the variant beats v0, which is the honest read on
# whether a gap is real.

# %%
from sklearn.metrics import roc_auc_score

gt = labelled.set_index("StudyInstanceUID")
pr = mar.set_index("StudyInstanceUID").loc[gt.index]
rng = np.random.default_rng(0)
BOOT = 2000

rows = []
for cond in WEAK:
    y = gt[cond].astype(int).to_numpy()
    base = pr[f"{cond}|v0"].to_numpy()
    auc_base = roc_auc_score(y, base)
    for var in VARIANTS[cond]:
        p = pr[f"{cond}|{var}"].to_numpy()
        auc = roc_auc_score(y, p)
        if var == "v0":
            beat = np.nan
        else:
            wins = 0
            for _ in range(BOOT):
                s = rng.integers(0, len(y), len(y))
                if len(np.unique(y[s])) < 2:
                    continue
                wins += roc_auc_score(y[s], p[s]) > roc_auc_score(y[s], base[s])
            beat = wins / BOOT
        rows.append({"condition": cond, "variant": var, "n_pos": int(y.sum()),
                     "auc": round(float(auc), 4),
                     "delta": round(float(auc - auc_base), 4),
                     "beats_v0": None if var == "v0" else round(float(beat), 3)})

res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv(f"{OUT}/sweep_v4_results.csv", index=False)

# %% [markdown]
# ## Best-of ensemble
#
# A second, cheaper lever: average the margins of several variants for the same
# condition. Different phrasings make partly independent errors, so the mean
# often ranks better than any single one — the same reason a fold ensemble
# works. Reported here so the relabel can adopt it if it wins.

# %%
ens_rows = []
for cond in WEAK:
    y = gt[cond].astype(int).to_numpy()
    cols = [f"{cond}|{v}" for v in VARIANTS[cond]]
    z = pr[cols].to_numpy()
    z = (z - z.mean(0)) / (z.std(0) + 1e-6)      # comparable scales before averaging
    auc_all = roc_auc_score(y, z.mean(1))
    aucs = {v: roc_auc_score(y, pr[f"{cond}|{v}"].to_numpy()) for v in VARIANTS[cond]}
    top2 = sorted(aucs, key=aucs.get, reverse=True)[:2]
    idx2 = [list(VARIANTS[cond]).index(v) for v in top2]
    auc_top2 = roc_auc_score(y, z[:, idx2].mean(1))
    ens_rows.append({"condition": cond,
                     "best_single": round(max(aucs.values()), 4),
                     "best_variant": max(aucs, key=aucs.get),
                     "mean_all5": round(float(auc_all), 4),
                     "mean_top2": round(float(auc_top2), 4),
                     "top2": "+".join(top2)})
ens = pd.DataFrame(ens_rows)
print(ens.to_string(index=False))
ens.to_csv(f"{OUT}/sweep_v4_ensemble.csv", index=False)

# %%
v0_mean = float(res[res.variant == "v0"].auc.mean())
best_mean = float(res.groupby("condition").auc.max().mean())
ens_mean = float(ens[["best_single", "mean_all5", "mean_top2"]].max(axis=1).mean())
print(f"\nweak-four mean AUC — v0 {v0_mean:.4f} | best single {best_mean:.4f} "
      f"| best incl. ensembles {ens_mean:.4f}")

json.dump({"model": MODEL, "n_labelled": int(len(labelled)),
           "pairs": len(PAIRS), "minutes": round((time.time() - t0) / 60, 2),
           "v0_mean": v0_mean, "best_single_mean": best_mean,
           "best_any_mean": ens_mean,
           "best_variant": {r.condition: r.best_variant for r in ens.itertuples()}},
          open(f"{OUT}/sweep_v4_meta.json", "w"), indent=2)
print("done")
