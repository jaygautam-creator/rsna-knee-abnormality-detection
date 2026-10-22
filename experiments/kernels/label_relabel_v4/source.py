# %% [markdown]
# # v4 — relabel the four weak conditions across all 4,407 studies
#
# The sweep (`knee-label-sweep-v4`) scored 5 phrasings per weak condition
# against the 58 expert labels. This notebook re-scores the whole corpus with
# the winners and emits replacement columns for Synovitis, Contusion, Effusion
# and PF OA. The other eight conditions keep their v3 values — they score
# 0.90-0.95 against gold and there is nothing to gain by re-running them at
# 3x the GPU cost.
#
# `BEST` is set by hand from the sweep table rather than by argmax. At n=58 the
# SE on an AUC is about ±0.06, so picking the top row automatically would often
# be picking noise; a variant is adopted only if the bootstrap says it beats v0
# in a clear majority of resamples. Where two variants are close and phrased
# differently, averaging their margins is preferred to choosing between them.

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

# Shard across kernels when the projected runtime approaches the session cap.
# The notebook prints its own ETA after the first few hundred prompts, so this
# is a decision made on measured throughput, not a guess.
SHARD = int(os.environ.get("KNEE_SHARD", "0"))
N_SHARDS = int(os.environ.get("KNEE_N_SHARDS", "1"))

print("GPUs:", torch.cuda.device_count(), torch.cuda.get_device_name(0))
print(f"shard {SHARD + 1}/{N_SHARDS}")


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
train = pd.read_csv(f"{COMP}/train.csv")
ALL_LABELS = [c for c in train.columns if c not in ("StudyInstanceUID", "Report")]
labelled = train[train[ALL_LABELS[0]].notna()].reset_index(drop=True)
WEAK = ["Synovitis", "Contusion", "Effusion", "PF OA"]
print(f"studies {len(train)}  labelled {len(labelled)}")

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

# %% [markdown]
# ## What to score, and what not to
#
# Chosen from the sweep by bootstrap confidence, not by argmax over AUC. With 25
# combinations tested per condition on 58 studies, the top row is very often the
# luckiest row; a combination is adopted only where it beats v0 in a clear
# majority of resamples.
#
# | condition | adopted | AUC vs gold | beats v0 |
# |---|---|---|---|
# | Synovitis | v0+v2 | 0.7467 → 0.7849 | 0.955 |
# | Contusion | v0+v2+v4 | 0.8097 → 0.8300 | 0.775 |
# | Effusion | v0+v4 | 0.8106 → 0.8398 | 0.810 |
# | PF OA | **v0 unchanged** | 0.8533 | best alternative only 0.649 |
#
# PF OA is left alone deliberately. No rewording beat it above coin-flip
# confidence, which is what the earlier keyword analysis predicted: the
# patellofemoral finding is usually absent from the report, so the failure is
# missing information rather than a badly posed question. Re-scoring it would
# spend a quarter of the GPU budget to buy noise.
#
# **v0 is never re-scored.** Every v0 margin for all 4,407 studies already exists
# in `artifacts/train_margins_v3.csv`, from the same Qwen3-8B teacher — verified
# against this sweep's own v0 columns at corr 1.0000, max |diff| 0.19 (float
# noise from different batch padding). So this kernel scores only the *new*
# variants and the local merge combines them with v3's v0. That drops the run
# from 6 prompts per study to 4: ~4.2 h instead of ~6.3 h, which matters when
# the whole week's quota is 30 h.

# %%
NEW = {
    "Synovitis": ["v2"],
    "Contusion": ["v2", "v4"],
    "Effusion": ["v4"],
}
# How src/labels/merge_v4.py combines them afterwards; recorded here so the two
# halves of the decision live in one place.
COMBINE = {
    "Synovitis": ["v0", "v2"],
    "Contusion": ["v0", "v2", "v4"],
    "Effusion": ["v0", "v4"],
    "PF OA": ["v0"],
}
assert set(COMBINE) == set(WEAK)
for _c, _vs in COMBINE.items():
    assert all(v in VARIANTS[_c] for v in _vs), (_c, _vs)
    assert [v for v in _vs if v != "v0"] == NEW.get(_c, []), (_c, _vs)
PAIRS = [(c, v) for c in NEW for v in NEW[c]]
print("scoring:", {c: "+".join(v) for c, v in NEW.items()})
print("will combine as:", {c: "+".join(v) for c, v in COMBINE.items()})

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
    nothing for activations, which is how the 14B attempt died."""
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

    Margin, not sigmoid: float32 sigmoid saturates to exactly 1.0 above ~17 and
    AUC scores ties at half credit, which silently capped earlier runs."""
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
        if i % (BATCH * 20) < BATCH:
            el = time.time() - t0
            print(f"  {i}/{len(order)}  {i/el:.2f} prompt/s  "
                  f"eta {(len(order)-i)/max(i/el,1e-9)/60:.1f} min", flush=True)
    return out


# %% [markdown]
# ## Score the corpus
#
# Studies are split by position, not by hash, so the shards are contiguous and
# a rerun of shard k covers exactly the same studies.

# %%
mine = train.iloc[SHARD::N_SHARDS].reset_index(drop=True)
print(f"shard covers {len(mine)}/{len(train)} studies, "
      f"{len(mine) * len(PAIRS)} prompts")

prompts, index = [], []
for sid, rep in zip(mine.StudyInstanceUID, mine.Report):
    rep = rep if isinstance(rep, str) else ""
    for cond, var in PAIRS:
        prompts.append(build_prompt(rep, VARIANTS[cond][var]))
        index.append((sid, cond, var))

t0 = time.time()
margins = score_prompts(prompts)
mins = (time.time() - t0) / 60
print(f"scored {len(prompts)} prompts in {mins:.1f} min")

wide = {}
for (sid, cond, var), m in zip(index, margins):
    wide.setdefault(sid, {})[f"{cond}|{var}"] = float(m)
raw = (pd.DataFrame.from_dict(wide, orient="index")
         .rename_axis("StudyInstanceUID").reset_index())
raw.to_csv(f"{OUT}/relabel_v4_raw_shard{SHARD}.csv", index=False)

# %% [markdown]
# ## Per-variant sanity check on the 58
#
# The kernel deliberately does **not** combine variants. Combining needs v3's v0
# column, which lives locally, and z-scoring has to happen across the whole
# corpus rather than per shard — two shards standardised against their own means
# would splice into one inconsistent column. `src/labels/merge_v4.py` does it.
#
# What is worth checking here is that each new variant reproduces the AUC the
# sweep measured for it. A large drift would mean the corpus run is not scoring
# what the sweep scored, and the whole relabel is suspect before we spend 8 h
# training on it.

# %%
from sklearn.metrics import roc_auc_score

SWEEP_AUC = {("Synovitis", "v2"): 0.7742, ("Contusion", "v2"): 0.8225,
             ("Contusion", "v4"): 0.8070, ("Effusion", "v4"): 0.8236}
gold = labelled[labelled.StudyInstanceUID.isin(raw.StudyInstanceUID)]
rows = []
if len(gold) >= 20:
    m = gold[["StudyInstanceUID"] + list(NEW)].merge(raw, on="StudyInstanceUID")
    for cond, var in PAIRS:
        y = m[cond].astype(int).to_numpy()
        p = m[f"{cond}|{var}"].to_numpy()
        auc = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
        rows.append({"condition": cond, "variant": var, "auc": round(auc, 4),
                     "sweep_auc": SWEEP_AUC.get((cond, var)),
                     "drift": round(auc - SWEEP_AUC.get((cond, var), auc), 4)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\ndrift should be ~0.00 — this shard holds all 58 gold studies only "
          "when N_SHARDS == 1")
else:
    print(f"only {len(gold)} gold studies in this shard — check after merge")

json.dump({"model": MODEL, "shard": SHARD, "n_shards": N_SHARDS,
           "scored": NEW, "combine": COMBINE, "n_studies": int(len(mine)),
           "n_prompts": len(prompts), "minutes": round(mins, 2), "check": rows},
          open(f"{OUT}/relabel_v4_meta_shard{SHARD}.json", "w"), indent=2)
print("done")
