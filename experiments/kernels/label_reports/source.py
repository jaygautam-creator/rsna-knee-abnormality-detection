# %% [markdown]
# # RSNA Knee — Phase 1: report -> soft labels
#
# Only 58 of 4,407 training studies carry labels; all 4,407 carry a radiology
# report, in 10+ languages. This notebook turns the reports into 12 calibrated
# per-condition probabilities for every study, producing the training targets
# the image model will actually learn from.
#
# **Method: logit scoring, not JSON generation.** For each (report, condition)
# we ask a yes/no question and read `P(Yes) / (P(Yes) + P(No))` off the first
# generated token. That gives a *calibrated probability* rather than a hard 0/1
# — which is what we want, since the competition metric is ranking-based and
# soft targets carry the model's uncertainty into training. It is also
# unparseable-output-proof and needs only one forward pass per question.
#
# Because we score a single token, this is one forward pass per prompt — no
# generation loop and therefore no need for vLLM. We use plain `transformers`,
# which is preinstalled on the Kaggle image. That is deliberate: `pip install
# vllm` downgrades numpy to 1.x and breaks the preinstalled pandas (which is
# built against numpy 2) with a C-ABI error, before the model even loads.
# No install means no dependency conflict.
#
# Internet must be ON for this notebook (to pull model weights). That is fine —
# this is a training-time artifact producer, not the submission notebook.

# %%
import os

MODEL = os.environ.get("KNEE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
BATCH = 32               # auto-halves on OOM; 4B fp16 leaves ~8 GB on a T4
USE_PREFIX_CACHE = True  # encode each report once instead of 12x (~3x faster)
OUT = "/kaggle/working"

# Set True for a fast wiring check on the 58 labelled studies only.
SMOKE_ONLY = False

# %%
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

print("GPUs:", torch.cuda.device_count(), torch.cuda.get_device_name(0))


def find_comp(max_depth: int = 3) -> Path:
    """Locate the competition mount. Kaggle neither guarantees the directory is
    named after the competition slug nor that it sits at the top level — it
    actually lands at /kaggle/input/competitions/<slug>. Walk a bounded depth
    instead of hardcoding, and never recurse into a directory holding the 820k
    DICOMs."""
    seen = []
    frontier = [(Path("/kaggle/input"), 0)]
    while frontier:
        d, depth = frontier.pop(0)
        if (d / "train.csv").exists() and (d / "train_series.csv").exists():
            return d
        if depth >= max_depth:
            continue
        for child in sorted(p for p in d.iterdir() if p.is_dir()):
            if child.name in ("train_series", "test_series"):
                continue          # hundreds of thousands of files, never the target
            seen.append(str(child))
            frontier.append((child, depth + 1))
    raise FileNotFoundError(f"competition data not found. searched: {seen[:40]}")


COMP = find_comp()
print("competition mount:", COMP)

train = pd.read_csv(f"{COMP}/train.csv")
LABELS = [c for c in train.columns if c not in ("StudyInstanceUID", "Report")]
labelled = train[train[LABELS[0]].notna()].reset_index(drop=True)
print(f"studies {len(train)}  labelled {len(labelled)}  conditions {len(LABELS)}")

# %% [markdown]
# ## Questions
#
# Wording matters more than model size here. Two things are encoded deliberately:
#
# - **Laterality is explicit.** `Medial OA` vs `Lateral OA` are the two
#   conditions the regex baseline scored worst on (recall 0.29 / 0.17), because
#   reports describe compartment disease obliquely ("cartilage thinning over the
#   medial femoral condyle") rather than naming the compartment.
# - **Severity thresholds are stated.** The 58 expert labels are stricter than
#   the report prose — one report says "moderate joint effusion" yet is labelled
#   `Effusion = 0`. Asking for "more than a trace" moves the question toward the
#   experts' threshold instead of the radiologist's phrasing.

# %%
QUESTIONS = {
    "ACL": "Does this report describe an injury to the anterior cruciate ligament (ACL) "
           "— a tear, rupture, sprain, or significant strain? Answer No if the ACL is "
           "described as intact, normal, or unremarkable.",
    "MCL": "Does this report describe an injury to the medial collateral ligament (MCL) "
           "— a tear, rupture, sprain, or significant strain? Answer No if the MCL is "
           "described as intact or normal.",
    "Medial Meniscus": "Does this report describe a tear of the MEDIAL meniscus? Include "
           "any tear morphology (horizontal, radial, oblique, complex, bucket-handle, "
           "flap), maceration, or a previous meniscectomy. Do not count the lateral "
           "meniscus. Answer No for isolated degeneration without a tear.",
    "Lateral Meniscus": "Does this report describe a tear of the LATERAL meniscus? Include "
           "any tear morphology (horizontal, radial, oblique, complex, bucket-handle, "
           "flap), maceration, or a previous meniscectomy. Do not count the medial "
           "meniscus. Answer No for isolated degeneration without a tear.",
    "Medial OA": "Does this report describe osteoarthritis or degenerative change in the "
           "MEDIAL tibiofemoral compartment? Count cartilage loss or thinning over the "
           "medial femoral condyle or medial tibial plateau, medial osteophytes, medial "
           "joint space narrowing, or medial subchondral sclerosis or cysts.",
    "Lateral OA": "Does this report describe osteoarthritis or degenerative change in the "
           "LATERAL tibiofemoral compartment? Count cartilage loss or thinning over the "
           "lateral femoral condyle or lateral tibial plateau, lateral osteophytes, "
           "lateral joint space narrowing, or lateral subchondral sclerosis or cysts.",
    "PF OA": "Does this report describe osteoarthritis or degenerative change in the "
           "PATELLOFEMORAL compartment? Count chondromalacia patellae, cartilage loss, "
           "thinning or ulceration of the patellar or trochlear cartilage, and "
           "patellofemoral osteophytes.",
    "Effusion": "Does this report describe a joint effusion or excess joint fluid that is "
           "MORE THAN a trace or physiologic amount? Answer No for 'trace', 'minimal', "
           "'physiologic', or no effusion.",
    "Synovitis": "Does this report describe synovitis — inflammation, thickening, "
           "hypertrophy or proliferation of the synovium or joint lining, synovial "
           "enhancement, or pannus?",
    "Baker's": "Does this report describe a Baker's cyst (also called a popliteal cyst or "
           "a gastrocnemio-semimembranosus bursal cyst)?",
    "Contusion": "Does this report describe a bone contusion, bone bruise, or bone marrow "
           "oedema pattern from injury? Answer No if marrow signal is normal, and No if "
           "the marrow oedema is purely degenerative or reactive to arthritis.",
    "Fracture": "Does this report describe a fracture? Include osteochondral fractures, "
           "avulsion fractures, insufficiency and stress fractures. Answer No if no "
           "fracture is seen.",
}
assert set(QUESTIONS) == set(LABELS), set(LABELS) ^ set(QUESTIONS)

SYSTEM = (
    "You are an expert musculoskeletal radiologist. You read knee MRI reports written "
    "in any language (English, Spanish, German, Turkish, Russian, Dutch, French, "
    "Italian and others) and answer questions about their findings. Base your answer "
    "only on what the report states or clearly implies. Answer with a single word."
)

# %%
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
# Left padding so the final position of every row is a real token, which is the
# position we read the Yes/No logits from.
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def build_prompt(report: str, condition: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"Knee MRI report:\n<report>\n{report.strip()}\n</report>\n\n"
            f"Question: {QUESTIONS[condition]}\n\nAnswer Yes or No."},
    ]
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def build_all(df: pd.DataFrame):
    prompts, index = [], []
    for sid, rep in zip(df.StudyInstanceUID, df.Report):
        rep = rep if isinstance(rep, str) else ""
        for cond in LABELS:
            prompts.append(build_prompt(rep, cond))
            index.append((sid, cond))
    return prompts, index


print(build_prompt(train.Report.iloc[0], "ACL")[:700])

# %%
from transformers import AutoModelForCausalLM

# Plain fp16, no quantisation. Kaggle allocated a single P100 16 GB (compute
# 6.0), where bitsandbytes 4-bit is both an extra install -- the thing that
# broke this notebook once already by downgrading numpy -- and poorly supported.
# A 4B in fp16 is ~8 GB, fits with room for activations, and runs faster than a
# 14B dequantising on every matmul.
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    # Pin to one GPU. `device_map="auto"` shards a model that already fits
    # (4B fp16 is ~8 GB, a T4 has 16 GB) across both cards, which is *pipeline*
    # parallelism: GPU 1 idles while GPU 0 computes, and every batch pays a
    # cross-device transfer. On v9 that capped us at 2.6 prompt/s.
    device_map={"": 0},
    trust_remote_code=True,
)
model.eval()
print(model.hf_device_map if hasattr(model, "hf_device_map") else "single device")

# %%
def _first_token_ids(words) -> list:
    """Token ids that a completion could start with for each surface form."""
    ids = set()
    for w in words:
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize(), w.upper()):
            enc = tok.encode(variant, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
    return sorted(ids)


# transformers renamed this arg; probe once so we fail here rather than per batch.
def _logit_kw() -> dict:
    import inspect
    params = inspect.signature(model.forward).parameters
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return {name: 1}
    print("WARNING: no logits_to_keep support; full logits will be materialised")
    return {}


LOGIT_KW = _logit_kw()
print("logit kwarg:", LOGIT_KW)

YES_IDS = _first_token_ids(["yes"])
NO_IDS = _first_token_ids(["no"])
assert YES_IDS and NO_IDS and not (set(YES_IDS) & set(NO_IDS))
print("yes ids", YES_IDS, "no ids", NO_IDS)


@torch.no_grad()
def score_prompts(prompts: list) -> np.ndarray:
    """Return the **logit margin** log P(Yes) - log P(No) at the first generated
    position. One forward pass per prompt, no generation.

    We deliberately return the margin rather than `sigmoid(margin)`. Sigmoid is
    monotonic, so in exact arithmetic it preserves ranking -- but in float32 it
    saturates to exactly 1.0 for any margin above ~17, which on v9 hit 32.6% of
    all predictions. AUC scores ties at half credit, so those collapsed values
    were capping the measured score (Effusion had just 11 distinct values across
    58 studies, and was our worst condition). The margin keeps full resolution;
    probabilities are derived from it afterwards for use as training targets."""
    global BATCH
    out = np.zeros(len(prompts), dtype=np.float32)
    # Sort by length so each batch pads to a similar width; padding on a batch
    # of mixed 300- and 2000-token prompts otherwise wastes most of the compute.
    order = np.argsort([len(p) for p in prompts])
    dev = next(model.parameters()).device
    t0, done = time.time(), 0
    i = 0
    while i < len(order):
        idx = order[i: i + BATCH]
        batch = [prompts[j] for j in idx]
        try:
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(dev)
            # Only the last position is ever read. Without this the model
            # materialises [batch, seq_len, vocab] logits -- at batch 8 x 2048
            # tokens x 152k vocab that is ~5 GB of fp16 we would immediately
            # discard, on a card that also holds the weights.
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
        done += len(idx)
        if done % (BATCH * 40) < BATCH:
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(prompts)}  {rate:.1f} prompt/s  "
                  f"eta {(len(prompts) - done) / rate / 60:.1f} min", flush=True)
    return out


@torch.no_grad()
def score_prompts_shared(df: pd.DataFrame) -> np.ndarray | None:
    """Same margins as `score_prompts`, but encoding each report once.

    All 12 questions for a study share the system+report prefix (~410 tokens)
    and differ only in a ~140-token suffix. Scoring them independently re-encodes
    that prefix 12 times: 12 x 550 = 6,600 tokens per study versus 410 + 12 x 140
    = 2,090 here, ~3x less prefill. Prefill is the entire cost since we generate
    nothing.

    Returns None if the installed transformers Cache API does not behave as
    expected, so the caller can fall back rather than fail a multi-hour run.
    """
    from transformers import DynamicCache

    dev = next(model.parameters()).device
    n_cond = len(LABELS)
    out = np.zeros(len(df) * n_cond, dtype=np.float32)
    t0 = time.time()

    for i, (sid, rep) in enumerate(zip(df.StudyInstanceUID, df.Report)):
        rep = rep if isinstance(rep, str) else ""
        prefix, suffixes = split_prompt(rep)

        pre_ids = tok(prefix, return_tensors="pt", truncation=True,
                      max_length=2048).to(dev)
        pre_out = model(**pre_ids, use_cache=True)
        pre_len = pre_ids.input_ids.shape[1]

        # Broadcast the single prefix cache across the 12 question rows.
        # `repeat` not `expand`: the next forward appends to these tensors and
        # needs real storage, not a stride-0 view.
        try:
            cache = DynamicCache()
            for layer, (k, v) in enumerate(zip(pre_out.past_key_values.key_cache,
                                               pre_out.past_key_values.value_cache)):
                cache.update(k.repeat(n_cond, 1, 1, 1),
                             v.repeat(n_cond, 1, 1, 1), layer)
        except (AttributeError, TypeError) as e:
            print(f"  prefix-cache unsupported ({e}); falling back")
            return None

        tok.padding_side = "right"      # suffixes continue the prefix
        suf = tok(suffixes, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev)
        tok.padding_side = "left"

        attn = torch.cat(
            [torch.ones(n_cond, pre_len, dtype=suf.attention_mask.dtype, device=dev),
             suf.attention_mask], dim=1)
        logits = model(input_ids=suf.input_ids, attention_mask=attn,
                       past_key_values=cache, **LOGIT_KW).logits

        # Right padding means the last real token differs per row.
        last = suf.attention_mask.sum(1) - 1
        if logits.shape[1] == 1:
            row = logits[:, 0, :].float()
        else:
            row = logits[torch.arange(n_cond, device=dev), last].float()
        y = torch.logsumexp(row[:, YES_IDS], dim=-1)
        n = torch.logsumexp(row[:, NO_IDS], dim=-1)
        out[i * n_cond:(i + 1) * n_cond] = (y - n).cpu().numpy()

        if i and i % 200 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{len(df)} studies  {rate:.1f} study/s  "
                  f"eta {(len(df) - i) / rate / 60:.1f} min", flush=True)
    return out


def split_prompt(report: str):
    """(shared prefix, [12 question suffixes]) — the concatenation of prefix and
    any suffix must equal the corresponding `build_prompt` output exactly."""
    full = [build_prompt(report, c) for c in LABELS]
    common = os.path.commonprefix(full)
    # Cut back to a token-safe boundary so prefix+suffix retokenises identically.
    cut = common.rfind("\n") + 1
    return common[:cut], [f[cut:] for f in full]


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Wide frame of logit margins, one row per study, one column per condition."""
    prompts, index = build_all(df)
    print(f"{len(df)} studies -> {len(prompts)} prompts")
    margins = None
    if USE_PREFIX_CACHE:
        margins = score_prompts_shared(df)
    if margins is None:
        margins = score_prompts(prompts)
    wide = {}
    for (sid, cond), m in zip(index, margins):
        wide.setdefault(sid, {})[cond] = float(m)
    return (pd.DataFrame.from_dict(wide, orient="index")[LABELS]
              .rename_axis("StudyInstanceUID").reset_index())


def to_prob(margins: pd.DataFrame, temperature: float = 1.0) -> pd.DataFrame:
    """Margins -> probabilities in float64, for use as BCE training targets.
    Ranking (and therefore AUC) is unaffected; only the target values change."""
    out = margins.copy()
    m = out[LABELS].to_numpy(dtype=np.float64) / temperature
    out[LABELS] = 1.0 / (1.0 + np.exp(-m))
    return out

# %% [markdown]
# ## Correctness check: prefix-cache path must match the plain path
#
# The fast path hand-manages the KV cache and splits a prompt into prefix +
# suffix, which is only valid if the two retokenise to the same ids and the
# broadcast cache is assembled correctly. Verify on a few studies *before* the
# long run — a silent mismatch here would corrupt all 4,407 labels while looking
# like a modelling problem later.

# %%
if USE_PREFIX_CACHE:
    _chk = labelled.head(3)

    # 1. prefix + suffix must retokenise identically to the whole prompt.
    _bad = 0
    for _rep in _chk.Report:
        _pre, _sufs = split_prompt(_rep)
        for _c, _s in zip(LABELS, _sufs):
            whole = tok(build_prompt(_rep, _c)).input_ids
            split = tok(_pre).input_ids + tok(_s, add_special_tokens=False).input_ids
            _bad += whole != split
    print(f"retokenisation mismatches: {_bad}/{len(_chk) * len(LABELS)}")

    # 2. the margins themselves must agree.
    _fast = score_prompts_shared(_chk)
    _slow = score_prompts([p for p in build_all(_chk)[0]])
    if _fast is None:
        print("prefix cache unavailable — will use the plain path")
        USE_PREFIX_CACHE = False
    else:
        _d = np.abs(_fast - _slow)
        print(f"margin |diff|: max {_d.max():.4f}  mean {_d.mean():.4f}")
        print(f"rank agreement: "
              f"{(np.sign(_fast) == np.sign(_slow)).mean():.3f} same sign")
        if _d.max() > 0.5:
            print("MISMATCH — disabling prefix cache, using the plain path")
            USE_PREFIX_CACHE = False
        else:
            print("prefix cache verified")

# %% [markdown]
# ## Gate: validate on the 58 before spending the GPU budget
#
# If mean AUC here is not clearly above the 0.818 regex baseline, stop and fix
# the prompt rather than labelling all 4,407 studies.

# %%
from sklearn.metrics import roc_auc_score

_t = time.time()
val = score(labelled)
_rate = len(labelled) * len(LABELS) / (time.time() - _t)
_eta = len(train) * len(LABELS) / _rate / 3600
print(f"\nthroughput {_rate:.1f} prompt/s -> full run ~{_eta:.1f} h "
      f"(submission cap is 9 h)")
gt = labelled.set_index("StudyInstanceUID")[LABELS].astype(int)
pr = val.set_index("StudyInstanceUID")[LABELS].loc[gt.index]

rows = []
for c in LABELS:
    auc = roc_auc_score(gt[c], pr[c]) if gt[c].nunique() > 1 else np.nan
    acc = ((pr[c] > 0).astype(int) == gt[c]).mean()   # margin > 0  <=>  p > 0.5
    rows.append((c, int(gt[c].sum()), round(auc, 3), round(acc, 3)))

report = pd.DataFrame(rows, columns=["condition", "n_pos", "auc", "acc@0.5"])
print(report.to_string(index=False))
print(f"\nMEAN AUC {report.auc.mean():.4f}   MEAN ACC {report['acc@0.5'].mean():.4f}")
print("regex baseline (English subset) was 0.818 acc")

val.to_csv(f"{OUT}/val_labelled_58_margin.csv", index=False)
to_prob(val).to_csv(f"{OUT}/val_labelled_58.csv", index=False)
_sat = float((to_prob(val)[LABELS].to_numpy() == 1.0).mean())
print(f"saturated-at-1.0 cells after fix: {_sat:.1%} (was 32.6% on v9)")

# Fail loudly rather than spending hours of GPU quota producing bad labels.
#
# Compare on AUC, which is the competition metric, and against the regex scored
# on *these same 58 studies* -- 0.6287. (The regex's often-quoted 0.818 is
# accuracy on the 28 English studies only; comparing that to AUC over all 58
# languages is two mismatches at once and wrongly failed this gate on v9.)
# Measured v9 with Qwen3-4B: 0.8448.
assert report.auc.mean() > 0.80, (
    f"mean AUC {report.auc.mean():.4f} — regex on these same 58 studies is "
    f"0.6287. Fix the prompts before labelling all 4,407 studies."
)

# %% [markdown]
# ## Full run

# %%
if _eta > 8:
    # Kaggle kills the session well before this notebook would finish. Stop with
    # the gate results in hand rather than being killed having written nothing.
    raise RuntimeError(
        f"projected {_eta:.1f} h is too long. Reduce the model, raise BATCH, or "
        f"reuse the report KV cache across the 12 questions (~3.5x saving)."
    )

if not SMOKE_ONLY:
    full = score(train)
    full.to_csv(f"{OUT}/train_margins_v1.csv", index=False)
    probs = to_prob(full)
    probs.to_csv(f"{OUT}/train_labels_v1.csv", index=False)
    print(full.shape)
    print(probs[LABELS].describe().T[["mean", "50%"]])

    meta = {
        "model": MODEL,
        "method": "first-token yes/no logit scoring",
        "n_studies": int(len(full)),
        "val_mean_auc": float(report.auc.mean()),
        "outputs": {"train_labels_v1.csv": "probabilities (BCE targets)",
                    "train_margins_v1.csv": "logit margins (ranking, full precision)"},
        "val_per_condition": report.to_dict("records"),
    }
    with open(f"{OUT}/labels_v1_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta["val_per_condition"], indent=1)[:400])
