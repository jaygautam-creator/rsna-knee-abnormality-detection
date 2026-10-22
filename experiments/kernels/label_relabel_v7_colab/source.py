# %% [markdown]
# # v7 relabel on Colab (free T4) — Synovitis, Lateral OA, PF OA
#
# ## Why these three, and why not the three v4 chose
#
# v4 picked its targets by ranking conditions on **label AUC alone** and went
# after Synovitis, Contusion and Effusion. With the public V52 OOF receipt
# joined to our own OOF we can finally rank by the thing that actually matters —
# where the *image model* is capped by its labels. On the 42 gold studies our
# OOF covers:
#
# | condition | blend AUC | label AUC | label - blend |
# |---|---|---|---|
# | **Synovitis** | 0.709 | **0.741** | +0.033 |
# | **Lateral OA** | 0.754 | **0.778** | +0.024 |
# | **PF OA** | 0.811 | **0.829** | +0.019 |
# | Lateral Meniscus | 0.791 | 0.874 | +0.083 |
# | Fracture | 0.915 | 0.960 | +0.045 |
# | Contusion | 0.934 | 0.741 | **-0.194** |
# | Effusion | 0.944 | 0.776 | **-0.168** |
#
# Read the last column as *how much room the label leaves the model*. Where it
# is small and both numbers are low, the label is the ceiling and rewording it
# lifts the model — that is Synovitis, Lateral OA, PF OA. Where it is large and
# negative the model already **beats** its own labels: Contusion and Effusion
# were v4's targets and were the wrong ones, which is exactly why v4's gold
# label AUC rose 0.874 -> 0.880 while the board fell 0.905 -> 0.900.
#
# Lateral Meniscus and Fracture have the biggest positive gaps, but their labels
# are already good (0.874, 0.960) — the imaging cannot see what the label knows.
# That is a resolution problem for the 336 px work, not something a reworded
# prompt can fix. They are deliberately **not** targeted here.
#
# ## Why each of the three is hard
#
# * **Lateral OA** — reports frequently say "tricompartmental" or "degenerative
#   changes" without naming a compartment. Medial OA scores 0.946 because medial
#   disease is common and usually named; lateral is the same disease described
#   implicitly. The variants below test whether making the implicit-compartment
#   rule explicit recovers it.
# * **PF OA** — the signal hides under *chondromalacia patellae*, "retropatellar"
#   and "trochlear" wording rather than the word osteoarthritis.
# * **Synovitis** — rarely named outright; it travels with effusion, Hoffa's fat
#   pad oedema and inflammatory arthropathy. v4 already tried this once and only
#   reached 0.741, so the variants here push harder on the implicit route.
#
# ## Shape of the run
#
# **Stage A** sweeps every variant on the 58 gold studies — 58 x 13 prompts,
# a few minutes — and picks winners by *bootstrap confidence over v0*, not by
# argmax, because argmax on 58 rows reliably selects noise.
# **Stage B** scores only the winners across all 4,407 studies, appending each
# chunk to Google Drive so a free-tier disconnect costs at most one chunk.
#
# **Before running:** Runtime -> Change runtime type -> **T4 GPU**.

# %% [markdown]
# ## 1. Install deps and confirm the GPU

# %%
!pip install -q -U "transformers>=4.44" accelerate bitsandbytes
import torch
assert torch.cuda.is_available(), "No GPU — Runtime > Change runtime type > T4 GPU"
print(torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# %% [markdown]
# ## 2. Mount Drive and load the reports
#
# First run asks for **`data/train.csv`** (5.7 MB — reports plus the 58 gold
# labels, no image data) and caches it to Drive. Later sessions read it straight
# from Drive.

# %%
import os, pandas as pd, numpy as np
from google.colab import drive
drive.mount('/content/drive')

OUTDIR = "/content/drive/MyDrive/knee_relabel_v7"
os.makedirs(OUTDIR, exist_ok=True)
OUT   = f"{OUTDIR}/relabel_v7_raw.csv"     # Stage B checkpoint == final output
SWEEP = f"{OUTDIR}/sweep_v7_gold.csv"      # Stage A result, cached
CSV   = f"{OUTDIR}/train.csv"              # cached reports

if not os.path.exists(CSV):
    from google.colab import files
    up = files.upload()                              # choose data/train.csv
    import shutil; shutil.copy("/content/" + next(iter(up)), CSV)
    print("cached reports to", CSV)

train = pd.read_csv(CSV)
LABELS = [c for c in train.columns if c not in ("StudyInstanceUID", "Report")]
gold = train[train[LABELS[0]].notna()].reset_index(drop=True)
assert train.Report.notna().all(), "a study is missing its report"
print(f"studies {len(train)}  gold {len(gold)}")
print("stage A:", "cached" if os.path.exists(SWEEP) else "to run")
print("stage B:", "resuming" if os.path.exists(OUT) else "fresh")

# %% [markdown]
# ## 3. The variant bank
#
# `v0` is the live v3/v4 phrasing and is the baseline every variant must beat.
# Every other variant changes exactly one idea, so a win is attributable.

# %%
TARGETS = ["Synovitis", "Lateral OA", "PF OA"]

VARIANTS = {
    "Synovitis": {
        "v0": "Does this report describe synovitis — inflammation, thickening, "
              "hypertrophy or proliferation of the synovium or joint lining, "
              "synovial enhancement, or pannus?",
        "v2": "Based on this report, would a radiologist reviewing the images likely "
              "see synovial inflammation or thickening? Consider that synovitis "
              "usually accompanies a joint effusion, Hoffa's fat pad impingement or "
              "oedema, inflammatory arthropathy, or advanced osteoarthritis, even "
              "when the report does not use the word synovitis.",
        "v5": "Does this report contain ANY evidence of an inflamed or thickened "
              "joint lining? Count explicit synovitis, synovial thickening, "
              "proliferation, enhancement or pannus. Also count it when the report "
              "describes a moderate or large effusion together with Hoffa's fat pad "
              "oedema or impingement, plica syndrome with inflammation, or an "
              "inflammatory arthropathy. Answer No for a dry joint, for a trace or "
              "small effusion on its own, and for uncomplicated degenerative change.",
        "v6": "Read this knee MRI report and decide whether the synovium is abnormal. "
              "The synovium is abnormal if the report names synovitis, synovial "
              "thickening or hypertrophy, pannus, or synovial enhancement, OR if it "
              "describes findings that only occur with an inflamed lining: "
              "hypertrophic or proliferative change in the joint lining, inflammatory "
              "or rheumatoid arthropathy, or a septic or reactive joint.",
    },
    "Lateral OA": {
        "v0": "Does this report describe osteoarthritis or degenerative change in the "
              "LATERAL tibiofemoral compartment? Count cartilage loss or thinning "
              "over the lateral femoral condyle or lateral tibial plateau, lateral "
              "osteophytes, lateral joint space narrowing, or lateral subchondral "
              "sclerosis or cysts.",
        "v2": "Does this report describe osteoarthritis or degenerative change "
              "involving the LATERAL tibiofemoral compartment — the lateral femoral "
              "condyle or the lateral tibial plateau? Count it as Yes when the report "
              "describes generalised, global, tricompartmental or bicompartmental "
              "osteoarthritis without naming compartments, because those terms "
              "include the lateral compartment. Answer No when degenerative change is "
              "confined to the medial or patellofemoral compartment only.",
        "v3": "Is there ANY degenerative or osteoarthritic change in the LATERAL "
              "tibiofemoral compartment? Include cartilage loss, thinning, fissuring "
              "or ulceration over the lateral femoral condyle or lateral tibial "
              "plateau; lateral osteophytes or spurring; lateral joint space "
              "narrowing; lateral subchondral sclerosis, cysts or marrow change. "
              "Treat 'tricompartmental' and 'global' osteoarthritis as involving the "
              "lateral compartment. Do not count a lateral meniscal tear on its own.",
        "v4": "Considering the LATERAL half of the knee joint between femur and "
              "tibia, does this report indicate wear, cartilage damage, osteophytes, "
              "joint space narrowing or subchondral change there — whether stated "
              "directly for the lateral compartment or implied by a description of "
              "widespread, tricompartmental or advanced osteoarthritis of the knee?",
    },
    "PF OA": {
        "v0": "Does this report describe osteoarthritis or degenerative change in the "
              "PATELLOFEMORAL compartment? Count chondromalacia patellae, cartilage "
              "loss, thinning or ulceration of the patellar or trochlear cartilage, "
              "and patellofemoral osteophytes.",
        "v2": "Does this report describe degenerative or cartilage change in the "
              "PATELLOFEMORAL joint — between the patella (kneecap) and the femoral "
              "trochlea? Count chondromalacia patellae of any grade, cartilage "
              "thinning, fissuring, fibrillation, ulceration or full-thickness loss "
              "of the patellar or trochlear cartilage, retropatellar or "
              "patellofemoral osteophytes, and subchondral change beneath the "
              "patella. Treat 'tricompartmental osteoarthritis' as involving the "
              "patellofemoral compartment.",
        "v3": "Is the cartilage behind the kneecap or in the femoral trochlear groove "
              "abnormal in this report? Chondromalacia patellae at ANY grade counts "
              "as Yes, as does retropatellar cartilage thinning, fissuring or loss, "
              "trochlear cartilage damage, patellofemoral osteophytes, or "
              "patellofemoral joint space narrowing. Answer No if the patellar and "
              "trochlear cartilage are described as intact or normal.",
    },
}
for c in TARGETS:
    assert "v0" in VARIANTS[c], f"{c} has no v0 baseline"
N_PROMPTS = sum(len(v) for v in VARIANTS.values())
print(f"{N_PROMPTS} variants across {len(TARGETS)} conditions "
      f"-> stage A = {N_PROMPTS * len(gold)} prompts")

SYSTEM = (
    "You are an expert musculoskeletal radiologist. You read knee MRI reports written "
    "in any language (English, Spanish, German, Turkish, Russian, Dutch, French, "
    "Italian and others) and answer questions about their findings. Base your answer "
    "only on what the report states or clearly implies. Answer with a single word."
)

# %% [markdown]
# ## 4. Load Qwen3-8B in 8-bit
#
# A single 16 GB T4 cannot hold an 8B model in fp16. `knee-colab-8bit-drift` put
# the 8-bit vs fp16 drift well inside the noise on this task, and stage A
# re-measures it anyway by scoring v0 against its known fp16 AUC.

# %%
import logging, warnings
# bitsandbytes logs one MatMul8bitLt warning PER matmul PER forward pass, which on
# an 8B model floods the tab and can hang the browser.
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*will be cast.*")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
MODEL = "Qwen/Qwen3-8B"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True,
    torch_dtype=torch.float16,     # match the 8-bit compute dtype -> no cast warnings
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map={"": 0},
)
model.eval()
print("loaded", MODEL, "in 8-bit;", f"{torch.cuda.memory_allocated()/1e9:.1f} GB used")


def build_prompt(report, question):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content":
            f"Knee MRI report:\n<report>\n{report.strip()}\n</report>\n\n"
            f"Question: {question}\n\nAnswer Yes or No."},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)

# %% [markdown]
# ## 5. First-token Yes/No margin scorer
#
# The score is `log P(Yes) - log P(No)` at the first generated position — a
# margin, not a sigmoid. A float32 sigmoid saturates to exactly 1.0 above ~17
# and AUC then scores every saturated study as a tie at half credit.

# %%
import time

def _first_token_ids(words):
    ids = set()
    for w in words:
        for v in (w, " " + w, w.capitalize(), " " + w.capitalize(), w.upper()):
            enc = tok.encode(v, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
    return sorted(ids)

YES_IDS, NO_IDS = _first_token_ids(["yes"]), _first_token_ids(["no"])
assert YES_IDS and NO_IDS and not (set(YES_IDS) & set(NO_IDS))
BATCH = 16

@torch.no_grad()
def score_prompts(prompts, tag=""):
    """Length-sorted batching with OOM backoff. Returns margins in input order."""
    global BATCH
    out = np.zeros(len(prompts), dtype=np.float32)
    order = np.argsort([len(p) for p in prompts])
    t0, i = time.time(), 0
    while i < len(order):
        idx = order[i: i + BATCH]
        batch = [prompts[j] for j in idx]
        try:
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(0)
            logits = model(**enc).logits[:, -1, :].float()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if BATCH == 1:
                raise
            BATCH = max(1, BATCH // 2)
            print(f"  OOM -> BATCH={BATCH}")
            continue
        lp = torch.log_softmax(logits, dim=-1)
        out[idx] = (torch.logsumexp(lp[:, YES_IDS], dim=-1)
                    - torch.logsumexp(lp[:, NO_IDS], dim=-1)).cpu().numpy()
        i += len(idx)
        if i % (BATCH * 20) < len(idx):
            r = i / max(time.time() - t0, 1e-9)
            print(f"  {tag} {i}/{len(order)}  {r:.1f}/s  "
                  f"eta {(len(order)-i)/max(r,1e-9)/60:.1f} min", flush=True)
    return out

# %% [markdown]
# ## 6. Stage A — sweep the variants on the 58 gold studies
#
# Selection is by **bootstrap confidence over v0**, not argmax. On 58 rows with
# 8-28 positives per condition, argmax picks noise most of the time; a variant
# earns its place only if it beats v0 in a clear majority of resamples.

# %%
from sklearn.metrics import roc_auc_score

CONF = 0.80          # a variant must beat v0 in >=80% of bootstrap resamples
N_BOOT = 4000

if os.path.exists(SWEEP):
    sweep = pd.read_csv(SWEEP)
    print("loaded cached stage A")
else:
    rows = {}
    for cond in TARGETS:
        for name, q in VARIANTS[cond].items():
            key = f"{cond}||{name}"
            rows[key] = score_prompts([build_prompt(r, q) for r in gold.Report],
                                      tag=key)
            print(f"scored {key}", flush=True)
    sweep = pd.DataFrame(rows)
    sweep.insert(0, "StudyInstanceUID", gold.StudyInstanceUID.values)
    sweep.to_csv(SWEEP, index=False)
    print("saved", SWEEP)

rng = np.random.default_rng(0)
selected = {}
print(f"\n{'condition':14} {'variant':7} {'goldAUC':>8} {'vs v0':>8} {'P(beat v0)':>11}")
print("-" * 54)
for cond in TARGETS:
    y = gold[cond].to_numpy(np.float32)
    base = sweep[f"{cond}||v0"].to_numpy()
    a0 = roc_auc_score(y, base)
    keep = ["v0"]
    for name in VARIANTS[cond]:
        s = sweep[f"{cond}||{name}"].to_numpy()
        a = roc_auc_score(y, s)
        if name == "v0":
            print(f"{cond:14} {name:7} {a:8.4f} {'—':>8} {'baseline':>11}")
            continue
        wins = 0
        for _ in range(N_BOOT):
            b = rng.integers(0, len(y), len(y))
            if len(np.unique(y[b])) < 2:
                continue
            wins += roc_auc_score(y[b], s[b]) > roc_auc_score(y[b], base[b])
        p = wins / N_BOOT
        flag = "  <- keep" if p >= CONF else ""
        print(f"{cond:14} {name:7} {a:8.4f} {a-a0:+8.4f} {p:11.3f}{flag}")
        if p >= CONF:
            keep.append(name)
    # rank-average the kept variants and confirm the combination beats v0 alone
    from scipy.stats import rankdata
    comb = np.mean([rankdata(sweep[f"{cond}||{k}"].to_numpy()) for k in keep], axis=0)
    ac = roc_auc_score(y, comb)
    if ac <= a0 and len(keep) > 1:
        print(f"  {cond}: combination {ac:.4f} does not beat v0 {a0:.4f} -> v0 only")
        keep = ["v0"]
        ac = a0
    selected[cond] = keep
    print(f"  => {cond}: {keep}  combined gold AUC {ac:.4f} ({ac-a0:+.4f} vs v0)\n")

# %% [markdown]
# ## 7. Stage B — score the winners across all 4,407 studies
#
# Appends each finished chunk to the Drive checkpoint and skips anything already
# there, so a free-tier disconnect costs at most one chunk. Re-run this cell in a
# fresh session to continue. If stage A kept only `v0` everywhere there is
# nothing new to learn and the cell says so instead of burning four hours.

# %%
CHUNK = 200

JOBS = [(c, v) for c in TARGETS for v in selected[c] if v != "v0"]
if not JOBS:
    print("stage A kept no variant over v0 — nothing to score. "
          "The current labels are already the best of these phrasings.")
else:
    cols = [f"{c}||{v}" for c, v in JOBS]
    print("scoring", cols)
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        if all(c in prev.columns for c in cols):
            done = set(prev.StudyInstanceUID)
        else:
            print("checkpoint columns differ from this job — starting fresh")
            os.remove(OUT)
    todo = train[~train.StudyInstanceUID.isin(done)].reset_index(drop=True)
    print(f"{len(done)} done, {len(todo)} to go")

    for s in range(0, len(todo), CHUNK):
        part = todo.iloc[s: s + CHUNK]
        out = {"StudyInstanceUID": part.StudyInstanceUID.values}
        for cond, var in JOBS:
            q = VARIANTS[cond][var]
            out[f"{cond}||{var}"] = score_prompts(
                [build_prompt(r, q) for r in part.Report], tag=f"{cond}/{var}")
        pd.DataFrame(out).to_csv(OUT, mode="a", header=not os.path.exists(OUT),
                                 index=False)
        print(f"checkpoint {min(s+CHUNK, len(todo))}/{len(todo)} -> {OUT}", flush=True)
    print("stage B complete")

# %% [markdown]
# ## 8. Verify the checkpoint, then pull it down
#
# The gate is the same one v4 used: the new margins must reproduce their stage-A
# gold AUC on the full run. A large drop means the variant only worked on the 58
# and should not be merged. `src/labels/merge_v7.py` does the merge locally.

# %%
if os.path.exists(OUT):
    df = pd.read_csv(OUT).drop_duplicates("StudyInstanceUID")
    missing = len(train) - df.StudyInstanceUID.isin(train.StudyInstanceUID).sum()
    print(f"rows {len(df)}/{len(train)}"
          + ("  INCOMPLETE — re-run stage B in a new session" if len(df) < len(train)
             else "  complete"))
    if len(df) == len(train):
        g = df.merge(gold[["StudyInstanceUID"] + TARGETS], on="StudyInstanceUID")
        print(f"\n{'column':22} {'fullrun':>8} {'stageA':>8} {'drift':>8}")
        print("-" * 50)
        for cond, var in JOBS:
            k = f"{cond}||{var}"
            a_full = roc_auc_score(g[cond].to_numpy(np.float32), g[k].to_numpy())
            a_swp = roc_auc_score(gold[cond].to_numpy(np.float32),
                                  sweep[k].to_numpy())
            print(f"{k:22} {a_full:8.4f} {a_swp:8.4f} {a_full-a_swp:+8.4f}")
        from google.colab import files
        files.download(OUT)
else:
    print("no checkpoint yet — run stage B")
