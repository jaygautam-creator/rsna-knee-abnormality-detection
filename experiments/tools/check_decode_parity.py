"""Train-time and test-time decoding must agree, or the model is scored on
volumes in an orientation it never saw. Compare the decisive expressions.

Two pairings are checked: v5 inference against the 224 cache, and v6 inference
against the 336 cache. Pass a pair name to check just one.
"""
import re
import sys

PAIRS = {
    "v5": ("experiments/kernels/cache_volumes_lat/source.py", "experiments/kernels/infer_v5/source.py"),
    "v6": ("experiments/kernels/cache_volumes_336/source.py", "experiments/kernels/infer_v6/source.py"),
}
WHICH = sys.argv[1] if len(sys.argv) > 1 else "v5"
CACHE_SRC, INFER_SRC = PAIRS[WHICH]
cache = open(CACHE_SRC).read()
infer = open(INFER_SRC).read()
print(f"=== {WHICH}: {CACHE_SRC} vs {INFER_SRC} ===")

REQUIRED = [
    # the flip decision
    "flip_w = (np.sign(row[0]) != side)",
    "flip_d = (np.sign(normal[0]) != side)",
    "if row is not None and abs(row[0]) > AXIS_TOL:",
    "if normal is not None and abs(normal[0]) > AXIS_TOL:",
    # how it is applied
    "idx = np.linspace(0, len(files) - 1, D_SLICES).round().astype(int)",
    "if flip_d:",
    "idx = idx[::-1]",
    "if flip_w:",
    "out = out[:, :, ::-1]",
    # side resolution — decided once per STUDY, over every slice of every
    # series. Per-series resolution from one unsorted slice is the bug that
    # left 9.4% of studies self-contradicting; these lines pin the fix.
    'for tag in ("Laterality", "ImageLaterality", "FrameLaterality"):',
    'if v.startswith("R"):',
    "votes = [s for hs in head_sets for _, ds in hs",
    "xs = [x for hs in head_sets for _, ds in hs if (x := _ipp_x(ds)) is not None]",
    # the laterality vocabulary must be the same in both, or the same study
    # resolves differently at train and test time
    r"\bright\b|\brt\b|\bderech\w+|\bdcha\b|\bdcho\b|\brechts\b",
    r"\bleft\b|\blt\b|\bizquierd\w+|\bizq\b|\blinks\b",
    "def load_volume(heads, side)",
    "heads = sorted(heads, key=lambda fh: _position_from(fh[1], normal))",
    # geometry + windowing must match too
    "return row, np.cross(row, col)",
    "lo, hi = np.percentile(out, [1, 99])",
    "cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA)",
]

ok = True
for line in REQUIRED:
    inc, ini = line in cache, line in infer
    if not (inc and ini):
        ok = False
        print(f"MISMATCH  cache={inc} infer={ini}  :: {line}")
if ok:
    print(f"PASS  all {len(REQUIRED)} decisive expressions present in both")

# Constants that must be equal
CONSTS = ("D_SLICES", "SIZE", "AXIS_TOL") if WHICH == "v5" else ("D_SLICES", "AXIS_TOL")
for name in CONSTS:
    vc = re.search(rf"^{name} = (\S+)", cache, re.M)
    vi = re.search(rf"^{name} = (\S+)", infer, re.M)
    got = (vc.group(1) if vc else None, vi.group(1) if vi else None)
    same = got[0] == got[1] and got[0] is not None
    ok = ok and same
    print(("PASS  " if same else "FAIL  ") + f"{name}: cache={got[0]} infer={got[1]}")

if WHICH == "v6":
    # v6 inference serves several resolutions, so there is no single SIZE to
    # compare. The requirement is that the cache's size is one of the sizes some
    # mounted checkpoint declares — otherwise no model ever sees this cache.
    vc = re.search(r"^SIZE = (\d+)", cache, re.M)
    csize = int(vc.group(1)) if vc else None
    declared = {int(m) for m in re.findall(r'"v\d_fold\d\.pt": (\d+),', infer)}
    same = csize is not None and csize in declared
    ok = ok and same
    print(("PASS  " if same else "FAIL  ")
          + f"cache SIZE={csize} is declared by some checkpoint {sorted(declared)}")
    # And nothing may resample between resolutions behind the model's back.
    no_resample = "F.interpolate" not in infer
    ok = ok and no_resample
    print(("PASS  " if no_resample else "FAIL  ")
          + "inference decodes each size natively (no F.interpolate resampling)")

# The thing that would silently undo everything
flip_tta = re.search(r"^TTA_FLIP = (\S+)", infer, re.M)
same = flip_tta and flip_tta.group(1) == "False"
ok = ok and bool(same)
print(("PASS  " if same else "FAIL  ") + f"infer TTA_FLIP = {flip_tta.group(1) if flip_tta else None}")


# --- intended divergences ---------------------------------------------------
# These two files are NOT identical, and should not be. Pin the differences so a
# future edit cannot quietly turn an intended one into an accidental one.
print()
div_ok = True

def diverge(label, cache_has, infer_has):
    global div_ok
    good = (cache_has in cache) == True and (infer_has in infer) == True
    print(("PASS  " if good else "FAIL  ") + label)
    div_ok = div_ok and good

# Training may use the radiology reports; test.csv has none.
diverge("cache uses reports, inference does not",
        "reports.get(sid", "study_signals(head_sets)")
assert "reports.get(sid" not in infer, \
    "inference references reports, but test studies have none"
print("PASS  inference never references reports")

# The mean-x cut is a site-dependent quantity, so inference refits it on the test
# set's own tagged studies rather than trusting the training value.
for needle, label in (
        ("_fit_threshold", "inference fits the mean-x threshold at run time"),
        ("SIDE_THR_TRAIN", "inference keeps the training threshold only as a fallback")):
    good = needle in infer
    print(("PASS  " if good else "FAIL  ") + label)
    div_ok = div_ok and good

ok = ok and div_ok
print("\nPARITY OK" if ok else "\nPARITY BROKEN")
raise SystemExit(0 if ok else 1)
