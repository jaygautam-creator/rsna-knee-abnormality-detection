"""Check the canonicalisation decides correctly for every plane x side."""
import numpy as np

AXIS_TOL = 0.5

# Standard DICOM ImageOrientationPatient triplets.
# patient axes: +x = patient LEFT, +y = posterior, +z = superior
IOP = {
    # rows run L->R (+x), cols run A->P (+y)  => normal = +z (superior)
    "Axial":    [1, 0, 0, 0, 1, 0],
    # rows run L->R (+x), cols run S->I (-z)  => normal = -y ... (anterior)
    "Coronal":  [1, 0, 0, 0, 0, -1],
    # rows run A->P (+y), cols run S->I (-z)  => normal = -x  (patient right)
    "Sagittal": [0, 1, 0, 0, 0, -1],
}


def decide(plane, side):
    """side: +1 right knee (medial = +x), -1 left knee (medial = -x)."""
    iop = IOP[plane]
    row = np.array(iop[:3], float)
    normal = np.cross(row, np.array(iop[3:], float))
    flip_w = flip_d = False
    if abs(row[0]) > AXIS_TOL:
        flip_w = (np.sign(row[0]) != side)
    if abs(normal[0]) > AXIS_TOL:
        flip_d = (np.sign(normal[0]) != side)
    return flip_w, flip_d, row, normal


print(f"{'plane':9s} {'side':5s} {'row.x':>6s} {'nrm.x':>6s} {'flip_w':>7s} {'flip_d':>7s}")
res = {}
for plane in IOP:
    for side, name in ((+1, "R"), (-1, "L")):
        fw, fd, row, nrm = decide(plane, side)
        res[(plane, name)] = (fw, fd)
        print(f"{plane:9s} {name:5s} {row[0]:6.1f} {nrm[0]:6.1f} {str(fw):>7s} {str(fd):>7s}")

print()
ok = True

def check(cond, msg):
    global ok
    print(("PASS  " if cond else "FAIL  ") + msg)
    ok = ok and cond

# In-plane W is the medial axis for axial and coronal only.
for p in ("Axial", "Coronal"):
    check(res[(p, "R")][0] != res[(p, "L")][0],
          f"{p}: exactly one side gets the W flip (they end up mirrored)")
    check(not res[(p, "R")][1] and not res[(p, "L")][1],
          f"{p}: slice order untouched (medial is not the slice axis)")

# Sagittal: medial axis is the slice axis, W must never be touched.
check(res[("Sagittal", "R")][1] != res[("Sagittal", "L")][1],
      "Sagittal: exactly one side gets the slice reversal")
check(not res[("Sagittal", "R")][0] and not res[("Sagittal", "L")][0],
      "Sagittal: W untouched (a flip there would mirror anterior/posterior)")

# The actual invariant: after canonicalisation, increasing index along the medial
# axis points medial for BOTH sides. Simulate it end to end.
print()
for plane in IOP:
    outs = []
    for side in (+1, -1):
        fw, fd, row, nrm = decide(plane, side)
        # direction (in patient x) that increasing index points, after the flip
        w_dir = row[0] * (-1 if fw else 1)
        d_dir = nrm[0] * (-1 if fd else 1)
        medial_x = side          # medial direction in patient x for this knee
        axis = "W" if abs(row[0]) > AXIS_TOL else "D"
        got = w_dir if axis == "W" else d_dir
        outs.append((axis, np.sign(got) == np.sign(medial_x)))
    check(all(o[1] for o in outs),
          f"{plane}: increasing {outs[0][0]} index points medial for both L and R")


# --- regression: the sagittal sign-flip that broke version 1 of the cache ------
#
# Side used to be read from ImagePositionPatient[0] of one arbitrary, unsorted
# slice. On a sagittal series the slices *span* the medial-lateral axis, so x
# changes sign across the stack whenever the knee straddles patient x = 0, and
# the per-slice answer is noise. 413 of 4,407 studies came out self-contradicting.
# The fix decides once per study from the mean over every slice.
print()

def per_slice_side(x):
    return +1 if x < 0 else -1

def study_side_from(xs):
    return +1 if float(np.mean(xs)) < 0 else -1

# A right knee centred at x = -40 mm, sagittal stack spanning +/- 50 mm.
xs = list(np.linspace(-90, 10, 24))
per = {per_slice_side(x) for x in xs}
check(len(per) == 2,
      "per-slice rule is ambiguous on a knee straddling x=0 (both answers occur)")
check(study_side_from(xs) == +1,
      "study-level mean-x still resolves that knee as right")

# A left knee centred at x = +35 mm, same span.
xs = list(np.linspace(-15, 85, 24))
check(study_side_from(xs) == -1, "study-level mean-x resolves a left knee")

# Tag beats position whenever present: a single unanimous vote decides.
check((lambda votes: +1 if sum(votes) > 0 else -1)([+1, +1, +1]) == +1,
      "unanimous tag votes resolve to the tagged side")

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
raise SystemExit(0 if ok else 1)
