"""Faster DICOM read paths for the RSNA knee inference kernel.

Every build_study_<name> here must return a (vol, mask) that is BYTE-IDENTICAL to
build_study_v2, the reference copied verbatim from
public/raptor_infer1_dense96_fast/raptor_infer1_dense96_fast.py (lines ~260-360; the r34
student kernel has the same read functions, only IMG/SLOTS differ). verify.py checks that.

Layouts: LAYOUTS["d96"] (IMG 384, 26/22/18/12/18) and LAYOUTS["d80"] (IMG 336, 22/18/15/10/15).

Variants (see VARIANTS at the bottom):
  v2        reference: stop_before_pixels dcmread of EVERY file (16 thr) + full dcmread of
            every pick (8 thr; a pick repeated when n < k is re-read each time)
  dedupe    v2 header pass, each UNIQUE pick read once
  onepass   one open per file, whole file read into memory, header parsed from the bytes,
            picks decoded from the same bytes -> opens = n, no second pass
  prefix    header pass reads only the first HDR_BYTES of each file (one pread), picks
            re-opened and read whole (opens = n + unique picks, far fewer header bytes)
  keepfd    like prefix but the fd of every file is kept open after the header pread; the
            picks' pixel bytes are pread from the SAME fd -> opens = n, bytes ~= n*hdr +
            picks*size (the minimum on both counts)
  hybrid    per series: onepass when unique picks / n >= HYBRID_FRAC, else keepfd
  study     keepfd I/O batched over the whole study: ONE header pool over all files of the 5
            series, then ONE pixel pool over all unique picks (fewest latency waves in a row)
  studyone  onepass I/O batched over the whole study (one whole-file read per file, one pool)
  *_lut     walker decode + exact percentile from raw-code counts + one clip/normalise table
            per series gathered over the 140 mm crop: the float image is never built
  *_fast    same I/O, but the header fields and uncompressed pixels are decoded by a small
            explicit/implicit-VR-LE walker instead of pydicom; anything unusual (other
            transfer syntax, modality LUT, odd lengths, undefined-length items, unparseable
            DS, missing tags, ...) falls back to the exact pydicom path on the same bytes.

I/O accounting (opens / read syscalls / bytes) is kept in STATS when ACCOUNT is True; v2's
pydicom opens are counted by patching pydicom.filereader.open.
"""
import glob, io, os, re, struct, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

CROP_MM = 140.0
LAYOUTS = {
    "d96": dict(IMG=384, SLOTS=[("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18),
                                ("Coronal", 0, 12), ("Axial", -1, 18)]),
    "d80": dict(IMG=336, SLOTS=[("Sagittal", 1, 22), ("Sagittal", 0, 18), ("Coronal", 1, 15),
                                ("Coronal", 0, 10), ("Axial", -1, 15)]),
}


# ============================================================================
# I/O accounting
# ============================================================================
class IOStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.opens = 0; self.reads = 0; self.bytes = 0

    def add(self, opens=0, reads=0, nbytes=0):
        with self.lock:
            self.opens += opens; self.reads += reads; self.bytes += nbytes

    def snap(self):
        with self.lock:
            return dict(opens=self.opens, reads=self.reads, bytes=self.bytes)


STATS = IOStats()
ACCOUNT = False


class _CountingFileIO(io.FileIO):
    """FileIO that counts opens, raw read calls and raw bytes (what the OS was asked for)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        STATS.add(opens=1)

    def readinto(self, b):
        n = super().readinto(b)
        STATS.add(reads=1, nbytes=n or 0)
        return n

    def readall(self):
        d = super().readall()
        STATS.add(reads=1, nbytes=len(d))
        return d

    def read(self, size=-1):
        d = super().read(size)
        STATS.add(reads=1, nbytes=len(d or b""))
        return d


def _counting_open(file, mode="rb", *a, **k):
    # mirrors io.open(file, "rb"): BufferedReader over FileIO with buffer = st_blksize
    raw = _CountingFileIO(file, "rb")
    bs = getattr(raw, "_blksize", 0) or 0
    return io.BufferedReader(raw, buffer_size=bs if bs > 1 else io.DEFAULT_BUFFER_SIZE)


def set_accounting(on):
    """Count opens/bytes for BOTH pydicom path reads (v2) and the os.pread variants."""
    global ACCOUNT
    import pydicom.filereader as _fr
    ACCOUNT = bool(on)
    if on:
        _fr.open = _counting_open
    elif "open" in _fr.__dict__:
        del _fr.open


# ============================================================================
# REFERENCE v2 -- verbatim from raptor_infer1_dense96_fast.py (only IMG / CROP_MM /
# HDR_THREADS / PX_THREADS / SLOTS / MAXS turned from module globals into parameters)
# ============================================================================
def make_reader_v2(IMG, HDR_THREADS=16):
    import pydicom, cv2
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def _hdr(f):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop = getattr(h, 'ImageOrientationPatient', None)
            ipp = getattr(h, 'ImagePositionPatient', None)
            if iop is not None and ipp is not None and len(iop) == 6:
                r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
            else:
                pos = float(getattr(h, 'InstanceNumber', 0) or 0)
            ps = getattr(h, 'PixelSpacing', None); ps = float(ps[0]) if ps is not None else 0.5
            return (pos, f, ps, True)
        except Exception:
            return (0.0, f, 0.5, False)

    def order_and_meta(sdir):
        # Header reads are latency-bound on Kaggle's mounted competition data (~240k opens on the hidden
        # set); a thread pool hides that latency. Records and sort are unchanged -> identical ordering.
        fs = glob.glob(sdir + "/*.dcm")
        with ThreadPoolExecutor(max_workers=HDR_THREADS) as ex:
            out = list(ex.map(_hdr, fs))
        recs = [(pos, f, ps) for pos, f, ps, _ in out]; ps_list = [ps for _, _, ps, ok in out if ok]
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return [(f, ps) for _, f, ps in recs], med_ps

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps):
        h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)

    return order_and_meta, read_px, mm_crop_resize


def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study_v2(sid, ser_records, tsdir, reader, SLOTS, IMG, PX_THREADS=8):
    MAXS = sum(s[2] for s in SLOTS)
    order_and_meta, read_px, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set()
    for plane, fluid, k in SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k; continue
        used.add(r['SeriesInstanceUID'])
        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k; continue
        # wide span: the collateral ligaments and lateral meniscus live in the
        # peripheral slices the old 0.15-0.85 crop threw away. Must match the corpus
        # the weights were trained on (build_corpus80.py, SPAN).
        n = len(files); lo, hi = int(n * 0.02), int(n * 0.98) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        sel = [files[min(p, n - 1)] for p in picks]
        def _px(fp_ps):
            fp, ps = fp_ps
            try: return read_px(fp), ps
            except Exception: return None, med_ps
        with ThreadPoolExecutor(max_workers=PX_THREADS) as ex:
            got = list(ex.map(_px, sel))                    # order preserved
        arrs = [a for a, _ in got]; pss = [ps for _, ps in got]
        valid = [a for a in arrs if a is not None]
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if idx >= MAXS: break
            if a is None: idx += 1; continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            vol[idx] = (aw * 255).astype(np.uint8); idx += 1
        if idx >= MAXS: break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


# ============================================================================
# Shared pieces for the variants (logic identical to v2, factored)
# ============================================================================
def pick_indices(n, k):
    """v2's pick rule -> list of indices into the sorted file list (len k, may repeat)."""
    lo, hi = int(n * 0.02), int(n * 0.98) - 1; hi = max(hi, lo)
    picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
    return [int(min(p, n - 1)) for p in picks]


def _fill_slot(vol, idx, MAXS, arrs, pss, med_ps, mm_crop_resize):
    """v2's per-series normalisation + crop + resize tail, verbatim."""
    valid = [a for a in arrs if a is not None]
    if valid:
        allpx = np.concatenate([a.ravel() for a in valid])
        loq, hiq = np.percentile(allpx, [2.0, 98.0])
    else:
        loq, hiq = 0.0, 1.0
    for a, ps in zip(arrs, pss):
        if idx >= MAXS: break
        if a is None: idx += 1; continue
        aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
        aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
        vol[idx] = (aw * 255).astype(np.uint8); idx += 1
    return idx


def _mm_crop_resize_fn(IMG):
    import cv2

    def mm_crop_resize(a, ps):
        h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return mm_crop_resize


# ---- raw file I/O (os-level so every open / pread is explicit and counted) ----
_O_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)


def _fopen(path, fadvise_random=False):
    fd = os.open(path, _O_FLAGS)
    if ACCOUNT: STATS.add(opens=1)
    if fadvise_random and hasattr(os, "posix_fadvise"):
        try: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        except OSError: pass
    return fd


def _pread(fd, n, off):
    b = os.pread(fd, n, off)
    if ACCOUNT: STATS.add(reads=1, nbytes=len(b))
    return b


def _read_rest(fd, have, off):
    """Read from `off` to EOF (fstat size first; loops on short reads)."""
    size = os.fstat(fd).st_size
    parts = [have]; pos = off
    while pos < size:
        b = _pread(fd, size - pos, pos)
        if not b: break
        parts.append(b); pos += len(b)
    return b"".join(parts) if len(parts) > 1 else have


_KEEP_MAX = None


def keep_max():
    """How many fds a keepfd/study batch may hold open in this process. Raises the soft
    RLIMIT_NOFILE to the hard limit once; a batch larger than this falls back to re-opening
    the picks (prefix mode) instead of risking EMFILE, which would silently turn a header into
    a failed record."""
    global _KEEP_MAX
    if _KEEP_MAX is None:
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            want = hard if hard != resource.RLIM_INFINITY else 1 << 16
            if soft != resource.RLIM_INFINITY and soft < want:
                try:
                    resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard)); soft = want
                except (ValueError, OSError):
                    pass
            soft = 1 << 16 if soft == resource.RLIM_INFINITY else soft
            _KEEP_MAX = max(0, (soft - 256) // 2)   # headroom for the other threads / pools
        except Exception:
            _KEEP_MAX = 256
    return _KEEP_MAX


def _read_whole(path):
    fd = _fopen(path)
    try:
        size = os.fstat(fd).st_size
        parts = []; pos = 0
        while pos < size:
            b = _pread(fd, size - pos, pos)
            if not b: break
            parts.append(b); pos += len(b)
        # v2 reads with pydicom until EOF; a file that grew after fstat is not a concern
        return b"".join(parts)
    finally:
        os.close(fd)


# ============================================================================
# pydicom on bytes (exact v2 semantics, just from memory instead of a path)
# ============================================================================
def _make_pyd():
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def hdr_from_bytes(f, data):
        # identical body to v2 _hdr, the file content comes from `data`
        try:
            h = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True)
            iop = getattr(h, 'ImageOrientationPatient', None)
            ipp = getattr(h, 'ImagePositionPatient', None)
            if iop is not None and ipp is not None and len(iop) == 6:
                r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
            else:
                pos = float(getattr(h, 'InstanceNumber', 0) or 0)
            ps = getattr(h, 'PixelSpacing', None); ps = float(ps[0]) if ps is not None else 0.5
            return (pos, f, ps, True)
        except Exception:
            return (0.0, f, 0.5, False)

    def px_from_bytes(data):
        # identical body to v2 read_px (raises on failure, like v2)
        d = pydicom.dcmread(io.BytesIO(data))
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    return hdr_from_bytes, px_from_bytes


# ============================================================================
# Minimal DICOM header walker (explicit / implicit VR little endian) + fast decoders.
# Returns None on ANYTHING unusual; callers then use the exact pydicom path.
# ============================================================================
_EXT_VR = frozenset((b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"SQ", b"SV", b"UC", b"UN",
                     b"UR", b"UT", b"UV"))
_STD_VR = frozenset((b"AE", b"AS", b"AT", b"CS", b"DA", b"DS", b"DT", b"FL", b"FD", b"IS",
                     b"LO", b"LT", b"PN", b"SH", b"SL", b"SS", b"ST", b"TM", b"UI", b"UL",
                     b"US"))
T_IN, T_IPP, T_IOP = 0x00200013, 0x00200032, 0x00200037
T_SPP, T_PI, T_NF, T_ROWS, T_COLS = 0x00280002, 0x00280004, 0x00280008, 0x00280010, 0x00280011
T_PS, T_BA, T_BS, T_PR = 0x00280030, 0x00280100, 0x00280101, 0x00280103
T_RI, T_RS, T_MLUT = 0x00281052, 0x00281053, 0x00283000
_WANT_VR = {T_IN: b"IS", T_IPP: b"DS", T_IOP: b"DS", T_SPP: b"US", T_PI: b"CS", T_NF: b"IS",
            T_ROWS: b"US", T_COLS: b"US", T_PS: b"DS", T_BA: b"US", T_BS: b"US", T_PR: b"US",
            T_RI: b"DS", T_RS: b"DS", T_MLUT: b"SQ"}
TS_EXPLICIT_LE = "1.2.840.10008.1.2.1"
TS_IMPLICIT_LE = "1.2.840.10008.1.2"
_TS_BAD_HDR = ("1.2.840.10008.1.2.2", "1.2.840.10008.1.2.1.99")   # big endian / deflated
_PIX_TAGS = (0x7FE00010, 0x7FE00009, 0x7FE00008)
_uH = struct.Struct("<H").unpack_from
_uI = struct.Struct("<I").unpack_from
_uHH = struct.Struct("<HH").unpack_from


class NeedMore(Exception):
    """The buffer ended before the Pixel Data element header."""


def scan_header(buf):
    """Walk the DICOM elements of `buf` up to (7FE0,0010).
    Returns dict(ts, implicit, pix_elem, pix_off, pix_len, vals={tag: (vr, raw bytes)}),
    None if the file is unusual (-> pydicom), raises NeedMore if `buf` is too short."""
    n = len(buf)
    if n < 132:
        raise NeedMore
    if buf[128:132] != b"DICM":
        return None
    pos = 132; ts = None
    while True:                                     # file meta group: explicit VR LE
        if pos + 8 > n: raise NeedMore
        grp, el = _uHH(buf, pos)
        if grp != 2: break
        vr = buf[pos + 4:pos + 6]
        if vr in _EXT_VR:
            if pos + 12 > n: raise NeedMore
            ln = _uI(buf, pos + 8)[0]; vpos = pos + 12
        elif vr in _STD_VR:
            ln = _uH(buf, pos + 6)[0]; vpos = pos + 8
        else:
            return None
        if ln == 0xFFFFFFFF: return None
        if vpos + ln > n: raise NeedMore
        if el == 0x0010:
            ts = buf[vpos:vpos + ln].decode("latin-1").rstrip("\x00 ")
        pos = vpos + ln
    if ts is None or ts in _TS_BAD_HDR:
        return None
    implicit = ts == TS_IMPLICIT_LE
    vals = {}; last = -1
    while True:
        if pos + 8 > n: raise NeedMore
        grp, el = _uHH(buf, pos); tag = (grp << 16) | el
        if tag <= last: return None                 # out of order / duplicate -> pydicom
        last = tag
        if implicit:
            ln = _uI(buf, pos + 4)[0]; vpos = pos + 8; vr = None
        else:
            vr = buf[pos + 4:pos + 6]
            if vr in _EXT_VR:
                if pos + 12 > n: raise NeedMore
                ln = _uI(buf, pos + 8)[0]; vpos = pos + 12
            elif vr in _STD_VR:
                ln = _uH(buf, pos + 6)[0]; vpos = pos + 8
            else:
                return None
        if tag in _PIX_TAGS:
            if tag != 0x7FE00010: return None
            return dict(ts=ts, implicit=implicit, pix_elem=pos, pix_off=vpos, pix_len=ln,
                        vals=vals)
        if ln == 0xFFFFFFFF: return None            # undefined-length item before pixels
        if tag in _WANT_VR:
            if vpos + ln > n: raise NeedMore
            if vr is not None and vr != _WANT_VR[tag]: return None
            vals[tag] = bytes(buf[vpos:vpos + ln])
        pos = vpos + ln


_NUM = re.compile(r"[ \t]*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[ \t]*\Z")


def _ds(raw):
    """DS values exactly as pydicom's MultiString(DSfloat) -> list of float; None if absent,
    empty or anything outside a strict numeric subset (-> pydicom decides)."""
    if raw is None: return None
    s = raw.decode("latin-1").rstrip()
    if s and s[-1] in " \x00": s = s[:-1]
    if not s: return None
    out = []
    for p in s.split("\\"):
        if not _NUM.match(p): return None
        out.append(float(p))
    return out


def _us(raw):
    if raw is None or len(raw) != 2: return None
    return _uH(raw, 0)[0]


def fast_hdr(info):
    """(pos, ps) exactly as v2 _hdr for the common case, else None (-> pydicom)."""
    v = info["vals"]
    iop = _ds(v.get(T_IOP)); ipp = _ds(v.get(T_IPP)); ps = _ds(v.get(T_PS))
    if iop is None or ipp is None or ps is None: return None
    if len(iop) != 6 or len(ipp) != 3 or len(ps) < 2: return None
    # np.cross(r, c) for 3-vectors is cp0 = a1*b2 - a2*b1 etc. as separate IEEE multiplies and
    # subtracts (numpy/_core/numeric.py), so Python float arithmetic gives the same bits;
    # np.dot is kept as is (BLAS may fuse multiply-adds).
    a0, a1, a2, b0, b1, b2 = iop
    n = np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])
    pos = float(np.dot(np.array(ipp, float), n))
    return pos, float(ps[0])


def _pydicom_major():
    import pydicom
    try: return int(pydicom.__version__.split(".")[0])
    except Exception: return -1


_CORRECT_UNUSED = None   # pydicom >= 3 masks/sign-extends bits above Bits Stored for native data


def fast_px(buf, info, want_hist=False):
    """v2 read_px for uncompressed monochrome 16-bit single-frame data, else None.
    With want_hist: returns (a, hist) where hist = (group key, dense int64 bincount of the raw
    integers) so the series percentile can be computed from counts (see _pct_from_hists)."""
    global _CORRECT_UNUSED
    if _CORRECT_UNUSED is None:
        _CORRECT_UNUSED = _pydicom_major() >= 3
    if info["ts"] not in (TS_EXPLICIT_LE, TS_IMPLICIT_LE): return None
    v = info["vals"]
    if T_MLUT in v: return None
    spp, rows, cols = _us(v.get(T_SPP)), _us(v.get(T_ROWS)), _us(v.get(T_COLS))
    ba, bs, pr = _us(v.get(T_BA)), _us(v.get(T_BS)), _us(v.get(T_PR))
    if spp != 1 or ba != 16 or pr not in (0, 1) or not rows or not cols: return None
    if bs is None or not (1 <= bs <= 16): return None
    if T_NF in v:
        nf = v[T_NF].decode("latin-1").strip(" \x00")
        if nf != "1": return None
    pi = v.get(T_PI)
    if pi is None: return None
    pi = pi.decode("latin-1").rstrip(" \x00")
    if pi not in ("MONOCHROME1", "MONOCHROME2"): return None
    npx = rows * cols
    # the file must end exactly where the pixels end: anything after Pixel Data (trailing
    # elements, padding, garbage) is left to pydicom, which parses it in a full dcmread
    if info["pix_len"] != 2 * npx or info["pix_off"] + 2 * npx != len(buf): return None
    has_rs, has_ri = T_RS in v, T_RI in v
    rs = ri = None
    if has_rs and has_ri:
        rs = _ds(v[T_RS]); ri = _ds(v[T_RI])
        if rs is None or ri is None or len(rs) != 1 or len(ri) != 1: return None
    dt = "<u2" if pr == 0 else "<i2"
    arr = np.frombuffer(buf, dtype=dt, count=npx, offset=info["pix_off"]).reshape(rows, cols)
    if _CORRECT_UNUSED and bs < 16:
        arr = arr.copy(); sh = 16 - bs
        np.left_shift(arr, sh, out=arr); np.right_shift(arr, sh, out=arr)
    tf = (dt, rs[0] if rs else None, ri[0] if ri else None)
    a = _transform(arr, tf)
    vmax = None
    if pi == "MONOCHROME1":
        vmax = a.max()
        a = vmax - a
    if not want_hist:
        return a
    cnt = np.bincount(arr.reshape(-1).view(np.uint16), minlength=65536)
    key = tf if vmax is None else tf + (id(a),)          # MONOCHROME1: one group per image
    return a, (key, vmax, cnt)


def _transform(raw, tf):
    """v2's apply_modality_lut(...).astype(float32) on an integer array (element-wise)."""
    dt, rs, ri = tf
    if rs is not None:
        x = raw.astype(np.float64) * rs
        x += ri
        return x.astype(np.float32)
    return raw.astype(np.float32)


def _lerp_percentile(n, stat, q=(2.0, 98.0)):
    """np.percentile(x, [2.0, 98.0]) (method linear, 1-D float32 x of length n) given
    stat(r) = the r-th smallest element of x. Replicates numpy's _quantile/_get_indexes/
    _get_gamma/_lerp for this case (unchanged numpy 1.22 .. 2.5); guarded by _pct_selftest."""
    quantiles = np.true_divide(np.asanyarray(list(q), dtype=float), 100)
    virtual = (n - 1) * quantiles
    prev = np.floor(virtual)
    nxt = prev + 1
    above = virtual >= n - 1
    prev[above] = -1; nxt[above] = -1
    below = virtual < 0
    prev[below] = 0; nxt[below] = 0
    prev = prev.astype(np.intp); nxt = nxt.astype(np.intp)
    gamma = np.asanyarray(virtual - prev, dtype=virtual.dtype)
    a = np.array([stat(int(i) % n) for i in prev], dtype=np.float32)
    b = np.array([stat(int(i) % n) for i in nxt], dtype=np.float32)
    diff = b - a
    res = np.add(a, diff * gamma)
    np.subtract(b, diff * (1 - gamma), out=res, where=gamma >= 0.5, casting="unsafe",
                dtype=type(res.dtype))
    return res


_ALL_CODES = np.arange(65536, dtype=np.uint16)


class RawSlice:
    """An uncompressed slice kept as its raw 16-bit codes (a view on the file bytes) plus the
    exact element-wise recipe v2 applies to them: unused-bit correction, modality rescale,
    MONOCHROME1 inversion. Never materialised as a float image in the *_lut variants."""
    __slots__ = ("raw", "dt", "shift", "rs", "ri", "mono1", "cnt", "shape", "size")

    def codes_to_float(self, codes):
        x = codes.view(np.int16) if self.dt == "<i2" else codes
        if self.shift:
            x = x.copy(); np.left_shift(x, self.shift, out=x); np.right_shift(x, self.shift, out=x)
        return _transform(x, (self.dt, self.rs, self.ri))

    def to_float(self):
        """The float32 slice v2's read_px returns (used only when a series mixes in a slice
        that needed the pydicom fallback)."""
        a = self.codes_to_float(self.raw)
        if self.mono1:
            a = a.max() - a
        return a


def fast_raw(buf, info):
    """RawSlice for the same cases fast_px handles, else None (-> pydicom)."""
    global _CORRECT_UNUSED
    if _CORRECT_UNUSED is None:
        _CORRECT_UNUSED = _pydicom_major() >= 3
    if info["ts"] not in (TS_EXPLICIT_LE, TS_IMPLICIT_LE): return None
    v = info["vals"]
    if T_MLUT in v: return None
    spp, rows, cols = _us(v.get(T_SPP)), _us(v.get(T_ROWS)), _us(v.get(T_COLS))
    ba, bs, pr = _us(v.get(T_BA)), _us(v.get(T_BS)), _us(v.get(T_PR))
    if spp != 1 or ba != 16 or pr not in (0, 1) or not rows or not cols: return None
    if bs is None or not (1 <= bs <= 16): return None
    if T_NF in v:
        nf = v[T_NF].decode("latin-1").strip(" \x00")
        if nf != "1": return None
    pi = v.get(T_PI)
    if pi is None: return None
    pi = pi.decode("latin-1").rstrip(" \x00")
    if pi not in ("MONOCHROME1", "MONOCHROME2"): return None
    npx = rows * cols
    # the file must end exactly where the pixels end: anything after Pixel Data (trailing
    # elements, padding, garbage) is left to pydicom, which parses it in a full dcmread
    if info["pix_len"] != 2 * npx or info["pix_off"] + 2 * npx != len(buf): return None
    rs = ri = None
    if T_RS in v and T_RI in v:
        rs = _ds(v[T_RS]); ri = _ds(v[T_RI])
        if rs is None or ri is None or len(rs) != 1 or len(ri) != 1: return None
        rs, ri = rs[0], ri[0]
    s = RawSlice()
    s.raw = np.frombuffer(buf, dtype=np.uint16, count=npx, offset=info["pix_off"]).reshape(rows, cols)
    s.dt = "<u2" if pr == 0 else "<i2"
    s.shift = (16 - bs) if (_CORRECT_UNUSED and bs < 16) else 0
    s.rs, s.ri = rs, ri
    s.mono1 = pi == "MONOCHROME1"
    s.cnt = np.bincount(s.raw.reshape(-1), minlength=65536)
    s.shape = (rows, cols); s.size = npx
    return s


def _fill_lut(vol, idx, MAXS, res, IMG):
    """v2's per-series tail for RawSlices: exact 2/98 percentile from code counts, then ONE
    65536-entry table per recipe holding v2's clip((a-loq)/(hiq-loq+1e-6),0,1) for every code,
    and a gather over the 140 mm crop. Element-wise identical to v2 by construction."""
    import cv2
    arrs, pss, med_ps, _ = res
    valid = [a for a in arrs if a is not None]
    if any(not isinstance(a, RawSlice) for a in valid) or not _pct_selftest():
        # mixed with a pydicom-fallback slice: materialise and take the plain crop-first path
        mat = {}
        arrs = [None if a is None else (a if not isinstance(a, RawSlice) else
                mat.setdefault(id(a), a.to_float())) for a in arrs]
        return _fill_plain(vol, idx, MAXS, arrs, pss, med_ps, IMG, True, None)
    if not valid:
        return _fill_plain(vol, idx, MAXS, arrs, pss, med_ps, IMG, True, None)
    groups = {}; n_total = 0; gkey = {}
    for a in valid:
        key = (a.dt, a.shift, a.rs, a.ri) + ((id(a),) if a.mono1 else ())
        gkey[id(a)] = key
        g = groups.get(key)
        if g is None:
            groups[key] = [a, a.cnt.copy()]
        else:
            g[1] += a.cnt
        n_total += a.size
    Vs, Cs, T = [], [], {}
    for key, (a0, cnt) in groups.items():
        full = a0.codes_to_float(_ALL_CODES)                 # float32 value of every code
        nz = np.flatnonzero(cnt)
        v = full[nz]
        if a0.mono1:
            vmax = v.max()                                   # == a.max() of that slice
            v = vmax - v
            full = vmax - full
        T[key] = full
        Vs.append(v); Cs.append(cnt[nz])
    V = np.concatenate(Vs); C = np.concatenate(Cs)
    if not np.isfinite(V).all():
        mat = {}
        arrs = [None if a is None else mat.setdefault(id(a), a.to_float()) for a in arrs]
        return _fill_plain(vol, idx, MAXS, arrs, pss, med_ps, IMG, True, None)
    order = np.argsort(V, kind="stable")
    V = V[order]; cum = np.cumsum(C[order])
    assert int(cum[-1]) == n_total
    loq, hiq = _lerp_percentile(n_total, lambda r: V[int(np.searchsorted(cum, r, side="right"))])
    luts = {}
    for a, ps in zip(arrs, pss):
        if idx >= MAXS: break
        if a is None: idx += 1; continue
        key = gkey[id(a)]
        lut = luts.get(key)
        if lut is None:
            lut = luts[key] = np.clip((T[key] - loq) / (hiq - loq + 1e-6), 0, 1)
        p = ps if ps > 0 else med_ps
        h, w = a.shape; cpx = int(round(CROP_MM / max(p, 1e-3)))
        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
        aw = lut.take(a.raw[y0:y0 + cpx, x0:x0 + cpx])      # == lut[crop], 2x faster
        aw = cv2.resize(aw, (IMG, IMG), interpolation=cv2.INTER_AREA)
        vol[idx] = (aw * 255).astype(np.uint8); idx += 1
    return idx


_PCT_OK = None


def _pct_selftest():
    """Compare _lerp_percentile with np.percentile on random float32 data (ties, tiny n,
    large n). Any mismatch disables the fast percentile in this process."""
    global _PCT_OK
    if _PCT_OK is not None:
        return _PCT_OK
    rng = np.random.default_rng(12345)
    ok = True
    sizes = list(range(1, 60)) + [97, 100, 101, 1000, 4097, 65536, 262144, 262145, 1048576]
    for n in sizes:
        for kind in range(3):
            if kind == 0: x = rng.integers(0, 50, n).astype(np.float32)          # heavy ties
            elif kind == 1: x = (rng.gamma(2.0, 300.0, n)).astype(np.float32)
            else: x = (rng.integers(0, 4096, n) * np.float64(2.55507) + 0.5).astype(np.float32)
            ref = np.percentile(x, [2.0, 98.0])
            xs = np.sort(x)
            got = _lerp_percentile(n, lambda r: xs[r])
            if not (ref.dtype == got.dtype and np.array_equal(ref, got)):
                ok = False; break
        if not ok: break
    _PCT_OK = ok
    return ok


def _pct_from_hists(arrs, hists):
    """Exact (loq, hiq) of np.percentile(concat(valid arrs), [2, 98]) from per-image counts of
    the raw integers, or None when any valid slice lacks a histogram (-> np.percentile)."""
    if not _pct_selftest():
        return None
    groups = {}; n_total = 0
    for a, h in zip(arrs, hists):
        if a is None: continue
        if h is None: return None
        key, vmax, cnt = h
        g = groups.get(key)
        if g is None:
            groups[key] = [vmax, cnt.copy()]
        else:
            g[1] += cnt
        n_total += a.size
    if n_total == 0:
        return None
    Vs, Cs = [], []
    for key, (vmax, cnt) in groups.items():
        nz = np.flatnonzero(cnt)
        raw = nz.astype(np.uint16)
        if key[0] == "<i2": raw = raw.view(np.int16)
        v = _transform(raw, key[:3])
        if vmax is not None:
            v = vmax - v
        Vs.append(v); Cs.append(cnt[nz])
    V = np.concatenate(Vs); C = np.concatenate(Cs)
    if not np.isfinite(V).all():
        return None
    order = np.argsort(V, kind="stable")
    V = V[order]; cum = np.cumsum(C[order])
    if int(cum[-1]) != n_total:
        return None
    res = _lerp_percentile(n_total, lambda r: V[int(np.searchsorted(cum, r, side="right"))])
    loq, hiq = res
    return loq, hiq


# ============================================================================
# Per-series readers. Each returns None (no files -> slot skipped, like v2) or
# (arrs, pss, med_ps) with arrs/pss in pick order, exactly as v2's `got`.
# ============================================================================
class Cfg:
    def __init__(self, hdr_threads=16, px_threads=8, hdr_bytes=4096, fast=False, pct=False,
                 fadvise=False, hybrid_frac=0.6, cropfirst=False, lut=False):
        self.hdr_threads = hdr_threads; self.px_threads = px_threads
        self.lut = lut; fast = fast or lut
        self.hdr_bytes = hdr_bytes; self.fast = fast; self.pct = pct and fast
        self.fadvise = fadvise; self.hybrid_frac = hybrid_frac; self.cropfirst = cropfirst


def _records(out):
    """v2's sort + median over the per-file (pos, f, ps, ok) records (glob order in)."""
    recs = [(pos, i, ps) for i, (pos, f, ps, ok) in enumerate(out)]
    ps_list = [ps for _, _, ps, ok in out if ok]
    recs.sort(key=lambda x: x[0])                   # stable -> ties keep glob order, as v2
    med_ps = float(np.median(ps_list)) if ps_list else 0.5
    return [(i, ps) for _, i, ps in recs], med_ps


def _picked(order, k):
    n = len(order)
    p = pick_indices(n, k)
    return [order[j] for j in p]                    # [(file index, ps)] in pick order


def _map_unique(fn, items, threads):
    """Apply fn once per unique key; items = list of hashable keys; returns dict."""
    uniq = list(dict.fromkeys(items))
    if threads <= 1 or len(uniq) <= 1:
        return {u: fn(u) for u in uniq}
    with ThreadPoolExecutor(max_workers=min(threads, len(uniq))) as ex:
        return dict(zip(uniq, ex.map(fn, uniq)))


class _Ctx:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hdr_b, self.px_b = _make_pyd()

    def hdr_rec(self, f, data, info):
        if self.cfg.fast and info is not None:
            r = fast_hdr(info)
            if r is not None:
                return (r[0], f, r[1], True)
        return self.hdr_b(f, data)

    def px(self, data, info):
        """-> (array or None, hist or None), v2's exception-to-None semantics."""
        if data is None:
            return None, None
        try:
            if self.cfg.lut and info is not None:
                r = fast_raw(data, info)
                if r is not None:
                    return r, None
            elif self.cfg.fast and info is not None:
                r = fast_px(data, info, want_hist=self.cfg.pct)
                if r is not None:
                    return r if self.cfg.pct else (r, None)
            return self.px_b(data), None
        except Exception:
            return None, None


def _scan_full(data):
    try:
        return scan_header(data)
    except NeedMore:
        return None


def _result(sel, got, med_ps):
    idxs = [i for i, _ in sel]
    arrs = [got[i][0] for i in idxs]
    hists = [got[i][1] for i in idxs]
    pss = [ps if got[i][0] is not None else med_ps for i, ps in sel]
    return arrs, pss, med_ps, hists


# ---- (a) dedupe: v2 header pass, unique picks read once through pydicom paths ----
def series_dedupe(sdir, k, ctx, v2reader):
    order_and_meta, read_px, _ = v2reader
    files, med_ps = order_and_meta(sdir)
    if not files:
        return None
    sel = [(j, files[j][1]) for j in pick_indices(len(files), k)]

    def _one(j):
        try: return read_px(files[j][0]), None
        except Exception: return None, None
    got = _map_unique(_one, [j for j, _ in sel], ctx.cfg.px_threads)
    return _result(sel, got, med_ps)


# ---- (b) onepass: open + read each file whole, once ----
def series_onepass(sdir, k, ctx, fs=None):
    fs = glob.glob(sdir + "/*.dcm") if fs is None else fs
    if not fs:
        return None

    def _one(f):
        try:
            data = _read_whole(f)
        except Exception:
            return (0.0, f, 0.5, False), None, None
        info = _scan_full(data) if ctx.cfg.fast else None
        return ctx.hdr_rec(f, data, info), data, info
    th = ctx.cfg.hdr_threads
    if th > 1:
        with ThreadPoolExecutor(max_workers=th) as ex:
            out = list(ex.map(_one, fs))
    else:
        out = [_one(f) for f in fs]
    order, med_ps = _records([o[0] for o in out])
    sel = _picked(order, k)
    got = _map_unique(lambda i: ctx.px(out[i][1], out[i][2]), [i for i, _ in sel], 1)
    return _result(sel, got, med_ps)


# ---- header read from a prefix; returns (rec, bytes, info, fd_or_None, is_full) ----
def _hdr_prefix(ctx, f, keep_fd):
    fd = None
    try:
        fd = _fopen(f, ctx.cfg.fadvise)
        data = _pread(fd, ctx.cfg.hdr_bytes, 0)
        try:
            info = scan_header(data)
            full = False
        except NeedMore:
            # header longer than the prefix (or a truncated file): read the rest, parse all
            data = _read_rest(fd, data, len(data))
            info = _scan_full(data); full = True
        if info is None and not full:
            # unusual file: v2 semantics need pydicom on the COMPLETE file
            data = _read_rest(fd, data, len(data)); full = True
        if ctx.cfg.fast and info is not None:
            r = fast_hdr(info)
            rec = (r[0], f, r[1], True) if r is not None else ctx.hdr_b(f, data)
        else:
            # pydicom on the prefix is exact: the walker proved the prefix holds every element
            # up to and including the Pixel Data tag/VR/length, where stop_before_pixels stops.
            rec = ctx.hdr_b(f, data)
        if not keep_fd:
            os.close(fd); fd = None
        return rec, data, info, fd, full
    except Exception:
        if fd is not None:
            try: os.close(fd)
            except OSError: pass
        return (0.0, f, 0.5, False), None, None, None, False


# ---- (c) prefix: header prefix pass, picks re-opened and read whole ----
def series_prefix(sdir, k, ctx, fs=None):
    fs = glob.glob(sdir + "/*.dcm") if fs is None else fs
    if not fs:
        return None
    with ThreadPoolExecutor(max_workers=ctx.cfg.hdr_threads) as ex:
        out = list(ex.map(lambda f: _hdr_prefix(ctx, f, False), fs))
    order, med_ps = _records([o[0] for o in out])
    sel = _picked(order, k)

    def _one(i):
        rec, data, info, _, full = out[i]
        try:
            if not full:
                data = _read_whole(fs[i])
                info = _scan_full(data) if ctx.cfg.fast else None
        except Exception:
            return None, None
        return ctx.px(data, info if ctx.cfg.fast else None)
    got = _map_unique(_one, [i for i, _ in sel], ctx.cfg.px_threads)
    return _result(sel, got, med_ps)


# ---- (f) keepfd: header prefix pass keeps every fd; picks pread from the same fd ----
def series_keepfd(sdir, k, ctx, fs=None):
    fs = glob.glob(sdir + "/*.dcm") if fs is None else fs
    if not fs:
        return None
    if len(fs) > keep_max():
        return series_prefix(sdir, k, ctx, fs)
    out = []
    try:
        with ThreadPoolExecutor(max_workers=ctx.cfg.hdr_threads) as ex:
            out = list(ex.map(lambda f: _hdr_prefix(ctx, f, True), fs))
        order, med_ps = _records([o[0] for o in out])
        sel = _picked(order, k)

        def _one(i):
            rec, data, info, fd, full = out[i]
            if data is None:
                return None, None
            try:
                if not full:
                    data = _read_rest(fd, data, len(data))
                    info = _scan_full(data) if ctx.cfg.fast else None
            except Exception:
                return None, None
            return ctx.px(data, info if ctx.cfg.fast else None)
        got = _map_unique(_one, [i for i, _ in sel], ctx.cfg.px_threads)
    finally:
        for o in out:
            if o[3] is not None:
                try: os.close(o[3])
                except OSError: pass
    return _result(sel, got, med_ps)


# ---- hybrid: onepass when most files get picked anyway, keepfd otherwise ----
def unique_pick_count(n, k):
    return len(set(pick_indices(n, k))) if n > 0 else 0


def series_hybrid(sdir, k, ctx):
    fs = glob.glob(sdir + "/*.dcm")
    if not fs:
        return None
    n = len(fs)
    if unique_pick_count(n, k) >= ctx.cfg.hybrid_frac * n:
        return series_onepass(sdir, k, ctx, fs)
    return series_keepfd(sdir, k, ctx, fs)


# ============================================================================
# Study driver shared by the variants (slot logic identical to v2)
# ============================================================================
def _fill_plain(vol, idx, MAXS, arrs, pss, med_ps, IMG, cropfirst, lh, mm=None):
    import cv2
    if lh is not None:
        loq, hiq = lh
    else:
        valid = [a for a in arrs if a is not None]
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
    for a, ps in zip(arrs, pss):
        if idx >= MAXS: break
        if a is None: idx += 1; continue
        if cropfirst:
            # the clip/normalise is element-wise, so apply it to the crop that is kept only
            p = ps if ps > 0 else med_ps
            h, w = a.shape; cpx = int(round(CROP_MM / max(p, 1e-3)))
            cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
            aw = np.clip((a[y0:y0 + cpx, x0:x0 + cpx] - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = cv2.resize(aw, (IMG, IMG), interpolation=cv2.INTER_AREA)
        else:
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = mm(aw, ps if ps > 0 else med_ps)
        vol[idx] = (aw * 255).astype(np.uint8); idx += 1
    return idx


def _fill(vol, idx, MAXS, res, IMG, mm, cfg):
    if cfg.lut:
        return _fill_lut(vol, idx, MAXS, res, IMG)
    arrs, pss, med_ps, hists = res
    lh = _pct_from_hists(arrs, hists) if cfg.pct else None
    if lh is None and not cfg.cropfirst:
        return _fill_slot(vol, idx, MAXS, arrs, pss, med_ps, mm)          # verbatim v2 tail
    return _fill_plain(vol, idx, MAXS, arrs, pss, med_ps, IMG, cfg.cropfirst, lh, mm)


def build_study_generic(sid, ser_records, tsdir, SLOTS, IMG, series_fn, cfg):
    MAXS = sum(s[2] for s in SLOTS)
    mm = _mm_crop_resize_fn(IMG)
    rows = ser_records.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set()
    for plane, fluid, k in SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k; continue
        used.add(r['SeriesInstanceUID'])
        res = series_fn(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}", k)
        if res is None:
            idx += k; continue
        idx = _fill(vol, idx, MAXS, res, IMG, mm, cfg)
        if idx >= MAXS: break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


def _hdr_whole(ctx, f):
    """onepass record in _hdr_prefix's shape: (rec, full bytes, info, None, full=True)."""
    try:
        data = _read_whole(f)
    except Exception:
        return (0.0, f, 0.5, False), None, None, None, False
    info = _scan_full(data) if ctx.cfg.fast else None
    return ctx.hdr_rec(f, data, info), data, info, None, True


def build_study_batched(sid, ser_records, tsdir, SLOTS, IMG, ctx, whole=False):
    """keepfd I/O for the WHOLE study at once: one pool pread-s the header prefix of every file
    of all 5 selected series (fds kept), the series are sorted, then one pool reads the rest of
    every unique pick of all series from the kept fds. v2 always runs all 5 slots (idx reaches
    MAXS only after the last one), so reading every series up front changes nothing but the
    number of latency round-trips in a row."""
    cfg = ctx.cfg
    MAXS = sum(s[2] for s in SLOTS)
    mm = _mm_crop_resize_fn(IMG)
    rows = ser_records.get(sid, [])
    used = set(); plan = []
    for plane, fluid, k in SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            plan.append((k, None)); continue
        used.add(r['SeriesInstanceUID'])
        plan.append((k, glob.glob(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}/*.dcm")))
    jobs = [(si, j, f) for si, (k, fs) in enumerate(plan) if fs for j, f in enumerate(fs)]
    keep = len(jobs) <= keep_max()
    outs = []
    per, sels, got = {}, {}, {}
    try:
        if jobs:
            with ThreadPoolExecutor(max_workers=min(cfg.hdr_threads, len(jobs))) as ex:
                outs = list(ex.map(lambda t: (_hdr_whole(ctx, t[2]) if whole
                                              else _hdr_prefix(ctx, t[2], keep)), jobs))
        for (si, j, f), o in zip(jobs, outs):
            per.setdefault(si, []).append(o)
        pj = []
        for si, (k, fs) in enumerate(plan):
            if not fs: continue
            order, med_ps = _records([o[0] for o in per[si]])
            sel = _picked(order, k)
            sels[si] = (sel, med_ps)
            pj += [(si, i) for i in dict.fromkeys(i for i, _ in sel)]

        def _one(job):
            si, i = job
            rec, data, info, fd, full = per[si][i]
            if data is None:
                return None, None
            try:
                if not full:
                    data = (_read_rest(fd, data, len(data)) if fd is not None
                            else _read_whole(plan[si][1][i]))
                    info = _scan_full(data) if cfg.fast else None
            except Exception:
                return None, None
            return ctx.px(data, info if cfg.fast else None)
        if pj and whole:                    # nothing left to read: decode in this thread
            got = {j: _one(j) for j in pj}
        elif pj:
            with ThreadPoolExecutor(max_workers=min(cfg.px_threads, len(pj))) as ex:
                got = dict(zip(pj, ex.map(_one, pj)))
    finally:
        for o in outs:
            if o[3] is not None:
                try: os.close(o[3])
                except OSError: pass
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0
    for si, (k, fs) in enumerate(plan):
        if not fs:
            idx += k; continue
        sel, med_ps = sels[si]
        res = _result(sel, {i: got[(si, i)] for i, _ in sel}, med_ps)
        idx = _fill(vol, idx, MAXS, res, IMG, mm, cfg)
        if idx >= MAXS: break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


def make_builder(name, IMG, SLOTS, **cfg_kw):
    """-> build(sid, ser_records, tsdir) for variant `name`: <io>[_fast][_pct][_crop]."""
    base, *tok = name.split("_")
    cfg_kw.setdefault("fast", "fast" in tok)
    cfg_kw.setdefault("pct", "pct" in tok)
    cfg_kw.setdefault("cropfirst", "crop" in tok)
    cfg_kw.setdefault("lut", "lut" in tok)
    cfg = Cfg(**cfg_kw)
    if base == "v2":
        reader = make_reader_v2(IMG, cfg.hdr_threads)
        return lambda sid, ser, tsdir: build_study_v2(sid, ser, tsdir, reader, SLOTS, IMG,
                                                      cfg.px_threads)
    ctx = _Ctx(cfg)
    if base == "dedupe":
        reader = make_reader_v2(IMG, cfg.hdr_threads)
        fn = lambda sdir, k: series_dedupe(sdir, k, ctx, reader)
    elif base == "onepass":
        fn = lambda sdir, k: series_onepass(sdir, k, ctx)
    elif base == "prefix":
        fn = lambda sdir, k: series_prefix(sdir, k, ctx)
    elif base == "keepfd":
        fn = lambda sdir, k: series_keepfd(sdir, k, ctx)
    elif base == "hybrid":
        fn = lambda sdir, k: series_hybrid(sdir, k, ctx)
    elif base == "study":
        return lambda sid, ser, tsdir: build_study_batched(sid, ser, tsdir, SLOTS, IMG, ctx)
    elif base == "studyone":
        return lambda sid, ser, tsdir: build_study_batched(sid, ser, tsdir, SLOTS, IMG, ctx, True)
    else:
        raise ValueError(name)
    return lambda sid, ser, tsdir: build_study_generic(sid, ser, tsdir, SLOTS, IMG, fn, cfg)


# I/O-only variants keep pydicom decoding + the verbatim tail; *_fast swaps in the header /
# pixel walker; *_pct adds the exact histogram percentile; *_crop normalises the crop only;
# *_lut = walker + code-count percentile + one clip/normalise table per series (no float image).
VARIANTS = ["v2", "dedupe", "onepass", "prefix", "keepfd", "hybrid",
            "onepass_fast", "prefix_fast", "keepfd_fast", "hybrid_fast",
            "onepass_fast_pct_crop", "prefix_fast_pct_crop", "keepfd_fast_pct_crop",
            "hybrid_fast_pct_crop", "onepass_lut", "prefix_lut", "keepfd_lut", "hybrid_lut",
            "study", "study_lut", "studyone", "studyone_lut"]


# convenience wrappers named as the task asks: build_study_<name>(sid, ser, tsdir, layout)
def _wrap(name):
    cache = {}

    def build(sid, ser_records, tsdir, layout="d96", **kw):
        key = (layout, tuple(sorted(kw.items())))
        if key not in cache:
            L = LAYOUTS[layout]
            cache[key] = make_builder(name, L["IMG"], L["SLOTS"], **kw)
        return cache[key](sid, ser_records, tsdir)
    build.__name__ = f"build_study_{name}"
    return build


for _n in VARIANTS:
    if _n != "v2":
        globals()[f"build_study_{_n}"] = _wrap(_n)
