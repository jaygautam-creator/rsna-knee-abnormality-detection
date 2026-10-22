import os
import subprocess
import time
from pathlib import Path

data_dir = Path("/content/data")
labels_dir = Path("/content/labels")
ckpt_dir = Path("/content/checkpoints")

data_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

# 1. Download calibrated labels
print("[1/2] Downloading calibrated labels (500 KB)...")
t0 = time.time()
subprocess.run([
    "kaggle", "datasets", "download",
    "-d", "jaygautam007/knee-labels-3state-calibrated",
    "-p", str(labels_dir),
    "--unzip", "-q"
], check=True)
print(f"Labels downloaded in {time.time() - t0:.1f}s. Files:")
for f in labels_dir.glob("*.csv"):
    print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

# 2. Check if volumes already exist (e.g. from previous run)
existing_npz = list(data_dir.rglob("*.npz"))
if len(existing_npz) > 1000:
    print(f"[2/2] Processed volumes already present! Found {len(existing_npz)} .npz files.")
else:
    print("[2/2] Downloading processed 3D MRI volumes (~17GB from barun2104)...")
    print("This takes ~2-3 minutes over Colab's high-speed connection.")
    t0 = time.time()
    res = subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "barun2104/rsna-knee-mri-processed-3d-volumes",
        "-p", str(data_dir),
        "--unzip", "-q"
    ])
    print(f"Volume download completed in {time.time() - t0:.1f}s (code: {res.returncode})")
    npz_count = len(list(data_dir.rglob("*.npz")))
    print(f"Total .npz volume files in {data_dir}: {npz_count}")
