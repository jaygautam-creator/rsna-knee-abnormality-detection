import os
import json
import shutil
import subprocess
from pathlib import Path

KAGGLE_USERNAME = "jaygautam007"
pkg_dir = Path("/content/kaggle_export")
pkg_dir.mkdir(parents=True, exist_ok=True)

ckpt_path = Path("/content/checkpoints/specialist_knee_colab_v1.pt")
if not ckpt_path.exists():
    latest = Path("/content/checkpoints/specialist_knee_colab_latest.pt")
    if latest.exists():
        ckpt_path = latest

assert ckpt_path.exists(), "No checkpoint found to export!"

dest_ckpt = pkg_dir / "specialist_knee_colab_v1.pt"
shutil.copy(ckpt_path, dest_ckpt)
print(f"Copied {ckpt_path} ({ckpt_path.stat().st_size / 1024 / 1024:.1f} MB) to {dest_ckpt}")

meta = {
    "title": "Knee Colab Specialist Weights",
    "id": f"{KAGGLE_USERNAME}/knee-colab-specialist-weights",
    "licenses": [{"name": "CC0-1.0"}]
}
with open(pkg_dir / "dataset-metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"Uploading dataset to Kaggle: {KAGGLE_USERNAME}/knee-colab-specialist-weights...")
# Try create, fallback to version
res_create = subprocess.run([
    "kaggle", "datasets", "create",
    "-p", str(pkg_dir),
    "-r", "tar"
], capture_output=True, text=True)

if res_create.returncode != 0 and "already exists" in (res_create.stderr + res_create.stdout).lower():
    print("Dataset already exists, pushing new version...")
    res_ver = subprocess.run([
        "kaggle", "datasets", "version",
        "-p", str(pkg_dir),
        "-m", "Updated Colab Specialist weights",
        "-r", "tar"
    ], capture_output=True, text=True)
    print("Version result:", res_ver.stdout or res_ver.stderr)
else:
    print("Create result:", res_create.stdout or res_create.stderr)

print(f"Done! Check dataset at: https://www.kaggle.com/datasets/{KAGGLE_USERNAME}/knee-colab-specialist-weights")
