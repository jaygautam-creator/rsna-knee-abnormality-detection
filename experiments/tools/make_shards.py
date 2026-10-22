"""Emit the three 336 px cache shard kernels from one template.

A 336 px cache is 35.8 GB and /kaggle/working caps at 20, so the run is split
three ways. The three kernels differ by exactly one integer, so they are generated
rather than copied: hand-maintained copies drift, and a shard whose SHARD constant
disagrees with its kernel id would overwrite a third of the corpus with a duplicate
of another third — and index.csv would look perfectly consistent afterwards.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
SRC = ROOT / "experiments" / "kernels" / "cache_volumes_336" / "source.py"
N_SHARDS = 3

template = SRC.read_text()
assert re.search(r"^SHARD = 0$", template, re.M), "template SHARD constant not found"
assert re.search(rf"^N_SHARDS = {N_SHARDS}$", template, re.M), "N_SHARDS mismatch"

for k in range(N_SHARDS):
    d = ROOT / "experiments" / "kernels" / f"cache_volumes_336_s{k}"
    d.mkdir(exist_ok=True)
    body = re.sub(r"^SHARD = 0$", f"SHARD = {k}", template, count=1, flags=re.M)
    body = body.replace("shard SHARD_PLACEHOLDER of 3", f"shard {k} of {N_SHARDS}")
    (d / "source.py").write_text(body)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"<kaggle-username>/knee-volume-cache-336-s{k}",
        "title": f"Knee Volume Cache 336 S{k}",
        "code_file": f"cache-volumes-336-s{k}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [],
        "kernel_sources": [],
    }, indent=2) + "\n")
    subprocess.run([sys.executable, str(ROOT / "src/knee/tools/py2nb.py"),
                    str(d / "source.py"), str(d / f"cache-volumes-336-s{k}.ipynb")],
                   check=True)
    print(f"built {d.relative_to(ROOT)}")
