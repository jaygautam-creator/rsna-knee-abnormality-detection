import os
import shutil
import subprocess
from pathlib import Path

kdir = Path.home() / ".kaggle"
kdir.mkdir(parents=True, exist_ok=True)
token_path = Path("/content/access_token")
if token_path.exists():
    shutil.copy(token_path, kdir / "access_token")
    os.chmod(kdir / "access_token", 0o600)
    token = token_path.read_text().strip()
    os.environ["KAGGLE_API_TOKEN"] = token
    print("Copied access_token to", kdir / "access_token")

res = subprocess.run(["kaggle", "datasets", "list", "-m"], capture_output=True, text=True)
print("Kaggle Auth Status:", "SUCCESS" if res.returncode == 0 else f"FAILED ({res.stderr})")
lines = res.stdout.strip().splitlines()
for line in lines[:5]:
    print(" ", line)
