"""Guards for a public repo: no credentials and no competition data in tracked files."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN = {
    "Kaggle access token": re.compile(r"KGAT_[A-Za-z0-9]{8,}"),
    "kaggle.json key": re.compile(r"\"key\"\s*:\s*\"[0-9a-f]{32}\""),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "Anthropic/OpenAI key": re.compile(r"sk-(ant-)?[A-Za-z0-9_-]{30,}"),
    "Google API key": re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "competition study UID": re.compile(r"1\.2\.826\.0\.1\.3680043\.8\.498\.\d{10,}"),
}


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("not a git checkout")
    return [ROOT / f for f in out.splitlines() if f]


def test_no_secrets_or_competition_ids_in_tracked_files():
    hits = []
    for path in tracked_files():
        try:
            text = path.read_text(errors="ignore")
        except (IsADirectoryError, FileNotFoundError):
            continue
        for name, rx in FORBIDDEN.items():
            if rx.search(text) and path.name != "test_repo_hygiene.py":
                hits.append(f"{path.relative_to(ROOT)}: {name}")
    assert not hits, hits


def test_no_competition_data_files_tracked():
    bad = [p for p in tracked_files() if p.suffix.lower() in {".dcm", ".pt", ".pth", ".npy", ".npz", ".safetensors"}
           or p.name in {"train.csv", "test.csv", "sample_submission.csv", "kaggle.json", "access_token", "s6_parity.json"}]
    assert not bad, bad
