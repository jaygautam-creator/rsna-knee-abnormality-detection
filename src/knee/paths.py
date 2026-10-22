"""Repository paths. Competition data never lives in the repo; point KNEE_DATA_DIR at your local copy."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("KNEE_DATA_DIR", REPO_ROOT / "data"))
ARTIFACTS_DIR = Path(os.environ.get("KNEE_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
PIPELINE_DIR = REPO_ROOT / "pipeline"
LOCAL_DIR = PIPELINE_DIR / "local"          # gitignored: values that must not be published
