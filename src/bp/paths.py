"""Where data and results live on disk.

All tracks import paths from here so nobody hard-codes a folder.
The data folder can be moved with the BP_DATA_DIR environment variable
(see .env.example). That lets several git worktrees share one copy of
the 3 GB dataset instead of downloading it once per worktree.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Empty or unset -> ./data. A relative path is taken relative to the repo root,
# so it means the same thing no matter which folder a script is started from.
DATA_DIR = Path(os.environ.get("BP_DATA_DIR") or "data").expanduser()
if not DATA_DIR.is_absolute():
    DATA_DIR = REPO_ROOT / DATA_DIR
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
RESULTS_DIR = REPO_ROOT / "results"

# The split lives in the repo (not in data/) so it is committed and reviewable.
SPLIT_FILE = REPO_ROOT / "splits" / "split.json"
