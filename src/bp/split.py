"""Record-level 70/15/15 train/val/test split with a fixed seed.

Plain-language summary:
- We split by record, never by window. Windows from one record are near copies of each
  other (same patient, minutes apart), so splitting by window would let the model see the
  test patient during training and look better than it is.
- Each record gets a fixed pseudo-random number between 0 and 1, computed from a hash of
  the seed and its record_id. Below 0.70 it goes to train, 0.70 to 0.85 to val, the rest
  to test. The same record always lands in the same split.
- Why a hash rather than shuffling the list of records: with a shuffle, removing one
  record (say after a bug fix in windows.py) moves every record after it, and records we
  tuned on in val could slide into test. With the hash a record's split depends only on
  its own id, so the others never move, even when more parts are added later.
- The fractions come out close to, not exactly, 70/15/15; main() prints the real ones.
- The split is saved to splits/split.json and committed, so anyone can check it.
  Only `evaluate.py --final` may use the test records: everything else calls
  `load_split()`, which leaves "test" out unless asked with final=True.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from bp.fileio import atomic_write
from bp.paths import INTERIM_DIR, SPLIT_FILE

SEED = 42
FRACTIONS = (0.70, 0.15, 0.15)  # train, val, test
SPLITS = ("train", "val", "test")


def record_number(record_id: str, seed: int = SEED) -> float:
    """A fixed pseudo-random number in [0, 1) for one record."""
    digest = hashlib.sha256(f"{seed}:{record_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign(record_id: str, seed: int = SEED, fractions=FRACTIONS) -> str:
    u = record_number(record_id, seed)
    if u < fractions[0]:
        return "train"
    if u < fractions[0] + fractions[1]:
        return "val"
    return "test"


def make_split(record_ids: Iterable[str], seed: int = SEED, fractions=FRACTIONS) -> dict:
    """{"seed", "fractions", "train", "val", "test"}, each id list sorted (contract §2)."""
    groups = {name: [] for name in SPLITS}
    for rid in sorted(set(record_ids)):
        groups[assign(rid, seed, fractions)].append(rid)
    return {"seed": seed, "fractions": list(fractions), **groups}


def save_split(split: dict, path: Path = SPLIT_FILE) -> None:
    with atomic_write(path, "w") as f:
        json.dump(split, f, indent=1)
        f.write("\n")


def load_split(path: Path = SPLIT_FILE, final: bool = False) -> dict:
    """Read the split. "test" is left out unless final=True (only `evaluate.py --final`)."""
    split = json.loads(Path(path).read_text())
    if not final:
        split.pop("test")
    return split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    windows_csv = INTERIM_DIR / "windows.csv"
    if not windows_csv.exists():
        print(f"{windows_csv} not found. Run `make windows` first.", file=sys.stderr)
        return 1
    record_ids = pd.read_csv(windows_csv, usecols=["record_id"], dtype=str)["record_id"]

    split = make_split(record_ids, args.seed)
    save_split(split)

    # Counts only: no labels are looked at here, so nothing about the test set leaks.
    windows_per_record = record_ids.value_counts()
    n_rec, n_win = len(windows_per_record), len(record_ids)
    print(f"Split of {n_rec} records / {n_win} windows (seed {args.seed}):")
    for name in SPLITS:
        recs = split[name]
        wins = int(windows_per_record[recs].sum())
        print(
            f"  {name:5s} {len(recs):5d} records ({100 * len(recs) / n_rec:4.1f}%)"
            f"  {wins:6d} windows ({100 * wins / n_win:4.1f}%)"
        )
    print(f"Wrote {SPLIT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
