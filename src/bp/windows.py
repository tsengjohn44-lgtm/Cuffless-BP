"""Cut records into 10 s windows, label SBP/DBP from arterial pressure, drop bad windows.

Plain-language summary:
- Each record is cut into back-to-back 10 s windows (1,250 samples at 125 Hz), starting at
  sample 0. The leftover tail shorter than 10 s is dropped, so records shorter than 10 s
  give no windows at all.
- The labels come from the invasive arterial pressure (ABP), the ground truth. We find the
  top (systolic) and bottom (diastolic) of every heartbeat in the window. SBP is the median
  of the tops and DBP the median of the bottoms, so one odd beat barely moves the label.
- A window is dropped when its labels or its PPG look unreliable (FILTERS below). Each
  dropped window is counted under the first filter it fails, in the order listed, so the
  counts add up to the total. They go to results/window_filters.json for the write-up.

Thresholds, checked on Part_1 by plotting windows (Phase 1):
- Beats: a peak must be at least 41 samples (0.33 s) after the previous one, the shortest
  beat at 180 bpm. It must also stand out from its surroundings (scipy's "prominence") by
  at least 30% of the window's pressure swing and at least 5 mmHg. That keeps the dicrotic
  notch (the small bump after each beat) from counting as a second beat. Windows where the
  pressure jumps or drifts (line flushes, transducer moves) then show too few beats and are
  dropped by the "fewer than 5 beats" filter.
- PPG: Part_1 has no flat windows (the smallest 10 s swing is 0.33 a.u.); the dataset
  authors already removed them, so that filter is a guard for other parts. Clipping is
  real: in about 1% of windows the PPG sits at the sensor's floor (0) or ceiling (4.002)
  for a while, and every such window we plotted was a dropout or motion artifact.
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from bp.fileio import atomic_write
from bp.load import FS_HZ, Record, iter_records
from bp.paths import INTERIM_DIR, RAW_DIR, RESULTS_DIR

WINDOW_S = 10
WINDOW = WINDOW_S * FS_HZ  # 1,250 samples

# Beat detection on ABP.
MAX_HR_BPM = 180
MIN_BEAT_DISTANCE = int(FS_HZ * 60 / MAX_HR_BPM)  # 41 samples
PROMINENCE_FRACTION = 0.3  # of the window's pressure swing (95th - 5th percentile)
MIN_PROMINENCE_MMHG = 5.0

# Quality filters.
MIN_BEATS = 5
SBP_RANGE_MMHG = (60.0, 220.0)
DBP_RANGE_MMHG = (30.0, 130.0)
PPG_MIN_RANGE = 0.1  # a.u.; below this the PPG is flat
PPG_FLOOR, PPG_CEILING = 0.0, 4.002  # sensor limits, verified on Part_1 (make inspect)
PPG_LIMIT_TOLERANCE = 0.01  # a.u.; "at the limit" means within this of 0 or 4.002
PPG_MAX_CLIPPED_S = 0.1  # clipped = at a limit for at least this long in total

# Filter names, in the order they are applied. These are the keys of window_filters.json.
MISSING = "missing samples (NaN)"
FEW_BEATS = f"fewer than {MIN_BEATS} beats"
SBP_OUT = "SBP outside {:g}-{:g} mmHg".format(*SBP_RANGE_MMHG)
DBP_OUT = "DBP outside {:g}-{:g} mmHg".format(*DBP_RANGE_MMHG)
SBP_LE_DBP = "SBP <= DBP"
FLAT_PPG = "flat PPG"
CLIPPED_PPG = "clipped PPG"
FILTERS = [MISSING, FEW_BEATS, SBP_OUT, DBP_OUT, SBP_LE_DBP, FLAT_PPG, CLIPPED_PPG]

COLUMNS = ["window_id", "record_id", "start_sample", "sbp_mmhg", "dbp_mmhg", "n_beats"]
SIGNALS = ["ppg", "abp", "ecg"]
WINDOWS_CSV = INTERIM_DIR / "windows.csv"
WINDOWS_NPZ = INTERIM_DIR / "windows.npz"
FILTERS_JSON = RESULTS_DIR / "window_filters.json"


def find_beats(abp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices of beat peaks (systolic) and troughs (diastolic) in one ABP window."""
    swing = np.percentile(abp, 95) - np.percentile(abp, 5)
    prominence = max(PROMINENCE_FRACTION * swing, MIN_PROMINENCE_MMHG)
    peaks, _ = find_peaks(abp, distance=MIN_BEAT_DISTANCE, prominence=prominence)
    troughs, _ = find_peaks(-abp, distance=MIN_BEAT_DISTANCE, prominence=prominence)
    return peaks, troughs


def label_window(abp: np.ndarray) -> tuple[float, float, int]:
    """(SBP, DBP, number of beats) for one ABP window. NaN when no peak/trough is found."""
    peaks, troughs = find_beats(abp)
    sbp = float(np.median(abp[peaks])) if len(peaks) else np.nan
    dbp = float(np.median(abp[troughs])) if len(troughs) else np.nan
    return sbp, dbp, len(peaks)


def ppg_is_flat(ppg: np.ndarray) -> bool:
    return bool(np.ptp(ppg) < PPG_MIN_RANGE)


def ppg_is_clipped(ppg: np.ndarray) -> bool:
    at_limit = (ppg <= PPG_FLOOR + PPG_LIMIT_TOLERANCE) | (ppg >= PPG_CEILING - PPG_LIMIT_TOLERANCE)
    return bool(at_limit.sum() >= PPG_MAX_CLIPPED_S * FS_HZ)


def failed_filter(ppg: np.ndarray, sbp: float, dbp: float, n_beats: int) -> str | None:
    """Name of the first filter this window fails, or None if it passes them all.

    Written as `not low <= x <= high` so a NaN label fails the range check.
    """
    if n_beats < MIN_BEATS:
        return FEW_BEATS
    if not SBP_RANGE_MMHG[0] <= sbp <= SBP_RANGE_MMHG[1]:
        return SBP_OUT
    if not DBP_RANGE_MMHG[0] <= dbp <= DBP_RANGE_MMHG[1]:
        return DBP_OUT
    if sbp <= dbp:
        return SBP_LE_DBP
    if ppg_is_flat(ppg):
        return FLAT_PPG
    if ppg_is_clipped(ppg):
        return CLIPPED_PPG
    return None


def check_window(
    ppg: np.ndarray, abp: np.ndarray, ecg: np.ndarray
) -> tuple[str | None, float, float, int]:
    """(first failed filter or None, SBP, DBP, number of beats) for one window."""
    if not all(np.isfinite(x).all() for x in (ppg, abp, ecg)):
        return MISSING, np.nan, np.nan, 0
    sbp, dbp, n_beats = label_window(abp)
    return failed_filter(ppg, sbp, dbp, n_beats), sbp, dbp, n_beats


def window_starts(n_samples: int) -> range:
    """First sample of every full 10 s window; the tail shorter than 10 s is dropped."""
    return range(0, n_samples - WINDOW + 1, WINDOW)


def process_record(rec: Record) -> tuple[list[dict], dict[str, np.ndarray], Counter]:
    """Windows of one record that pass every filter, their signals, and the drop counts."""
    rows, kept_starts, dropped = [], [], Counter()
    for start in window_starts(len(rec.ppg)):
        sl = slice(start, start + WINDOW)
        reason, sbp, dbp, n_beats = check_window(rec.ppg[sl], rec.abp[sl], rec.ecg[sl])
        if reason is not None:
            dropped[reason] += 1
            continue
        rows.append(
            {
                "record_id": rec.record_id,
                "start_sample": start,
                "sbp_mmhg": sbp,
                "dbp_mmhg": dbp,
                "n_beats": n_beats,
            }
        )
        kept_starts.append(start)
    signals = {
        name: np.array(
            [getattr(rec, name)[s : s + WINDOW] for s in kept_starts], dtype=np.float32
        ).reshape(-1, WINDOW)
        for name in SIGNALS
    }
    return rows, signals, dropped


def build_windows(records: Iterable[Record]) -> tuple[pd.DataFrame, dict, dict, int]:
    """Run every record. Returns (table, signals, filter counts, records shorter than 10 s)."""
    rows, chunks, dropped = [], {name: [] for name in SIGNALS}, Counter()
    n_short = 0
    for rec in records:
        if len(rec.ppg) < WINDOW:
            n_short += 1
            continue
        rec_rows, rec_signals, rec_dropped = process_record(rec)
        rows += rec_rows
        dropped += rec_dropped
        for name in SIGNALS:
            chunks[name].append(rec_signals[name])

    table = pd.DataFrame(rows, columns=COLUMNS[1:])
    table.insert(0, "window_id", np.arange(len(table)))
    signals = {
        name: np.concatenate(chunks[name]) if chunks[name] else np.empty((0, WINDOW), np.float32)
        for name in SIGNALS
    }
    counts = {name: dropped[name] for name in FILTERS}
    counts["kept"] = len(table)
    return table, signals, counts, n_short


def write_outputs(
    table: pd.DataFrame,
    signals: dict,
    counts: dict,
    interim_dir: Path = INTERIM_DIR,
    filters_json: Path = FILTERS_JSON,
) -> None:
    with atomic_write(interim_dir / "windows.csv", "w") as f:
        table.to_csv(f, index=False, float_format="%.3f")
    with atomic_write(interim_dir / "windows.npz") as f:
        np.savez(f, **signals)
    with atomic_write(filters_json, "w") as f:
        json.dump(counts, f, indent=2)
        f.write("\n")


def load_windows(interim_dir: Path = INTERIM_DIR) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Read windows.csv and windows.npz. Row i of the table = row i of each array."""
    table = pd.read_csv(interim_dir / "windows.csv", dtype={"record_id": str})
    with np.load(interim_dir / "windows.npz") as npz:
        signals = {name: npz[name] for name in SIGNALS}
    return table, signals


def print_summary(counts: dict, n_short: int, table: pd.DataFrame, parts: list[int]) -> None:
    total = sum(counts.values())
    print(f"Parts {parts}: {n_short} records shorter than {WINDOW_S} s give no windows.")
    print(f"{total} full {WINDOW_S} s windows from the other records:")
    for name, n in counts.items():
        print(f"  {name:28s} {n:7d}  ({100 * n / max(total, 1):5.1f}%)")
    print(f"{table['record_id'].nunique()} records keep at least one window.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", type=int, nargs="+", default=[1], help="Part_N.mat to use")
    args = parser.parse_args(argv)

    paths = [RAW_DIR / f"Part_{p}.mat" for p in args.parts]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(f"Not found: {', '.join(missing)}. Run `make data` first.", file=sys.stderr)
        return 1

    records = (rec for path in paths for rec in iter_records(path))
    table, signals, counts, n_short = build_windows(records)
    write_outputs(table, signals, counts)
    print_summary(counts, n_short, table, args.parts)
    print(f"Wrote {WINDOWS_CSV}, {WINDOWS_NPZ.name} and {FILTERS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
