"""Read records from the UCI Part_N.mat files.

Plain-language summary:
- The .mat files are MATLAB v7.3, which is really HDF5, so we read them with h5py.
- `python -m bp.load --inspect` prints the raw layout of a file without assuming
  anything, so we can check the shape and channel order before trusting a loader.
- `iter_records()` walks one file and yields each record's three signals.

Layout, verified on Part_1.mat with `make inspect` (Phase 0):
- The file holds one dataset named like the file ("Part_1") of shape (3000, 1).
  Each entry is a reference to one record (a MATLAB cell array).
- Each record is a float64 array of shape (n_samples, 3): one column per signal.
  MATLAB saves it as 3 x n_samples; h5py shows it transposed.
- Column 0 = PPG (fingertip, arbitrary units, range 0 to 4.002; hits both limits
  rarely, i.e. clipping), column 1 = ABP (mmHg, roughly 50-160), column 2 = ECG.
  Checked three ways: ABP is the only channel in a mmHg range, ECG is the only spiky
  one (kurtosis 7-13 vs about -1), and all three beat at the same heart rate.
- Records are 1,000 to 74,000 samples (8 s to 592 s at 125 Hz), in steps of 1,000.
  293 of the 3,000 records in Part_1 are shorter than one 10 s window.
"""

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np

from bp.paths import RAW_DIR

FS_HZ = 125  # sampling rate stated by the dataset authors
PPG, ABP, ECG = 0, 1, 2  # column order inside each record (verified, see above)


class Record(NamedTuple):
    record_id: str
    ppg: np.ndarray  # 1-D, arbitrary units
    abp: np.ndarray  # 1-D, mmHg
    ecg: np.ndarray  # 1-D


def record_id(part: int, index: int) -> str:
    """'1_0042' = Part_1, record 42 (0-based). Matches docs/data-contract.md."""
    return f"{part}_{index:04d}"


def part_number(path: Path) -> int:
    """Part_3.mat -> 3."""
    return int(Path(path).stem.split("_")[-1])


def iter_records(path: Path) -> Iterator[Record]:
    """Yield every record in one Part_N.mat file, in file order."""
    part = part_number(path)
    with h5py.File(path, "r") as f:
        refs = np.asarray(f[f"Part_{part}"]).ravel()
        for i, ref in enumerate(refs):
            rec = f[ref][()]
            if rec.ndim != 2 or rec.shape[1] != 3:
                raise ValueError(f"record {i} has shape {rec.shape}, expected (n_samples, 3)")
            yield Record(record_id(part, i), rec[:, PPG], rec[:, ABP], rec[:, ECG])


def describe(name: str, obj) -> str:
    if isinstance(obj, h5py.Dataset):
        return f"  dataset {name!r}: shape={obj.shape} dtype={obj.dtype}"
    return f"  group   {name!r}: keys={list(obj.keys())[:10]}"


def channel_stats(arr: np.ndarray, axis: int) -> list[str]:
    """One line per channel along `axis`: min / max / mean / std."""
    lines = []
    for i in range(arr.shape[axis]):
        ch = np.take(arr, i, axis=axis).astype(float)
        lines.append(
            f"    channel {i}: min={ch.min():9.3f} max={ch.max():9.3f} "
            f"mean={ch.mean():9.3f} std={ch.std():8.3f}  first 5={np.round(ch[:5], 3).tolist()}"
        )
    return lines


def inspect(path: Path, n_records: int = 3) -> None:
    print(f"file: {path} ({path.stat().st_size / 1e9:.2f} GB)")
    with h5py.File(path, "r") as f:
        print("top-level objects:")
        for key in f.keys():
            print(describe(key, f[key]))

        # A MATLAB cell array is stored as a dataset of object references.
        cell_keys = [
            k for k in f.keys() if isinstance(f[k], h5py.Dataset) and f[k].dtype == h5py.ref_dtype
        ]
        if not cell_keys:
            print("no cell array of references found; layout differs from expectation")
            return
        cells = f[cell_keys[0]]
        refs = np.asarray(cells).ravel()
        print(f"\ncell array {cell_keys[0]!r}: stored shape {cells.shape} -> {refs.size} records")

        lengths = [f[r].shape for r in refs]
        shapes = {tuple(sorted(s)) for s in lengths}
        n_samples = np.array([max(s) for s in lengths])
        more = " ..." if len(shapes) > 5 else ""
        print(f"record shapes (sorted dims) seen: {sorted(shapes)[:5]}{more}")
        print(
            f"samples per record: min={n_samples.min()} median={int(np.median(n_samples))} "
            f"max={n_samples.max()}  (= {n_samples.min() / FS_HZ:.0f} s to "
            f"{n_samples.max() / FS_HZ:.0f} s at {FS_HZ} Hz)"
        )

        for j in range(min(n_records, refs.size)):
            rec = np.asarray(f[refs[j]])
            ch_axis = int(np.argmin(rec.shape))  # the short axis holds the channels
            print(f"\nrecord {j}: array shape {rec.shape}, dtype {rec.dtype}")
            print(f"  channels on axis {ch_axis}")
            print("\n".join(channel_stats(rec, ch_axis)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect or load UCI Part_N.mat files")
    parser.add_argument("--inspect", action="store_true", help="print the raw layout")
    parser.add_argument("--file", type=Path, default=RAW_DIR / "Part_1.mat")
    parser.add_argument("--n", type=int, default=3, help="records to print with --inspect")
    args = parser.parse_args(argv)
    if not args.file.exists():
        print(f"{args.file} not found. Run `make data` first.", file=sys.stderr)
        return 1
    if args.inspect:
        inspect(args.file, args.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
