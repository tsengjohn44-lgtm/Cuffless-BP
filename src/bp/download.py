"""Download the UCI Cuff-Less Blood Pressure dataset (id 340) into data/raw.

Plain-language summary:
- UCI serves the dataset as one zip (about 3 GB) holding Part_1.mat .. Part_4.mat.
- The server does not support partial downloads, so we must fetch the whole
  zip once. We keep the zip so other parts can be unzipped later without
  downloading again.
- By default we unzip only Part_1.mat (about 0.8 GB), which is what we develop on.
- If a file is already there, we skip it. Sizes are printed so you can see progress.

Usage:
    python -m bp.download            # Part_1 only (development)
    python -m bp.download --parts 1 2 3 4   # all parts (final run)

Dataset: UCI "Cuff-Less Blood Pressure Estimation", CC BY 4.0.
https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation
"""

import argparse
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from bp.paths import RAW_DIR

URL = "https://archive.ics.uci.edu/static/public/340/cuff+less+blood+pressure+estimation.zip"
ZIP_NAME = "cuff_less_bp.zip"
CHUNK = 8 * 1024 * 1024  # 8 MB


def human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def download_zip(dest: Path) -> None:
    """Stream the zip to dest. Writes to a .part file first so a broken
    download never looks like a finished one."""
    if dest.exists():
        print(f"zip already present: {dest} ({human(dest.stat().st_size)}), skipping download")
        return
    tmp = dest.with_suffix(".zip.part")
    print(f"downloading {URL}\n  -> {dest}")
    start = time.time()
    done = 0
    last_print = 0.0
    with urllib.request.urlopen(URL) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(CHUNK):
            out.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last_print > 10:
                rate = done / max(now - start, 1e-9)
                print(f"  {human(done)} downloaded ({human(rate)}/s)", flush=True)
                last_print = now
    tmp.rename(dest)
    print(f"download finished: {human(done)} in {time.time() - start:.0f} s")


def find_member(zf: zipfile.ZipFile, part: int) -> zipfile.ZipInfo:
    """Find Part_<n>.mat inside the zip, wherever it sits in the folder tree."""
    wanted = f"part_{part}.mat"
    for info in zf.infolist():
        if Path(info.filename).name.lower() == wanted:
            return info
    names = ", ".join(i.filename for i in zf.infolist())
    raise FileNotFoundError(f"Part_{part}.mat not found in zip. Zip contains: {names}")


def extract_parts(zip_path: Path, parts: list[int]) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for part in parts:
            target = RAW_DIR / f"Part_{part}.mat"
            if target.exists():
                print(f"{target.name} already present ({human(target.stat().st_size)}), skipping")
                continue
            info = find_member(zf, part)
            print(f"unzipping {info.filename} ({human(info.file_size)}) -> {target}")
            tmp = target.with_suffix(".mat.part")
            with zf.open(info) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, CHUNK)
            tmp.rename(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--parts", type=int, nargs="+", default=[1], choices=[1, 2, 3, 4])
    args = parser.parse_args(argv)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    targets = [RAW_DIR / f"Part_{p}.mat" for p in args.parts]
    if all(t.exists() for t in targets):
        for t in targets:
            print(f"{t.name} already present ({human(t.stat().st_size)}), nothing to do")
        return 0

    zip_path = RAW_DIR / ZIP_NAME
    download_zip(zip_path)
    extract_parts(zip_path, args.parts)
    for t in targets:
        print(f"ready: {t} ({human(t.stat().st_size)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
