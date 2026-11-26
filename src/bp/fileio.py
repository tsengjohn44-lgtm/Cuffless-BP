"""Write output files so another terminal never reads half a file.

The three track worktrees share one data folder. If one terminal is rewriting
features.csv while another reads it, the reader could see a half-written file.
`atomic_write` writes to a hidden temporary file next to the target and renames
it into place only when writing finished without error. A rename on the same
disk is atomic: readers see either the old file or the new one, never a mix.

    with atomic_write(INTERIM_DIR / "windows.npz") as f:
        np.savez(f, ppg=ppg, abp=abp, ecg=ecg)   # pass the file object, not a path
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO


@contextmanager
def atomic_write(path: Path, mode: str = "wb") -> Iterator[IO]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, mode) as f:
            yield f
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
