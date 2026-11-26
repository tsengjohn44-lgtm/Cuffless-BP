"""Download the UCI Cuff-Less Blood Pressure dataset (id 340) into data/raw.

Plain-language summary:
- UCI serves one zip (about 3 GB) holding Part_1.mat .. Part_4.mat, in that order.
- The UCI server cuts every download off at exactly 1 GiB and does not support
  resuming. So we never try to save the whole zip. Instead we read the zip
  as it streams in, unpack the members we want on the fly, and stop as soon as
  we have them. Part_1.mat is first and ends about 840 MB into the stream, so it
  always fits under the cutoff. Parts 2-4 lie past it and cannot be fetched this way.
- Every unpacked file is checked against the CRC32 checksum stored in the zip,
  so a cut-off or corrupted download is caught, never silently used.
- If a file is already there, we skip it.

Usage:
    python -m bp.download                     # Part_1 from UCI (development set)
    python -m bp.download --zip path/to.zip   # unpack from a zip you already have
    python -m bp.download --parts 1 2         # more parts (needs a full local zip)

Dataset: UCI "Cuff-Less Blood Pressure Estimation", CC BY 4.0.
https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation
"""

import argparse
import http.client
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

from bp.paths import RAW_DIR

URL = "https://archive.ics.uci.edu/static/public/340/cuff+less+blood+pressure+estimation.zip"
CHUNK = 8 * 1024 * 1024  # 8 MB

LOCAL_HEADER = b"PK\x03\x04"
CENTRAL_HEADER = b"PK\x01\x02"
DESCRIPTOR = b"PK\x07\x08"


class DownloadCutOff(Exception):
    """The stream ended before the files we wanted were complete."""


def human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


class Stream:
    """A byte stream we can read exact amounts from and push bytes back into."""

    def __init__(self, raw, progress: bool = False):
        self.raw = raw
        self.buf = b""
        self.n_read = 0
        self.progress = progress
        self.start = self.last_print = time.time()

    def read(self, n: int) -> bytes:
        if self.buf:
            out, self.buf = self.buf[:n], self.buf[n:]
            return out
        try:
            data = self.raw.read(n)
        except (http.client.IncompleteRead, ConnectionError, TimeoutError) as e:
            raise DownloadCutOff(f"connection closed after {human(self.n_read)}") from e
        self.n_read += len(data)
        if self.progress and time.time() - self.last_print > 10:
            rate = self.n_read / max(time.time() - self.start, 1e-9)
            print(f"  {human(self.n_read)} read ({human(rate)}/s)", flush=True)
            self.last_print = time.time()
        return data

    def read_upto(self, n: int) -> bytes:
        """Read n bytes, or fewer only if the stream ends."""
        parts, need = [], n
        while need:
            data = self.read(need)
            if not data:
                break
            parts.append(data)
            need -= len(data)
        return b"".join(parts)

    def read_exact(self, n: int) -> bytes:
        data = self.read_upto(n)
        if len(data) < n:
            raise DownloadCutOff(f"stream ended after {human(self.n_read)}")
        return data

    def unread(self, data: bytes) -> None:
        self.buf = data + self.buf


def _zip64_sizes(extra: bytes, csize: int, usize: int) -> tuple[int, int]:
    """Real sizes for files over 4 GB live in the 'zip64' extra field."""
    i = 0
    while i + 4 <= len(extra):
        field_id, field_len = struct.unpack("<HH", extra[i : i + 4])
        if field_id == 1:
            values = list(struct.unpack(f"<{field_len // 8}Q", extra[i + 4 : i + 4 + field_len]))
            if usize == 0xFFFFFFFF and values:
                usize = values.pop(0)
            if csize == 0xFFFFFFFF and values:
                csize = values.pop(0)
        i += 4 + field_len
    return csize, usize


def _read_descriptor_crc(s: Stream) -> int:
    """Read the data descriptor that follows a streamed member; return its CRC32."""
    first = s.read_exact(4)
    if first == DESCRIPTOR:
        first = s.read_exact(4)
    (crc,) = struct.unpack("<I", first)
    s.read_exact(8)  # compressed + uncompressed size, 4 bytes each
    # Normally the next zip header follows now. If not, this was a zip64
    # descriptor whose sizes are 8 bytes each: skip the 8 bytes still left.
    peek = s.read_upto(4)
    if len(peek) == 4 and peek not in (LOCAL_HEADER, CENTRAL_HEADER):
        s.read_exact(4)
        peek = s.read_upto(4)
    s.unread(peek)
    return crc


def _copy_member(s: Stream, name, method, streamed, csize, header_crc, out) -> tuple[int, int]:
    """Decode one member's data from the stream into `out` (or nowhere if None).
    Returns (CRC32 of what we decoded, CRC32 the zip says it should be)."""
    got_crc = 0
    inflater = zlib.decompressobj(-15) if method == 8 else None

    def next_chunk(size: int) -> bytes:
        chunk = s.read(size)
        if not chunk:
            raise DownloadCutOff(f"stream ended inside {name} after {human(s.n_read)}")
        return chunk

    def keep(data: bytes) -> None:
        nonlocal got_crc
        got_crc = zlib.crc32(data, got_crc)
        if out:
            out.write(data)

    if streamed:  # size unknown up front: decode until the deflate stream ends
        while not inflater.eof:
            keep(inflater.decompress(next_chunk(CHUNK)))
        s.unread(inflater.unused_data)
        return got_crc, _read_descriptor_crc(s)

    left = csize  # size known: read exactly csize bytes
    while left > 0:
        chunk = next_chunk(min(CHUNK, left))
        left -= len(chunk)
        keep(inflater.decompress(chunk) if inflater else chunk)
    if inflater:
        keep(inflater.flush())
    return got_crc, header_crc


def extract_from_stream(raw, targets: dict[str, Path], progress: bool = False) -> list[Path]:
    """Unpack the zip members named in `targets` (lowercase base name -> output path)
    from a zip byte stream. Stops reading once every target is written.
    Each file is written to <name>.part first and renamed only after its CRC matches."""
    s = Stream(raw, progress)
    remaining = dict(targets)
    written = []
    while remaining:
        sig = s.read(4)
        if sig != LOCAL_HEADER:
            missing = ", ".join(sorted(remaining))
            raise FileNotFoundError(f"reached the end of the zip without finding: {missing}")
        header = s.read_exact(26)
        _, flag, method, _, _, crc, csize, usize, nlen, xlen = struct.unpack("<HHHHHIIIHH", header)
        name = s.read_exact(nlen).decode("utf-8" if flag & 0x800 else "cp437")
        csize, usize = _zip64_sizes(s.read_exact(xlen), csize, usize)
        key = Path(name).name.lower()
        target = remaining.get(key)
        tmp = target.with_name(target.name + ".part") if target else None
        if target:
            print(f"unpacking {name} -> {target}")
        if method not in (0, 8):
            raise ValueError(f"{name}: unsupported compression method {method}")
        streamed = bool(flag & 0x8)
        if streamed and method != 8:
            raise ValueError(f"{name}: cannot stream an uncompressed member without its size")

        # Members we don't want are still decoded (and thrown away): with a
        # streamed member that's the only way to find where it ends.
        out = open(tmp, "wb") if tmp else None
        try:
            got_crc, crc = _copy_member(s, name, method, streamed, csize, crc, out)
        except BaseException:
            if out:
                out.close()
                tmp.unlink(missing_ok=True)
            raise
        if out:
            out.close()
            if got_crc != crc:
                tmp.unlink(missing_ok=True)
                raise ValueError(f"{name}: checksum mismatch, the download is corrupt")
            tmp.rename(target)
            print(f"  ok: {human(target.stat().st_size)}, CRC32 {crc:#010x} matches")
            written.append(target)
            del remaining[key]
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download UCI dataset 340 into data/raw")
    parser.add_argument("--parts", type=int, nargs="+", default=[1], choices=[1, 2, 3, 4])
    parser.add_argument("--zip", type=Path, help="unpack from this local zip instead of UCI")
    args = parser.parse_args(argv)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    targets = {}
    for p in args.parts:
        path = RAW_DIR / f"Part_{p}.mat"
        if path.exists():
            print(f"{path.name} already present ({human(path.stat().st_size)}), skipping")
        else:
            targets[f"part_{p}.mat"] = path
    if not targets:
        return 0

    try:
        if args.zip:
            with open(args.zip, "rb") as f:
                extract_from_stream(f, targets)
        else:
            print(f"streaming {URL}")
            with urllib.request.urlopen(URL, timeout=60) as resp:
                extract_from_stream(resp, targets, progress=True)
    except DownloadCutOff as e:
        print(
            f"\nDownload cut off: {e}.\nThe UCI server stops every download at 1 GiB. "
            "Part_1 fits under that; re-run `make data` if Part_1 was cut off.\n"
            "Parts 2-4 need a complete zip from elsewhere: pass it with --zip.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
