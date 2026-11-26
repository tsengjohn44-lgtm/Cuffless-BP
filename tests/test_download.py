"""Streaming zip extraction, the way bp.download reads the UCI zip."""

import io
import os
import zipfile

import pytest

from bp.download import DownloadCutOff, extract_from_stream

PART_1 = os.urandom(50_000) + b"ppg" * 20_000  # mix of random and compressible bytes
PART_2 = b"abp" * 30_000


class Unseekable(io.RawIOBase):
    """A write-only stream zipfile can't seek in, so it writes 'streamed' members
    (sizes in a data descriptor after the data), exactly like the UCI zip."""

    def __init__(self):
        self.buf = io.BytesIO()

    def writable(self):
        return True

    def write(self, b):
        return self.buf.write(b)


def zip_bytes(streamed=True):
    target = Unseekable() if streamed else io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dataset/readme.txt", b"hello")
        zf.writestr("dataset/Part_1.mat", PART_1)
        zf.writestr("dataset/Part_2.mat", PART_2)
    return (target.buf if streamed else target).getvalue()


@pytest.mark.parametrize("streamed", [True, False])
def test_extracts_wanted_member_and_skips_others(tmp_path, streamed):
    data = zip_bytes(streamed)
    out = tmp_path / "Part_2.mat"
    written = extract_from_stream(io.BytesIO(data), {"part_2.mat": out})
    assert written == [out]
    assert out.read_bytes() == PART_2
    assert not (tmp_path / "Part_1.mat").exists()


def test_stops_reading_once_targets_are_done(tmp_path):
    """Part_1 must come out even if the stream is cut right after it (the UCI 1 GiB cap)."""
    data = zip_bytes()
    cut = data.index(b"PK\x03\x04", data.index(b"Part_1.mat")) + 4
    out = tmp_path / "Part_1.mat"
    extract_from_stream(io.BytesIO(data[:cut]), {"part_1.mat": out})
    assert out.read_bytes() == PART_1


def test_cut_off_stream_raises_and_leaves_no_file(tmp_path):
    data = zip_bytes()
    cut = data.index(b"Part_2.mat") + 100  # inside Part_2's data
    out = tmp_path / "Part_2.mat"
    with pytest.raises(DownloadCutOff):
        extract_from_stream(io.BytesIO(data[:cut]), {"part_2.mat": out})
    assert list(tmp_path.iterdir()) == []


def test_corrupt_checksum_is_rejected(tmp_path):
    data = bytearray(zip_bytes())
    start = data.index(b"Part_2.mat")
    crc_at = data.index(b"PK\x07\x08", start) + 4  # Part_2's data descriptor CRC
    data[crc_at] ^= 0xFF
    out = tmp_path / "Part_2.mat"
    with pytest.raises(ValueError, match="checksum"):
        extract_from_stream(io.BytesIO(bytes(data)), {"part_2.mat": out})
    assert not out.exists()


def test_missing_member_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="part_3.mat"):
        extract_from_stream(io.BytesIO(zip_bytes()), {"part_3.mat": tmp_path / "Part_3.mat"})
