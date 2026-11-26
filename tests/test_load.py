"""Loader test on a tiny synthetic file laid out like the real Part_N.mat
(dataset of object references, each record stored as (n_samples, 3))."""

import h5py
import numpy as np
import pytest

from bp.load import iter_records, part_number, record_id


def write_fake_part(path, records):
    with h5py.File(path, "w") as f:
        refs_group = f.create_group("#refs#")
        cells = f.create_dataset(f"Part_{part_number(path)}", (len(records), 1), h5py.ref_dtype)
        for i, rec in enumerate(records):
            cells[i, 0] = refs_group.create_dataset(str(i), data=rec).ref


def make_record(n, offset):
    """Columns: PPG, ABP, ECG, each filled with a recognisable value."""
    return np.column_stack(
        [np.full(n, 1.0 + offset), np.full(n, 100.0 + offset), np.full(n, -0.5 + offset)]
    )


def test_record_id_format():
    assert record_id(1, 42) == "1_0042"
    assert record_id(4, 2999) == "4_2999"


def test_iter_records_reads_columns_in_order(tmp_path):
    path = tmp_path / "Part_2.mat"
    write_fake_part(path, [make_record(1000, 0), make_record(2000, 1)])

    records = list(iter_records(path))

    assert [r.record_id for r in records] == ["2_0000", "2_0001"]
    assert len(records[1].ppg) == 2000
    first = records[0]
    assert first.ppg[0] == 1.0 and first.abp[0] == 100.0 and first.ecg[0] == -0.5


def test_iter_records_rejects_wrong_shape(tmp_path):
    path = tmp_path / "Part_1.mat"
    write_fake_part(path, [np.zeros((1000, 2))])
    with pytest.raises(ValueError, match="expected"):
        list(iter_records(path))
