"""Windowing, labels and quality filters on synthetic signals with known answers."""

import json

import numpy as np
import pytest
from scipy.signal import find_peaks

from bp.load import FS_HZ, Record
from bp.windows import (
    CLIPPED_PPG,
    DBP_OUT,
    FEW_BEATS,
    FILTERS,
    FLAT_PPG,
    MIN_BEAT_DISTANCE,
    MISSING,
    SBP_LE_DBP,
    SBP_OUT,
    WINDOW,
    build_windows,
    check_window,
    failed_filter,
    label_window,
    load_windows,
    process_record,
    write_outputs,
)


def synthetic_abp(sbp=120.0, dbp=80.0, hr_bpm=60.0, n=WINDOW, dicrotic=0.15):
    """Arterial-pressure-like wave: each beat rises to `sbp` in 0.12 s, then decays
    exponentially back to `dbp` at the start of the next beat, with a small dicrotic bump
    0.4 s after the peak (50 samples, far enough that distance alone would not reject it)."""
    t = np.arange(n) / FS_HZ
    period = 60 / hr_bpm
    phase = t % period
    rise, tau = 0.12, 0.3
    end = np.exp(-(period - rise) / tau)
    decay = (np.exp(-(phase - rise) / tau) - end) / (1 - end)
    wave = np.where(phase < rise, np.sin(np.pi / 2 * phase / rise) ** 2, decay)
    bump = dicrotic * np.exp(-0.5 * ((phase - rise - 0.4) / 0.04) ** 2)
    return dbp + (sbp - dbp) * (wave + bump)


def good_ppg(n=WINDOW):
    """Pulsing between 1 and 3 a.u., well inside the sensor's 0 to 4.002 range."""
    return 2.0 + np.sin(2 * np.pi * np.arange(n) / FS_HZ)


def ecg(n=WINDOW):
    return np.zeros(n)


# --- labels -------------------------------------------------------------------------


def test_label_window_finds_sbp_dbp_and_beats():
    sbp, dbp, n_beats = label_window(synthetic_abp(sbp=120, dbp=80, hr_bpm=60))
    assert sbp == pytest.approx(120, abs=0.5)
    assert dbp == pytest.approx(80, abs=0.5)
    assert n_beats == 10  # 60 bpm for 10 s


def test_dicrotic_bump_is_not_counted_as_a_beat():
    abp = synthetic_abp(hr_bpm=60)
    naive, _ = find_peaks(abp, distance=MIN_BEAT_DISTANCE)
    assert len(naive) == 20  # without the prominence rule every bump looks like a beat
    assert label_window(abp)[2] == 10


def test_fast_heart_rate_keeps_every_beat():
    # 150 bpm = one beat every 50 samples, above the 41-sample minimum distance
    assert label_window(synthetic_abp(hr_bpm=150))[2] == 25


# --- filters ------------------------------------------------------------------------


def clipped_ppg(n_at_limit, value):
    ppg = good_ppg()
    ppg[100 : 100 + n_at_limit] = value
    return ppg


@pytest.mark.parametrize(
    ("ppg", "abp", "expected"),
    [
        (good_ppg(), synthetic_abp(), None),
        (good_ppg(), synthetic_abp(hr_bpm=24), FEW_BEATS),  # 4 beats in 10 s
        (good_ppg(), 80 + np.random.default_rng(0).normal(0, 0.5, WINDOW), FEW_BEATS),
        (good_ppg(), synthetic_abp(sbp=240, dbp=100), SBP_OUT),
        (good_ppg(), synthetic_abp(sbp=150, dbp=135), DBP_OUT),
        (good_ppg(), synthetic_abp(sbp=90, dbp=20), DBP_OUT),
        (np.full(WINDOW, 2.0), synthetic_abp(), FLAT_PPG),
        (clipped_ppg(20, 0.0), synthetic_abp(), CLIPPED_PPG),  # 0.16 s at the floor
        (clipped_ppg(20, 4.002), synthetic_abp(), CLIPPED_PPG),  # 0.16 s at the ceiling
        (clipped_ppg(10, 4.002), synthetic_abp(), None),  # 0.08 s: under the 0.1 s limit
    ],
    ids=[
        "good",
        "few-beats",
        "flat-abp-noise",
        "sbp-high",
        "dbp-high",
        "dbp-low",
        "flat-ppg",
        "clip-floor",
        "clip-ceiling",
        "short-clip-ok",
    ],
)
def test_check_window_filters(ppg, abp, expected):
    assert check_window(ppg, abp, ecg())[0] == expected


def test_missing_samples_in_any_signal_drop_the_window():
    bad = ecg()
    bad[5] = np.nan
    assert check_window(good_ppg(), synthetic_abp(), bad)[0] == MISSING


def test_sbp_not_above_dbp_is_dropped():
    assert failed_filter(good_ppg(), sbp=80.0, dbp=90.0, n_beats=10) == SBP_LE_DBP


def test_nan_label_fails_range_check():
    assert failed_filter(good_ppg(), sbp=np.nan, dbp=80.0, n_beats=10) == SBP_OUT


# --- records and outputs ------------------------------------------------------------


def make_record(record_id, n_samples, clip_second_window=False):
    ppg = good_ppg(n_samples)
    if clip_second_window:
        ppg[WINDOW + 100 : WINDOW + 130] = 0.0
    return Record(record_id, ppg, synthetic_abp(n=n_samples), ecg(n_samples))


def test_process_record_cuts_full_windows_and_drops_tail():
    rec = make_record("1_0007", int(25 * FS_HZ), clip_second_window=True)  # 2 windows + 5 s

    rows, signals, dropped = process_record(rec)

    assert [r["start_sample"] for r in rows] == [0]  # second window clipped, tail dropped
    assert dropped == {CLIPPED_PPG: 1}
    assert signals["ppg"].shape == (1, WINDOW) and signals["ppg"].dtype == np.float32
    np.testing.assert_allclose(signals["abp"][0], rec.abp[:WINDOW], rtol=1e-6)


def test_build_windows_counts_add_up():
    records = [
        make_record("1_0000", 8 * FS_HZ),  # shorter than one window
        make_record("1_0001", 25 * FS_HZ, clip_second_window=True),
        make_record("1_0002", 30 * FS_HZ),
    ]

    table, signals, counts, n_short = build_windows(records)

    assert n_short == 1
    assert list(counts) == FILTERS + ["kept"]
    assert sum(counts.values()) == 2 + 3  # every full window is kept or counted once
    assert counts["kept"] == len(table) == 4 and counts[CLIPPED_PPG] == 1
    assert table["window_id"].tolist() == [0, 1, 2, 3]
    assert table["record_id"].tolist() == ["1_0001", "1_0002", "1_0002", "1_0002"]
    assert all(signals[name].shape == (4, WINDOW) for name in ("ppg", "abp", "ecg"))


def test_build_windows_with_no_records():
    table, signals, counts, n_short = build_windows([])
    assert len(table) == 0 and counts["kept"] == 0 and n_short == 0
    assert signals["ppg"].shape == (0, WINDOW)


def test_outputs_round_trip(tmp_path):
    table, signals, counts, _ = build_windows([make_record("1_0042", 20 * FS_HZ)])
    filters_json = tmp_path / "results" / "window_filters.json"

    write_outputs(table, signals, counts, interim_dir=tmp_path, filters_json=filters_json)
    loaded, loaded_signals = load_windows(tmp_path)

    assert loaded["record_id"].tolist() == ["1_0042", "1_0042"]  # stays a string
    assert list(loaded.columns) == list(table.columns)
    np.testing.assert_array_equal(loaded_signals["ppg"], signals["ppg"])
    assert json.loads(filters_json.read_text()) == counts
