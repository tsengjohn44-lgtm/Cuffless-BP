"""Feature tests on synthetic signals whose right answers can be worked out by hand."""

import numpy as np
import pandas as pd
import pytest

from bp import features as F
from bp.features import FEATURES, FS_HZ

ONE_SAMPLE_S = 1 / FS_HZ


def triangle_beat(rise_s=0.2, fall_s=0.6):
    """Rises linearly 0 -> 1 in rise_s, falls linearly back to 0 in fall_s."""
    up = np.linspace(0, 1, round(rise_s * FS_HZ) + 1)
    down = np.linspace(1, 0, round(fall_s * FS_HZ) + 1)[1:]
    return np.concatenate([up, down])


def synthetic_ppg(hr_bpm=75.0, delay_s=0.0, seconds=10):
    """A PPG-like window: every 60/hr seconds a systolic bump (Gaussian, sd 0.05 s,
    0.15 s after the beat start) plus a smaller diastolic bump 0.40 s after it.
    The steepest point of each upstroke is one sd before the systolic peak, so it sits
    exactly 0.10 s + delay_s after each beat start."""
    t = np.arange(seconds * FS_HZ) / FS_HZ
    period = 60 / hr_bpm
    x = np.full_like(t, 1.5)  # DC offset, like the real 0-4 range
    for start in np.arange(-2 * period, seconds + period, period) + delay_s:
        x += np.exp(-((t - start - 0.15) ** 2) / (2 * 0.05**2))
        x += 0.4 * np.exp(-((t - start - 0.40) ** 2) / (2 * 0.07**2))
    return x


def synthetic_ecg(hr_bpm=75.0, seconds=10):
    """Narrow R spikes at the start of every beat (same beat starts as synthetic_ppg)."""
    t = np.arange(seconds * FS_HZ) / FS_HZ
    x = np.zeros_like(t)
    for start in np.arange(0, seconds, 60 / hr_bpm):
        x += np.exp(-((t - start) ** 2) / (2 * 0.01**2))
    return x


# ------------------------------------------------------------- beat measurements


def test_pulse_widths_of_a_triangle():
    # Rise 0.2 s, fall 0.6 s, so the beat spends a fraction (1 - level) of each side
    # above `level`: width = (1 - level) * 0.8 s.
    beat = triangle_beat()
    peak = int(np.argmax(beat))
    assert peak / FS_HZ == pytest.approx(0.2)  # upstroke time
    for level, expected_s in [(0.25, 0.6), (0.5, 0.4), (0.75, 0.2)]:
        assert F.pulse_width(beat, peak, level) / FS_HZ == pytest.approx(expected_s)


def test_area_ratio_of_a_triangle_split_at_the_peak():
    # Areas of the two right triangles: 0.5 * 25 * 1 vs 0.5 * 75 * 1 samples -> 1/3.
    beat = triangle_beat()
    assert F.area_ratio(beat, int(np.argmax(beat))) == pytest.approx(1 / 3)


def test_level_beat_removes_a_straight_line():
    beat = triangle_beat()
    drift = np.linspace(0.3, 0.9, len(beat))
    np.testing.assert_allclose(F.level_beat(beat + drift), beat, atol=1e-12)


def test_apg_b_over_a_of_a_gaussian():
    # For g(t) = exp(-t^2 / 2s^2): g'' is lowest at t = 0 (-1/s^2, the b wave) and
    # highest at t = -sqrt(3) s (2 e^-1.5 / s^2, the a wave), so b/a = -e^1.5 / 2.
    sigma = 12.5  # samples (0.1 s)
    t = np.arange(201)
    x = np.exp(-((t - 100) ** 2) / (2 * sigma**2))
    d1 = np.gradient(x)
    d2 = np.gradient(d1)
    ms = int(np.argmax(d1[:101]))
    waves = F.apg_waves(d2, foot=0, ms=ms, peak=100, end=200)
    assert waves["b"] == 100
    assert F.apg_ratios(d2, waves)["apg_b_a"] == pytest.approx(-np.exp(1.5) / 2, rel=0.03)


def test_feet_are_evenly_spaced_even_with_a_flat_diastole():
    # The synthetic beats have ~0.3 s of nearly flat signal before each upstroke, where
    # "lowest point" lands at random. Tangent feet must repeat every 100 samples (75 bpm).
    from scipy.signal import sosfiltfilt

    ppg_f = sosfiltfilt(F.PPG_BAND, synthetic_ppg(hr_bpm=75))
    d1 = np.gradient(ppg_f)
    feet, floors = F.find_feet(ppg_f, d1, F.find_systolic_peaks(ppg_f))
    assert len(feet) >= 10
    np.testing.assert_allclose(np.diff(feet), 100, atol=1)
    # The tangent foot sits at the start of the upstroke, at or above the valley floor.
    assert np.all(ppg_f[feet] >= floors)


def test_level_beat_on_given_floor_levels():
    # Levelled on floors 0.2 below the end samples, the beat sits 0.2 above 0 at both ends.
    beat = triangle_beat() + 1.0
    levelled = F.level_beat(beat, start=0.8, end=0.8)
    assert levelled[0] == pytest.approx(0.2) and levelled[-1] == pytest.approx(0.2)


def test_pulse_arrival_times_pairs_each_r_peak_with_the_next_upstroke():
    r_peaks = np.array([100, 200, 300])
    ms_points = np.array([131, 231, 331])  # 31 samples = 0.248 s later
    np.testing.assert_allclose(F.pulse_arrival_times(r_peaks, ms_points), [0.248] * 3)
    # A missing upstroke is skipped, not paired with the following beat.
    assert len(F.pulse_arrival_times(r_peaks, np.array([131, 331]))) == 2
    # An upstroke only 0.08 s after R is impossible (wrong-beat pairing) and dropped.
    assert len(F.pulse_arrival_times(r_peaks, np.array([110, 231, 331]))) == 2


# ------------------------------------------------------------------ whole window


def test_clean_synthetic_window():
    f = F.window_features(synthetic_ppg(hr_bpm=75), synthetic_ecg(hr_bpm=75))
    assert set(f) == set(FEATURES)
    assert f["hr_bpm"] == pytest.approx(75, abs=1)
    assert f["ibi_cv"] < 0.02
    assert f["n_ppg_beats"] >= 9  # 12.5 beats in 10 s, minus the cut-off ones at the edges
    assert f["beat_corr"] > 0.99
    assert f["width25_s"] > f["width50_s"] > f["width75_s"] > 0
    assert f["apg_b_a"] < 0
    assert f["pulse_amp_au"] > 0


def test_r_peak_is_not_moved_onto_a_sharp_s_wave():
    # Seen in record 1_1847: a sharp R spike followed 30 ms later by a smaller but broad
    # S dip. A sharp spike has most of its energy above 15 Hz, so in the 5-15 Hz band
    # the broad S looks bigger than R; a polarity check done in that band flips the ECG
    # and lands on S. The R peak must stay on the tallest point of the real ECG.
    t = np.arange(1250) / FS_HZ
    ecg = np.zeros_like(t)
    r_times = np.arange(0.3, 10, 0.8)
    for r in r_times:
        ecg += np.exp(-((t - r) ** 2) / (2 * 0.008**2))  # R: height 1, sharp
        ecg -= 0.7 * np.exp(-((t - r - 0.03) ** 2) / (2 * 0.02**2))  # S: -0.7, broad
    found = F.find_r_peaks(ecg) / FS_HZ
    assert len(found) == len(r_times)
    np.testing.assert_allclose(found, r_times, atol=ONE_SAMPLE_S)


def test_pat_matches_the_built_in_delay():
    # Upstrokes are steepest 0.10 s + delay after each R spike: PAT = 0.25 s.
    f = F.window_features(synthetic_ppg(delay_s=0.15), synthetic_ecg())
    assert f["pat_s"] == pytest.approx(0.25, abs=2 * ONE_SAMPLE_S)


def test_r_peak_stays_on_r_when_s_is_deeper_than_r():
    # Seen in records like 1_0543: R spike of height 1, then an S dip of depth 1.3.
    # A rule that flips the ECG when its downward deflections are bigger lands on S.
    t = np.arange(1250) / FS_HZ
    ecg = np.zeros_like(t)
    r_times = np.arange(0.3, 10, 0.8)
    for r in r_times:
        ecg += np.exp(-((t - r) ** 2) / (2 * 0.008**2))  # R: height 1
        ecg -= 1.3 * np.exp(-((t - r - 0.03) ** 2) / (2 * 0.01**2))  # S: depth 1.3
    found = F.find_r_peaks(ecg) / FS_HZ
    assert len(found) == len(r_times)
    np.testing.assert_allclose(found, r_times, atol=ONE_SAMPLE_S)


def test_heart_rate_follows_the_signal():
    for hr in (50, 100, 140):
        assert F.window_features(synthetic_ppg(hr), synthetic_ecg(hr))["hr_bpm"] == pytest.approx(
            hr, abs=1.5
        )


def test_flat_ppg_gives_nan_not_zero():
    f = F.window_features(np.zeros(1250), synthetic_ecg())
    assert f["n_ppg_beats"] == 0
    assert all(np.isnan(v) for k, v in f.items() if k != "n_ppg_beats")


def test_ppg_with_nan_gives_nan():
    ppg = synthetic_ppg()
    ppg[500] = np.nan
    assert np.isnan(F.window_features(ppg, synthetic_ecg())["hr_bpm"])


def test_flat_ecg_only_blanks_pat():
    f = F.window_features(synthetic_ppg(), np.zeros(1250))
    assert np.isnan(f["pat_s"])
    assert f["hr_bpm"] == pytest.approx(75, abs=1)


def test_noise_does_not_crash_or_give_infinities():
    rng = np.random.default_rng(0)
    for _ in range(20):
        f = F.window_features(rng.normal(size=1250), rng.normal(size=1250))
        assert not any(np.isinf(v) for v in f.values())


# --------------------------------------------------------------------- file I/O


def write_fake_windows(folder, n=3, with_abp=False):
    windows = pd.DataFrame(
        {
            "window_id": [10, 11, 12][:n],
            "record_id": ["1_0000", "1_0000", "1_0042"][:n],
            "start_sample": [0, 1250, 0][:n],
            "sbp_mmhg": [120.0, 121.0, 140.0][:n],
            "dbp_mmhg": [70.0, 71.0, 80.0][:n],
            "n_beats": [12, 12, 13][:n],
        }
    )
    windows.to_csv(folder / "windows.csv", index=False)
    arrays = {
        "ppg": np.stack([synthetic_ppg(hr) for hr in (60, 75, 90)][:n]).astype(np.float32),
        "ecg": np.stack([synthetic_ecg(hr) for hr in (60, 75, 90)][:n]).astype(np.float32),
    }
    if with_abp:
        arrays["abp"] = np.full((n, 1250), 100, dtype=np.float32)
    np.savez(folder / "windows.npz", **arrays)


def test_main_writes_one_row_per_window_without_labels(tmp_path):
    # windows.npz has no "abp" here: if features.py ever read the ground truth, this
    # would fail with a KeyError.
    write_fake_windows(tmp_path, with_abp=False)
    assert F.main(["--interim", str(tmp_path)]) == 0

    out = pd.read_csv(tmp_path / "features.csv", dtype={"record_id": str})
    assert list(out.columns) == ["window_id", "record_id", *FEATURES]
    assert out["window_id"].tolist() == [10, 11, 12]
    assert out["record_id"].tolist() == ["1_0000", "1_0000", "1_0042"]
    assert all(pd.api.types.is_numeric_dtype(out[c]) for c in FEATURES)
    np.testing.assert_allclose(out["hr_bpm"], [60, 75, 90], atol=1.5)


def test_main_limit(tmp_path):
    write_fake_windows(tmp_path, with_abp=True)
    assert F.main(["--interim", str(tmp_path), "--limit", "2"]) == 0
    assert len(pd.read_csv(tmp_path / "features.csv")) == 2


def test_main_explains_missing_inputs(tmp_path, capsys):
    assert F.main(["--interim", str(tmp_path)]) == 1
    assert "make windows" in capsys.readouterr().err


def test_mismatched_row_counts_are_refused():
    windows = pd.DataFrame({"window_id": [1, 2], "record_id": ["1_0000", "1_0000"]})
    with pytest.raises(ValueError, match="row counts differ"):
        F.build_feature_table(windows, np.zeros((3, 1250)), np.zeros((3, 1250)))
