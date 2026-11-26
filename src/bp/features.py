"""PPG (and ECG) features per 10 s window, each a median over the beats in the window.

Owner: Model track. Formats: docs/data-contract.md, section 3.

Plain-language summary:
- Reads the windows written by windows.py, using only the PPG and ECG arrays. ABP is the
  ground truth, so it is never loaded here (that would leak the answer into the inputs).
- For each window: clean the PPG, find the beats, measure the shape of every beat, and
  take the median over beats so one bad beat can't move the result much.
- Writes one row of numbers per window to interim/features.csv. Anything that can't be
  measured is NaN, never 0.

Steps for one window (window_features):
1. Band-pass the PPG at 0.5-8 Hz (2nd-order Butterworth, run forward and backward so
   nothing shifts in time). This removes slow baseline drift and fast noise.
2. Systolic peaks come from find_peaks, at least one beat at 180 bpm apart and standing
   out by at least 30% of the signal's typical swing. The foot of a beat is where the
   tangent at the steepest point of the upstroke meets the level of the preceding
   minimum (find_feet). One beat runs from one foot to the next, so the cut-off beats
   at the window edges drop out. Beats outside 30-180 bpm are skipped.
3. Each beat is levelled (the straight line joining the valley floors before and after
   it is subtracted, so it sits on 0) and then measured; see the table.
4. ECG QRS complexes are found on 5-15 Hz band-passed ECG (the band where the QRS
   lives); the R peak is the main deflection of the real ECG near each one
   (find_r_peaks). PAT is the time from each R peak to the next PPG max-slope point.

Caveat on pat_s, from plotting Part_1 windows: R peaks usually fall near the PPG
systolic peak, while at the finger the PPG upstroke should come 0.2-0.4 s after R.
So pat_s (median about 0.56 s) most likely includes an ECG-to-PPG recording delay that
differs between records. It is stable within a record and still follows heart rate,
so it is kept as a feature, but it means "R peak to PPG upstroke in this dataset",
not a true physiological pulse arrival time.

Columns (median over beats unless noted; fewer than 3 usable beats -> NaN):
| column         | meaning                                                            |
|----------------|--------------------------------------------------------------------|
| hr_bpm         | 60 / median time between systolic peaks                            |
| ibi_cv         | SD / mean of the time between peaks (beat-to-beat variability)     |
| pulse_amp_au   | valley-floor-to-peak height of the beat, arbitrary units           |
| upstroke_s     | foot-to-peak time (systolic upstroke time)                         |
| width{25,50,75}_s | time the beat stays above 25/50/75% of its height                |
| area_ratio     | area before the dicrotic notch / area after it (NaN if no notch)   |
| apg_{b,c,d,e}_a | second-derivative (APG) wave heights divided by the a-wave height |
| pat_s          | pulse arrival time: ECG R peak to PPG max-slope point              |
| n_ppg_beats    | complete beats measured in the window (a count, not a median)      |
| beat_corr      | mean correlation of each beat with the window's median beat        |
| ppg_skew       | skewness of the filtered PPG window (Elgendi 2016 quality index)   |

The last three are mostly signal-quality measures, kept for error analysis
("are errors worst on noisy PPG?") as well as for the model.

APG waves (apg_waves): the second derivative of a PPG beat has a fixed pattern of
bumps. a = the big positive peak early in the upstroke, b = the dip after it. Walking
forward from b: the next bump up is c, the next dip is d, the next bump up is e, which
marks the dicrotic notch. If only one bump follows b, the c and d waves have merged
(common in older patients), so that bump is e and c, d are NaN.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.stats import skew

from bp.fileio import atomic_write
from bp.load import FS_HZ
from bp.paths import INTERIM_DIR

MAX_HR_BPM = 180
MIN_HR_BPM = 30
MIN_GAP = int(FS_HZ * 60 / MAX_HR_BPM)  # 41 samples: the closest two beats can be
MIN_BEATS = 3  # fewer usable beats than this -> NaN
PEAK_PROMINENCE = 0.3  # a systolic peak must rise 30% of the 5th-95th percentile swing
A_WAVE_LOOKBACK = int(0.05 * FS_HZ)  # the APG a-wave can sit a few samples before the foot
CDE_SEARCH = 0.6  # look for APG c, d, e waves in the first 60% of the beat
# Outside this range the R-peak/upstroke pairing is surely wrong: no real PAT is under
# 0.15 s (that happens when a long delay makes us pair with the wrong beat).
PAT_RANGE_S = (0.15, 1.0)
TEMPLATE_LEN = 100  # beats are resampled to this many points to compare their shapes

WIDTH_LEVELS = (25, 50, 75)
FEATURES = [
    "hr_bpm",
    "ibi_cv",
    "pulse_amp_au",
    "upstroke_s",
    *[f"width{p}_s" for p in WIDTH_LEVELS],
    "area_ratio",
    "apg_b_a",
    "apg_c_a",
    "apg_d_a",
    "apg_e_a",
    "pat_s",
    "n_ppg_beats",
    "beat_corr",
    "ppg_skew",
]
# Per-beat measurements whose median becomes a column.
BEAT_FEATURES = [
    "pulse_amp_au",
    "upstroke_s",
    *[f"width{p}_s" for p in WIDTH_LEVELS],
    "area_ratio",
    "apg_b_a",
    "apg_c_a",
    "apg_d_a",
    "apg_e_a",
]

PPG_BAND = butter(2, [0.5, 8.0], btype="bandpass", fs=FS_HZ, output="sos")
ECG_BAND = butter(2, [5.0, 15.0], btype="bandpass", fs=FS_HZ, output="sos")  # QRS band
ECG_DRIFT = butter(2, 0.5, btype="highpass", fs=FS_HZ, output="sos")  # drift removal only
QRS_HALF_WIDTH = int(0.06 * FS_HZ)  # the R peak lies within 60 ms of the QRS energy peak


# ---------------------------------------------------------------- beat finding


def find_systolic_peaks(ppg_f: np.ndarray) -> np.ndarray:
    """Indices of the systolic peaks in a filtered PPG window."""
    swing = np.percentile(ppg_f, 95) - np.percentile(ppg_f, 5)
    if not swing > 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(ppg_f, distance=MIN_GAP, prominence=PEAK_PROMINENCE * swing)
    return peaks


def find_feet(
    ppg_f: np.ndarray, d1: np.ndarray, peaks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The foot before each peak (except the first), plus the valley-floor value there.

    Foot = intersecting-tangent method: draw the tangent line at the steepest point of
    the upstroke; the foot is where it meets the level of the lowest point since the
    previous peak. Simply taking the lowest point is fragile: late diastole is often
    nearly flat, so the minimum can land anywhere on the flat stretch. The tangent
    doesn't care. The floor *value* (the minimum itself) is stable, so it is the
    baseline the beat is measured from (see level_beat)."""
    feet, floors = [], []
    for p0, p1 in zip(peaks[:-1], peaks[1:], strict=True):
        trough = p0 + int(np.argmin(ppg_f[p0:p1]))
        ms = trough + int(np.argmax(d1[trough : p1 + 1]))
        foot = trough
        if d1[ms] > 0:
            tangent_foot = ms - (ppg_f[ms] - ppg_f[trough]) / d1[ms]
            foot = int(np.clip(round(tangent_foot), trough, ms))
        feet.append(foot)
        floors.append(ppg_f[trough])
    return np.array(feet, dtype=int), np.array(floors)


def level_beat(segment: np.ndarray, start: float | None = None, end: float | None = None):
    """Subtract the straight line from level `start` to level `end` (default: the
    first and last samples), so the beat sits on 0 at both ends. This removes leftover
    drift within one beat. Beats are levelled on the valley floors before and after
    them, so the whole beat, diastole included, is above 0."""
    start = segment[0] if start is None else start
    end = segment[-1] if end is None else end
    return segment - np.linspace(start, end, len(segment))


# ------------------------------------------------------------ beat measurements


def pulse_width(beat: np.ndarray, peak: int, level: float) -> float:
    """Samples between the beat rising through `level` (last crossing before the peak)
    and falling back through it (first crossing after the peak). Linear interpolation
    between samples gives sub-sample precision."""
    below_before = np.flatnonzero(beat[:peak] < level)
    below_after = np.flatnonzero(beat[peak + 1 :] < level)
    if below_before.size == 0 or below_after.size == 0:
        return np.nan
    i = below_before[-1]  # beat[i] < level <= beat[i + 1]
    rise = i + (level - beat[i]) / (beat[i + 1] - beat[i])
    j = peak + 1 + below_after[0]  # beat[j - 1] >= level > beat[j]
    fall = (j - 1) + (beat[j - 1] - level) / (beat[j - 1] - beat[j])
    return fall - rise


def area_ratio(beat: np.ndarray, notch: int) -> float:
    """Area under the levelled beat before the notch divided by the area after it."""
    systolic = np.trapezoid(beat[: notch + 1])
    diastolic = np.trapezoid(beat[notch:])
    return systolic / diastolic if systolic > 0 and diastolic > 0 else np.nan


def apg_waves(d2: np.ndarray, foot: int, ms: int, peak: int, end: int) -> dict[str, int | None]:
    """Locate the APG a-e waves of one beat. All indices are positions in `d2`.

    foot/end = the two feet of the beat, ms = max-slope point, peak = systolic peak.
    A wave that can't be found is None.
    """
    lo = max(foot - A_WAVE_LOOKBACK, 0)
    a = lo + int(np.argmax(d2[lo : ms + 1]))
    # d2 crosses zero at the max-slope point; b is its deepest dip around the peak.
    b_stop = min(peak + (peak - foot) // 2, end)
    b = ms + int(np.argmin(d2[ms : b_stop + 1]))
    waves: dict[str, int | None] = {"a": a, "b": b, "c": None, "d": None, "e": None}

    stop = min(foot + int(CDE_SEARCH * (end - foot)), end)
    if stop - b < 3:
        return waves
    bumps = b + 1 + find_peaks(d2[b + 1 : stop + 1])[0]
    if len(bumps) >= 2:
        c, e = int(bumps[0]), int(bumps[1])
        waves.update(c=c, d=c + int(np.argmin(d2[c : e + 1])), e=e)
    elif len(bumps) == 1:
        waves["e"] = int(bumps[0])
    return waves


def apg_ratios(d2: np.ndarray, waves: dict[str, int | None]) -> dict[str, float]:
    """b/a, c/a, d/a, e/a from the wave heights. NaN if a wave is missing or a <= 0."""
    a_height = d2[waves["a"]]
    out = {}
    for name in "bcde":
        idx = waves[name]
        ok = idx is not None and a_height > 0
        out[f"apg_{name}_a"] = d2[idx] / a_height if ok else np.nan
    return out


def measure_beat(
    beat: np.ndarray, d1: np.ndarray, d2: np.ndarray, foot: int, end: int
) -> tuple[dict[str, float], int] | None:
    """Shape measurements of one levelled beat, plus the index of its max-slope point
    (used for PAT). `beat` is the levelled PPG from `foot` to `end` (the next foot);
    d1, d2 are the whole window's derivatives. None if the beat has no clear peak."""
    peak = int(np.argmax(beat))
    amp = beat[peak]
    if amp <= 0 or peak == 0 or peak == len(beat) - 1:
        return None

    m = {"pulse_amp_au": amp, "upstroke_s": peak / FS_HZ}
    for pct in WIDTH_LEVELS:
        m[f"width{pct}_s"] = pulse_width(beat, peak, pct / 100 * amp) / FS_HZ

    ms = foot + int(np.argmax(d1[foot : foot + peak + 1]))
    waves = apg_waves(d2, foot, ms, foot + peak, end)
    m.update(apg_ratios(d2, waves))
    notch = waves["e"]
    has_notch = notch is not None and notch > foot + peak
    m["area_ratio"] = area_ratio(beat, notch - foot) if has_notch else np.nan
    return m, ms


def mean_template_corr(beats: list[np.ndarray]) -> float:
    """Resample every beat to the same length, take the point-by-point median as the
    window's typical beat, and average each beat's correlation with it."""
    grid = np.linspace(0, 1, TEMPLATE_LEN)
    shapes = np.array([np.interp(grid, np.linspace(0, 1, len(b)), b) for b in beats])
    template = np.median(shapes, axis=0)
    if template.std() == 0:
        return np.nan
    corrs = [np.corrcoef(s, template)[0, 1] for s in shapes if s.std() > 0]
    return float(np.mean(corrs)) if corrs else np.nan


# ------------------------------------------------------------------------ ECG


def find_r_peaks(ecg: np.ndarray) -> np.ndarray:
    """R-peak indices in one ECG window, or an empty array if the ECG is unusable.

    1. Find the QRS complexes on the 5-15 Hz band-passed ECG, either polarity (peaks of
       its absolute value). P waves, T waves and drift barely show up in that band.
    2. The R peak is the highest point of the real ECG (drift removed) within 60 ms of
       each QRS. In lead II the R wave is, by definition, the upward spike of the QRS,
       so the ECG is never flipped.

    Why never flip: deciding "this ECG is upside down" from which deflection is bigger
    is a coin toss when the S wave is about as deep as R is tall, which is common here.
    Judged on the raw ECG, it lands on S in records with a deep S (e.g. 1_0543); judged
    on the 5-15 Hz band, it lands on S in records with a sharp R spike, whose energy is
    mostly above 15 Hz (e.g. 1_1847). Either way "R" ends up 30-40 ms late. Never
    flipping puts R on the R apex in both. On 4,000 train windows all three rules gave
    equally steady beat-to-beat PAT (median SD about 6 ms), so the choice rests on the
    definition. Limitation: if a record's lead has no clear upward R wave (about 5% of
    Part_1 train windows: QS-shaped, paced-looking or clipped complexes), the most
    positive point near the QRS is used, which is a weaker landmark."""
    empty = np.array([], dtype=int)
    if not np.all(np.isfinite(ecg)) or np.ptp(ecg) == 0:
        return empty
    band = np.abs(sosfiltfilt(ECG_BAND, ecg))
    hi = np.percentile(band, 99.5)
    if not hi > 0:
        return empty
    qrs, _ = find_peaks(band, height=0.4 * hi, distance=MIN_GAP)

    x = sosfiltfilt(ECG_DRIFT, ecg)
    spans = [(max(q - QRS_HALF_WIDTH, 0), min(q + QRS_HALF_WIDTH + 1, len(x))) for q in qrs]
    return np.array([a + int(np.argmax(x[a:b])) for a, b in spans], dtype=int)


def pulse_arrival_times(r_peaks: np.ndarray, ms_points: np.ndarray) -> np.ndarray:
    """For each R peak, seconds to the first PPG max-slope point after it and before
    the next R peak. Pairs outside PAT_RANGE_S are dropped."""
    pats = []
    ms_points = np.sort(ms_points)
    for k, r in enumerate(r_peaks):
        nxt = r_peaks[k + 1] if k + 1 < len(r_peaks) else np.inf
        after = ms_points[(ms_points > r) & (ms_points <= nxt)]
        if after.size:
            pat = (after[0] - r) / FS_HZ
            if PAT_RANGE_S[0] <= pat <= PAT_RANGE_S[1]:
                pats.append(pat)
    return np.array(pats)


# --------------------------------------------------------------------- window


def robust_median(values) -> float:
    """Median of the finite values, or NaN if there are fewer than MIN_BEATS of them."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size >= MIN_BEATS else np.nan


def window_features(ppg: np.ndarray, ecg: np.ndarray) -> dict[str, float]:
    """All feature columns for one window. Never raises on bad signals: what can't be
    measured is NaN."""
    out = dict.fromkeys(FEATURES, np.nan)
    out["n_ppg_beats"] = 0.0
    ppg = np.asarray(ppg, dtype=float)
    ecg = np.asarray(ecg, dtype=float)
    if not np.all(np.isfinite(ppg)) or np.ptp(ppg) == 0:
        return out

    ppg_f = sosfiltfilt(PPG_BAND, ppg)
    d1 = np.gradient(ppg_f)
    d2 = np.gradient(d1)
    out["ppg_skew"] = float(skew(ppg_f))

    # Heart rate from the time between neighbouring systolic peaks.
    peaks = find_systolic_peaks(ppg_f)
    ibis = np.diff(peaks) / FS_HZ
    ibis = ibis[ibis <= 60 / MIN_HR_BPM]  # longer gaps mean a missed beat
    if ibis.size >= MIN_BEATS - 1:
        out["hr_bpm"] = 60 / float(np.median(ibis))
        out["ibi_cv"] = float(ibis.std() / ibis.mean())

    # Shape of each complete beat (foot to next foot).
    if len(peaks) < 2:
        return out
    feet, floors = find_feet(ppg_f, d1, peaks)
    per_beat, ms_points, levelled = [], [], []
    for k in range(len(feet) - 1):
        foot, end = int(feet[k]), int(feet[k + 1])
        if not 60 / MAX_HR_BPM <= (end - foot) / FS_HZ <= 60 / MIN_HR_BPM:
            continue
        beat = level_beat(ppg_f[foot : end + 1], floors[k], floors[k + 1])
        measured = measure_beat(beat, d1, d2, foot, end)
        if measured is None:
            continue
        per_beat.append(measured[0])
        ms_points.append(measured[1])
        levelled.append(beat)

    out["n_ppg_beats"] = float(len(per_beat))
    if len(per_beat) < MIN_BEATS:
        return out
    for name in BEAT_FEATURES:
        out[name] = robust_median([m[name] for m in per_beat])
    out["beat_corr"] = mean_template_corr(levelled)

    # Pulse arrival time needs the ECG; a bad ECG only blanks this one column.
    out["pat_s"] = robust_median(pulse_arrival_times(find_r_peaks(ecg), np.array(ms_points)))
    return out


def build_feature_table(windows: pd.DataFrame, ppg: np.ndarray, ecg: np.ndarray) -> pd.DataFrame:
    """One row per window, in the same order: window_id, record_id, then FEATURES."""
    if not len(windows) == len(ppg) == len(ecg):
        raise ValueError(
            f"row counts differ: windows.csv {len(windows)}, ppg {len(ppg)}, ecg {len(ecg)}"
        )
    rows = []
    t0 = time.time()
    for i in range(len(windows)):
        rows.append(window_features(ppg[i], ecg[i]))
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(windows)} windows ({time.time() - t0:.0f} s)")
    keys = windows[["window_id", "record_id"]].reset_index(drop=True)
    return pd.concat([keys, pd.DataFrame(rows, columns=FEATURES)], axis=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PPG/ECG features per window")
    parser.add_argument("--interim", type=Path, default=INTERIM_DIR, help="folder with windows.*")
    parser.add_argument("--limit", type=int, help="only the first N windows (quick checks)")
    args = parser.parse_args(argv)

    windows_csv = args.interim / "windows.csv"
    windows_npz = args.interim / "windows.npz"
    missing = [p for p in (windows_csv, windows_npz) if not p.exists()]
    if missing:
        print(
            f"{', '.join(map(str, missing))} not found. Run `make windows` first.", file=sys.stderr
        )
        return 1

    windows = pd.read_csv(windows_csv, dtype={"record_id": str})
    with np.load(windows_npz) as z:
        ppg, ecg = z["ppg"], z["ecg"]  # "abp" is deliberately never loaded
    if args.limit:
        windows, ppg, ecg = windows.iloc[: args.limit], ppg[: args.limit], ecg[: args.limit]

    print(f"computing features for {len(windows)} windows")
    t0 = time.time()
    table = build_feature_table(windows, ppg, ecg)
    out = args.interim / "features.csv"
    with atomic_write(out, "w") as f:
        table.to_csv(f, index=False)

    print(f"wrote {out} ({len(table)} rows) in {time.time() - t0:.0f} s")
    print("missing values per feature (% of windows):")
    for name, pct in (table[FEATURES].isna().mean() * 100).items():
        print(f"  {name:<14} {pct:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
