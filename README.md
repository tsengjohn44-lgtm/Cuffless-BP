# Cuffless-BP

Estimates systolic (SBP) and diastolic (DBP) blood pressure from a fingertip pulse signal (PPG) and reports the error in mmHg against the AAMI and BHS standards that cuff monitors must meet.

> **Research demo, not a medical device.** Do not use these estimates to make health decisions.

**Status (Nov 26, 2026):** data pipeline, features, metrics and evaluation code are in place.
Baseline 2 and the main model are still being built, so this README has `TODO` markers
where results will go. Every number below comes from a file in `results/` or `splits/`.

## The problem and who has it

High blood pressure is a leading risk factor for stroke and heart disease, and it often
goes unnoticed because measuring it needs a cuff and a deliberate check. An optical pulse
sensor (PPG) is already in every pulse oximeter and most smartwatches. If blood pressure
could be estimated from that signal, people could be checked far more often. The open
question is how accurate such an estimate really is, measured in mmHg against the bar
that cuff monitors have to clear. This project answers that question honestly on a public
dataset, with a held-out test set and simple baselines to compare against.

Who it is for: researchers and students who want a transparent, reproducible cuffless BP
baseline, and, in the long run, people without easy access to a cuff.

## Install

Requires Python 3.11.

```bash
git clone https://github.com/tsengjohn44-lgtm/Cuffless-BP.git
cd Cuffless-BP
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # versions are pinned
cp .env.example .env                # optional: change where the data is stored
```

## Run

| Command | What it does |
|---|---|
| `make all` | Downloads the data and rebuilds every validation result from scratch |
| `make demo` | Starts the Streamlit demo (no dataset or wifi needed) |
| `make test` | Runs the unit tests (pytest) |
| `make data` | Downloads `Part_1.mat` (about 0.8 GB) into `data/raw` |
| `make inspect` | Prints and explains the layout of one record |
| `make windows` | Cuts 10 s windows, labels SBP/DBP, applies quality filters, makes the split |
| `make features` | Computes PPG and ECG features per window |
| `make baselines` | Fits baseline 1 (and baseline 2) on the training split |
| `make train` | Trains the main model (TODO: not built yet, this step currently fails) |
| `make eval` | Scores every saved model on the **validation** split |
| `make final` | Scores every model on the **test** split. Run once, in Phase 4 only |

### Reproducing the evaluation

`make all` runs `data → windows → features → baselines → train → eval` and writes
`results/metrics.json`, `results/comparison_val.md`, `results/predictions_val.csv` and
the figures in `results/figures/val/`. The held-out test numbers come from a single
`make final` run, written to the same places with `test` in the name.

The split is committed in `splits/split.json`, and all random steps use fixed seeds, so the
same commit gives the same numbers on any machine with the pinned versions.

> TODO: confirm reproduction in a fresh environment (Phase 5) and note the run time.

## Data

[UCI "Cuff-Less Blood Pressure Estimation"](https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation)
(dataset 340, CC BY 4.0), derived from ICU recordings in MIMIC-II. It holds 12,000 records
in four MATLAB v7.3 files (3.1 GB). Each record has three signals sampled at 125 Hz:

| Signal | Role |
|---|---|
| PPG, fingertip pulse wave | Model input |
| ABP, invasive arterial blood pressure (mmHg) | Ground truth labels |
| ECG lead II | Optional input, for pulse arrival time |

`make data` streams the UCI zip and unpacks `Part_1.mat` (3,000 records) on the fly, checking
its CRC32 checksum. The UCI server cuts downloads off at 1 GiB, and Parts 2 to 4 sit past that
point in the zip, so **all results here use Part_1 only**. With a full local copy of the zip,
`python -m bp.download --zip path/to.zip --parts 1 2 3 4` unpacks the rest.

Raw data is never committed (`data/` is in `.gitignore`).

## Method

**1. Windows and labels** (`src/bp/windows.py`). Each record is cut into back-to-back
10 s windows (1,250 samples). In each window we find the top and bottom of every heartbeat
in the arterial pressure. SBP is the median of the tops and DBP the median of the bottoms, so
one odd beat barely moves the label. Beats must be at least 0.33 s apart (a heart rate of
180 bpm) and stand out clearly, so the small dicrotic bump after each beat is not counted.

**2. Quality filters.** A window is dropped if its labels or its PPG look unreliable. Each
dropped window is counted under the first filter it fails (`results/window_filters.json`):

| Filter | Windows dropped (Part_1) |
|---|---|
| Missing samples (NaN) | 0 |
| Fewer than 5 beats | 48 |
| SBP outside 60–220 mmHg | 0 |
| DBP outside 30–130 mmHg | 33 |
| SBP ≤ DBP | 0 |
| Flat PPG | 0 |
| Clipped PPG (stuck at the sensor's floor or ceiling) | 698 |
| **Kept** | **64,093** of 64,872 |

Clipping is the main loss: every clipped window we plotted was a sensor dropout or a motion
artifact. 293 of the 3,000 records are shorter than 10 s and give no windows at all.

**3. Split by record, not by window** (`src/bp/split.py`). Windows from one record are near
copies of each other, so splitting by window would let the model see a test patient during
training. Each record goes to train, validation or test by a hash of the seed (42) and its
record id, aiming at 70/15/15. The result is 1,910 train, 390 validation and 403 test records
(`splits/split.json`). Only `evaluate.py --final` may read the test records.

**4. Features per window** (`src/bp/features.py`). The PPG is band-passed at 0.5–8 Hz, the
beats are found, and each feature is the median over the beats in the window:
heart rate and beat-to-beat variability; pulse amplitude; systolic upstroke time; pulse width
at 25, 50 and 75% of the beat's height; the area before vs. after the dicrotic notch; and the
second-derivative (APG) wave ratios b/a, c/a, d/a and e/a. Three signal-quality measures
(number of beats, how alike the beats are, PPG skewness) are kept for the model and for error
analysis. The arterial pressure is never read here, so the answer cannot leak into the inputs.

*Caveat on pulse arrival time (`pat_s`).* We measure it from the ECG R peak to the steepest
point of the next PPG upstroke. In this dataset the median is about 0.56 s, longer than the
0.2–0.4 s expected at the finger, which most likely means a recording delay between the ECG
and PPG channels that differs between records. We keep it as a feature, but it means "R peak
to PPG upstroke in this dataset", not a true physiological pulse arrival time.

**5. Models.** All are fit on the training split only and tuned on validation only.

| Model | What it is |
|---|---|
| Baseline 1 (`mean_baseline`) | Predicts the training mean SBP and DBP for every window. Anything worse than this has learned nothing |
| Baseline 2 (`pat_linear`) | Pulse arrival time + heart rate into linear regression. TODO: not built yet |
| Main (`hgb`) | `HistGradientBoostingRegressor`, one for SBP and one for DBP, on the PPG features. TODO: not built yet |

## Evaluation

**What we measure.** For SBP and DBP separately, the error of every window (predicted minus
true, in mmHg) against the invasive arterial pressure:

- **Mean error (ME) ± standard deviation (SD).** The AAMI standard for cuff monitors requires
  |ME| ≤ 5 mmHg and SD ≤ 8 mmHg.
- **BHS grade.** The % of windows within 5, 10 and 15 mmHg. Grade A needs 60/85/95%,
  B 50/75/90%, C 40/65/85%, otherwise D.
- **Mean absolute error (MAE)**, the headline number, with a **95% confidence interval from
  1,000 bootstrap resamples of whole records**. Windows of one record are not independent,
  so resampling single windows would give an interval that is too narrow.

**What we compare against.** Every model is scored on the same windows as baseline 1 (the
training mean) and baseline 2 (pulse arrival time + heart rate). The main model is only
useful if it clearly beats both. `metrics.py` is unit-tested on hand-computed examples.

**How.** `make eval` scores validation; `make final` scores the held-out test split once,
after all tuning is done. Nobody looks at test numbers before that run, and nothing is
tuned after it.

### Results

> TODO: paste the test table from `results/comparison_test.md` after `make final` (Phase 4).
> Until then, the validation table from `results/comparison_val.md` can go here, labelled as
> validation.

| Model | Target | ME ± SD (mmHg) | MAE (95% CI) | ≤5 / ≤10 / ≤15 mmHg | BHS | AAMI |
|---|---|---|---|---|---|---|
| Baseline 1 | SBP | TODO | TODO | TODO | TODO | TODO |
| Baseline 1 | DBP | TODO | TODO | TODO | TODO | TODO |
| Baseline 2 | SBP | TODO | TODO | TODO | TODO | TODO |
| Baseline 2 | DBP | TODO | TODO | TODO | TODO | TODO |
| Main (`hgb`) | SBP | TODO | TODO | TODO | TODO | TODO |
| Main (`hgb`) | DBP | TODO | TODO | TODO | TODO | TODO |

> TODO: add the figures from `results/figures/test/`: predicted vs. true, Bland-Altman,
> error histogram for the main model.

> TODO: one paragraph on what the numbers mean, in plain words. Missing AAMI is normal for
> cuffless methods; say so if it happens.

### Where it works and where it fails

> TODO (Phase 3 error analysis): error by true BP range, by heart rate and by PPG quality
> (`n_ppg_beats`, `beat_corr`, `ppg_skew`).

Conditions under which the result stops being useful, known before any numbers:

- **Outside ICU patients with arterial lines.** The model has only seen sedated, monitored
  ICU patients. It is untested on healthy people, on people moving, or on phone and watch
  sensors, whose PPG looks different from a clinical finger probe.
- **Extreme pressures.** Very high and very low BP are rare in the data, and tree models
  cannot predict outside the range they were trained on, so the error is likely largest
  exactly where a reading matters most.
- **Noisy or clipped PPG.** Windows with clipped PPG are dropped; on a real device they would
  need to be detected and rejected, not estimated.
- **A new patient.** There are no patient IDs, so the same person may appear in train and test.
  Scores may look better than they would on someone the model has never seen.
- **A passed AAMI check is not AAMI validation.** The real protocol needs at least 85 subjects
  measured with a reference cuff under a fixed protocol, which this dataset cannot provide.

## Demo

`make demo` starts a Streamlit app: pick a held-out window, see its PPG waveform, the model's
SBP/DBP next to the true arterial values, the model's typical error band, and its AAMI/BHS
result. It runs from `app/sample.npz` (a small sample of test windows written by `make final`)
with no download or wifi, and falls back to a validation preview (`app/sample_val.npz`) until
then. If anything fails to load, it shows a message instead of crashing.

> TODO: demo link, screenshot and video link.

## Who benefits, and who could be harmed

**Who benefits.** People without easy access to a cuff, who could one day get more frequent
checks from a sensor they already own; and researchers who want an honest, reproducible
baseline with errors stated in the clinical unit.

**Who could be harmed.**

- **False reassurance.** A reading that is too low could lead someone with high blood pressure
  to skip seeing a doctor. The demo shows the error band next to every number and says it is
  not a medical device.
- **People with darker skin.** Optical pulse sensors can read less accurately on darker skin.
  This dataset has no demographic data, so we cannot check whether our errors differ by skin
  tone, and we make no claim that they don't.
- **Over-trust in a number.** A single mmHg value looks precise. Without the error band, a
  user or clinician could treat it as a measurement rather than an estimate.

## Limitations

- **ICU patients with arterial lines** (MIMIC-II), not healthy people wearing a watch. Not
  tested on phone or watch sensors.
- **Possible patient leakage across records.** The dataset has no patient IDs, so splitting by
  record cannot rule out one person appearing in both train and test.
- **No demographic data**, so effects of skin tone, age or sex on PPG cannot be checked.
- **Part_1 only** (3,000 of 12,000 records), because the UCI server cuts downloads at 1 GiB.
- **Pulse arrival time includes a recording delay** in this dataset (see Method), so baseline 2
  is weaker than pulse arrival time on a properly synchronised device would be.
- **10 s windows, not readings.** Each window is scored as one estimate; a cuff reading
  protocol averages repeated measurements under rest conditions.

## What we would do with another month

- Train and test on all four parts (12,000 records) for more patients and a tighter interval.
- Try a small 1D CNN on the raw PPG and compare it with the feature model on the same split.
- Per-patient calibration (one cuff reading to set the offset), the way most cuffless devices
  work, and measure how much it helps.
- Test on PPG from a consumer wearable or a phone camera, with a cuff as reference, to see how
  much accuracy is lost outside the ICU.
- Validate on a dataset with demographics (for example later MIMIC versions or VitalDB) to check
  for differences by skin tone, age and sex.

## Team

> TODO: names and roles (lead + model, data + baselines, evaluation + write-up, demo + video).

## Credits

- **Dataset:** M. Kachuee, M. M. Kiani, H. Mohammadzade, M. Shabany, "Cuff-Less Blood Pressure
  Estimation", UCI Machine Learning Repository, dataset 340,
  https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation, licensed CC BY 4.0.
  It is derived from the MIMIC-II waveform database (PhysioNet). The demo sample
  `app/sample.npz` contains a few hundred PPG windows from this dataset, shared under the same
  license with this credit.
- **Libraries:** numpy, scipy, h5py, pandas, scikit-learn, matplotlib, streamlit, pytest and
  ruff (versions in `requirements.txt`).
- **Methods from the literature:** the PPG skewness quality index follows Elgendi (2016);
  the AAMI and BHS criteria come from the published standards.
- No pretrained models or third-party code are included.
- **AI assistance:** Claude Code helped write the code and this README.
