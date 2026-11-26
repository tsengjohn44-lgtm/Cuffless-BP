# Data contract (v0.1 — draft, needs a team OK)

Every track builds against this file. Don't change it from a track branch:
propose the change in a PR or chat and wait for an OK.

All paths below are relative to `BP_DATA_DIR` (default `./data`, see `.env.example`)
unless they start with `results/` or `app/`.

## Conventions
- Sampling rate: **125 Hz**. One window = **10 s = 1,250 samples**, no overlap.
- Pressures in **mmHg**, times in **seconds**, heart rate in **beats per minute**.
- PPG and ECG amplitudes: arbitrary units. Verified on Part_1 (`make inspect`): PPG ranges
  0 to 4.002 and sits at either limit in about 0.02% of samples (clipping); the record
  layout is documented in `src/bp/load.py`.
- Every output in the shared data folder is written with `bp.fileio.atomic_write`, so a
  terminal reading it while another rewrites it never sees half a file.
- `record_id` is a string `"<part>_<index>"`, e.g. `"1_0042"` = Part_1, record 42 (0-based,
  4-digit zero padding). It is the unit of the train/val/test split.
- `window_id` is an int, unique across all parts, assigned in order by `windows.py`.
- Missing values are `NaN`, never 0 or -1.

## 1. Windows (`windows.py` → everyone)
Two files whose rows line up one-to-one (row *i* of the table = row *i* of each array).

`interim/windows.csv`

| column | type | meaning |
|---|---|---|
| window_id | int | unique id |
| record_id | str | see conventions |
| start_sample | int | first sample of the window inside its record |
| sbp_mmhg | float | median of ABP beat peaks in the window |
| dbp_mmhg | float | median of ABP beat troughs in the window |
| n_beats | int | ABP beats found in the window |

`interim/windows.npz`: `ppg`, `abp`, `ecg`, each `float32` with shape `(n_windows, 1250)`.

Only windows that pass every quality filter are written. The count dropped by each filter
is written to `results/window_filters.json` as `{"filter name": n_dropped, ...}` plus
`"kept"`.

## 2. Split (`split.py` → everyone)
`splits/split.json` (repo-relative, committed to git so the split can be reviewed)
```json
{"seed": 42, "fractions": [0.70, 0.15, 0.15],
 "train": ["1_0000", "..."], "val": ["..."], "test": ["..."]}
```
Split by `record_id`, never by window. **Only `evaluate.py --final` may read `"test"`.**

## 3. Features (`features.py` → `baselines.py`, `train.py`)
`interim/features.csv`: one row per `window_id` from `windows.csv`.
- Key columns: `window_id`, `record_id`.
- Every other column is a numeric feature, `snake_case` with a unit suffix where it has one
  (`hr_bpm`, `upstroke_s`, `width50_s`, `apg_b_a`, `pat_s`, ...). `NaN` allowed.
- No label columns here. Join labels from `windows.csv` on `window_id`.
- The model track owns the list of features; adding a column is not a contract change.

## 4. Saved models (`baselines.py`, `train.py` → `evaluate.py`)
`models/<model>.joblib`: one object per model with `predict(X) -> (n, 2)` array,
column 0 = SBP, column 1 = DBP, in mmHg, where `X` is the feature table with the key
columns dropped. `<model>` is `snake_case`, e.g. `mean_baseline`, `pat_linear`, `hgb`.

## 5. Predictions (`evaluate.py` → `evaluate.py`, demo)
`results/predictions_<split>.csv`, `<split>` = `val` or `test` (test only from `--final`).

| column | type |
|---|---|
| window_id | int |
| record_id | str |
| model | str |
| sbp_true, dbp_true | float, mmHg |
| sbp_pred, dbp_pred | float, mmHg |

Long format: one row per window per model.

## 6. Metrics (`evaluate.py` → README, demo)
`results/metrics.json`
```json
{"split": "val",
 "models": {
   "<model>": {
     "sbp": {"n_windows": 0, "n_records": 0, "me": 0.0, "sd": 0.0, "mae": 0.0,
             "mae_ci95": [0.0, 0.0], "pct_within_5": 0.0, "pct_within_10": 0.0,
             "pct_within_15": 0.0, "bhs_grade": "D", "aami_pass": false},
     "dbp": {"...": "same keys"}}}}
```
Errors are `pred - true`. `me`, `sd`, `mae` in mmHg; `pct_*` in percent (0–100).
`mae_ci95` comes from bootstrapping records, 1,000 resamples.

## 7. Demo sample (`evaluate.py --final` → `app/demo.py`)
`app/sample.npz`, under 10 MB, test windows only, from the main model:
`ppg` (k, 1250) float32, `record_id` (k,) str, `sbp_true`, `dbp_true`, `sbp_pred`,
`dbp_pred` (k,) float, plus `metrics_json` (0-d str) = the main model's test entry
from `metrics.json`.
