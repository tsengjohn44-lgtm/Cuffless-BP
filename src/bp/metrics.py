"""How far off the blood pressure estimates are, in mmHg, judged by the AAMI and BHS standards.

Owner: Evaluation + write-up track. Output format: docs/data-contract.md, section 6.

Plain-language summary:
- Everything starts from the error of each window: predicted minus true, in mmHg.
  A positive error means the model reads too high.
- AAMI (the US standard for cuff monitors) looks at two numbers: the mean error
  (ME, the bias) and the standard deviation of the errors (SD, the spread).
  It passes if |ME| <= 5 and SD <= 8 mmHg.
- BHS (the British standard) counts how often the error is small: the % of windows
  within 5, 10 and 15 mmHg. Those three numbers give a grade from A (best) to D.
- MAE (mean absolute error) is the average size of the error, ignoring its sign.
- Windows cut from the same record are alike, so they are not independent samples.
  The 95% confidence interval for MAE therefore resamples whole records, not single
  windows. Resampling windows would act as if we had far more independent data than
  we do, and give an interval that is too narrow.

Honesty note: `aami_pass` only checks the ME/SD numbers. A real AAMI validation also
needs at least 85 subjects measured with a reference cuff under a fixed protocol,
which this dataset (ICU records, invasive arterial line) cannot provide.
"""

import numpy as np
from numpy.typing import ArrayLike

AAMI_MAX_ABS_ME = 5.0  # mmHg
AAMI_MAX_SD = 8.0  # mmHg

BHS_LIMITS_MMHG = (5, 10, 15)
# Grade -> minimum % of windows within 5, 10 and 15 mmHg. All three must be met.
BHS_GRADES = {"A": (60, 85, 95), "B": (50, 75, 90), "C": (40, 65, 85)}

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0


def errors(pred: ArrayLike, true: ArrayLike) -> np.ndarray:
    """Error per window = predicted - true (mmHg). Refuses missing values.

    A NaN prediction is never dropped quietly: leaving out the windows a model
    failed on would make it look better than it is."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    if pred.ndim != 1 or pred.shape != true.shape:
        raise ValueError(
            f"pred and true must be 1-D and equal length, got {pred.shape} and {true.shape}"
        )
    if pred.size == 0:
        raise ValueError("no windows to score")
    n_bad = int((~np.isfinite(pred)).sum() + (~np.isfinite(true)).sum())
    if n_bad:
        raise ValueError(f"{n_bad} NaN or infinite value(s) in pred/true; fix them upstream")
    return pred - true


def mean_error(err: np.ndarray) -> float:
    """ME, the bias: the average signed error."""
    return float(np.mean(err))


def sd_error(err: np.ndarray) -> float:
    """SD of the errors, the spread. Uses n - 1 (sample SD), as AAMI reports do."""
    if len(err) < 2:
        raise ValueError("SD needs at least 2 windows")
    return float(np.std(err, ddof=1))


def mae(err: np.ndarray) -> float:
    """Mean absolute error: the average size of the error, ignoring its sign."""
    return float(np.mean(np.abs(err)))


def pct_within(err: np.ndarray, limit_mmhg: float) -> float:
    """% of windows (0-100) whose absolute error is at most `limit_mmhg`.
    An error of exactly 5 counts as within 5."""
    return float(100.0 * np.mean(np.abs(err) <= limit_mmhg))


def bhs_grade(pct_5: float, pct_10: float, pct_15: float) -> str:
    """BHS grade A/B/C/D: the best grade whose three minimum percentages are all met."""
    for grade, minimums in BHS_GRADES.items():
        if all(p >= lo for p, lo in zip((pct_5, pct_10, pct_15), minimums, strict=True)):
            return grade
    return "D"


def aami_pass(me: float, sd: float) -> bool:
    """True if |ME| <= 5 and SD <= 8 mmHg (the AAMI accuracy numbers only, see top)."""
    return bool(abs(me) <= AAMI_MAX_ABS_ME and sd <= AAMI_MAX_SD)


def limits_of_agreement(me: float, sd: float) -> tuple[float, float]:
    """Bland-Altman 95% limits of agreement: about 95% of errors fall in ME +/- 1.96 SD."""
    return me - 1.96 * sd, me + 1.96 * sd


def bootstrap_mae_ci(
    err: np.ndarray,
    record_ids: ArrayLike,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% confidence interval for MAE, resampling records (not windows).

    Each resample draws as many records as we have, with replacement, pools all
    windows of the drawn records and computes their MAE. The interval runs from the
    2.5th to the 97.5th percentile of those 1,000 MAEs. A fixed seed makes it repeatable.
    """
    record_ids = np.asarray(record_ids)
    if record_ids.shape != err.shape:
        raise ValueError(f"record_ids has shape {record_ids.shape}, errors have {err.shape}")
    # Per record: total absolute error and window count. A resample's MAE is then
    # (sum of its records' totals) / (sum of their counts), with no per-window loop.
    _, rec_index = np.unique(record_ids, return_inverse=True)
    abs_sum = np.bincount(rec_index, weights=np.abs(err))
    n_windows = np.bincount(rec_index)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(abs_sum), size=(n_resamples, len(abs_sum)))
    maes = abs_sum[picks].sum(axis=1) / n_windows[picks].sum(axis=1)
    lo, hi = np.percentile(maes, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(pred: ArrayLike, true: ArrayLike, record_ids: ArrayLike) -> dict:
    """Every metric for one target (SBP or DBP), keyed as in the data contract, section 6.
    All values are plain Python types, so the dict can go straight into json.dump."""
    err = errors(pred, true)
    record_ids = np.asarray(record_ids)
    me, sd = mean_error(err), sd_error(err)
    pcts = {f"pct_within_{lim}": pct_within(err, lim) for lim in BHS_LIMITS_MMHG}
    lo, hi = bootstrap_mae_ci(err, record_ids)
    return {
        "n_windows": int(err.size),
        "n_records": int(np.unique(record_ids).size),
        "me": me,
        "sd": sd,
        "mae": mae(err),
        "mae_ci95": [lo, hi],
        **pcts,
        "bhs_grade": bhs_grade(*pcts.values()),
        "aami_pass": aami_pass(me, sd),
    }
