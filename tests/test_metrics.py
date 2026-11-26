"""bp.metrics against examples worked out by hand. Each expected number is derived in a comment."""

import json
import math

import numpy as np
import pytest

from bp import metrics as m

# Main example: 5 windows, true SBP 120 mmHg in each.
TRUE = [120, 120, 120, 120, 120]
PRED = [114, 117, 120, 124, 130]
# errors = pred - true = -6, -3, 0, 4, 10
RECORDS = ["1_0001", "1_0001", "1_0002", "1_0002", "1_0003"]

CONTRACT_KEYS = [  # docs/data-contract.md, section 6, in order
    "n_windows",
    "n_records",
    "me",
    "sd",
    "mae",
    "mae_ci95",
    "pct_within_5",
    "pct_within_10",
    "pct_within_15",
    "bhs_grade",
    "aami_pass",
]


def test_error_is_pred_minus_true():
    assert m.errors(PRED, TRUE).tolist() == [-6, -3, 0, 4, 10]


def test_main_example_by_hand():
    err = m.errors(PRED, TRUE)
    # ME = (-6 - 3 + 0 + 4 + 10) / 5 = 5 / 5 = 1.0
    assert m.mean_error(err) == pytest.approx(1.0)
    # deviations from ME: -7, -4, -1, 3, 9 -> squares 49 + 16 + 1 + 9 + 81 = 156
    # sample SD = sqrt(156 / (5 - 1)) = sqrt(39) = 6.245
    assert m.sd_error(err) == pytest.approx(math.sqrt(39))
    # MAE = (6 + 3 + 0 + 4 + 10) / 5 = 23 / 5 = 4.6
    assert m.mae(err) == pytest.approx(4.6)
    # |error| <= 5: 3, 0, 4 -> 3 of 5 = 60%.  <= 10: all 5 (10 itself counts) = 100%.
    assert m.pct_within(err, 5) == pytest.approx(60.0)
    assert m.pct_within(err, 10) == pytest.approx(100.0)
    assert m.pct_within(err, 15) == pytest.approx(100.0)


def test_summarize_main_example():
    s = m.summarize(PRED, TRUE, RECORDS)
    assert list(s) == CONTRACT_KEYS
    assert s["n_windows"] == 5
    assert s["n_records"] == 3
    assert s["me"] == pytest.approx(1.0)
    assert s["sd"] == pytest.approx(math.sqrt(39))
    assert s["mae"] == pytest.approx(4.6)
    # 60 / 100 / 100 % meets grade A's minimums 60 / 85 / 95 (exactly on the 60 edge)
    assert s["bhs_grade"] == "A"
    # |1.0| <= 5 and 6.245 <= 8
    assert s["aami_pass"] is True
    lo, hi = s["mae_ci95"]
    assert lo <= 4.6 <= hi


def test_summary_is_plain_json():
    json.dumps(m.summarize(PRED, TRUE, RECORDS))  # numpy bools/ints would raise here


def test_sd_divides_by_n_minus_1():
    # errors -2, 0, 2: squares sum to 8. 8 / (3 - 1) = 4 -> SD 2.0.
    # (Dividing by n would give sqrt(8 / 3) = 1.633.)
    assert m.sd_error(np.array([-2.0, 0.0, 2.0])) == pytest.approx(2.0)


def test_pct_within_counts_the_boundary():
    # 5 and -5 are within 5, 5.01 is not -> 2 of 3 = 66.67%
    assert m.pct_within(np.array([5.0, -5.0, 5.01]), 5) == pytest.approx(200 / 3)


@pytest.mark.parametrize(
    "pcts, grade",
    [
        ((60, 85, 95), "A"),  # exactly A's minimums
        ((59.9, 85, 95), "B"),  # just misses A on the 5 mmHg count
        ((100, 100, 94), "B"),  # great at 5 and 10 mmHg, but 15 mmHg misses A's 95
        ((50, 75, 90), "B"),
        ((40, 65, 85), "C"),
        ((39.9, 100, 100), "D"),
        ((0, 0, 0), "D"),
    ],
)
def test_bhs_grade(pcts, grade):
    assert m.bhs_grade(*pcts) == grade


@pytest.mark.parametrize(
    "me, sd, ok",
    [(5, 8, True), (-5, 8, True), (5.01, 8, False), (0, 8.01, False), (-5.01, 0, False)],
)
def test_aami_pass(me, sd, ok):
    assert m.aami_pass(me, sd) is ok


def test_limits_of_agreement():
    # ME 1, SD 2: 1.96 * 2 = 3.92 -> (1 - 3.92, 1 + 3.92) = (-2.92, 4.92)
    lo, hi = m.limits_of_agreement(1.0, 2.0)
    assert (lo, hi) == (pytest.approx(-2.92), pytest.approx(4.92))


def test_bootstrap_resamples_records_not_windows():
    """Record A: 100 windows with error 0. Record B: 100 windows with error 10.

    Each resample draws 2 records with replacement:
      A,A (chance 1/4) -> MAE 0;  A,B or B,A (1/2) -> MAE 5;  B,B (1/4) -> MAE 10.
    25% of resamples give 0 and 25% give 10, so the 2.5th and 97.5th percentiles
    are exactly 0 and 10. Resampling the 200 windows instead would wrongly give
    about [4.3, 5.7], as if we had 200 independent patients rather than 2.
    """
    err = np.array([0.0] * 100 + [10.0] * 100)
    records = np.array(["A"] * 100 + ["B"] * 100)
    assert m.bootstrap_mae_ci(err, records) == (pytest.approx(0.0), pytest.approx(10.0))


def test_bootstrap_with_identical_records_has_no_width():
    # every window has |error| 3, so every resample's MAE is 3
    err = np.array([3.0, -3.0] * 50)
    records = np.repeat([f"r{i}" for i in range(10)], 10)
    assert m.bootstrap_mae_ci(err, records) == (pytest.approx(3.0), pytest.approx(3.0))


def test_bootstrap_is_repeatable():
    rng = np.random.default_rng(1)
    err = rng.normal(0, 8, 300)
    records = np.repeat([f"r{i}" for i in range(30)], 10)
    assert m.bootstrap_mae_ci(err, records) == m.bootstrap_mae_ci(err, records)


@pytest.mark.parametrize(
    "pred, true, message",
    [
        ([1.0, 2.0], [1.0], "equal length"),
        ([], [], "no windows"),
        ([1.0, np.nan], [1.0, 2.0], "NaN"),
        ([1.0, 2.0], [np.inf, 2.0], "NaN or infinite"),
    ],
)
def test_bad_input_is_refused(pred, true, message):
    with pytest.raises(ValueError, match=message):
        m.errors(pred, true)


def test_sd_needs_two_windows():
    with pytest.raises(ValueError, match="at least 2"):
        m.sd_error(np.array([1.0]))


def test_record_ids_must_match_errors():
    with pytest.raises(ValueError, match="record_ids"):
        m.bootstrap_mae_ci(np.array([1.0, 2.0]), ["a"])
