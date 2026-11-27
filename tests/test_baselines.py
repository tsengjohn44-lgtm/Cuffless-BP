"""Baselines 1 (training mean) and 2 (PAT + heart rate, linear): hand-checked numbers,
train-only fitting, saving."""

import joblib
import numpy as np
import pandas as pd

from bp.baselines import (
    MeanBaseline,
    fit_mean_baseline,
    fit_pat_linear,
    mae,
    make_pat_linear,
    save_model,
)


def synthetic_features(n: int, seed: int = 0) -> pd.DataFrame:
    """Feature table like features.csv, with a known straight-line BP relation built in."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "hr_bpm": rng.uniform(60, 120, n),
            "upstroke_s": rng.uniform(0.1, 0.3, n),  # unrelated column: must be ignored
            "pat_s": rng.uniform(0.3, 0.8, n),
        }
    )


def true_bp(X: pd.DataFrame) -> np.ndarray:
    sbp = 200 - 100 * X["pat_s"] + 0.2 * X["hr_bpm"]  # -10 mmHg per +100 ms PAT
    dbp = 100 - 40 * X["pat_s"] + 0.1 * X["hr_bpm"]
    return np.column_stack([sbp, dbp])


def test_pat_linear_recovers_a_known_relation():
    X = synthetic_features(200)
    model = make_pat_linear().fit(X, true_bp(X))

    X_new = synthetic_features(50, seed=1)
    np.testing.assert_allclose(model.predict(X_new), true_bp(X_new), atol=1e-6)
    linear = model.named_steps["linear"]
    np.testing.assert_allclose(linear.coef_, [[-100, 0.2], [-40, 0.1]], atol=1e-6)


def test_pat_linear_ignores_other_columns_and_their_order():
    X = synthetic_features(200)
    model = make_pat_linear().fit(X, true_bp(X))

    shuffled = X[["pat_s", "upstroke_s", "hr_bpm"]].assign(extra=7.0)
    np.testing.assert_allclose(model.predict(shuffled), model.predict(X))


def test_missing_pat_gets_the_training_median():
    X = synthetic_features(201)
    model = make_pat_linear().fit(X, true_bp(X))

    row = pd.DataFrame({"hr_bpm": [80.0], "upstroke_s": [0.2], "pat_s": [np.nan]})
    filled = row.assign(pat_s=X["pat_s"].median())
    np.testing.assert_allclose(model.predict(row), model.predict(filled))


def test_pat_linear_fits_on_train_records_only():
    X = synthetic_features(30)
    y = true_bp(X)
    records = ["1_0000"] * 10 + ["1_0001"] * 10 + ["1_0002"] * 10
    table = pd.DataFrame(
        {"window_id": range(30), "record_id": records, "sbp_mmhg": y[:, 0], "dbp_mmhg": y[:, 1]}
    )
    table.loc[10:, ["sbp_mmhg", "dbp_mmhg"]] += 500.0  # val/test labels: wrong on purpose
    features = X.assign(window_id=range(30), record_id=records)
    split = {"train": ["1_0000"], "val": ["1_0001"], "test": ["1_0002"]}

    model, val_mae = fit_pat_linear(table, features, split)

    # Fit on train only, so it predicts the true relation and is off by 500 on val.
    np.testing.assert_allclose(model.predict(X), y, atol=1e-6)
    np.testing.assert_allclose(val_mae, [500.0, 500.0], atol=1e-6)


def test_saved_pat_linear_is_plain_scikit_learn(tmp_path):
    """No class of ours inside, so any checkout can load it, whatever branch it is on."""
    X = synthetic_features(50)
    path = save_model(make_pat_linear().fit(X, true_bp(X)), "pat_linear", models_dir=tmp_path)

    loaded = joblib.load(path)
    modules = {type(step).__module__ for _, step in loaded.steps} | {type(loaded).__module__}
    assert all(m.startswith("sklearn.") for m in modules), modules
    assert loaded.predict(X).shape == (50, 2)


def test_mean_baseline_predicts_training_mean_for_every_row():
    y = np.array([[120.0, 80.0], [100.0, 60.0], [140.0, 70.0]])
    model = MeanBaseline().fit(None, y)

    pred = model.predict(np.zeros((4, 7)))  # X content is ignored, only its length counts

    assert pred.shape == (4, 2)
    np.testing.assert_allclose(pred, [[120.0, 70.0]] * 4)


def test_mae_by_hand():
    y_true = np.array([[120.0, 80.0], [100.0, 60.0]])
    y_pred = np.array([[110.0, 85.0], [100.0, 70.0]])
    # SBP errors 10 and 0 -> 5.0; DBP errors 5 and 10 -> 7.5
    np.testing.assert_allclose(mae(y_true, y_pred), [5.0, 7.5])


def test_fit_uses_train_records_only():
    table = pd.DataFrame(
        {
            "record_id": ["1_0000", "1_0000", "1_0001", "1_0002"],
            "sbp_mmhg": [110.0, 130.0, 150.0, 999.0],
            "dbp_mmhg": [70.0, 90.0, 60.0, 999.0],
        }
    )
    split = {"train": ["1_0000"], "val": ["1_0001"], "test": ["1_0002"]}

    model, val_mae = fit_mean_baseline(table, split)

    np.testing.assert_allclose(model.mean_, [120.0, 80.0])  # val/test labels not used
    np.testing.assert_allclose(val_mae, [30.0, 20.0])  # |120-150|, |80-60|


def test_saved_model_loads_and_predicts(tmp_path):
    model = MeanBaseline().fit(None, [[120.0, 80.0]])

    path = save_model(model, "mean_baseline", models_dir=tmp_path)

    loaded = joblib.load(path)
    assert path.name == "mean_baseline.joblib"
    np.testing.assert_allclose(loaded.predict([0, 0]), [[120.0, 80.0]] * 2)
