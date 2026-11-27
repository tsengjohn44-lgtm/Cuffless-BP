"""Main model (train.py): grid, NaN handling, validation-only tuning, train/val only, saving."""

import joblib
import numpy as np
import pandas as pd

from bp.train import load_training_data, make_hgb, param_grid, save_model, tune


def synthetic(n: int, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Features with a curved (non-linear) BP relation and some missing values."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"pat_s": rng.uniform(0.3, 0.8, n), "hr_bpm": rng.uniform(60, 120, n)})
    sbp = 120 + 40 * np.sin(6 * X["pat_s"]) + 0.2 * X["hr_bpm"]
    dbp = 70 + 15 * np.cos(6 * X["pat_s"])
    X.loc[::10, "pat_s"] = np.nan  # every 10th window has no PAT
    return X, np.column_stack([sbp, dbp])


def test_param_grid_lists_every_combination():
    combos = param_grid({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(combos) == 6
    assert {"a": 2, "b": "z"} in combos


def test_model_predicts_sbp_and_dbp_with_missing_features():
    X, y = synthetic(300)
    pred = make_hgb(max_iter=20).fit(X, y).predict(X)
    assert pred.shape == (300, 2)
    assert np.isfinite(pred).all()


def test_tune_keeps_the_setting_with_the_lowest_validation_error():
    X_tr, y_tr = synthetic(600)
    X_va, y_va = synthetic(200, seed=1)

    model, params, table = tune(X_tr, y_tr, X_va, y_va, grid={"max_iter": [1, 100]})

    assert params == {"max_iter": 100}  # 1 tree can't fit the curve; 100 can
    assert list(table["max_iter"]) == [100, 1]  # sorted best first
    np.testing.assert_allclose(table["score"], (table["val_mae_sbp"] + table["val_mae_dbp"]) / 2)
    mean_mae = np.abs(y_va - y_tr.mean(axis=0)).mean()
    assert table.loc[0, "score"] < mean_mae / 2  # far better than predicting the mean


def test_training_data_has_train_and_val_records_only():
    windows = pd.DataFrame(
        {
            "window_id": [0, 1, 2, 3],
            "record_id": ["1_0000", "1_0000", "1_0001", "1_0002"],
            "start_sample": 0,
            "sbp_mmhg": [110.0, 120.0, 130.0, 140.0],
            "dbp_mmhg": [70.0, 75.0, 80.0, 85.0],
        }
    )
    features = pd.DataFrame(
        {"window_id": [3, 2, 1, 0], "record_id": windows["record_id"][::-1], "hr_bpm": [4, 3, 2, 1]}
    )
    split = {"train": ["1_0000"], "val": ["1_0001"], "test": ["1_0002"]}

    X_tr, y_tr, X_va, y_va = load_training_data(windows, features, split)

    assert list(X_tr.columns) == ["hr_bpm"]  # key columns dropped
    np.testing.assert_array_equal(X_tr["hr_bpm"], [1, 2])  # matched on window_id
    np.testing.assert_array_equal(y_tr, [[110, 70], [120, 75]])
    np.testing.assert_array_equal(y_va, [[130, 80]])  # the test record never appears


def test_saved_model_is_plain_scikit_learn(tmp_path):
    X, y = synthetic(100)
    path = save_model(make_hgb(max_iter=5).fit(X, y), models_dir=tmp_path)

    loaded = joblib.load(path)
    assert path.name == "hgb.joblib"
    modules = {type(loaded).__module__} | {type(e).__module__ for e in loaded.estimators_}
    assert all(m.startswith("sklearn.") for m in modules), modules
    assert loaded.predict(X).shape == (100, 2)


def test_grid_still_contains_the_phase2_best():
    """Retuning can then only match or beat the Phase 2 validation error, never lose to it."""
    from bp.train import GRID, PHASE2_BEST

    assert PHASE2_BEST in param_grid(GRID)
