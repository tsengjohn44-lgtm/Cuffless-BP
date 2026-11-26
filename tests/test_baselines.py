"""Baseline 1 (training mean): hand-checked numbers, train-only fitting, saving."""

import joblib
import numpy as np
import pandas as pd

from bp.baselines import MeanBaseline, fit_mean_baseline, mae, save_model


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
