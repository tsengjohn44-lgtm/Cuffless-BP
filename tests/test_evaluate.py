"""bp.evaluate end to end on a tiny made-up data folder: 2 records per split, 3 windows each."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from bp import evaluate
from bp.evaluate import Paths

SPLIT = {
    "seed": 42,
    "fractions": [0.70, 0.15, 0.15],
    "train": ["1_0000", "1_0001"],
    "val": ["1_0002", "1_0003"],
    "test": ["1_0004", "1_0005"],
}
# True SBP of each record's 3 windows; DBP = SBP - 40. The test windows are far off
# (200 mmHg), so any test window leaking into a validation result would show.
SBP = {
    "1_0000": [100, 120, 140],
    "1_0001": [110, 130, 150],
    "1_0002": [110, 115, 120],
    "1_0003": [125, 130, 135],
    "1_0004": [200, 200, 200],
    "1_0005": [200, 200, 200],
}
FEATURES = ["sbp_hint", "dbp_hint"]
CONTRACT_KEYS = [
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
SAMPLE_KEYS = {"ppg", "record_id", "sbp_true", "dbp_true", "sbp_pred", "dbp_pred", "metrics_json"}


@pytest.fixture
def paths(tmp_path):
    """A data folder laid out as in docs/data-contract.md, with two saved models:
    'constant' always predicts 120/80; 'hgb' (a linear model standing in for the main
    model) reads the answer from its features, so it is exact."""
    p = Paths(
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        app_dir=tmp_path / "app",
        split_file=tmp_path / "splits" / "split.json",
    )
    rows = [
        {
            "window_id": None,
            "record_id": rec,
            "start_sample": 1250 * k,
            "sbp_mmhg": float(sbp),
            "dbp_mmhg": sbp - 40.0,
            "n_beats": 12,
        }
        for rec, sbps in SBP.items()
        for k, sbp in enumerate(sbps)
    ]
    windows = pd.DataFrame(rows).assign(window_id=range(len(rows)))
    p.windows_csv.parent.mkdir(parents=True)
    windows.to_csv(p.windows_csv, index=False)
    # Row i of the PPG array is filled with the value i, so a demo sample row can be
    # traced back to the window it came from.
    ppg = np.repeat(np.arange(len(windows), dtype=np.float32)[:, None], 1250, axis=1)
    np.savez(p.windows_npz, ppg=ppg, abp=ppg, ecg=ppg)
    features = windows[["window_id", "record_id"]].assign(
        sbp_hint=windows["sbp_mmhg"], dbp_hint=windows["dbp_mmhg"]
    )
    features.to_csv(p.features_csv, index=False)
    p.split_file.parent.mkdir()
    p.split_file.write_text(json.dumps(SPLIT))

    train = windows["record_id"].isin(SPLIT["train"])
    X, y = features.loc[train, FEATURES], windows.loc[train, ["sbp_mmhg", "dbp_mmhg"]]
    p.models_dir.mkdir()
    constant = DummyRegressor(strategy="constant", constant=[120.0, 80.0]).fit(X, y)
    joblib.dump(constant, p.models_dir / "constant.joblib")
    joblib.dump(LinearRegression().fit(X, y), p.models_dir / "hgb.joblib")
    return p


def read_predictions(path):
    return pd.read_csv(path, dtype={"record_id": str})


def test_split_records_reads_test_only_when_final():
    assert evaluate.split_records(SPLIT, final=False) == ("val", SPLIT["val"])
    assert evaluate.split_records(SPLIT, final=True) == ("test", SPLIT["test"])


def test_validation_run_never_touches_test_records(paths):
    evaluate.run(paths, final=False)
    preds = read_predictions(paths.results_dir / "predictions_val.csv")
    assert list(preds.columns) == evaluate.PREDICTION_COLUMNS
    assert len(preds) == 6 * 2  # 6 validation windows x 2 models
    assert set(preds["record_id"]) == set(SPLIT["val"])
    assert not (paths.results_dir / "predictions_test.csv").exists()
    assert not (paths.app_dir / "sample.npz").exists()


def test_metrics_match_hand_computed_values(paths):
    result = evaluate.run(paths, final=False)
    on_disk = json.loads((paths.results_dir / "metrics.json").read_text())
    assert on_disk == result
    assert on_disk["split"] == "val"
    for t in ("sbp", "dbp"):
        for model in ("constant", "hgb"):
            assert list(on_disk["models"][model][t]) == CONTRACT_KEYS

        # 'constant' predicts 120 (SBP) and 80 (DBP). Validation SBP is 110, 115, 120,
        # 125, 130, 135 (DBP 40 lower), so the errors are 10, 5, 0, -5, -10, -15 for both.
        s = on_disk["models"]["constant"][t]
        assert (s["n_windows"], s["n_records"]) == (6, 2)
        assert s["me"] == pytest.approx(-2.5)  # -15 / 6
        assert s["mae"] == pytest.approx(7.5)  # (10 + 5 + 0 + 5 + 10 + 15) / 6 = 45 / 6
        # deviations from -2.5: +-12.5, +-7.5, +-2.5 -> squares 2 * (156.25 + 56.25 + 6.25)
        # = 437.5; / (6 - 1) = 87.5 -> SD = sqrt(87.5) = 9.354
        assert s["sd"] == pytest.approx(87.5**0.5, abs=1e-3)
        assert s["pct_within_5"] == pytest.approx(50.0)  # 5, 0, -5 -> 3 of 6
        assert s["pct_within_10"] == pytest.approx(83.333, abs=1e-3)  # all but -15 -> 5 of 6
        assert s["pct_within_15"] == pytest.approx(100.0)
        assert s["bhs_grade"] == "B"  # 50 / 83 / 100 meets B (50/75/90), misses A's 60
        assert s["aami_pass"] is False  # SD 9.35 > 8

        h = on_disk["models"]["hgb"][t]  # exact model: no error at all
        assert h["mae"] == pytest.approx(0.0, abs=1e-3)
        assert (h["bhs_grade"], h["aami_pass"]) == ("A", True)


def test_figures_and_comparison_table(paths):
    evaluate.run(paths, final=False)
    figures = sorted(p.name for p in (paths.results_dir / "figures" / "val").iterdir())
    kinds = ("scatter", "bland_altman", "error_hist")
    assert figures == sorted(f"{m}_{k}.png" for m in ("constant", "hgb") for k in kinds)
    table = (paths.results_dir / "comparison_val.md").read_text()
    assert "**validation**" in table
    assert table.index("| constant |") < table.index("| hgb |")  # worst SBP MAE first


def test_validation_run_writes_demo_preview(paths):
    result = evaluate.run(paths, final=False)
    windows = pd.read_csv(paths.windows_csv, dtype={"record_id": str})
    with np.load(paths.app_dir / "sample_val.npz") as s:
        assert set(s.files) == SAMPLE_KEYS
        assert s["ppg"].shape == (4, 1250)  # 2 windows from each of the 2 records
        assert s["ppg"].dtype == np.float32
        assert set(s["record_id"]) == set(SPLIT["val"])
        for k in range(4):  # each row's PPG must belong to the window its labels came from
            window = windows.iloc[int(s["ppg"][k, 0])]
            assert window["record_id"] == s["record_id"][k]
            assert window["sbp_mmhg"] == s["sbp_true"][k]
            assert window["dbp_mmhg"] == s["dbp_true"][k]
        assert json.loads(str(s["metrics_json"])) == result["models"]["hgb"]


def test_final_run_scores_the_test_split(paths, capsys):
    assert evaluate.main(["--final"], paths=paths) == 0
    assert "FINAL RUN" in capsys.readouterr().out
    metrics = json.loads((paths.results_dir / "metrics.json").read_text())
    assert metrics["split"] == "test"
    preds = read_predictions(paths.results_dir / "predictions_test.csv")
    assert set(preds["record_id"]) == set(SPLIT["test"])
    sample = paths.app_dir / "sample.npz"
    assert sample.stat().st_size < 10 * 1024 * 1024
    with np.load(sample) as s:
        assert set(s.files) == SAMPLE_KEYS
        assert set(s["record_id"]) == set(SPLIT["test"])


def test_no_demo_sample_without_the_main_model(paths, capsys):
    (paths.models_dir / "hgb.joblib").unlink()
    evaluate.run(paths, final=False)
    assert not (paths.app_dir / "sample_val.npz").exists()
    assert "no 'hgb' model" in capsys.readouterr().out


@pytest.mark.parametrize(
    "missing, make_step",
    [
        ("windows_csv", "make windows"),
        ("features_csv", "make features"),
        ("split_file", "make windows"),
    ],
)
def test_missing_input_names_the_make_step(paths, capsys, missing, make_step):
    getattr(paths, missing).unlink()
    assert evaluate.main([], paths=paths) == 1
    err = capsys.readouterr().err
    assert getattr(paths, missing).name in err and make_step in err


def test_no_models_says_how_to_make_them(paths, capsys):
    for f in paths.models_dir.iterdir():
        f.unlink()
    assert evaluate.main([], paths=paths) == 1
    assert "make train" in capsys.readouterr().err


def test_features_missing_a_window_is_caught(paths, capsys):
    features = pd.read_csv(paths.features_csv)
    features[features["window_id"] != 6].to_csv(paths.features_csv, index=False)  # a val window
    assert evaluate.main([], paths=paths) == 1
    assert "lacks 1" in capsys.readouterr().err


def test_model_with_wrong_output_shape_is_named(paths):
    X = pd.DataFrame({name: [0.0, 0.0, 0.0] for name in FEATURES})
    one_column = DummyRegressor().fit(X, [120.0, 121.0, 122.0])  # predicts SBP only
    joblib.dump(one_column, paths.models_dir / "one_column.joblib")
    with pytest.raises(ValueError, match="one_column"):
        evaluate.run(paths, final=False)


def test_pick_sample_rows_caps_total_and_per_record():
    labels = pd.DataFrame({"record_id": np.repeat([f"1_{i:04d}" for i in range(300)], 3)})
    rows = evaluate.pick_sample_rows(labels, n=200, per_record=2, seed=0)
    assert len(rows) == 200 and len(set(rows)) == 200
    assert labels.iloc[rows]["record_id"].value_counts().max() <= 2
    assert (rows == evaluate.pick_sample_rows(labels, n=200, per_record=2, seed=0)).all()
