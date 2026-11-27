"""Baselines: the simplest predictors that every real model has to beat.

Plain-language summary:
- Baseline 1 (Phase 1, this file): predict the mean SBP and DBP of the training windows
  for every window, ignoring the signal completely. Its error tells us how much blood
  pressure varies between windows; a model that cannot beat it has learned nothing.
- Baseline 2 (Phase 2): SBP and DBP as straight-line functions of pulse arrival time
  (pat_s) and heart rate (hr_bpm). This is the classic cuffless idea: a stiffer, more
  pressurised artery carries the pulse faster, so a shorter PAT should mean higher BP.
  Windows with no usable ECG beats have no PAT (about 10%); they get the training median.
  Caveat from features.py: pat_s here likely includes an ECG-to-PPG recording delay that
  differs between records, so this baseline may do little better than baseline 1.
- Each model is fit on the train split only and saved to models/<name>.joblib with
  `predict(X) -> (n, 2)` array, column 0 = SBP, column 1 = DBP, in mmHg (contract §4).
  X is the whole feature table (key columns dropped), as evaluate.py passes it.
- Baseline 2 is built only from scikit-learn parts (no class of ours), so the saved file
  loads in any checkout, whatever branch it is on.
- main() also prints the validation MAE as a quick check. The official numbers (ME, SD,
  BHS, confidence intervals) come from evaluate.py. Nothing here reads the test split.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from bp.fileio import atomic_write
from bp.paths import DATA_DIR, INTERIM_DIR, SPLIT_FILE
from bp.split import load_split

MODELS_DIR = DATA_DIR / "models"
TARGETS = ["sbp_mmhg", "dbp_mmhg"]
KEY_COLUMNS = ["window_id", "record_id"]
PAT_HR = ["pat_s", "hr_bpm"]  # baseline 2 inputs


class MeanBaseline:
    """Baseline 1: predict the training mean SBP and DBP for every window, whatever X is."""

    def fit(self, X, y):
        self.mean_ = np.asarray(y, dtype=float).mean(axis=0)  # [SBP, DBP]
        return self

    def predict(self, X) -> np.ndarray:
        return np.tile(self.mean_, (len(X), 1))


def rows_in(table: pd.DataFrame, record_ids: list[str]) -> pd.DataFrame:
    """The windows that belong to the given records."""
    return table[table["record_id"].isin(set(record_ids))]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Mean absolute error per column: [SBP MAE, DBP MAE] in mmHg."""
    return np.abs(np.asarray(y_pred) - np.asarray(y_true)).mean(axis=0)


def fit_mean_baseline(table: pd.DataFrame, split: dict) -> tuple[MeanBaseline, np.ndarray]:
    """Fit on the train records; return the model and its validation MAE [SBP, DBP]."""
    train, val = rows_in(table, split["train"]), rows_in(table, split["val"])
    model = MeanBaseline().fit(train, train[TARGETS])
    return model, mae(val[TARGETS], model.predict(val))


def make_pat_linear() -> Pipeline:
    """Baseline 2: pick PAT and heart rate from the feature table, fill a missing value with
    the training median, then one linear regression for SBP and DBP together."""
    pick = ColumnTransformer([("pat_hr", SimpleImputer(strategy="median"), PAT_HR)])
    return Pipeline([("pick", pick), ("linear", LinearRegression())])


def join_features(table: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Windows (with labels) plus their features, matched on window_id."""
    data = table.merge(features.drop(columns="record_id"), on="window_id", validate="one_to_one")
    if len(data) != len(table):
        raise ValueError(
            f"features.csv lacks {len(table) - len(data)} windows. Re-run `make features`."
        )
    return data


def fit_pat_linear(
    table: pd.DataFrame, features: pd.DataFrame, split: dict
) -> tuple[Pipeline, np.ndarray]:
    """Fit baseline 2 on the train records; return it and its validation MAE [SBP, DBP]."""
    data = join_features(table, features)
    feature_cols = [c for c in features.columns if c not in KEY_COLUMNS]
    train, val = rows_in(data, split["train"]), rows_in(data, split["val"])
    model = make_pat_linear().fit(train[feature_cols], train[TARGETS])
    return model, mae(val[TARGETS], model.predict(val[feature_cols]))


def save_model(model, name: str, models_dir: Path = MODELS_DIR) -> Path:
    path = models_dir / f"{name}.joblib"
    with atomic_write(path) as f:
        joblib.dump(model, f)
    return path


def main(argv: list[str] | None = None) -> int:
    windows_csv = INTERIM_DIR / "windows.csv"
    features_csv = INTERIM_DIR / "features.csv"
    for needed, step in (
        (windows_csv, "windows"),
        (SPLIT_FILE, "windows"),
        (features_csv, "features"),
    ):
        if not needed.exists():
            print(f"{needed} not found. Run `make {step}` first.", file=sys.stderr)
            return 1
    table = pd.read_csv(windows_csv, dtype={"record_id": str})
    features = pd.read_csv(features_csv, dtype={"record_id": str})
    split = load_split()  # train and val only

    model, val_mae = fit_mean_baseline(table, split)
    path = save_model(model, "mean_baseline")

    train, val = rows_in(table, split["train"]), rows_in(table, split["val"])
    print(
        f"Baseline 1 (training mean), fit on {len(train)} windows / "
        f"{train['record_id'].nunique()} records:"
    )
    print(f"  predicts SBP {model.mean_[0]:.1f} mmHg, DBP {model.mean_[1]:.1f} mmHg")
    print(
        f"  validation MAE ({len(val)} windows / {val['record_id'].nunique()} records): "
        f"SBP {val_mae[0]:.2f} mmHg, DBP {val_mae[1]:.2f} mmHg"
    )
    print(f"Wrote {path}")

    model2, val_mae2 = fit_pat_linear(table, features, split)
    path2 = save_model(model2, "pat_linear")
    linear = model2.named_steps["linear"]
    print("\nBaseline 2 (PAT + heart rate, linear), fit on the same train windows:")
    for i, name in enumerate(["SBP", "DBP"]):
        pat, hr = linear.coef_[i]
        print(
            f"  {name} = {linear.intercept_[i]:.1f} {pat * 0.1:+.2f} per +100 ms PAT "
            f"{hr:+.3f} per +1 bpm"
        )
    print(f"  validation MAE: SBP {val_mae2[0]:.2f} mmHg, DBP {val_mae2[1]:.2f} mmHg")
    print(f"Wrote {path2}")
    return 0


if __name__ == "__main__":
    # Run main from the imported module, not from __main__, so the saved class is
    # pickled as bp.baselines.MeanBaseline and evaluate.py can load it.
    from bp.baselines import main as _main

    sys.exit(_main())
