"""Baselines: the simplest predictors that every real model has to beat.

Plain-language summary:
- Baseline 1 (Phase 1, this file): predict the mean SBP and DBP of the training windows
  for every window, ignoring the signal completely. Its error tells us how much blood
  pressure varies between windows; a model that cannot beat it has learned nothing.
- Baseline 2 (Phase 2, not built yet): pulse arrival time + heart rate into a linear
  regression, once features.py exists.
- Each model is fit on the train split only and saved to models/<name>.joblib with
  `predict(X) -> (n, 2)` array, column 0 = SBP, column 1 = DBP, in mmHg (contract §4).
- main() also prints the validation MAE as a quick check. The official numbers (ME, SD,
  BHS, confidence intervals) come from evaluate.py. Nothing here reads the test split.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from bp.fileio import atomic_write
from bp.paths import DATA_DIR, INTERIM_DIR, SPLIT_FILE
from bp.split import load_split

MODELS_DIR = DATA_DIR / "models"
TARGETS = ["sbp_mmhg", "dbp_mmhg"]


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


def save_model(model, name: str, models_dir: Path = MODELS_DIR) -> Path:
    path = models_dir / f"{name}.joblib"
    with atomic_write(path) as f:
        joblib.dump(model, f)
    return path


def main(argv: list[str] | None = None) -> int:
    windows_csv = INTERIM_DIR / "windows.csv"
    for needed in (windows_csv, SPLIT_FILE):
        if not needed.exists():
            print(f"{needed} not found. Run `make windows` first.", file=sys.stderr)
            return 1
    table = pd.read_csv(windows_csv, dtype={"record_id": str})
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
    return 0


if __name__ == "__main__":
    # Run main from the imported module, not from __main__, so the saved class is
    # pickled as bp.baselines.MeanBaseline and evaluate.py can load it.
    from bp.baselines import main as _main

    sys.exit(_main())
