"""Main model: gradient-boosted trees for SBP and DBP, tuned on the validation split only.

Owner: Model track. Formats: docs/data-contract.md, section 4.

Plain-language summary:
- Inputs: every PPG/ECG feature in features.csv (16 per window). Labels: SBP and DBP from
  windows.csv. The arterial pressure itself is never an input.
- Model: HistGradientBoostingRegressor. It builds many small decision trees, each one
  correcting the errors left by the ones before. It handles missing features (NaN) on its
  own, needs no scaling, and can learn curved relations (BP need not move in a straight
  line with PAT or pulse shape), which the linear baseline 2 cannot.
- One model for SBP and one for DBP: MultiOutputRegressor fits a separate copy per
  target, with the same settings for both.
- Tuning: a small grid of settings (GRID). Each is fit on the train records and scored on
  the validation records by the mean of the SBP and DBP MAE. The best one is kept. The
  test split is never read (load_split() leaves it out).
- Deterministic: no early stopping and no subsampling, so `make train` builds the same
  model every time.
- The model is fit on train only, so its validation numbers stay honest. Whether to refit
  on train + val before the one final test run is a Phase 4 decision.
- Saved to models/hgb.joblib (plain scikit-learn objects, so it loads in any checkout);
  predict(X) -> (n, 2): SBP, DBP in mmHg. The tuning table goes to models/hgb_tuning.csv
  and is printed (results/ belongs to the eval track).
"""

import itertools
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from bp.fileio import atomic_write
from bp.paths import DATA_DIR, INTERIM_DIR, SPLIT_FILE
from bp.split import load_split

MODEL_NAME = "hgb"  # evaluate.py's MAIN_MODEL: the model the demo shows
MODELS_DIR = DATA_DIR / "models"
TARGETS = ["sbp_mmhg", "dbp_mmhg"]
KEY_COLUMNS = ["window_id", "record_id"]
RANDOM_STATE = 0

# 16 settings. learning_rate x max_iter: how far each tree corrects and how many trees.
# max_leaf_nodes: how detailed one tree can be. min_samples_leaf: the fewest windows a leaf
# may hold; larger values smooth over noisy windows.
# Phase 2 searched learning_rate {0.05, 0.1} x max_iter {200, 500} x max_leaf_nodes {15, 63}
# x min_samples_leaf {20, 200}. The best (0.1, 500, 63, 20) sat at the edge of three of the
# four ranges, so Phase 3 moves each range outward from it. PHASE2_BEST stays in the grid,
# so the validation error can only stay the same or improve.
PHASE2_BEST = {"learning_rate": 0.1, "max_iter": 500, "max_leaf_nodes": 63, "min_samples_leaf": 20}
GRID = {
    "learning_rate": [0.05, 0.1],
    "max_iter": [500, 1000],
    "max_leaf_nodes": [63, 127],
    "min_samples_leaf": [10, 20],
}


def param_grid(grid: dict[str, list]) -> list[dict]:
    """Every combination of the grid's values, as a list of keyword dicts."""
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]


def make_hgb(**params) -> MultiOutputRegressor:
    """One HistGradientBoostingRegressor per target (SBP, DBP), same settings for both."""
    hgb = HistGradientBoostingRegressor(early_stopping=False, random_state=RANDOM_STATE, **params)
    return MultiOutputRegressor(hgb)


def load_training_data(
    windows: pd.DataFrame, features: pd.DataFrame, split: dict
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """(X_train, y_train, X_val, y_val). Only train and val records; test is never touched.
    X keeps every feature column in features.csv order, as evaluate.py passes it."""
    data = windows[[*KEY_COLUMNS, *TARGETS]].merge(
        features.drop(columns="record_id"), on="window_id", validate="one_to_one"
    )
    if len(data) != len(windows):
        raise ValueError(
            f"features.csv lacks {len(windows) - len(data)} windows. Re-run `make features`."
        )
    feature_cols = [c for c in features.columns if c not in KEY_COLUMNS]
    parts = []
    for name in ("train", "val"):
        rows = data[data["record_id"].isin(set(split[name]))]
        parts += [rows[feature_cols].reset_index(drop=True), rows[TARGETS].to_numpy()]
    return tuple(parts)


def val_mae(model, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """[SBP MAE, DBP MAE] in mmHg."""
    return np.abs(model.predict(X) - y).mean(axis=0)


def tune(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    grid: dict[str, list] = GRID,
    verbose: bool = False,
) -> tuple[MultiOutputRegressor, dict, pd.DataFrame]:
    """Fit one model per setting on train, score it on val. Returns the best fitted model,
    its settings, and the table of all settings sorted best first.
    Score = mean of SBP and DBP validation MAE (lower is better)."""
    rows, best, best_score = [], None, np.inf
    for i, params in enumerate(param_grid(grid), 1):
        start = time.time()
        model = make_hgb(**params).fit(X_train, y_train)
        sbp, dbp = val_mae(model, X_val, y_val)
        score = (sbp + dbp) / 2
        rows.append({**params, "val_mae_sbp": sbp, "val_mae_dbp": dbp, "score": score})
        if verbose:
            print(f"  [{i:2d}] {params} -> score {score:.3f} ({time.time() - start:.1f} s)")
        if score < best_score:
            best, best_score = (model, params), score
    table = pd.DataFrame(rows).sort_values("score", ignore_index=True)
    return best[0], best[1], table


def save_model(model, name: str = MODEL_NAME, models_dir: Path = MODELS_DIR) -> Path:
    path = models_dir / f"{name}.joblib"
    with atomic_write(path) as f:
        joblib.dump(model, f)
    return path


def main(argv: list[str] | None = None) -> int:
    windows_csv, features_csv = INTERIM_DIR / "windows.csv", INTERIM_DIR / "features.csv"
    for needed, step in (
        (windows_csv, "windows"),
        (SPLIT_FILE, "windows"),
        (features_csv, "features"),
    ):
        if not needed.exists():
            print(f"{needed} not found. Run `make {step}` first.", file=sys.stderr)
            return 1
    windows = pd.read_csv(windows_csv, dtype={"record_id": str})
    features = pd.read_csv(features_csv, dtype={"record_id": str})
    X_train, y_train, X_val, y_val = load_training_data(windows, features, load_split())

    n_settings = len(param_grid(GRID))
    print(
        f"Tuning {n_settings} settings: fit on {len(X_train)} train windows, "
        f"scored on {len(X_val)} validation windows, {X_train.shape[1]} features."
    )
    model, params, table = tune(X_train, y_train, X_val, y_val, verbose=True)

    path = save_model(model)
    with atomic_write(MODELS_DIR / f"{MODEL_NAME}_tuning.csv", "w") as f:
        table.to_csv(f, index=False, float_format="%.3f")

    print("\nBest 5 settings (validation MAE in mmHg):")
    print(table.head(5).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    sbp, dbp = table.loc[0, ["val_mae_sbp", "val_mae_dbp"]]
    print(f"\nKept: {params}\n  validation MAE: SBP {sbp:.2f} mmHg, DBP {dbp:.2f} mmHg")
    print(f"Wrote {path} and {MODEL_NAME}_tuning.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
