"""Patient-leakage check: does the main model's gain survive a stricter split?

Plain-language summary:
- The dataset has no patient IDs. If neighbouring records come from the same patient,
  our record-level split could put one person in both train and test, and the model
  could learn "this person's pulse shape -> this person's BP" instead of a general rule.
- This script uses the TRAIN and VALIDATION records only. The test list is never read
  (load_split() leaves it out), so nothing here touches the held-out test set.
- Check A, neighbour similarity: are records with adjacent ids (1_0041, 1_0042) more
  alike in mean SBP/DBP than random pairs? A high lag-1 correlation that fades with the
  gap means runs of consecutive records come from one patient.
- Check B, block cross-validation: 5-fold CV over train+val records, done three ways.
  (1) records assigned to folds at random (like our real split);
  (2) and (3) consecutive records kept together in blocks of BLOCKS[0] and BLOCKS[1],
  so a run of one patient's records cannot sit on both sides.
  If the main model's MAE rises a lot under blocks, the random split was flattering it.
  The mean baseline and a linear model on the same features are scored too, as anchors.
- Main model settings are read from models/hgb_tuning.csv (the ones chosen on validation);
  nothing is tuned here.

Usage (after `make all`):  PYTHONPATH=src python tools/leakage_check.py
Writes results/leakage_check.json and prints a table.
"""

import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bp.paths import DATA_DIR, INTERIM_DIR, RESULTS_DIR
from bp.split import load_split

TARGETS = ["sbp_mmhg", "dbp_mmhg"]
KEYS = ["window_id", "record_id"]
N_FOLDS = 5
BLOCKS = (20, 100)  # consecutive records kept together
LAGS = (1, 2, 5, 10, 20, 50, 100, 200)
SEED = 0


def record_order(record_id: str) -> tuple[int, int]:
    part, idx = record_id.split("_")
    return int(part), int(idx)


def neighbour_similarity(per_record: pd.DataFrame) -> dict:
    """Correlation of record-mean BP between a record and the one `lag` ids later.
    Only pairs where both records are in train+val are used."""
    per_record = per_record.copy()
    per_record["part"], per_record["idx"] = zip(*per_record.index.map(record_order), strict=True)
    out = {}
    for target in TARGETS:
        rows = {}
        for lag in LAGS:
            a, b = [], []
            for _, grp in per_record.groupby("part"):
                s = grp.set_index("idx")[target]
                later = s.reindex(s.index + lag)
                ok = later.notna().to_numpy()
                a.append(s.to_numpy()[ok])
                b.append(later.to_numpy()[ok])
            a, b = np.concatenate(a), np.concatenate(b)
            rows[str(lag)] = {
                "n_pairs": int(a.size),
                "corr": float(np.corrcoef(a, b)[0, 1]),
                "median_abs_diff": float(np.median(np.abs(a - b))),
            }
        # Random pairs as the reference.
        rng = np.random.default_rng(SEED)
        v = per_record[target].to_numpy()
        i, j = rng.integers(0, v.size, (2, 20000))
        keep = i != j
        rows["random"] = {
            "n_pairs": int(keep.sum()),
            "corr": float(np.corrcoef(v[i[keep]], v[j[keep]])[0, 1]),
            "median_abs_diff": float(np.median(np.abs(v[i[keep]] - v[j[keep]]))),
        }
        out[target] = rows
    return out


def fold_of(record_ids: pd.Series, block: int | None) -> np.ndarray:
    """Fold number per record. block=None: each record on its own (random);
    otherwise consecutive ids are grouped in blocks of `block`, and whole blocks go to folds."""
    rng = np.random.default_rng(SEED)
    if block is None:
        units = record_ids.to_numpy()
    else:
        units = np.array([f"{p}_{i // block}" for p, i in map(record_order, record_ids)])
    uniq = np.unique(units)
    fold_of_unit = dict(zip(uniq, rng.permutation(np.arange(uniq.size) % N_FOLDS), strict=True))
    return np.array([fold_of_unit[u] for u in units])


def models(hgb_params: dict) -> dict:
    hgb = HistGradientBoostingRegressor(early_stopping=False, random_state=0, **hgb_params)

    class Mean:
        def fit(self, X, y):
            self.m = y.mean(axis=0)
            return self

        def predict(self, X):
            return np.tile(self.m, (len(X), 1))

    return {
        "mean_baseline": Mean,
        "linear_all_features": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()
        ),
        "hgb": lambda: MultiOutputRegressor(hgb),
    }


def main() -> int:
    windows = pd.read_csv(INTERIM_DIR / "windows.csv", dtype={"record_id": str})
    features = pd.read_csv(INTERIM_DIR / "features.csv", dtype={"record_id": str})
    split = load_split()  # train + val only; "test" is removed
    keep = set(split["train"]) | set(split["val"])
    data = windows[[*KEYS, *TARGETS]].merge(
        features.drop(columns="record_id"), on="window_id", validate="one_to_one"
    )
    data = data[data["record_id"].isin(keep)].reset_index(drop=True)
    feature_cols = [c for c in features.columns if c not in KEYS]
    print(f"Train+val only: {data['record_id'].nunique()} records, {len(data)} windows.")

    tuning = pd.read_csv(DATA_DIR / "models" / "hgb_tuning.csv")
    best = tuning.iloc[0]
    hgb_params = {
        "learning_rate": float(best["learning_rate"]),
        "max_iter": int(best["max_iter"]),
        "max_leaf_nodes": int(best["max_leaf_nodes"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
    }
    print(f"Main model settings (from validation tuning): {hgb_params}")

    # Check A
    per_record = data.groupby("record_id")[TARGETS].mean()
    sim = neighbour_similarity(per_record)
    print("\nCheck A: correlation of record-mean BP, record vs. the one `lag` ids later")
    for t in TARGETS:
        cells = "  ".join(f"{k}:{v['corr']:+.2f}" for k, v in sim[t].items())
        print(f"  {t}: {cells}")

    # Check B
    rec_series = pd.Series(sorted(keep, key=record_order))
    schemes = {"random_records": None, **{f"blocks_of_{b}": b for b in BLOCKS}}
    X, y = data[feature_cols], data[TARGETS].to_numpy()
    results = {}
    for scheme, block in schemes.items():
        fold_map = dict(zip(rec_series, fold_of(rec_series, block), strict=True))
        folds = data["record_id"].map(fold_map).to_numpy()
        results[scheme] = {}
        for name, make in models(hgb_params).items():
            abs_err = np.zeros_like(y, dtype=float)
            t0 = time.time()
            for k in range(N_FOLDS):
                tr, te = folds != k, folds == k
                model = make().fit(X[tr], y[tr])
                abs_err[te] = np.abs(model.predict(X[te]) - y[te])
            sbp, dbp = abs_err.mean(axis=0)
            results[scheme][name] = {"sbp_mae": float(sbp), "dbp_mae": float(dbp)}
            print(
                f"  {scheme:15s} {name:20s} SBP {sbp:6.2f} DBP {dbp:5.2f} ({time.time() - t0:.0f}s)"
            )

    out = {
        "note": "Train+val records only; the test split is not read.",
        "hgb_params": hgb_params,
        "n_records": int(data["record_id"].nunique()),
        "n_windows": int(len(data)),
        "neighbour_similarity": sim,
        "cv_mae": results,
    }
    path = RESULTS_DIR / "leakage_check.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
