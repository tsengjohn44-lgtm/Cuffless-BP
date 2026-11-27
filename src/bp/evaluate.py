"""Score every saved model on the validation split, or (once, in Phase 4) on the test split.

Owner: Evaluation + write-up track. Formats: docs/data-contract.md, sections 4-7.

Plain-language summary:
1. Pick the records to score from splits/split.json: the validation records, or the
   test records with --final. `split_records` is the only code that reads the test
   list, and only `make final` passes --final (CLAUDE.md: the test split is scored
   once, in Phase 4, and nothing is tuned afterwards).
2. Load the true SBP/DBP of those records' windows (windows.csv) and their features
   (features.csv), matched on window_id.
3. Load every saved model (models/*.joblib) and ask it for SBP and DBP per window.
4. Write, for that split:
     results/predictions_<split>.csv   one row per window per model
     results/metrics.json              every metric for every model (see bp.metrics)
     results/comparison_<split>.md     the comparison table that goes in the README
     results/figures/<split>/          scatter, Bland-Altman and error histogram per model
     app/sample_val.npz                validation run: demo preview from the main model
     app/sample.npz                    --final run: the demo's real sample (test windows)
     results/error_analysis_<split>.md where the main model's errors are largest (Phase 3):
     results/error_analysis_<split>.json  by true BP, heart rate, PPG quality and PAT; how
                                       much is a fixed per-record offset; a one-reading
                                       calibration what-if; which features it relies on
If an input is missing, it names the file and the make step that creates it.

Usage:
    python -m bp.evaluate            # validation split (make eval)
    python -m bp.evaluate --final    # test split, Phase 4 only (make final)
"""

import argparse
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bp import metrics
from bp.fileio import atomic_write
from bp.paths import DATA_DIR, REPO_ROOT, RESULTS_DIR, SPLIT_FILE

KEY_COLUMNS = ["window_id", "record_id"]
TARGETS = ("sbp", "dbp")
PREDICTION_COLUMNS = [
    "window_id",
    "record_id",
    "model",
    "sbp_true",
    "dbp_true",
    "sbp_pred",
    "dbp_pred",
]
SPLIT_LABELS = {"val": "validation", "test": "test"}

MAIN_MODEL = "hgb"  # the model the demo shows (data contract, section 4)
SAMPLE_SIZE = 200  # demo windows: 200 x 1,250 float32 samples = 1 MB
SAMPLE_PER_RECORD = 2  # at most 2 windows per record, so the demo shows many patients
SAMPLE_SEED = 0
MAX_SAMPLE_BYTES = 10 * 1024 * 1024  # CLAUDE.md: the demo sample stays under 10 MB

# Error analysis (Phase 3): (heading, column, cut points). The cut points are fixed, so the
# validation and test splits are grouped the same way. A bin includes its lower edge:
# "100–120" means 100 <= value < 120. The PPG-quality cut points are roughly the quartiles
# of the validation windows.
ERROR_SLICES = [
    ("True SBP (mmHg)", "sbp_true", [100, 120, 140, 160]),
    ("True DBP (mmHg)", "dbp_true", [50, 60, 70, 80]),
    ("Heart rate (bpm, measured on the PPG)", "hr_bpm", [60, 80, 100, 120]),
    ("PPG skewness: a signal-quality index, lower = noisier", "ppg_skew", [0.45, 0.65, 0.8]),
    (
        "Beat-shape consistency (beat_corr): lower = beats less alike",
        "beat_corr",
        [0.987, 0.997, 0.999],
    ),
]
PAT_PRESENT, PAT_MISSING = "available", "missing (no usable ECG beats)"
WORST_FRACTION = 0.1  # "the 10% of records with the largest offset"
IMPORTANCE_REPEATS = 3
IMPORTANCE_SEED = 0

FINAL_BANNER = """\
==========================================================================
FINAL RUN on the TEST split. Per CLAUDE.md this happens once, in Phase 4.
Do not tune anything after seeing these numbers.
=========================================================================="""

# Figure colours: one data colour and neutral greys for text, grid and reference lines.
SERIES = "#257de0"
INK = "#191919"
INK_2 = "#666666"
GRID = "#e5e5e5"
LIMIT_LINE = "#9c9c9c"


class MissingInput(Exception):
    """An input is missing or incomplete. The message names the make step that fixes it."""


@dataclass(frozen=True)
class Paths:
    """Every folder evaluate reads or writes. Tests point these at a temporary folder."""

    data_dir: Path = DATA_DIR
    results_dir: Path = RESULTS_DIR
    app_dir: Path = REPO_ROOT / "app"
    split_file: Path = SPLIT_FILE

    @property
    def windows_csv(self) -> Path:
        return self.data_dir / "interim" / "windows.csv"

    @property
    def windows_npz(self) -> Path:
        return self.data_dir / "interim" / "windows.npz"

    @property
    def features_csv(self) -> Path:
        return self.data_dir / "interim" / "features.csv"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"


def require(path: Path, make_step: str) -> Path:
    if not path.exists():
        raise MissingInput(f"{path} not found. Run `{make_step}` first.")
    return path


# ---------------------------------------------------------------- inputs


def split_records(split: dict, final: bool) -> tuple[str, list[str]]:
    """(split name, record ids) to score: validation, or test with --final.
    This is the only code in the project that reads split["test"]."""
    name = "test" if final else "val"
    return name, list(split[name])


def load_split_windows(paths: Paths, records: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The windows of `records`, sorted by window_id, as two row-aligned tables:
    labels   window_id, record_id, sbp_true, dbp_true, npz_row (its row in windows.npz)
    features only the feature columns, exactly what the models were trained on."""
    # record_id is read as text: "1_0042" must not turn into a number.
    windows = pd.read_csv(require(paths.windows_csv, "make windows"), dtype={"record_id": str})
    windows["npz_row"] = np.arange(len(windows))  # row i of windows.csv = row i of windows.npz
    # Keep only what we need, so a feature that shares a name with another windows.csv
    # column (e.g. n_beats) is not renamed n_beats_x / n_beats_y by the merge below.
    in_split = windows["record_id"].isin(set(records))
    windows = windows.loc[in_split, [*KEY_COLUMNS, "sbp_mmhg", "dbp_mmhg", "npz_row"]]
    if windows.empty:
        raise MissingInput(
            f"no window in {paths.windows_csv} belongs to the split. Re-run `make windows`."
        )

    features = pd.read_csv(require(paths.features_csv, "make features"), dtype={"record_id": str})
    table = windows.merge(features, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(table) != len(windows):
        raise MissingInput(
            f"{paths.features_csv} lacks {len(windows) - len(table)} of the split's "
            "windows. Re-run `make features`."
        )
    table = table.sort_values("window_id", ignore_index=True)

    labels = pd.DataFrame(
        {
            "window_id": table["window_id"],
            "record_id": table["record_id"],
            "sbp_true": table["sbp_mmhg"],
            "dbp_true": table["dbp_mmhg"],
            "npz_row": table["npz_row"],
        }
    )
    feature_columns = [c for c in features.columns if c not in KEY_COLUMNS]
    return labels, table[feature_columns]


def load_models(models_dir: Path) -> dict:
    """Every saved model, by name. joblib files can run code when loaded, so this only
    ever loads the ones our own `make baselines` and `make train` wrote."""
    files = sorted(models_dir.glob("*.joblib"))
    if not files:
        raise MissingInput(
            f"no saved models in {models_dir}. Run `make baselines` and `make train` first."
        )
    return {f.stem: joblib.load(f) for f in files}


def predict(models: dict, features: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each model's (n_windows, 2) predictions: column 0 = SBP, column 1 = DBP, in mmHg."""
    preds = {}
    for name, model in models.items():
        out = np.asarray(model.predict(features), dtype=float)
        if out.shape != (len(features), 2):
            raise ValueError(
                f"model {name!r} returned shape {out.shape}; the data contract (section 4) "
                f"needs ({len(features)}, 2): SBP, DBP"
            )
        preds[name] = out
    return preds


# ---------------------------------------------------------------- scoring


def predictions_table(labels: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """Long format, data contract section 5: one row per window per model."""
    parts = [
        labels.assign(model=name, sbp_pred=p[:, 0], dbp_pred=p[:, 1]) for name, p in preds.items()
    ]
    return pd.concat(parts, ignore_index=True)[PREDICTION_COLUMNS]


def rounded(obj, digits: int = 3):
    """Round every float in nested dicts/lists, so metrics.json stays readable."""
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: rounded(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rounded(v, digits) for v in obj]
    return obj


def compute_metrics(table: pd.DataFrame, split_name: str) -> dict:
    """metrics.json content, data contract section 6."""
    models = {
        name: {
            t: metrics.summarize(rows[f"{t}_pred"], rows[f"{t}_true"], rows["record_id"])
            for t in TARGETS
        }
        for name, rows in table.groupby("model", sort=False)
    }
    return rounded({"split": split_name, "models": models})


def comparison_table(result: dict) -> str:
    """Markdown table of every model and baseline, worst to best SBP MAE."""
    split = SPLIT_LABELS[result["split"]]
    models = result["models"]
    order = sorted(models, key=lambda name: -models[name]["sbp"]["mae"])
    lines = [
        f"Errors on the **{split}** split, in mmHg (error = predicted - true), "
        "worst to best SBP MAE.",
        "",
        "| Model | Target | Windows | Records | ME | SD | MAE (95% CI) "
        "| ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | BHS | AAMI |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|:-:|:-:|",
    ]
    for name in order:
        for t in TARGETS:
            s = models[name][t]
            lo, hi = s["mae_ci95"]
            lines.append(
                f"| {name} | {t.upper()} | {s['n_windows']:,} | {s['n_records']:,} "
                f"| {s['me']:+.1f} | {s['sd']:.1f} | {s['mae']:.1f} ({lo:.1f}–{hi:.1f}) "
                f"| {s['pct_within_5']:.0f}% | {s['pct_within_10']:.0f}% "
                f"| {s['pct_within_15']:.0f}% | {s['bhs_grade']} "
                f"| {'pass' if s['aami_pass'] else 'fail'} |"
            )
    lines += [
        "",
        "95% CI: 1,000 bootstrap resamples of records. AAMI here means |ME| ≤ 5 and SD ≤ 8 "
        "mmHg only, not a full AAMI validation. Generated by `python -m bp.evaluate` from "
        "results/metrics.json; do not edit by hand.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- figures


def _style(ax: Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)


def _dots(ax: Axes, x: np.ndarray, y: np.ndarray) -> None:
    ax.scatter(x, y, s=5, alpha=0.3, color=SERIES, linewidths=0, rasterized=True)


def draw_scatter(ax: Axes, true: np.ndarray, pred: np.ndarray, label: str) -> None:
    """Predicted vs true. Perfect predictions would sit on the grey diagonal."""
    lo, hi = min(true.min(), pred.min()) - 5, max(true.max(), pred.max()) + 5
    _dots(ax, true, pred)
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1)
    ax.annotate("predicted = true", xy=(hi, hi), ha="right", va="bottom", fontsize=8, color=INK)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
    mae = metrics.mae(pred - true)
    _style(ax, f"True {label} (mmHg)", f"Predicted {label} (mmHg)", f"{label}: MAE {mae:.1f} mmHg")


def draw_bland_altman(ax: Axes, true: np.ndarray, pred: np.ndarray, label: str) -> None:
    """Error against the average of predicted and true. The middle line is the bias
    (ME); about 95% of errors fall between the outer lines (ME +/- 1.96 SD)."""
    err = pred - true
    me, sd = metrics.mean_error(err), metrics.sd_error(err)
    lo, hi = metrics.limits_of_agreement(me, sd)
    _dots(ax, (pred + true) / 2, err)
    for y, text, color in (
        (hi, f"+1.96 SD {hi:+.1f}", INK_2),
        (me, f"mean {me:+.1f}", INK),
        (lo, f"-1.96 SD {lo:+.1f}", INK_2),
    ):
        ax.axhline(y, color=color, linewidth=1)
        ax.annotate(
            text,
            xy=(1, y),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
        )
    _style(
        ax,
        f"Mean of predicted and true {label} (mmHg)",
        "Predicted - true (mmHg)",
        f"{label}: ME {me:+.1f}, SD {sd:.1f} mmHg",
    )


def draw_error_hist(ax: Axes, true: np.ndarray, pred: np.ndarray, label: str) -> None:
    """How often each error size occurs. Grey lines mark the BHS limits of 5/10/15 mmHg."""
    err = pred - true
    edge = max(20.0, 5 * np.ceil(np.percentile(np.abs(err), 99) / 5))
    bins = np.arange(-edge, edge + 2, 2)
    counts, _, _ = ax.hist(
        np.clip(err, -edge, edge), bins=bins, color=SERIES, edgecolor="white", linewidth=1
    )
    ax.set_ylim(0, counts.max() * 1.15)  # headroom so the limit labels clear the bars
    for limit in metrics.BHS_LIMITS_MMHG:
        for x in (-limit, limit):
            ax.axvline(x, color=LIMIT_LINE, linewidth=0.8, zorder=0)  # behind the bars
        ax.annotate(
            f"±{limit}",
            xy=(limit, 1),
            xycoords=("data", "axes fraction"),
            xytext=(2, -2),
            textcoords="offset points",
            va="top",
            fontsize=7,
            color=INK_2,
        )
    p5, p10, p15 = (metrics.pct_within(err, lim) for lim in metrics.BHS_LIMITS_MMHG)
    _style(
        ax,
        f"Predicted - true {label} (mmHg); errors beyond ±{edge:.0f} piled at the edges",
        "Windows",
        f"{label}: {p5:.0f}% within 5, {p10:.0f}% within 10, {p15:.0f}% within 15 mmHg",
    )
    ax.grid(axis="x", visible=False)  # the limit lines are the only vertical lines


FIGURES = {
    "scatter": ("predicted vs true", draw_scatter),
    "bland_altman": ("Bland-Altman", draw_bland_altman),
    "error_hist": ("error histogram", draw_error_hist),
}


def save_figures(table: pd.DataFrame, out_dir: Path, split_name: str) -> list[Path]:
    """Three PNGs per model, each with an SBP panel and a DBP panel."""
    written = []
    for name, rows in table.groupby("model", sort=False):
        for kind, (title, draw) in FIGURES.items():
            fig = Figure(figsize=(10, 4.2), layout="constrained")
            for ax, t in zip(fig.subplots(1, 2), TARGETS, strict=True):
                draw(ax, rows[f"{t}_true"].to_numpy(), rows[f"{t}_pred"].to_numpy(), t.upper())
            fig.suptitle(
                f"{name}: {title}, {SPLIT_LABELS[split_name]} split ({len(rows):,} windows)",
                x=0.01,
                ha="left",
                color=INK,
                fontsize=11,
            )
            path = out_dir / f"{name}_{kind}.png"
            with atomic_write(path) as f:
                fig.savefig(f, format="png", dpi=150, bbox_inches="tight")
            written.append(path)
    return written


# ---------------------------------------------------------------- demo sample


def pick_sample_rows(
    labels: pd.DataFrame,
    n: int = SAMPLE_SIZE,
    per_record: int = SAMPLE_PER_RECORD,
    seed: int = SAMPLE_SEED,
) -> np.ndarray:
    """Positions in `labels` of up to n windows, at most `per_record` from each record,
    picked at random with a fixed seed so the demo shows many different patients."""
    rng = np.random.default_rng(seed)
    picked = []
    for idx in labels.groupby("record_id").indices.values():
        picked.extend(rng.choice(idx, size=min(per_record, len(idx)), replace=False))
    picked = np.asarray(picked, dtype=int)
    if len(picked) > n:
        picked = rng.choice(picked, size=n, replace=False)
    return np.sort(picked)


def write_demo_sample(
    labels: pd.DataFrame, pred: np.ndarray, entry: dict, npz_path: Path, out_path: Path
) -> Path:
    """The demo's data file, data contract section 7. `entry` is the main model's
    metrics.json entry, so the demo can show the error band and AAMI/BHS result."""
    rows = pick_sample_rows(labels)
    chosen = labels.iloc[rows]
    with np.load(require(npz_path, "make windows")) as npz:
        ppg = npz["ppg"][chosen["npz_row"].to_numpy()].astype(np.float32)
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        ppg=ppg,
        record_id=chosen["record_id"].to_numpy(dtype=str),
        sbp_true=chosen["sbp_true"].to_numpy(dtype=float),
        dbp_true=chosen["dbp_true"].to_numpy(dtype=float),
        sbp_pred=pred[rows, 0],
        dbp_pred=pred[rows, 1],
        metrics_json=np.array(json.dumps(entry)),
    )
    if buf.tell() > MAX_SAMPLE_BYTES:
        raise ValueError(f"demo sample is {buf.tell() / 1e6:.1f} MB, over the 10 MB limit")
    with atomic_write(out_path) as f:
        f.write(buf.getvalue())
    return out_path


# ---------------------------------------------------------------- main


# ---------------------------------------------------------------- error analysis (Phase 3)


def bin_labels(values: np.ndarray, edges: list[float]) -> tuple[np.ndarray, list[str]]:
    """Label each value with its bin ("< a", "a–b", ..., "≥ z"); NaN -> "missing".
    Returns (one label per value, every label in order)."""
    names = [f"< {edges[0]:g}"]
    names += [f"{lo:g}–{hi:g}" for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    names += [f"≥ {edges[-1]:g}"]
    values = np.asarray(values, dtype=float)
    index = np.searchsorted(edges, values, side="right")  # a value on an edge goes up
    labels = np.array(names, dtype=object)[np.minimum(index, len(names) - 1)]
    labels[np.isnan(values)] = "missing"
    return labels, [*names, "missing"]


def permutation_importance(model, features: pd.DataFrame, true: np.ndarray) -> list[dict]:
    """How much the MAE rises when one feature's values are shuffled across windows,
    averaged over IMPORTANCE_REPEATS shuffles. Sorted by the total rise, biggest first."""
    rng = np.random.default_rng(IMPORTANCE_SEED)
    base = np.abs(model.predict(features) - true).mean(axis=0)
    rows = []
    for col in features.columns:
        rises = []
        for _ in range(IMPORTANCE_REPEATS):
            shuffled = features.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            rises.append(np.abs(model.predict(shuffled) - true).mean(axis=0) - base)
        sbp, dbp = np.mean(rises, axis=0)
        rows.append({"feature": col, "sbp_mae_rise": float(sbp), "dbp_mae_rise": float(dbp)})
    return sorted(rows, key=lambda r: -(r["sbp_mae_rise"] + r["dbp_mae_rise"]))


def error_analysis(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    pred: np.ndarray,
    model,
    split_name: str,
    model_name: str = MAIN_MODEL,
) -> dict:
    """Where one model's errors are largest (Phase 3). Rows of labels, features and pred
    line up, sorted by window_id, so each record's windows are in time order."""
    record_ids = labels["record_id"].to_numpy()
    true = labels[["sbp_true", "dbp_true"]].to_numpy(dtype=float)
    errs = {t: pred[:, i] - true[:, i] for i, t in enumerate(TARGETS)}

    slices = []
    for heading, col, edges in ERROR_SLICES:
        source = labels if col in ("sbp_true", "dbp_true") else features
        if col not in source.columns:
            continue  # e.g. a feature the current features.csv doesn't have
        groups, order = bin_labels(source[col].to_numpy(), edges)
        rows = metrics.by_group(errs, record_ids, groups, order)
        slices.append({"heading": heading, "column": col, "rows": rows})
    if "pat_s" in features.columns:
        groups = np.where(features["pat_s"].isna(), PAT_MISSING, PAT_PRESENT)
        rows = metrics.by_group(errs, record_ids, groups, [PAT_PRESENT, PAT_MISSING])
        slices.append({"heading": "Pulse arrival time (PAT)", "column": "pat_s", "rows": rows})

    offset = {
        t: {
            "offset_share": metrics.record_offset_share(e, record_ids),
            "worst_records_share": metrics.worst_records_share(e, record_ids, WORST_FRACTION),
        }
        for t, e in errs.items()
    }
    calibration = {}
    for i, t in enumerate(TARGETS):
        after, before, cal_ids = metrics.one_point_calibration(pred[:, i], true[:, i], record_ids)
        if after.size < 2:
            calibration[t] = None  # too few windows after each record's first to score
            continue
        calibration[t] = {
            "n_windows": int(after.size),
            "n_records": int(np.unique(cal_ids).size),
            "mae_before": metrics.mae(before),
            "mae_after": metrics.mae(after),
            "sd_before": metrics.sd_error(before),
            "sd_after": metrics.sd_error(after),
        }
    return {
        "split": split_name,
        "model": model_name,
        "n_windows": int(record_ids.size),
        "n_records": int(np.unique(record_ids).size),
        "slices": slices,
        "record_offset": offset,
        "one_point_calibration": calibration,
        "permutation_importance": permutation_importance(model, features, true),
    }


def error_analysis_markdown(a: dict) -> str:
    split = SPLIT_LABELS[a["split"]]
    lines = [
        f"# Error analysis: main model ({a['model']}), {split} split",
        "",
        f"Where the main model's errors are largest, on {a['n_windows']:,} {split} windows "
        f"from {a['n_records']:,} records. Error = estimated − true, in mmHg. MAE is the "
        "typical size of the error; ME is the bias (positive = reads too high). Groups use "
        "fixed cut points, so the validation and test splits are cut the same way; "
        '"100–120" means 100 ≤ value < 120.',
        "",
    ]
    for s in a["slices"]:
        lines += [
            f"## {s['heading']}",
            "",
            "| Group | Windows | Records | SBP MAE | SBP ME | DBP MAE | DBP ME |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        lines += [
            f"| {r['group']} | {r['n_windows']:,} | {r['n_records']:,} | {r['sbp_mae']:.1f} "
            f"| {r['sbp_me']:+.1f} | {r['dbp_mae']:.1f} | {r['dbp_me']:+.1f} |"
            for r in s["rows"]
        ]
        lines.append("")

    o = a["record_offset"]
    pct = f"{WORST_FRACTION:.0%}"
    lines += [
        "## How much of the error is a fixed offset per record",
        "",
        "| | SBP | DBP |",
        "|---|---:|---:|",
        "| Share of the squared error that is each record's constant offset "
        f"| {o['sbp']['offset_share']:.0%} | {o['dbp']['offset_share']:.0%} |",
        f"| Share of the squared error in the {pct} of records with the largest offset "
        f"| {o['sbp']['worst_records_share']:.0%} | {o['dbp']['worst_records_share']:.0%} |",
        "",
        "A high offset share means the model is off by about the same amount for every "
        "window of a record: it misjudges that patient's baseline, rather than being noisy "
        "from window to window. Records have no patient IDs, so a record stands in for a "
        "patient.",
        "",
    ]

    cal = a["one_point_calibration"]
    if all(cal.values()):
        lines += [
            "## What-if: one calibration reading per record",
            "",
            "Cuffless devices are often calibrated once against a cuff. To estimate what "
            "that would do, the first window of each record plays the calibration reading: "
            "its error is subtracted from every later prediction of that record, and only "
            "the later windows are scored. This is a different use case (it needs a "
            "reference reading per patient), not the headline result.",
            "",
            "| | Windows | Records | MAE before | MAE after | SD before | SD after |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        lines += [
            f"| {t.upper()} | {c['n_windows']:,} | {c['n_records']:,} | {c['mae_before']:.1f} "
            f"| {c['mae_after']:.1f} | {c['sd_before']:.1f} | {c['sd_after']:.1f} |"
            for t, c in cal.items()
        ]
        lines.append("")

    lines += [
        "## Which features the model relies on",
        "",
        f"Permutation importance: how much the {split} MAE rises when one feature's values "
        f"are shuffled across windows ({IMPORTANCE_REPEATS} repeats, fixed seed). A bigger "
        "rise means the model leans on that feature more. A feature can rank low because "
        "another feature carries the same information.",
        "",
        "| Feature | SBP MAE rise | DBP MAE rise |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {r['feature']} | {r['sbp_mae_rise']:+.2f} | {r['dbp_mae_rise']:+.2f} |"
        for r in a["permutation_importance"]
    ]
    lines += ["", "Generated by `python -m bp.evaluate`; do not edit by hand."]
    return "\n".join(lines) + "\n"


def run(paths: Paths, final: bool, main_model: str = MAIN_MODEL) -> dict:
    """Score every model on one split and write every output. Returns metrics.json content."""
    split = json.loads(require(paths.split_file, "make windows").read_text())
    split_name, records = split_records(split, final)
    labels, features = load_split_windows(paths, records)
    models = load_models(paths.models_dir)
    preds = predict(models, features)
    table = predictions_table(labels.drop(columns="npz_row"), preds)
    result = compute_metrics(table, split_name)

    out = paths.results_dir
    written = []
    with atomic_write(out / f"predictions_{split_name}.csv", "w") as f:
        table.to_csv(f, index=False, float_format="%.3f")
    written.append(out / f"predictions_{split_name}.csv")
    with atomic_write(out / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    written.append(out / "metrics.json")
    comparison = comparison_table(result)
    with atomic_write(out / f"comparison_{split_name}.md", "w") as f:
        f.write(comparison)
    written.append(out / f"comparison_{split_name}.md")
    written += save_figures(table, out / "figures" / split_name, split_name)

    if main_model in preds:
        sample = paths.app_dir / ("sample.npz" if final else "sample_val.npz")
        entry = result["models"][main_model]
        written.append(
            write_demo_sample(labels, preds[main_model], entry, paths.windows_npz, sample)
        )
        analysis = error_analysis(
            labels, features, preds[main_model], models[main_model], split_name, main_model
        )
        with atomic_write(out / f"error_analysis_{split_name}.json", "w") as f:
            json.dump(rounded(analysis), f, indent=2)
            f.write("\n")
        with atomic_write(out / f"error_analysis_{split_name}.md", "w") as f:
            f.write(error_analysis_markdown(analysis))
        written += [out / f"error_analysis_{split_name}.{ext}" for ext in ("json", "md")]
    else:
        print(f"note: no {main_model!r} model in {paths.models_dir} yet, so no demo sample")

    print(comparison)
    print("wrote:\n" + "\n".join(f"  {p}" for p in written))
    return result


def main(argv: list[str] | None = None, paths: Paths | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score every saved model on one split")
    parser.add_argument(
        "--final", action="store_true", help="score the TEST split (Phase 4 only, once)"
    )
    parser.add_argument(
        "--main-model", default=MAIN_MODEL, help="model whose windows go into the demo sample"
    )
    args = parser.parse_args(argv)
    if args.final:
        print(FINAL_BANNER)
    try:
        run(paths or Paths(), args.final, args.main_model)
    except MissingInput as e:
        print(f"bp.evaluate: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
