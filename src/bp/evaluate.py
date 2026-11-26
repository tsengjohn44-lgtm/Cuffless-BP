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
    windows = windows[windows["record_id"].isin(set(records))]
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


def run(paths: Paths, final: bool, main_model: str = MAIN_MODEL) -> dict:
    """Score every model on one split and write every output. Returns metrics.json content."""
    split = json.loads(require(paths.split_file, "make windows").read_text())
    split_name, records = split_records(split, final)
    labels, features = load_split_windows(paths, records)
    preds = predict(load_models(paths.models_dir), features)
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
