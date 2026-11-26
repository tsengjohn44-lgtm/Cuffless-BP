"""Streamlit demo: pick a held-out window, see its fingertip pulse signal (PPG) and the
model's SBP/DBP estimate next to the true values from the arterial line, with the
model's typical error and its AAMI/BHS result.

Owner: Eval track (with 3 terminals the Demo track folds into Evaluation).
Reads app/sample.npz (data contract, section 7), written once by `make final`. Until
then it falls back to app/sample_val.npz from `make eval`, labelled as a validation
preview. It needs neither the dataset nor wifi, and if anything fails to load it shows
a message on screen instead of a stack trace.

Run: make demo
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
FS_HZ = 125  # samples per second in the dataset
# The first file that exists wins: the real test sample, else the validation preview.
SAMPLES = [(APP_DIR / "sample.npz", "test"), (APP_DIR / "sample_val.npz", "validation")]
ARRAY_KEYS = ["ppg", "record_id", "sbp_true", "dbp_true", "sbp_pred", "dbp_pred"]
TARGETS = {"sbp": "Systolic (SBP)", "dbp": "Diastolic (DBP)"}
METRIC_KEYS = [  # one metrics.json entry (data contract, section 6)
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
DISCLAIMER = "Research demo, not a medical device. Do not use it for any health decision."


class SampleError(Exception):
    """The demo sample is there but not in the expected format."""


@st.cache_data
def load_sample(path: str, modified: float) -> dict:
    """Read and check the sample. `modified` (the file's mtime) makes a rewritten file reload."""
    with np.load(path, allow_pickle=False) as npz:
        missing = [k for k in [*ARRAY_KEYS, "metrics_json"] if k not in npz.files]
        if missing:
            raise SampleError(f"it lacks {', '.join(missing)}")
        data = {k: npz[k] for k in ARRAY_KEYS}
        data["metrics"] = json.loads(str(npz["metrics_json"]))
    n = len(data["ppg"])
    if data["ppg"].ndim != 2 or n == 0:
        raise SampleError(f"ppg has shape {data['ppg'].shape}, expected (windows, samples)")
    if any(len(data[k]) != n for k in ARRAY_KEYS):
        raise SampleError("its arrays have different numbers of windows")
    for t in TARGETS:  # check now, so a bad entry never surfaces later as a stack trace
        missing = [k for k in METRIC_KEYS if k not in data["metrics"].get(t, {})]
        if missing:
            raise SampleError(f"its metrics for {t.upper()} lack {', '.join(missing)}")
    return data


def find_sample() -> tuple[Path, str] | None:
    for path, split in SAMPLES:
        if path.exists():
            return path, split
    return None


def pick_random_window(n: int) -> None:
    st.session_state.window = int(np.random.default_rng().integers(n))


st.set_page_config(page_title="Cuffless BP demo", layout="wide")
st.title("Blood pressure from a fingertip pulse signal")
st.warning(DISCLAIMER)

found = find_sample()
if found is None:
    st.error(
        "No demo data found. Expected app/sample.npz (made by `make final`) "
        "or app/sample_val.npz (made by `make eval`)."
    )
    st.stop()
path, split = found
try:
    data = load_sample(str(path), path.stat().st_mtime)
except Exception as e:  # any load failure goes on screen, never as a stack trace
    st.error(f"Could not read {path.name}: {e}")
    st.stop()
if split == "validation":
    st.info(
        "Validation preview: these windows come from the validation split. The final "
        "demo uses held-out test windows, scored once at the end of the project."
    )

metrics = data["metrics"]
n_windows = len(data["ppg"])

# ---- 1. pick a window
pick, rand = st.columns([5, 1], vertical_alignment="bottom")
window = pick.selectbox(
    f"Pick one of {n_windows} held-out 10-second windows",
    range(n_windows),
    format_func=lambda i: f"Window {i + 1}: record {data['record_id'][i]}",
    key="window",
)
rand.button("Random window", on_click=pick_random_window, args=(n_windows,))

# ---- 2. the signal the model sees
ppg = data["ppg"][window]
st.subheader("Fingertip pulse signal (PPG)")
st.line_chart(
    pd.DataFrame({"PPG": ppg}, index=np.arange(len(ppg)) / FS_HZ),
    x_label="Time (s)",
    y_label="PPG (arbitrary units)",
    height=240,
)

# ---- 3. the estimate vs the truth
st.subheader("Estimated vs true blood pressure")
for col, (t, name) in zip(st.columns(2), TARGETS.items(), strict=True):
    pred, true = float(data[f"{t}_pred"][window]), float(data[f"{t}_true"][window])
    with col:
        st.metric(
            f"{name}, estimated from PPG",
            f"{pred:.0f} mmHg",
            delta=f"{pred - true:+.1f} mmHg vs true",
            delta_color="off",
        )
        st.markdown(f"True value (invasive arterial line): **{true:.0f} mmHg**")
        st.caption(
            f"Typical error ± {metrics[t]['mae']:.1f} mmHg (mean absolute error on the {split} set)"
        )

# ---- 4. how good the model is overall
s0 = metrics["sbp"]
st.subheader("How accurate is the model overall?")
st.caption(
    f"{split.capitalize()} set: {s0['n_windows']:,} windows from {s0['n_records']:,} records. "
    "Error = estimated - true, in mmHg."
)
rows = []
for t in TARGETS:
    s = metrics[t]
    lo, hi = s["mae_ci95"]
    rows.append(
        {
            "": t.upper(),
            "Mean error (bias)": f"{s['me']:+.1f}",
            "SD of error": f"{s['sd']:.1f}",
            "95% of errors between": f"{s['me'] - 1.96 * s['sd']:+.1f} and "
            f"{s['me'] + 1.96 * s['sd']:+.1f}",
            "MAE (95% CI)": f"{s['mae']:.1f} ({lo:.1f}–{hi:.1f})",
            "Within 5 / 10 / 15 mmHg": f"{s['pct_within_5']:.0f}% / {s['pct_within_10']:.0f}% "
            f"/ {s['pct_within_15']:.0f}%",
            "BHS grade": s["bhs_grade"],
            "AAMI": "pass" if s["aami_pass"] else "fail",
        }
    )
st.table(pd.DataFrame(rows).set_index(""))
st.caption(
    "AAMI, the US standard for cuff monitors, asks for |mean error| ≤ 5 and SD ≤ 8 mmHg; "
    "a full AAMI validation also needs 85+ people measured against a cuff, which this "
    "dataset cannot provide. BHS grades A (best) to D by the share of errors within "
    "5, 10 and 15 mmHg. Cuffless methods usually miss AAMI."
)
st.caption(
    "Data: UCI Cuff-Less Blood Pressure Estimation (dataset 340), ICU recordings from "
    "MIMIC-II, CC BY 4.0."
)
