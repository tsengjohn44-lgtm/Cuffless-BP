"""The demo runs headlessly (Streamlit AppTest) from a made-up sample, and fails gracefully.

Each test copies app/demo.py into a temporary app/ folder, because the demo looks for its
sample next to itself."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

DEMO = Path(__file__).resolve().parents[1] / "app" / "demo.py"
K = 5  # windows in the made-up sample
ENTRY = {  # a metrics.json entry (data contract, section 6)
    t: {
        "n_windows": 500,
        "n_records": 40,
        "me": 1.2,
        "sd": 9.8,
        "mae": 7.6,
        "mae_ci95": [6.9, 8.4],
        "pct_within_5": 45.0,
        "pct_within_10": 72.0,
        "pct_within_15": 86.0,
        "bhs_grade": "C",
        "aami_pass": False,
    }
    for t in ("sbp", "dbp")
}


def write_sample(app_dir: Path, name: str = "sample.npz", drop: str | None = None) -> None:
    t = np.arange(1250) / 125
    data = {
        "ppg": np.sin(2 * np.pi * 1.2 * t)[None].repeat(K, axis=0).astype(np.float32),
        "record_id": np.array([f"1_{i:04d}" for i in range(K)]),
        "sbp_true": np.full(K, 120.0),
        "dbp_true": np.full(K, 70.0),
        "sbp_pred": 121.0 + np.arange(K) * 5,  # 121, 126, 131, ... so windows differ
        "dbp_pred": np.full(K, 73.0),
        "metrics_json": np.array(json.dumps(ENTRY)),
    }
    data.pop(drop, None)
    np.savez(app_dir / name, **data)


@pytest.fixture
def app_dir(tmp_path):
    d = tmp_path / "app"
    d.mkdir()
    shutil.copy(DEMO, d / "demo.py")
    return d


def run(app_dir: Path) -> AppTest:
    at = AppTest.from_file(str(app_dir / "demo.py"), default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def texts(elements) -> str:
    return "\n".join(str(e.value) for e in elements)


def test_demo_shows_window_estimate_and_accuracy(app_dir):
    write_sample(app_dir)
    at = run(app_dir)
    assert "not a medical device" in texts(at.warning)
    assert not at.info  # no "validation preview" banner for the real sample
    sbp, dbp = at.metric
    assert (sbp.value, dbp.value) == ("121 mmHg", "73 mmHg")
    assert sbp.delta == "+1.0 mmHg vs true"
    assert "Typical error ± 7.6 mmHg" in texts(at.caption)
    table = at.table[0].value
    assert list(table.index) == ["SBP", "DBP"]
    assert table.loc["SBP", "BHS grade"] == "C"
    assert table.loc["SBP", "AAMI"] == "fail"
    assert table.loc["SBP", "95% of errors between"] == "-18.0 and +20.4"  # 1.2 -/+ 1.96 * 9.8


def test_picking_another_window_updates_the_estimate(app_dir):
    write_sample(app_dir)
    at = run(app_dir)
    at.selectbox[0].set_value(2).run()
    assert at.metric[0].value == "131 mmHg"  # 121 + 2 * 5
    at.button[0].click().run()  # "Random window" must not crash
    assert not at.exception


def test_validation_preview_is_labelled(app_dir):
    write_sample(app_dir, name="sample_val.npz")
    at = run(app_dir)
    assert "Validation preview" in texts(at.info)
    assert "validation set" in texts(at.caption)


def test_missing_sample_shows_a_message(app_dir):
    at = run(app_dir)
    assert "make final" in texts(at.error)


def test_corrupt_sample_shows_a_message(app_dir):
    (app_dir / "sample.npz").write_bytes(b"not a numpy file")
    at = run(app_dir)
    assert "Could not read sample.npz" in texts(at.error)


def test_sample_missing_a_key_says_which(app_dir):
    write_sample(app_dir, drop="metrics_json")
    at = run(app_dir)
    assert "lacks metrics_json" in texts(at.error)
