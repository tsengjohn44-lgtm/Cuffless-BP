"""Phase 0 checks: data folder override and zip member lookup."""

import importlib
import io
import zipfile

import pytest

import bp.paths
from bp.download import find_member


@pytest.fixture(autouse=True)
def restore_paths(monkeypatch):
    """Put bp.paths back to normal after each test that reloads it."""
    yield
    monkeypatch.undo()
    importlib.reload(bp.paths)


def reload_paths(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("BP_DATA_DIR", raising=False)
    else:
        monkeypatch.setenv("BP_DATA_DIR", value)
    return importlib.reload(bp.paths)


def test_data_dir_defaults_to_repo_data(monkeypatch):
    paths = reload_paths(monkeypatch, None)
    assert paths.DATA_DIR == paths.REPO_ROOT / "data"
    assert paths.RAW_DIR == paths.DATA_DIR / "raw"


def test_empty_data_dir_means_default(monkeypatch):
    paths = reload_paths(monkeypatch, "")
    assert paths.DATA_DIR == paths.REPO_ROOT / "data"


def test_relative_data_dir_is_relative_to_repo(monkeypatch):
    paths = reload_paths(monkeypatch, "elsewhere")
    assert paths.DATA_DIR == paths.REPO_ROOT / "elsewhere"


def test_absolute_data_dir_is_used_as_is(monkeypatch, tmp_path):
    paths = reload_paths(monkeypatch, str(tmp_path))
    assert paths.DATA_DIR == tmp_path


def make_zip(names):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in names:
            zf.writestr(name, b"x")
    buf.seek(0)
    return zipfile.ZipFile(buf)


def test_find_member_in_subfolder_any_case():
    zf = make_zip(["readme.txt", "dataset/part_1.mat", "dataset/Part_2.mat"])
    assert find_member(zf, 1).filename == "dataset/part_1.mat"
    assert find_member(zf, 2).filename == "dataset/Part_2.mat"


def test_find_member_missing_part_says_what_is_there():
    zf = make_zip(["Part_1.mat"])
    with pytest.raises(FileNotFoundError, match="Part_1.mat"):
        find_member(zf, 3)
