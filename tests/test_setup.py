"""Phase 0 checks: where the data folder resolves to."""

import importlib

import pytest

import bp.paths


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


def test_atomic_write_replaces_file_only_on_success(tmp_path):
    from bp.fileio import atomic_write

    target = tmp_path / "out.csv"
    target.write_text("old")
    with pytest.raises(RuntimeError):
        with atomic_write(target, "w") as f:
            f.write("half a fi")
            raise RuntimeError("crash mid-write")
    assert target.read_text() == "old"
    assert list(tmp_path.iterdir()) == [target]  # no temp file left behind

    with atomic_write(target, "w") as f:
        f.write("new")
    assert target.read_text() == "new"


def test_split_file_is_committed_not_in_data(monkeypatch, tmp_path):
    paths = reload_paths(monkeypatch, str(tmp_path))
    assert paths.SPLIT_FILE == paths.REPO_ROOT / "splits" / "split.json"
