"""File ownership rules for the three track terminals (tools/tracks.py)."""

import pytest
from tracks import TRACKS, claude_settings, deny_patterns, owner_of, violations


@pytest.mark.parametrize(
    "path, owner",
    [
        ("src/bp/windows.py", "data"),
        ("splits/split.json", "data"),
        ("results/window_filters.json", "data"),  # exact path beats eval's results/**
        ("tests/test_windows_filters.py", "data"),
        ("src/bp/features.py", "model"),
        ("tests/test_features.py", "model"),
        ("results/metrics.json", "eval"),
        ("results/figures/bland_altman.png", "eval"),
        ("app/demo.py", "eval"),
        ("README.md", "eval"),
        ("Makefile", "shared"),
        ("docs/data-contract.md", "shared"),
        ("src/bp/paths.py", "shared"),
        ("src/bp/new_module.py", "shared"),  # unlisted -> shared, needs an OK
        ("tests/test_setup.py", "shared"),
    ],
)
def test_owner_of(path, owner):
    assert owner_of(path) == owner


def test_owned_files_pass_and_foreign_files_fail():
    changed = ["src/bp/windows.py", "tests/test_windows.py", "src/bp/features.py", "Makefile"]
    assert violations("data", changed) == [
        ("src/bp/features.py", "model"),
        ("Makefile", "shared"),
    ]


def test_tracks_never_share_a_file():
    exact = [p for spec in TRACKS.values() for p in spec["owns"] if "*" not in p]
    assert len(exact) == len(set(exact))


def test_deny_rules_block_other_tracks_and_shared_but_not_own_files():
    data = deny_patterns("data")
    assert "src/bp/features.py" in data and "Makefile" in data and "app/**" in data
    assert "src/bp/windows.py" not in data
    # eval's results/** would also block data's own results/window_filters.json
    assert "results/**" not in data
    assert "results/window_filters.json" in deny_patterns("eval")


def test_claude_settings_use_absolute_paths(tmp_path):
    rules = claude_settings("model", tmp_path)["permissions"]["deny"]
    root = tmp_path.resolve().as_posix()  # macOS: /tmp resolves to /private/tmp
    assert f"Edit(/{root}/src/bp/windows.py)" in rules
    assert all(r.startswith("Edit(//") or r.startswith("Bash(") for r in rules)
    assert "Bash(gh pr merge:*)" in rules
