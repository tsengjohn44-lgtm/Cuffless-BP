"""Record-level split: deterministic, disjoint, close to 70/15/15, and stable."""

import pytest

from bp.split import SPLITS, load_split, make_split, save_split

IDS = [f"1_{i:04d}" for i in range(3000)]


def test_same_seed_gives_same_split():
    assert make_split(IDS, seed=42) == make_split(list(reversed(IDS)), seed=42)


def test_different_seed_gives_different_split():
    assert make_split(IDS, seed=1)["test"] != make_split(IDS, seed=2)["test"]


def test_every_record_in_exactly_one_split():
    split = make_split(IDS)
    groups = [set(split[name]) for name in SPLITS]
    assert sum(len(g) for g in groups) == len(IDS)
    assert set().union(*groups) == set(IDS)
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_fractions_close_to_target():
    split = make_split(IDS)
    for name, target in zip(SPLITS, (0.70, 0.15, 0.15), strict=True):
        assert len(split[name]) / len(IDS) == pytest.approx(target, abs=0.03)


def test_removing_a_record_does_not_move_the_others():
    full = make_split(IDS)
    fewer = make_split(IDS[1:])  # e.g. a bug fix in windows.py drops record 1_0000
    for name in SPLITS:
        assert set(fewer[name]) == set(full[name]) - {"1_0000"}


def test_lists_are_sorted_and_schema_matches_contract():
    split = make_split(["1_0002", "1_0001", "1_0001"])
    assert list(split) == ["seed", "fractions", "train", "val", "test"]
    assert split["fractions"] == [0.70, 0.15, 0.15]
    for name in SPLITS:
        assert split[name] == sorted(split[name])


def test_load_split_hides_test_unless_final(tmp_path):
    path = tmp_path / "split.json"
    save_split(make_split(IDS), path)

    assert "test" not in load_split(path)
    assert load_split(path, final=True) == make_split(IDS)
