"""Run the three tracks in three Claude Code terminals without stepping on each other.

Plain-language summary:
- Every file has one owner: a track, or "shared" (nobody edits it on a track branch).
  The table below is the single source of truth.
- `setup` makes one git worktree per track (its own folder and branch), points it at
  the shared data folder and venv, and writes Claude Code deny rules so the Claude in
  that terminal can't edit files its track doesn't own.
- `check` looks at everything the current branch changed since main (committed or not)
  and fails if any file belongs to another track or is shared. Deny rules only stop
  Claude's edit tools; `check` also catches edits made any other way.

Usage (from any checkout of the repo):
    python tools/tracks.py setup [--from BRANCH]     # make tracks
    python tools/tracks.py check [--track NAME]      # make check-owner
"""

import argparse
import fnmatch
import json
import subprocess
import sys
from pathlib import Path

# Track -> branch for its first task, demo port, owned path patterns.
# Exact paths win over globs, so results/window_filters.json belongs to data
# even though eval owns results/**. Anything not listed is shared.
TRACKS = {
    "data": {
        "branch": "data/phase1-windows-split",
        "port": 8503,
        "owns": [
            "src/bp/download.py",
            "src/bp/load.py",
            "src/bp/windows.py",
            "src/bp/split.py",
            "src/bp/baselines.py",
            "splits/**",
            "results/window_filters.json",
            "tests/test_download*.py",
            "tests/test_load*.py",
            "tests/test_windows*.py",
            "tests/test_split*.py",
            "tests/test_baselines*.py",
        ],
    },
    "model": {
        "branch": "model/phase2-features",
        "port": 8502,
        "owns": [
            "src/bp/features.py",
            "src/bp/train.py",
            "tests/test_features*.py",
            "tests/test_train*.py",
        ],
    },
    "eval": {  # with 3 terminals the Demo track folds into Evaluation (CLAUDE.md)
        "branch": "eval/metrics-evaluate",
        "port": 8501,
        "owns": [
            "src/bp/metrics.py",
            "src/bp/evaluate.py",
            "results/**",
            "app/**",
            "README.md",
            "tests/test_metrics*.py",
            "tests/test_evaluate*.py",
            "tests/test_demo*.py",
        ],
    },
}

# Shared files Claude must not edit from a track terminal (CLAUDE.md "Shared files"
# plus the setup itself). Used for deny rules; `check` treats every unowned file as shared.
SHARED_DENY = [
    "requirements.txt",
    "Makefile",
    "pyproject.toml",
    ".gitignore",
    ".env",
    ".env.example",
    "CLAUDE.md",
    "docs/**",
    "tools/**",
    ".claude/**",
    "src/bp/__init__.py",
    "src/bp/paths.py",
    "src/bp/fileio.py",
    "tests/test_setup.py",
    "tests/test_tracks.py",
]

# Shell commands blocked in track terminals: the venv is shared, and Claude never merges.
BASH_DENY = ["pip install:*", "uv pip install:*", "uv add:*", "gh pr merge:*"]

# Branches a track branch may have started from. The closest one is the base for `check`.
BASE_CANDIDATES = ["origin/main", "main", "model/phase0-setup"]


def is_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[")


def matches(path: str, pattern: str) -> bool:
    if pattern.endswith("/**"):
        return path.startswith(pattern[:-2])
    return fnmatch.fnmatchcase(path, pattern)


def owner_of(path: str) -> str:
    """Track that owns a repo-relative path, or 'shared'."""
    for track, spec in TRACKS.items():  # exact paths first
        if path in (p for p in spec["owns"] if not is_glob(p)):
            return track
    for track, spec in TRACKS.items():
        if any(is_glob(p) and matches(path, p) for p in spec["owns"]):
            return track
    return "shared"


def violations(track: str, changed: list[str]) -> list[tuple[str, str]]:
    """(path, real owner) for every changed path the track doesn't own."""
    return [(p, o) for p in changed if (o := owner_of(p)) != track]


def deny_patterns(track: str) -> list[str]:
    """Repo-relative patterns the track's Claude must not edit. A glob from another track
    that would also cover one of this track's own files (eval's results/** vs data's
    results/window_filters.json) is left out; `check` still catches those edits."""
    own = TRACKS[track]["owns"]
    patterns = list(SHARED_DENY)
    for other, spec in TRACKS.items():
        if other == track:
            continue
        for pattern in spec["owns"]:
            if is_glob(pattern) and any(matches(mine, pattern) for mine in own):
                continue
            patterns.append(pattern)
    return patterns


def claude_settings(track: str, worktree: Path) -> dict:
    root = worktree.resolve().as_posix()  # "//" prefix = absolute path in Claude Code rules
    deny = [f"Edit(/{root}/{p})" for p in deny_patterns(track)]
    deny += [f"Bash({c})" for c in BASH_DENY]
    return {"permissions": {"deny": deny}}


def git(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def main_checkout() -> Path:
    """The main checkout (not a worktree), wherever this is run from."""
    common = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir"))
    return common.parent


def setup(from_branch: str | None) -> int:
    main = main_checkout()
    start = from_branch or git("branch", "--show-current", cwd=main)
    print(f"main checkout: {main}\nnew track branches start from: {start}\n")
    for track, spec in TRACKS.items():
        wt = main / ".claude" / "worktrees" / track
        if wt.exists():
            print(f"[{track}] worktree exists: {wt}")
        else:
            exists = git("rev-parse", "--verify", "--quiet", spec["branch"], cwd=main, check=False)
            if exists:
                git("worktree", "add", str(wt), spec["branch"], cwd=main)
            else:
                git("worktree", "add", "-b", spec["branch"], str(wt), start, cwd=main)
            print(f"[{track}] created {wt} on {spec['branch']}")
        (wt / ".env").write_text(
            f"BP_DATA_DIR={main / 'data'}\nPYTHON={main / '.venv' / 'bin' / 'python'}\n"
            f"DEMO_PORT={spec['port']}\n"
        )
        settings = claude_settings(track, wt)
        path = wt / ".claude" / "settings.local.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(settings, indent=2) + "\n")
        print(f"[{track}] wrote .env and {len(settings['permissions']['deny'])} deny rules")
    print("\nOpen one terminal per track:")
    for track in TRACKS:
        print(f"  cd {main / '.claude' / 'worktrees' / track} && claude")
    return 0


def changed_files() -> list[str]:
    """Files changed on this branch since its base, including uncommitted and untracked."""
    bases = []
    for cand in BASE_CANDIDATES:
        if git("rev-parse", "--verify", "--quiet", cand, check=False):
            mb = git("merge-base", "HEAD", cand, check=False)
            if mb:
                bases.append((int(git("rev-list", "--count", mb)), mb, cand))
    if not bases:
        raise SystemExit("no base branch found (tried: " + ", ".join(BASE_CANDIDATES) + ")")
    _, base, name = max(bases)
    print(f"comparing against {name} ({base[:7]})")
    tracked = git("diff", "--name-only", base).splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    return sorted(set(tracked) | set(untracked))


def check(track: str | None) -> int:
    branch = git("branch", "--show-current")
    track = track or branch.split("/")[0]
    if track not in TRACKS:
        print(f"branch {branch!r} is not a track branch ({', '.join(TRACKS)}); nothing to check")
        return 0
    files = changed_files()
    bad = violations(track, files)
    if not bad:
        print(f"ok: all {len(files)} changed file(s) belong to the {track} track")
        return 0
    print(f"FAIL: the {track} track changed files it doesn't own:")
    for path, owner in bad:
        print(f"  {path}  (owner: {owner})")
    print("Revert them, or ask the owner / lead to make the change on their branch.")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_setup = sub.add_parser("setup", help="create the three track worktrees")
    p_setup.add_argument("--from", dest="from_branch", help="branch to start from")
    p_check = sub.add_parser("check", help="fail if this branch touches files it doesn't own")
    p_check.add_argument("--track", choices=list(TRACKS))
    args = parser.parse_args(argv)
    return setup(args.from_branch) if args.cmd == "setup" else check(args.track)


if __name__ == "__main__":
    sys.exit(main())
