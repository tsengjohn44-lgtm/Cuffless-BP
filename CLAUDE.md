# Cuffless BP — Hackathon project

<!-- Hackathon.md template merged with the build plan (PLAN.md).
     Put this file in the repo root as CLAUDE.md so Claude Code reads it every session.
     Keep it short. When Claude makes the same mistake twice, add a rule under "Rules for Claude Code".
     Fill in the [brackets] for teammate names at kickoff. -->

## Project
- **What it does (one sentence):** Estimates systolic (SBP) and diastolic (DBP) blood pressure from a fingertip pulse signal (PPG) and reports the error in mmHg against the AAMI and BHS standards that cuff monitors must meet.
- **What judges will see in the demo:** pick a held-out test window → see the PPG waveform → see predicted vs. true SBP/DBP with the model's typical error band and the AAMI/BHS result.
- **Stack:** Python 3.11, numpy, scipy, h5py, pandas, scikit-learn, matplotlib, streamlit, pytest, ruff. Optional later: PyTorch for a small 1D CNN. Pin versions in `requirements.txt`.
- **Submission deadline:** Fri Dec 4, 11:45 PM ET, AI for Medicine student hackathon on Devpost.

## Contest context (read first)
- Judging, 25% each: technical depth; rigor and reproducibility; clinical relevance and honesty; demo quality. Ties go to the stronger held-out evaluation. Missing quantitative evaluation caps two categories at half.
- The team must be able to explain the method. **Explain every step in a few plain sentences as you go**, and keep code readable over clever.
- AI tool use must be disclosed: the README has one line saying Claude Code helped write the code.

## Commands
- Install: `pip install -r requirements.txt`
- Run locally: `make all` (downloads data, rebuilds every result from scratch); demo: `make demo`
- Single steps: `make data`, `make windows`, `make features`, `make baselines`, `make train`, `make eval`
- Test: `make test` (pytest)
- Lint/format: `ruff check . && ruff format .`

## Team & tracks
Each track has its own worktree and branch. Only edit files in your own track unless I ask.

| Track | Owner | Branch prefix | Owns these folders/files |
|-------|-------|---------------|--------------------------|
| Model (lead) | Jonathan | `model/` | `src/bp/features.py`, `src/bp/train.py` |
| Data + baselines | [name] | `data/` | `src/bp/download.py`, `load.py`, `windows.py`, `split.py`, `baselines.py` |
| Evaluation + write-up | [name] | `eval/` | `src/bp/metrics.py`, `evaluate.py`, `results/`, `README.md` |
| Demo + video | [name] | `demo/` | `app/` |

If the team is smaller, tracks merge from the bottom up. `tests/` belongs to whoever owns the code under test.

**Shared files.** Never change these without asking first: `requirements.txt`, `Makefile`, `data/split.json`, the data contract (`docs/data-contract.md`), and `.gitignore`.

## Data contract
The tracks build at the same time, so agree on the file formats first and write them in `docs/data-contract.md`: the columns and units of the windows table, the feature table, and the predictions file that `evaluate.py` and the demo read. Every track builds against that file. Don't change the contract from a track branch. Propose the change and wait for an OK instead.

## Data
- UCI "Cuff-Less Blood Pressure Estimation" (dataset 340), CC BY 4.0. Get the download link from the dataset page: https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation
- 3.1 GB, four files `Part_1.mat` to `Part_4.mat`, 12,000 records total, MATLAB v7.3 (use `h5py`, not `scipy.io.loadmat`).
- Each record holds three signals at 125 Hz: PPG (fingertip), ABP (invasive arterial pressure, the ground truth), ECG lead II. **Verify the exact layout (array shape, row order) by inspecting Part_1 before writing loaders.** Do not assume.
- Records come from ICU patients (MIMIC-II) with no patient IDs, so the same person may appear in several records.
- **Develop on Part_1 only** (about 0.8 GB). Use all four parts for the final run only if disk and time allow.
- Never commit raw data. `data/` goes in `.gitignore`.

## Repo layout
```
CLAUDE.md  README.md  requirements.txt  Makefile  .env.example
docs/data-contract.md
src/bp/
  download.py   # fetch + unzip into data/raw (prints size, skips if present)
  load.py       # read .mat records -> numpy arrays
  windows.py    # cut records into 10 s windows, quality filters, SBP/DBP labels
  split.py      # record-level train/val/test split, fixed seed, saved to data/split.json
  features.py   # PPG (and optional ECG) features per window
  baselines.py  # mean baseline, pulse-arrival-time + linear regression
  train.py      # gradient boosting models for SBP and DBP
  metrics.py    # all metrics below
  evaluate.py   # writes results/metrics.json and figures
app/demo.py     # streamlit demo
tests/          # unit tests for windows, metrics, split
results/        # metrics.json, figures (committed)
```

## Method
1. **Windows and labels.** Cut each record into non-overlapping 10 s windows (1,250 samples). SBP label = median of ABP beat peaks in the window; DBP = median of beat troughs (`scipy.signal.find_peaks` with a minimum distance set by a max heart rate of about 180 bpm). Drop windows with SBP outside 60 to 220 mmHg, DBP outside 30 to 130, SBP <= DBP, fewer than 5 beats, or flat/clipped PPG. Log how many windows each filter drops (this goes in the write-up).
2. **Split by record, not by window.** 70/15/15 train/val/test with a fixed seed. All windows of one record stay in one split. Save the split. **The test split is only touched by `evaluate.py --final`.**
3. **Features per window (PPG):** heart rate, pulse amplitude, systolic upstroke time, pulse width at 25/50/75% of amplitude, systolic/diastolic area ratio, and second-derivative (APG) ratios such as b/a and d/a, each as a median over beats. Optional ECG feature: pulse arrival time (ECG R peak to PPG max-slope point).
4. **Models.**
   - Baseline 1: predict the training mean SBP and DBP for every window.
   - Baseline 2: pulse arrival time + heart rate into linear regression.
   - Main: `HistGradientBoostingRegressor`, one for SBP and one for DBP, tuned on validation only.
   - Stretch (only if Phase 3 is done by Dec 1): small 1D CNN on raw PPG windows.

## Metrics (metrics.py), reported for SBP and DBP separately
- Mean error (ME) and standard deviation of error (SD), in mmHg. AAMI pass = |ME| <= 5 and SD <= 8.
- MAE in mmHg.
- BHS: % of windows with absolute error <= 5, <= 10, <= 15 mmHg, and the grade (A: 60/85/95, B: 50/75/90, C: 40/65/85, else D).
- 95% confidence interval for MAE by bootstrapping **records** (not windows), 1,000 resamples.
- Every model and baseline goes in one comparison table.
- Figures: predicted vs. true scatter, Bland-Altman plot, error histogram.
- Unit tests for metrics.py with hand-computed examples.

## Demo (app/demo.py)
- Demo mode by default: runs without the 3 GB dataset or wifi, from a small sample of test windows plus predictions saved to `app/sample.npz` (under 10 MB; allowed under CC BY 4.0 with credit).
- Pick a test window → show the PPG waveform, predicted SBP/DBP vs. the true ABP values, and the model's test-set error band (e.g. "typical error +/- X mmHg").
- Show the AAMI/BHS result on screen and a clear line: "Research demo, not a medical device."
- If anything fails to load, show a clear message on screen instead of a stack trace.

## Priorities (in order)
1. The demo flow works from start to finish, and `make all` reproduces every number.
2. It doesn't crash on stage. Show errors clearly; the demo runs from `app/sample.npz` with no download or wifi.
3. Polish comes last. Don't refactor working code close to the deadline.

## Phases and checkpoints (dates are targets)
| Phase | Target | Done when |
| --- | --- | --- |
| 0. Setup | Wed Nov 25 | Repo, requirements, `docs/data-contract.md`, `make data` downloads Part_1, layout of one record printed and explained |
| 1. Windows + split + baseline 1 | Thu Nov 26 | Window counts per filter logged, split saved, baseline 1 MAE on validation |
| 2. Features + baseline 2 + main model | Fri to Sun Nov 27 to 29 | Validation table: baseline 1, baseline 2, main model |
| 3. Error analysis + tuning | Tue Dec 1 | Where errors are worst (high BP, high HR, noisy PPG) written down |
| 4. Final test run + demo | Wed Dec 2 | `evaluate.py --final` run once, results committed, demo works |
| 5. README + write-up help | Thu Dec 3 | README complete, reproduction tested in a fresh environment |

At each checkpoint: run the verification checklist, commit with a clear message, and summarise what changed and what the numbers mean.

## Workflow for each task
1. Start from the latest `main` (`git pull`) and create a branch with your track's prefix.
2. Plan: list the files you'll touch and how you'll check that it works. Wait for an OK.
3. Build in small steps (each phase fits in about 2 hours), and commit after each step that works.
4. Run the verification checklist below.
5. Open a PR into `main` that says what changed and how you tested it. Add screenshots for any demo change. One feature per PR.
6. Never merge a PR yourself. A teammate reviews it and merges it.
7. Ask before creating or pushing to a GitHub repo, or making it public.

## Verification (before every PR)
- [ ] `make test` and `ruff check .` pass
- [ ] The demo starts with `make demo`, with no errors in the terminal or the browser console
- [ ] The demo flow still works from start to finish
- [ ] The test split was not touched (only `evaluate.py --final` reads it)
- [ ] Every number in the README or docs comes from `results/metrics.json`
- [ ] No secrets or raw data in the diff, and any new env variables are added to `.env.example`
- [ ] Say clearly if there's anything you couldn't check

## README must include (contest requirements)
What it does and who it is for; how to install and run; how to reproduce the evaluation (`make all`); the results table and figures; the demo link and the team; dataset credit (UCI dataset 340, CC BY 4.0) and any borrowed code or models; limitations (ICU patients with arterial lines, possible patient leakage across records, no demographic data so skin-tone effects on PPG cannot be checked, not tested on phone or watch sensors); one line disclosing AI coding assistance.

## Repo hygiene
- `.gitignore` includes `.env`, `data/`, and `.claude/worktrees/`.
- Keep the README up to date as results come in.

## Rules for Claude Code
- Do not look at test-set metrics before Phase 4, and do not tune anything after seeing them.
- Never invent numbers in docs.
- If a result is bad, report it honestly; missing AAMI is normal for cuffless methods.
