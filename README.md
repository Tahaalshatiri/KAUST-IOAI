# KAUST IOAI Training — High School Program

Training materials from the **KAUST program preparing high school students for the
[International Olympiad in Artificial Intelligence (IOAI)](https://ioai-official.org/)**.

During the program, students competed in a series of private Kaggle competitions,
each targeting a core AI/ML skill — from classic tabular machine learning to
computer vision and NLP with deep learning. This repository collects, for every
competition:

- a short **explanation of the problem** (task, data, metric),
- a clean, commented **baseline notebook** students can run and improve,
- (where available) the **top solution / reference solution**,
- a **link to the Kaggle competition** so the data can be downloaded.

The goal is that a future student (or instructor) can open any competition folder,
read the README, download the data with one command, run the baseline, and start
climbing the leaderboard.

## Repository structure

```
KAUST-IOAI/
├── README.md                  ← you are here
├── competitions/
│   ├── 01-<comp-name>/
│   │   ├── README.md          ← problem statement, data, metric, comp link
│   │   ├── baseline.ipynb     ← starter notebook (runs top-to-bottom)
│   │   ├── solution.ipynb     ← reference / winning solution (optional)
│   │   └── data/              ← NOT in git — downloaded from Kaggle (see below)
│   ├── 02-<comp-name>/
│   └── ...
├── scripts/
│   └── download_data.sh       ← downloads a competition's data via the Kaggle API
└── docs/
    └── SETUP.md               ← environment setup for students
```

> **Note:** raw data is intentionally **not stored in this repository** (the
> combined datasets are ~1 GB). Each competition README links to its Kaggle page,
> and `scripts/download_data.sh` fetches the data into the right folder.

## Quick start (for students)

1. **Set up Kaggle API access** (once):
   - Create an account on [kaggle.com](https://www.kaggle.com) and, for private
     competitions, accept the invite link in the competition README.
   - Go to *Kaggle → Settings → API → Create New Token*; save the downloaded
     `kaggle.json` to `~/.kaggle/kaggle.json` (and `chmod 600 ~/.kaggle/kaggle.json`).
   - `pip install kaggle`

2. **Pick a competition** from `competitions/` and read its `README.md`.

3. **Download its data:**
   ```bash
   ./scripts/download_data.sh <kaggle-competition-slug> competitions/<folder-name>/data
   ```

4. **Run the baseline** (`baseline.ipynb`) top-to-bottom, then try to beat it.
   Ideas for improvement are listed at the end of each baseline.

## Competitions

<!-- Table is filled in as competitions are added. Keep it sorted by number. -->

| # | Competition | Topic | Task type | Metric | Link |
|---|-------------|-------|-----------|--------|------|
| 01 | _TBD_ | _e.g. Tabular ML_ | _e.g. binary classification_ | _e.g. AUC_ | _[Kaggle](#)_ |

## About the program

This training was run at **KAUST (King Abdullah University of Science and
Technology)** to prepare high school students for the **IOAI**, the international
olympiad covering machine learning, computer vision, and natural language
processing. The competitions here mirror the olympiad's style: short, focused
problems where a well-understood baseline plus careful iteration beats
complexity.

## Contributing / maintaining

- One folder per competition under `competitions/`, numbered in the order they
  were run. Copy `competitions/_template/` to start a new one.
- Never commit data files — `.gitignore` blocks `competitions/*/data/` and common
  data extensions. Keep the repo small; data lives on Kaggle.
- Clear notebook outputs before committing **unless** the outputs are small and
  instructive (a final score, a few plots).
