# KAUST IOAI Training — High School Program

Training materials from the **KAUST program preparing high school students for the
[International Olympiad in Artificial Intelligence (IOAI)](https://ioai-official.org/)**.

The program ran as a series of training **days**. Each day combined hands-on
**labs** with one or more private **Kaggle competitions**, covering the core
IOAI skills — classic tabular machine learning, computer vision, and NLP with
deep learning. This repository keeps, for every day, the **lab notebooks** and
each competition's **baseline / solution notebook**.

## Repository structure

```
KAUST-IOAI/
├── days/
│   ├── Day-1/
│   │   ├── <lab notebooks>.ipynb            ← labs, at day level
│   │   └── <competition folder>/
│   │       ├── <baseline notebook>.ipynb    ← baseline / solution
│   │       └── README.md                    ← comp description (when available)
│   ├── Day-2/
│   └── ...
├── scripts/
│   ├── extract_notebooks_from_zips.py   ← pulls notebooks out of comp dataset zips
│   └── download_data.sh                 ← downloads comp data via the Kaggle API
└── docs/SETUP.md                        ← environment setup for students
```

> **Where is the data?** Datasets are intentionally **not stored in git** — they
> total ~1 GB. The `.gitignore` is in *whitelist mode*: only notebooks, scripts,
> and markdown files are ever committed, so the raw training folders (with all
> their data) can sit inside `days/` on a maintainer's machine without bloating
> the repository. Get the data from:
>
> - **Google Drive (all datasets):** _TODO: add link_
> - **Kaggle:** each competition's link — _TODO: being added below_

## Quick start (for students)

1. Pick a day under `days/` and open its lab notebooks, or pick a competition
   folder.
2. Download that competition's data (Kaggle link below, or the Drive folder
   above) and place it next to the notebook — or update the paths at the top of
   the notebook.
3. Run the baseline top-to-bottom, then try to beat it.

Environment setup (Python, Kaggle API token): see [`docs/SETUP.md`](docs/SETUP.md).

## Competitions index

<!-- One row per competition; fill the links as they are added. -->

| Day | Competition | Topic | Kaggle link |
|-----|-------------|-------|-------------|
| 1 | Sensory readings classification | Tabular ML | _TODO_ |
| 3 | EMNIST 4-digit classification | Computer vision | _TODO_ |
| 4 | Animal classification using crops | Computer vision | _TODO_ |
| 4 | Food image matching | Computer vision / CLIP | _TODO_ |
| 10 | Intent classification (RU→EN) | NLP | _TODO_ |
| 10 | Math questions classification | NLP | _TODO_ |

## Maintaining this repo

Adding new material is designed to be brainless:

1. Copy the raw `Day-N` folder(s) — data and all — into `days/`.
2. If a day contains competition **zips**, run
   `python scripts/extract_notebooks_from_zips.py` once from the repo root. For
   every zip it creates a matching folder with the baseline notebook and README
   from inside the zip — or a placeholder README if the zip holds only data —
   so every competition is visible in the repo.
3. `git add .` → commit → push. The whitelist `.gitignore` guarantees only
   `.ipynb` / `.py` / `.md` / `.sh` / `.pdf` / `.pptx` files are committed —
   notebooks and slides, never data, zips, or junk files.

Then fill in the competition's Kaggle link in the table above.

## About the program

This training was run at **KAUST (King Abdullah University of Science and
Technology)** to prepare high school students for the **IOAI**, the
international olympiad covering machine learning, computer vision, and natural
language processing. The competitions mirror the olympiad's style: short,
focused problems where a well-understood baseline plus careful iteration beats
complexity.
