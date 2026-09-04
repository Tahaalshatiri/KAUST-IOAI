# KAUST IOAI Training — High School Program

Training materials from the **KAUST program preparing high school students for the
[International Olympiad in Artificial Intelligence (IOAI)](https://ioai-official.org/)**.

The program ran as a series of training **days**. Each day combined hands-on
**labs** with one or more private **Kaggle competitions**, covering the core IOAI
skills — classic tabular machine learning, computer vision, and NLP with deep
learning. This repository collects, for every day:

- the **lab notebooks** used in class,
- for each competition: a **README** (problem, data, metric, link) and the
  **baseline / solution notebook**.

## Repository structure

```
KAUST-IOAI/
├── README.md                  ← you are here
├── days/
│   ├── Day-01/
│   │   ├── <Lab Name>.ipynb           ← labs, kept at day level
│   │   └── <competition-name>/
│   │       ├── README.md              ← problem, data, metric, comp link
│   │       └── <baseline>.ipynb       ← baseline / solution notebook
│   ├── Day-02/
│   └── ...
├── scripts/
│   ├── organize_days.py       ← turns the raw training dump into this layout
│   └── download_data.sh       ← downloads a competition's data via the Kaggle API
└── docs/
    └── SETUP.md               ← environment setup for students
```

> **Where is the data?** Competition datasets are intentionally **not stored in
> git** (they total ~1 GB). Get them either from the Kaggle competition pages or
> from the shared Google Drive folder:
>
> - **Google Drive (all datasets):** _TODO: add link_
> - Each competition's `README.md` has (or will have) its Kaggle link.

## Quick start (for students)

1. Pick a day under `days/` and open its lab notebooks, or pick a competition
   folder and read its `README.md`.
2. Download that competition's data (Kaggle link in the README, or the Drive
   folder above) and place it next to the notebook (or update the paths at the
   top of the notebook).
3. Run the baseline top-to-bottom, then try to beat it.

Environment setup (Python, Kaggle API token): see [`docs/SETUP.md`](docs/SETUP.md).

## Competitions index

<!-- TODO: fill links as they are added. One row per competition. -->

| Day | Competition | Topic | Kaggle link |
|-----|-------------|-------|-------------|
| 01 | [Sensory readings classification](days/Day-01/sensory-readings/) | Tabular ML | _TODO_ |
| 03 | [EMNIST 4-digit classification](days/Day-03/emnist-4-digit/) | Computer vision | _TODO_ |
| 04 | [Animal classification using crops](days/Day-04/animal-classification-using-crops/) | Computer vision | _TODO_ |
| 04 | [Food image matching](days/Day-04/food-image-matching/) | Computer vision / CLIP | _TODO_ |
| 10 | [Intent classification (RU→EN)](days/Day-10/intent-classification-rus-to-eng/) | NLP | _TODO_ |
| 10 | [Math questions classification](days/Day-10/math-questions-classification/) | NLP | _TODO_ |

## Maintaining this repo

The raw training folder (with datasets, slides, etc.) is turned into this clean
layout by one script:

```bash
python scripts/organize_days.py /path/to/raw/KAUST-IOAI-folder
```

It walks every `Day-N` folder, digs into competition subfolders **and** dataset
zips, and keeps only `.ipynb`/`.py` files plus competition descriptions —
skipping all data files, slides, and junk. It also generates a `README.md` stub
(with a `TODO` link placeholder) for each competition, and never overwrites a
README you've already edited. Re-running it is safe.

Rules of thumb:

- **Never commit data** — `.gitignore` blocks `*.csv`, `*.zip`, images, etc.
  Data lives on Kaggle / Google Drive.
- One folder per competition inside its day, lowercase-hyphenated name.
- When adding a new day, just drop the raw folder into the dump and re-run the
  script, then fill in the links.

## About the program

This training was run at **KAUST (King Abdullah University of Science and
Technology)** to prepare high school students for the **IOAI**, the
international olympiad covering machine learning, computer vision, and natural
language processing. The competitions mirror the olympiad's style: short,
focused problems where a well-understood baseline plus careful iteration beats
complexity.
