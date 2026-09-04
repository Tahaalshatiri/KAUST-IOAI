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

Every competition folder has a `README.md` (task, data, metric) and a
`baseline.ipynb` (executed baseline) or `starter.ipynb` (GPU starter kit).
Baseline scores below are honest cross-validation / holdout numbers from the
committed notebooks — beat them!

<!-- One row per competition; fill the Kaggle links as they are added. -->

| Day | Competition | Topic | Baseline (score) | Kaggle link |
|-----|-------------|-------|------------------|-------------|
| 1 | [Sensory readings classification](days/Day-1/sensory-readings%20comp/) | Tabular ML | student notebook | _TODO_ |
| 3 | [EMNIST 4-digit classification](days/Day-3/EMNIST%204%20digit%20comp/) | Vision | student notebook | _TODO_ |
| 4 | [Animal classification using crops](days/Day-4/Animal%20classification%20using%20crops%20comp/) | Vision | student notebook | _TODO_ |
| 4 | [Food image matching](days/Day-4/Food%20%20Image%20matching%20Comp/) | Vision / CLIP | student notebook | _TODO_ |
| 5 | [Pet segmentation](days/Day-5(exam)/Pet%20segmentation%20competition/) | Vision | student notebook | _TODO_ |
| 5 | [Agent rumble](days/Day-5(exam)/agent%20rumble%20competition/) | Agents | student notebook | _TODO_ |
| 6 | [Decode schizophrenia](days/Day-6/Decode%20schizophrenia%20comp/) | ML / neuroscience | student notebook | _TODO_ |
| 9 | [Animal voice detection](days/Day-9/animal-voice-detection/) | Audio | ResNet18+LR (acc 0.60) | _TODO_ |
| 9 | [Coins counting](days/Day-9/coins%20counting/) | Vision | ResNet18+Ridge (MAE 322 grosz) | _TODO_ |
| 9 | [Tricy table](days/Day-9/tricy-table/) | Tabular ML | LightGBM (MAE 3.37) | _TODO_ |
| 10 | [Intent classification (RU→EN)](days/Day-10/intent-classification-rus-to-eng/) | NLP | student notebook | _TODO_ |
| 10 | [Math questions classification](days/Day-10/math-questions-classification/) | NLP | student notebook | _TODO_ |
| 13 | [Classifying corrupted images](days/Day-13/classifying-curropted-images/) | Vision | ResNet18+LR (acc 0.86) | _TODO_ |
| 13 | [KAUST residents power consumption](days/Day-13/predict-kaust-residents-power-consumption/) | Time series | LightGBM (MAE 67.1) | _TODO_ |
| 14 | [Patent phrases similarity](days/Day-14/measure-patents-phrases-similarity/) | NLP | TF-IDF+Ridge (Pearson 0.56) | _TODO_ |
| 15 | [Molecular energy prediction](days/Day-15/molecular-energy-prediction/) | ML / chemistry | LightGBM (MAE 66 eV) | _TODO_ |
| 15 | [Predict image rotation](days/Day-15/predict-image-rotation/) | Vision | ResNet18+LR (acc 0.64) | _TODO_ |
| 16 | [Fashion-IQ retrieval](days/Day-16/fashion-iq/) | Multimodal / CLIP | zero-shot CLIP (acc 0.59) | _TODO_ |
| 16 | [Predict blindness](days/Day-16/predict-blindness-before-it-happen/) | Vision / medical | ResNet18+LR (QWK 0.82) | _TODO_ |
| 17 | [Image colorization from scratch](days/Day-17/image-colorization-from-scratch/) | Vision / generative | scratch CNN (MAE 12.2/255) | _TODO_ |
| 17 | [Molecular scalar couplings](days/Day-17/predicting-molecular-scalar-couplings/) | ML / chemistry | LightGBM (MAE 2.79) | _TODO_ |
| 19 | [Lobotomize the generator](days/Day-19/lobotomize-the-generator/) | Diffusion / erasure | GPU starter | _TODO_ |
| 22 | [Concepts guessing](days/Day-22/concepts-guessing/) | NLP | TF-IDF+LR (top-10 0.47) | _TODO_ |
| 23 | [Steer LLMs (ocean)](days/Day-23/steer-ll-ms-to-yap-about-the-ocean/) | LLM steering | GPU starter | _TODO_ |
| 24 | [AI hallucination detector](days/Day-24/ai-hallucination-detector/) | NLP | TF-IDF+LR (acc 0.81) | _TODO_ |
| 25 | [Solve misconceptions](days/Day-25/solve-misconceptions/) | NLP / education | TF-IDF matching | _TODO_ |
| 26 | [Classify NLP with noisy labels](days/Day-26/classify-nlp-with-noisy-labels/) | NLP / robustness | TF-IDF+LR + cleaning | _TODO_ |
| 28 | [IOAI 2026 home task 2](days/Day-28/ioai-2026-home-task-2/) | RL / imitation | BC (acc 0.76) + BFS planner | _TODO_ |
| 30 | [Rosetta embedding alignment](days/Day-30/rosetta-embedding-alignment/) | Embeddings | linear map + cosine NN (acc ~0.2) | _TODO_ |

Days 27, 29, 31–33 contain the **Muraqam** Arabic-NLP project notebooks
(diacritization / punctuation), and Day-21 holds reference papers.

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
