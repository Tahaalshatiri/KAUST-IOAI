# Environment setup

## Python

Any recent Python (3.10+) works. Recommended: create a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip
pip install jupyter numpy pandas scikit-learn matplotlib seaborn kaggle
# For deep-learning competitions:
pip install torch torchvision transformers
```

Alternatively, run everything directly on **Kaggle Notebooks** or
**Google Colab** — both have all of the above pre-installed and free GPUs,
which is what most students did during the program.

## Kaggle API token

1. Log in to [kaggle.com](https://www.kaggle.com) → *Settings* → *API* →
   **Create New Token**. This downloads `kaggle.json`.
2. Put it in place:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Test it: `kaggle competitions list` should print a table.

> ⚠️ `kaggle.json` is a secret. Never commit it to git (it's in `.gitignore`)
> and never paste it into a notebook.

## Private competitions

The program's competitions are private Kaggle competitions. You must open the
invite link in each competition's README (once) before the API will let you
download its data.
