#!/usr/bin/env python3
"""Make every competition zip visible in the repo, without committing its data.

Many training days store a competition as a single zip (data + sometimes a
baseline notebook + sometimes a README). Git ignores zips, so without this
script those competitions would be invisible on GitHub.

For every .zip found under days/, this script creates a folder named after the
zip, right next to it, containing:

  - every .ipynb / .py file found inside the zip (the baseline), and
  - the zip's README.md if it has one, or otherwise
  - a generated placeholder README.md naming the competition, with TODO spots
    for the Kaggle link — so the competition always shows up in the repo even
    if its zip holds only data.

Data files are never extracted, and the zip itself is left untouched.

Example:  days/Day-10/math-questions-classification.zip
     ->   days/Day-10/math-questions-classification/notebook.ipynb
     ->   days/Day-10/math-questions-classification/README.md

Usage (from the repo root — no arguments needed):
    python scripts/extract_notebooks_from_zips.py

Safe to re-run: it never overwrites a README.md that already exists on disk,
so links you add by hand survive.
"""

import zipfile
from pathlib import Path

KEEP = {".ipynb", ".py"}
ROOT = Path(__file__).resolve().parents[1] / "days"

PLACEHOLDER = """# {title}

**Kaggle competition link:** _TODO: add link_

**Data:** not stored in git — see `{zipname}` in the shared Google Drive
folder (link in the main README), or the Kaggle page above.

_TODO: short description of the competition (task, data, metric)._
"""

extracted = 0
for zpath in sorted(ROOT.rglob("*.zip")):
    outdir = zpath.with_suffix("")
    try:
        zf = zipfile.ZipFile(zpath)
    except zipfile.BadZipFile:
        print(f"skipping (not a valid zip): {zpath}")
        continue

    zip_readme = None
    with zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if info.is_dir():
                continue
            if Path(name).suffix.lower() in KEEP:
                outdir.mkdir(parents=True, exist_ok=True)
                (outdir / name).write_bytes(zf.read(info))
                print(f"extracted: {outdir / name}")
                extracted += 1
            elif name.lower() == "readme.md":
                zip_readme = zf.read(info)

    readme_path = outdir / "README.md"
    if not readme_path.exists():  # never clobber manual edits
        outdir.mkdir(parents=True, exist_ok=True)
        if zip_readme is not None:
            readme_path.write_bytes(zip_readme)
            print(f"extracted: {readme_path}")
        else:
            title = zpath.stem.replace("-", " ").replace("_", " ").title()
            readme_path.write_text(
                PLACEHOLDER.format(title=title, zipname=zpath.name))
            print(f"created placeholder: {readme_path}")

print(f"\nDone — {extracted} notebooks/scripts extracted. "
      "Every zip now has a visible folder; data stayed inside the zips.")
