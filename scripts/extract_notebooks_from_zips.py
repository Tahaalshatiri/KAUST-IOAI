#!/usr/bin/env python3
"""Pull notebooks out of competition dataset zips, in place.

Some training days store each competition as a single zip (data + a baseline
notebook + a README describing the competition). Git ignores zips, so the
notebook inside would be lost. This script walks the days/ folder, and for
every .zip it finds, extracts ONLY the .ipynb / .py / README.md files into a
folder named after the zip, right next to it. Data inside the zip is not
extracted. The zip itself is left untouched.

Example:  days/Day-10/math-questions-classification.zip
     ->   days/Day-10/math-questions-classification/notebook.ipynb
     ->   days/Day-10/math-questions-classification/README.md

Usage (from the repo root — no arguments needed):
    python scripts/extract_notebooks_from_zips.py
"""

import zipfile
from pathlib import Path

KEEP = {".ipynb", ".py"}
ROOT = Path(__file__).resolve().parents[1] / "days"

count = 0
for zpath in sorted(ROOT.rglob("*.zip")):
    outdir = zpath.with_suffix("")
    try:
        zf = zipfile.ZipFile(zpath)
    except zipfile.BadZipFile:
        print(f"skipping (not a valid zip): {zpath}")
        continue
    with zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if info.is_dir():
                continue
            if Path(name).suffix.lower() in KEEP or name.lower() == "readme.md":
                outdir.mkdir(parents=True, exist_ok=True)
                (outdir / name).write_bytes(zf.read(info))
                print(f"extracted: {outdir / name}")
                count += 1

print(f"\nDone — extracted {count} files. Data files were left inside the zips.")
