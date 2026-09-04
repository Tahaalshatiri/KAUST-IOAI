#!/usr/bin/env python3
"""Organize the raw KAUST-IOAI training folder into a clean, git-friendly repo.

Walks a source folder of Day-N directories (the raw ~800MB dump) and rebuilds
it under <repo>/days/ keeping ONLY lightweight teaching material:

  - .ipynb / .py files        -> kept (labs and competition baselines)
  - README.md inside comp zips -> kept (competition description)
  - everything else (data files, csv, pdf slides, desktop.ini, ...) -> skipped

Layout produced:

  days/Day-01/
      <Lab Name>.ipynb                    # loose labs kept at day level
      <comp-slug>/
          README.md                       # description + comp-link placeholder
          <baseline>.ipynb                # notebook(s) found for that comp

Competition sources handled:
  1. A subfolder inside a day (e.g. "Food  Image matching Comp/") -> its
     notebooks/py files are kept, data files skipped.
  2. A dataset zip inside a day (e.g. "math-questions-classification.zip") ->
     opened in memory; only .ipynb/.py/README.md are extracted.

Usage:
    python scripts/organize_days.py <raw-source-dir> [--repo <repo-root>]

Safe to re-run: existing generated README.md files are NOT overwritten (so your
manual edits / added links survive), notebooks are only copied if missing or
changed.
"""

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

KEEP_SUFFIXES = {".ipynb", ".py"}
JUNK_NAMES = {"desktop.ini", "thumbs.db", ".ds_store"}

README_TEMPLATE = """# {title}

**Kaggle competition link:** _TODO: add link_

**Dataset download:** _TODO: add Google Drive link_ (folder: `{day}/{slug}`)

{description}

## Files

- `{notebooks}` — baseline / solution notebook(s). Run top-to-bottom after
  placing the competition data next to the notebook (or updating the paths).
"""


def slugify(name: str) -> str:
    name = re.sub(r"\b(comp|competition)\b", "", name, flags=re.I)
    name = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return re.sub(r"-{2,}", "-", name) or "unnamed"


def day_sort_key(p: Path):
    m = re.search(r"(\d+)", p.name)
    return int(m.group(1)) if m else 999


def norm_day(name: str) -> str:
    m = re.search(r"(\d+)", name)
    return f"Day-{int(m.group(1)):02d}" if m else name


def copy_if_needed(src_bytes: bytes, dest: Path) -> bool:
    if dest.exists() and dest.read_bytes() == src_bytes:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src_bytes)
    return True


def write_comp_readme(comp_dir: Path, title: str, day: str, slug: str,
                      description: str, notebooks: list[str]):
    readme = comp_dir / "README.md"
    if readme.exists():
        return  # never clobber manual edits (added links etc.)
    desc = description.strip() or "_TODO: short explanation of the competition._"
    nb_list = "`, `".join(notebooks) if notebooks else "TODO"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(README_TEMPLATE.format(
        title=title, day=day, slug=slug, description=desc, notebooks=nb_list))


def handle_comp_folder(src: Path, dest_day: Path, day: str, stats: dict):
    slug = slugify(src.name)
    comp_dir = dest_day / slug
    notebooks = []
    for f in sorted(src.rglob("*")):
        if not f.is_file() or f.name.lower() in JUNK_NAMES:
            continue
        if f.suffix.lower() in KEEP_SUFFIXES:
            if copy_if_needed(f.read_bytes(), comp_dir / f.name):
                stats["copied"] += 1
            notebooks.append(f.name)
        else:
            stats["skipped"] += 1
    desc = ""
    src_readme = src / "README.md"
    if src_readme.exists():
        desc = src_readme.read_text(errors="replace")
    write_comp_readme(comp_dir, src.name.strip(), day, slug, desc, notebooks)


def handle_comp_zip(zpath: Path, dest_day: Path, day: str, stats: dict):
    slug = slugify(zpath.stem)
    comp_dir = dest_day / slug
    notebooks, desc = [], ""
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if info.is_dir() or name.lower() in JUNK_NAMES:
                continue
            suffix = Path(name).suffix.lower()
            if suffix in KEEP_SUFFIXES:
                if copy_if_needed(zf.read(info), comp_dir / name):
                    stats["copied"] += 1
                notebooks.append(name)
            elif name.lower() == "readme.md":
                desc = zf.read(info).decode("utf-8", errors="replace")
            else:
                stats["skipped"] += 1
    write_comp_readme(comp_dir, zpath.stem.replace("-", " ").title(), day, slug,
                      desc, notebooks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="raw folder containing Day-N dirs")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1],
                    help="repo root (default: this repo)")
    args = ap.parse_args()

    day_dirs = sorted([d for d in args.source.iterdir()
                       if d.is_dir() and re.match(r"day[\s_-]*\d+", d.name, re.I)],
                      key=day_sort_key)
    if not day_dirs:
        sys.exit(f"No Day-N folders found in {args.source}")

    stats = {"copied": 0, "skipped": 0}
    for src_day in day_dirs:
        day = norm_day(src_day.name)
        dest_day = args.repo / "days" / day
        for entry in sorted(src_day.iterdir()):
            if entry.name.lower() in JUNK_NAMES:
                continue
            if entry.is_dir():
                handle_comp_folder(entry, dest_day, day, stats)
            elif entry.suffix.lower() == ".zip":
                handle_comp_zip(entry, dest_day, day, stats)
            elif entry.suffix.lower() in KEEP_SUFFIXES:
                # loose lab notebook/script at day level
                dest_day.mkdir(parents=True, exist_ok=True)
                if copy_if_needed(entry.read_bytes(), dest_day / entry.name):
                    stats["copied"] += 1
            else:
                stats["skipped"] += 1  # slides/pdf/data — not kept in git

    print(f"Copied/updated {stats['copied']} files, "
          f"skipped {stats['skipped']} non-notebook files.")
    print(f"Output: {args.repo / 'days'}")


if __name__ == "__main__":
    main()
