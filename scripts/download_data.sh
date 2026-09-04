#!/usr/bin/env bash
# Download a Kaggle competition's data into a competition folder.
#
# Usage:
#   ./scripts/download_data.sh <kaggle-slug> <target-dir>
#   ./scripts/download_data.sh kaust-ioai-titanic competitions/01-titanic/data
#
# Requires: pip install kaggle, and ~/.kaggle/kaggle.json (see docs/SETUP.md)

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <kaggle-competition-slug> <target-dir>" >&2
  exit 1
fi

SLUG="$1"
TARGET="$2"

mkdir -p "$TARGET"
kaggle competitions download -c "$SLUG" -p "$TARGET"
( cd "$TARGET" && for z in *.zip; do [[ -e "$z" ]] && unzip -o "$z" && rm "$z"; done )

echo "Done. Data extracted to $TARGET"
