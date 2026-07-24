#!/usr/bin/env bash
# Insert the reserved Zenodo version DOI into every metadata file.
# Usage:  bash set_doi.sh 10.5281/zenodo.NNNNNNNN
# (run from the package root, before rebuilding/uploading the ZIP)
set -euo pipefail
if [ $# -ne 1 ]; then
  echo "Usage: bash set_doi.sh 10.5281/zenodo.NNNNNNNN" >&2
  exit 1
fi
DOI="$1"
NUM="${DOI##*.}"   # the numeric suffix, e.g. 20863560
for f in README.md CITATION.cff .zenodo.json; do
  if [ -f "$f" ]; then
    # replace both the full 10.5281/zenodo.XXXXXXXX token and a bare XXXXXXXX
    sed -i "s#10.5281/zenodo.XXXXXXXX#${DOI}#g; s#zenodo.XXXXXXXX#zenodo.${NUM}#g" "$f"
    echo "updated $f"
  fi
done
echo "Done. Remember to also set the DOI in the manuscript Data Availability statement."
