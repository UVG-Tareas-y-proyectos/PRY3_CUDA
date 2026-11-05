#!/usr/bin/env bash
set -euo pipefail
binary="${1:-./build/hough}"
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
python3 - "$scratch/corners.pgm" <<'PY'
import sys
from pathlib import Path
w, h = 17, 19
Path(sys.argv[1]).write_bytes(b'P5\n17 19\n255\n' + bytes(255 if x in (0, w-1) or y in (0, h-1) else 0 for y in range(h) for x in range(w)))
PY
mkdir -p results
for tool in memcheck racecheck synccheck; do
  compute-sanitizer --tool "$tool" --error-exitcode 99 "$binary" "$scratch/corners.pgm" --runs 1 --output "$scratch/checked" > "results/sanitizer_${tool}.txt" 2>&1
  tail -n 4 "results/sanitizer_${tool}.txt"
done
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 "$binary" data/runway.pgm --runs 1 --output "$scratch/runway" > results/sanitizer_runway.txt 2>&1
tail -n 5 results/sanitizer_runway.txt
