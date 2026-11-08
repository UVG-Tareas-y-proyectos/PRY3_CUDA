#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f .toolchain/env.sh ]]; then
  source .toolchain/env.sh
elif ! command -v nvcc >/dev/null; then
  if ! command -v g++-14 >/dev/null; then
    echo 'Se requiere g++-14: sudo apt-get install g++-14 make python3' >&2
    exit 1
  fi
  python3 scripts/setup_toolchain.py
  source .toolchain/env.sh
fi
make
if [[ $# -eq 0 ]]; then make run; else make "$@"; fi
