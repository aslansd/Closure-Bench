#!/usr/bin/env bash
# Run a notebook headless (e.g. overnight), keeping the Mac awake, with a timestamped
# executed copy and log in results/executed/.  Re-running resumes from checkpoints.
#
# Uses the project's .venv automatically if it exists (no need to activate it).
# Usage (from the Closure-Bench folder):
#   scripts/run_notebook.sh notebooks/02_Synthetic_Worms_Ladder_Validation.ipynb laptop
#   scripts/run_notebook.sh notebooks/04_E1_Closure_Ladder_Real_Data.ipynb laptop
# Modes: smoke | laptop | full   (default: laptop)
set -euo pipefail
# Apple Silicon Terminal running under Rosetta: re-launch this script natively (arm64)
if [ "$(uname -s)" = "Darwin" ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ] \
   && [ -z "${CB_REEXEC:-}" ] && command -v arch >/dev/null 2>&1; then
  echo "Note: this Terminal runs under Rosetta; re-running natively with 'arch -arm64'."
  echo "      Permanent fix: Finder > Applications > Utilities > Terminal > Get Info > untick 'Open using Rosetta'."
  CB_REEXEC=1 exec arch -arm64 /bin/bash "$0" "$@"
fi
NB="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
MODE="${2:-laptop}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/executed"; mkdir -p "$OUT"
NAME="$(basename "$NB" .ipynb)_${MODE}_$(date +%Y%m%d-%H%M%S)"
export CLOSUREBENCH_MODE="$MODE" CLOSUREBENCH_ROOT="$ROOT"
# Always run nbconvert inside the project's venv Python when it exists, so the notebook
# kernel (python3 = the Python running nbconvert) is guaranteed to be the venv's.
if [ -x "$ROOT/.venv/bin/python" ]; then
  # equivalent of 'source .venv/bin/activate': kernels that launch plain "python" must
  # find the venv's Python, not e.g. conda's (base) Python
  export VIRTUAL_ENV="$ROOT/.venv"
  export PATH="$ROOT/.venv/bin:$PATH"
  unset PYTHONHOME
  NBCONVERT=("$ROOT/.venv/bin/python" -m nbconvert)   # run nbconvert IN the venv's Python
else
  NBCONVERT=(python3 -m nbconvert)
fi
echo "Python: $("${NBCONVERT[0]}" -c 'import sys, platform; print(sys.executable, platform.machine())' 2>/dev/null || echo "${NBCONVERT[0]}")"
KEEP_AWAKE=""
if command -v caffeinate >/dev/null 2>&1; then KEEP_AWAKE="caffeinate -is"; fi   # macOS: no idle/system sleep
echo "Running $NB in mode '$MODE' -> $OUT/$NAME.ipynb"
cd "$(dirname "$NB")"
$KEEP_AWAKE "${NBCONVERT[@]}" --to notebook --execute "$NB" \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 --output-dir "$OUT" --output "$NAME.ipynb" 2>&1 | tee "$OUT/$NAME.log"
echo "Done: $OUT/$NAME.ipynb"
