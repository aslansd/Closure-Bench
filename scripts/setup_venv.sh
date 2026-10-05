#!/usr/bin/env bash
# One-command local setup with Python's built-in venv + pip from PyPI (no conda).
#
# Usage (from anywhere):
#   scripts/setup_venv.sh                        # tested versions (requirements-lock*.txt)
#   scripts/setup_venv.sh --latest               # newest compatible versions (requirements.txt)
#   scripts/setup_venv.sh --python /path/to/python3.12
# pip settings such as PIP_INDEX_URL / PIP_PROXY / PIP_TRUSTED_HOST are honoured.
set -euo pipefail
# Apple Silicon Terminal running under Rosetta: re-launch this script natively (arm64)
if [ "$(uname -s)" = "Darwin" ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ] \
   && [ -z "${CB_REEXEC:-}" ] && command -v arch >/dev/null 2>&1; then
  echo "Note: this Terminal runs under Rosetta; re-running natively with 'arch -arm64'."
  echo "      Permanent fix: Finder > Applications > Utilities > Terminal > Get Info > untick 'Open using Rosetta'."
  CB_REEXEC=1 exec arch -arm64 /bin/bash "$0" "$@"
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REQ="$ROOT/requirements-lock.txt"
PY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --latest) REQ="$ROOT/requirements.txt" ;;
    --python) PY="$2"; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
  shift
done

# 1) pick a Python >= 3.11 (3.12 preferred). On Apple Silicon, skip Intel-only (x86_64) builds.
APPLE_SILICON=0
if [ "$(uname -s)" = "Darwin" ] && [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = "1" ]; then APPLE_SILICON=1; fi
FW=/Library/Frameworks/Python.framework/Versions
if [ -z "$PY" ]; then
  for c in $FW/3.12/bin/python3.12 /opt/homebrew/bin/python3.12 python3.12 \
           $FW/3.13/bin/python3.13 /opt/homebrew/bin/python3.13 python3.13 \
           $FW/3.11/bin/python3.11 /opt/homebrew/bin/python3.11 python3.11 python3; do
    command -v "$c" >/dev/null 2>&1 || continue
    c="$(command -v "$c")"
    if ! "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
      continue
    fi
    if [ "$APPLE_SILICON" = "1" ] && [ "$("$c" -c 'import platform; print(platform.machine())')" != "arm64" ]; then
      echo "Skipping $c (Intel-only build; JAX needs a native arm64 Python)"
      continue
    fi
    PY="$c"; break
  done
fi
if [ -z "$PY" ]; then
  echo "No suitable Python >= 3.11 found. Install Python 3.12 from https://www.python.org/downloads/macos/"
  echo "(macOS 64-bit universal2 installer), then re-run this script."; exit 1
fi
VER="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
ARCH="$("$PY" -c 'import platform; print(platform.machine())')"
echo "Using $PY  (Python $VER, $ARCH)"

# Apple Silicon: the Python must run natively (arm64). JAX publishes no Intel-Mac
# builds after 0.4.38, and Intel builds need AVX, which Rosetta does not emulate.
if [ "$(uname -s)" = "Darwin" ] && [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = "1" ] && [ "$ARCH" != "arm64" ]; then
  echo
  echo "ERROR: this is an Apple Silicon Mac, but $PY runs as Intel (x86_64 / Rosetta)."
  echo "       JAX cannot be installed for it. Fix one of these, then re-run this script:"
  if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ]; then
    echo "  * This Terminal runs under Rosetta. Quit Terminal; in Finder > Applications > Utilities,"
    echo "    right-click Terminal > Get Info > untick 'Open using Rosetta'; reopen it."
    echo "    (One-off alternative: run  arch -arm64 zsh  and then this script again.)"
  fi
  if ! file -L "$PY" 2>/dev/null | grep -q arm64; then
    echo "  * $PY is an Intel-only build. Install Python 3.12 from https://www.python.org/downloads/macos/"
    echo "    (macOS 64-bit universal2 installer), then run:"
    echo "      scripts/setup_venv.sh --python /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
  fi
  exit 1
fi
if [ "$REQ" = "$ROOT/requirements-lock.txt" ] && [ "$VER" = "3.11" ]; then
  echo "Python 3.11: using requirements-lock-py311.txt (tested; JAX 0.10.2 because JAX 0.11 needs Python >= 3.12)."
  REQ="$ROOT/requirements-lock-py311.txt"
elif [ "$REQ" = "$ROOT/requirements-lock.txt" ] && [ "$VER" != "3.12" ] && [ "$VER" != "3.13" ]; then
  echo "No lock file for Python $VER; using version ranges (requirements.txt)."
  REQ="$ROOT/requirements.txt"
fi

# 2) create the venv and install
# rebuild an existing .venv that was made by a different Python version or architecture
if [ -x "$ROOT/.venv/bin/python" ]; then
  OLD="$("$ROOT/.venv/bin/python" -c 'import sys, platform; print(f"{sys.version_info.major}.{sys.version_info.minor} {platform.machine()}")' 2>/dev/null || echo broken)"
  if [ "$OLD" != "$VER $ARCH" ]; then
    echo "Existing .venv was made by Python $OLD; rebuilding it for Python $VER $ARCH."
    rm -rf "$ROOT/.venv"
  fi
fi
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  "$PY" -m venv "$ROOT/.venv"
fi
VPY="$ROOT/.venv/bin/python"
"$VPY" -m pip install --upgrade pip
if ! "$VPY" -m pip install --prefer-binary -r "$REQ"; then
  echo
  echo "pip install failed. If the error mentions jax/jaxlib versions, your Python is probably not"
  echo "a native build for this machine (see the architecture printed above)."
  exit 1
fi

# 3) register the Jupyter kernel (for VS Code / any Jupyter)
"$VPY" -m ipykernel install --user --name closurebench --display-name "Python (closurebench)"

# 4) check
"$VPY" "$ROOT/scripts/check_setup.py"
echo
echo "Done. Next time, activate with:   source \"$ROOT/.venv/bin/activate\""
echo "Interactive:   jupyter lab        (kernel: 'Python (closurebench)')"
echo "Long runs:     scripts/run_notebook.sh notebooks/02_Synthetic_Worms_Ladder_Validation.ipynb laptop"
