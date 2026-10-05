"""Check that the local environment can run Closure-Bench.
Usage (from the Closure-Bench folder):  .venv/bin/python scripts/check_setup.py
(scripts/setup_venv.sh runs this automatically)"""
import os, sys, time, platform
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ok = True
print(f"python {platform.python_version()} on {platform.system()} {platform.machine()}")
def _sysctl(name):
    import subprocess
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""
if platform.system() == "Darwin" and _sysctl("hw.optional.arm64") == "1" and platform.machine() != "arm64":
    sys.exit("ERROR: Apple Silicon Mac, but this Python runs as Intel (Rosetta). JAX needs a native arm64 "
             "Python: untick 'Open using Rosetta' for Terminal, or install the python.org universal2 "
             "Python 3.12, then re-run scripts/setup_venv.sh.")
if sys.prefix == sys.base_prefix:
    print("  note: not running inside a virtual environment (run scripts/setup_venv.sh, then use .venv/bin/python)")
if sys.version_info < (3, 11):
    print("  WARNING: Python >= 3.11 required (3.12 recommended)")
for mod in ["numpy", "scipy", "matplotlib", "h5py", "jax", "optax", "wormneuroatlas", "nbformat", "brian2"]:
    try:
        m = __import__(mod)
        print(f"  ok   {mod:15s} {getattr(m, '__version__', '')}")
    except ImportError as e:
        ok = False
        print(f"  MISSING {mod}: {e}")
    except Exception as e:   # e.g. Brian2 2.9 with NumPy >= 2.4 ('ndarray' has no attribute 'ptp')
        ok = False
        print(f"  BROKEN  {mod}: {type(e).__name__}: {e}")
        if mod == "brian2":
            print("          fix: .venv/bin/pip install 'numpy==2.3.5'  (Python 3.11), or use Python 3.12+ with brian2 2.10")
if not ok:
    sys.exit("Install the missing packages: scripts/setup_venv.sh (or .venv/bin/pip install -r requirements-lock.txt)")
import numpy as np, jax
from closurebench import data as D, ladder as lad, config as C
print(C.describe_machine())
print("loading the atlas ...")
ds = D.load_atlas_dataset()
print(ds.summary())
st = lad.Structure.from_dataset(ds)
t = time.perf_counter()
R = lad.predict("L1", lad.init_params("L1", st, jax.random.PRNGKey(0)), st, ds.stim[:8], ds.strains, lad.SimConfig())
R.block_until_ready()
print(f"simulated 8 stimulations x 2 strains of the full worm (incl. compile): {time.perf_counter() - t:.1f} s, "
      f"finite={bool(np.isfinite(np.asarray(R)).all())}")
print("\nAll good. Start Jupyter from this folder (jupyter lab) and run notebooks/00 ... 04.")
