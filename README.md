# Closure-Bench

**Where does the software end? Decisive tests of computational closure in a whole nervous system.**
Implementation of the Kempner Institute Research Fellowship proposal by Aslan S. Dizaji.

The project asks, by measurement rather than argument, whether brain activity in *C. elegans* is reducible to a self-contained computational level. It fits a ladder of formal descriptions (L0 wiring-only → L4 self-modifying), plus a black box, to the whole-brain signal-propagation atlas (Randi et al. 2023). Each rung is scored on held-out optogenetic interventions against the noise ceiling.

## Run locally (Python venv + pip; macOS Apple Silicon, Linux, Intel)

Everything runs on the CPU and installs from PyPI with plain `pip`; no conda is needed. Every dependency has a prebuilt wheel for Apple Silicon (arm64), so nothing compiles. Apple's Metal plugin for JAX is experimental and not used.

**1. Python 3.12.** Check with `python3.12 --version`. If it's missing, install it from the macOS 64-bit universal2 installer at [python.org/downloads/macos](https://www.python.org/downloads/macos/). Python 3.13 and 3.11 also work. On 3.11 the script installs `requirements-lock-py311.txt`, which is identical except for JAX 0.10.2 (JAX 0.11 needs Python ≥ 3.12). That version was tested and gives identical results.

**2. One command creates `.venv`, installs everything, registers the Jupyter kernel, and checks the setup:**

```bash
cd /path/to/Closure-Bench
scripts/setup_venv.sh
```

The script picks a native Python ≥ 3.11 automatically. It prefers the python.org installs in `/Library/Frameworks`, and on Apple Silicon it skips Intel-only builds such as an old Homebrew in `/usr/local`. It then installs the exact tested versions from `requirements-lock.txt` (or `requirements-lock-py311.txt` on Python 3.11). Add `--latest` for the newest compatible versions, or `--python /path/to/python3.12` to choose the interpreter. If PyPI is reachable only through a mirror or proxy, set `PIP_INDEX_URL` (or `PIP_PROXY`) before running; pip honours these.

The same steps by hand:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --prefer-binary -r requirements-lock.txt
python -m ipykernel install --user --name closurebench --display-name "Python (closurebench)"
python scripts/check_setup.py
```

The checker verifies the imports and warns if Python runs under Rosetta (x86_64) instead of arm64. It also loads the atlas and simulates the worm. The data ship inside `wormneuroatlas`, so nothing else is downloaded.

**3. Interactive work:**

```bash
source .venv/bin/activate
jupyter lab
```

Open `notebooks/` and use the kernel **Python (closurebench)**. In VS Code, open the folder and select the same kernel, or select `.venv` as the interpreter. Run in this order: 00 → 01 → 03 → 02 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18 → 19 → 20 → 21 → 22.

**4. Long runs (Notebooks 02 and 04): run them headless.** The script uses `.venv` automatically; you don't need to activate it.

```bash
scripts/run_notebook.sh notebooks/02_Synthetic_Worms_Ladder_Validation.ipynb laptop
scripts/run_notebook.sh notebooks/04_E1_Closure_Ladder_Real_Data.ipynb laptop
```

The script uses `caffeinate` to keep the Mac awake. Keep it plugged in, with the lid open. The executed notebook and a log go to `results/executed/`. If anything interrupts the run, run the same command again: finished fits load from checkpoints.

### Troubleshooting (Apple Silicon)

*`Could not find a version that satisfies the requirement jaxlib==...`*, or `setup_venv.sh` reports `x86_64` on an M1/M2/M3 Mac: your Python runs as Intel (Rosetta). JAX has no Intel-Mac builds after 0.4.38, and its Intel builds need AVX, which Rosetta lacks. Check with:

```bash
uname -m                              # must print arm64
sysctl -n sysctl.proc_translated      # 1 = this Terminal runs under Rosetta
file "$(which python3.12)"            # must list arm64
```

* **Terminal under Rosetta:** quit it, then Finder → Applications → Utilities → Terminal → Get Info → untick *Open using Rosetta*. Jupyter launched from a Rosetta Terminal also runs as Intel, so this fix is needed for interactive work too.
* **Intel-only Python** (e.g. an old Homebrew in `/usr/local`): install the python.org *universal2* Python 3.12, then run `scripts/setup_venv.sh --python /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12`.

`setup_venv.sh` detects both cases and skips Intel-only interpreters when searching. If you already have a native python.org Python 3.11 or newer, it uses that. It rebuilds a `.venv` that was created by the wrong Python. When the Terminal is under Rosetta and the Python is universal, it re-launches itself natively.

### Run modes

Each notebook's setup cell sets `MODE` from the environment variable `CLOSUREBENCH_MODE`.

| Mode | Used by default | Notebook 02 | Notebook 04 |
|---|---|---|---|
| `smoke` | opt-in | minutes, code check only | minutes, code check only |
| `laptop` | when there is no GPU | 60-neuron real wiring; a few hours | 120 most-sampled neurons; overnight (preliminary E1) |
| `full` | when a GPU is present | 120 neurons, 3 seeds, full power grid | all 300 neurons: the registered E1 test |

Before the long fits start, Notebooks 02 and 04 print a runtime estimate measured on your machine. For reference, the same estimator gave these figures on a single slow cloud CPU core:

* Notebook 02 `laptop`: about 4.5 h.
* Notebook 04 `laptop`: about 5.5 h (primary and secondary protocol).

An Apple Silicon laptop should be similar or faster. Of the rungs, the black box B and L1 (with 3 restarts) take the most time. To force a mode in Jupyter, start it as `CLOSUREBENCH_MODE=full jupyter lab`.

## Run on Colab

1. Push this folder to GitHub, e.g. `github.com/aslansd/Closure-Bench`. If you use a different URL, edit `REPO_URL` in each notebook's setup cell.
2. Open a notebook in Colab (File → Open notebook → GitHub), choose a GPU runtime, and run it. The setup cell installs the dependencies and clones the repo.

## Notebooks

| Notebook | What it does | Runtime |
|---|---|---|
| 00 Project plan and setup | roadmap, design decisions, smoke test | CPU, 1 min |
| 01 Closure theory on toy systems | exact informational, causal and computational closure on Markov chains; finite-data estimator | CPU, 2 min |
| 02 Synthetic worms | does the pipeline recover a known closed level? power for L2 vs L3 | laptop mode: hours |
| 03 Real data pipeline | atlas → `Dataset`; anatomy vs function; noise ceiling; unc-31 peptide test | CPU, 2 min |
| 04 E1 on real data | pre-registration, cross-validated ladder, registered decision, compression-cost curve | laptop mode: overnight |
| 05 E2 substrate transfer | fitted L* on RK4, Brian2 and fixed-point substrates; degeneracy; Lyapunov exponent | laptop mode: ~15 min |
| 06 E1b timing information | same synthetic trials analysed as window means vs time courses: does timing make L2/L3 detectable? real kernel timing | laptop mode: a few hours |
| 07 E1c identifiability | optimization vs identifiability; Fisher spectra; ensemble estimator; necessity bracket on real E1 | laptop mode: ~2 h |
| 08 E1d experimental design | which new interventions would constrain the deeper mechanisms? information-gain screening, synthetic validation, real-data screening | laptop mode: ~3–4 h |
| 09 E1e calibrated decision rule | one null-calibrated threshold replaces the sufficiency/necessity pair; recovery rates on test worms; 5-s pulse vs more of the same | laptop mode: ~4 h |
| 10 E3 program vs hardware | does a fixed program keep forecasting spontaneous activity, or does a compact meta-program (or only per-window rewriting) absorb the drift? synthetic validation + real recordings | laptop mode: ~30–60 min (+ real data) |
| 11 E3b what is the drift? | is drift concentrated in neurons whose behavioral encoding changed? does slow behavioral state explain it (vs time-shifted control)? | laptop mode: ~1–2 min per recording |
| 12 E3c form of the drift | which restricted update (drives, measurement, gains, low-rank or dense couplings) explains the drift, out of sample, against planted-drift surrogates? do the drives move together? does drift track raw-signal fading? | laptop mode: ~1–2 min per recording (68 recordings: ~1.5–2 h) |
| 13 E3d wander or return? | do the drives return to set points (OU, homeostasis) or random-walk? can one online error-driven tracker follow the drift? | laptop mode: ~1 min per recording (68 recordings: ~1–1.5 h); reuses Notebook 12's checkpoints for drift detection |
| 14 E3e drift or colored input? | are E3's gap, staleness slope and tracker gain larger than in stationary surrogates driven by the real residuals' fast (< 1 min) autocorrelation? | laptop mode: ~1–1.5 min per recording; reuses Notebooks 12–13 checkpoints |
| 15 E4a brain or brain + body? | does modelling the body as part of a closed system improve autonomous forecasts of the brain, beyond shifted behavior? does the observed body carry information the closed model cannot generate? | laptop mode: ~30–60 s per recording |
| 16 E2b spiking realization | how many spiking neurons (LIF, Poisson; NEF-compiled) does one graded unit of the fitted L1 need before its interventional responses match? | laptop mode: ~1–2 h (dominated by spiking LIF at M = 1000); needs the Notebook 04 fits |
| 17 E2c conductance synapses | does the fitted L1 survive synapses that act through reversal potentials (shunting), with naive or leak-compensated compilation, across voltage scales? | laptop mode: ~10–20 min; needs the Notebook 04 fits |
| 18 E2d slow intrinsic currents | how slow may a K+-like current in the substrate be (share ρ, time constant τ_n) before the fitted L1's interventional responses change? on current and conductance synapses | laptop mode: ~15–20 min; needs the Notebook 04 fits |
| 19 E1f adaptation revisited | does a two-number slow adaptation, chosen on training pairs, beat L1 on held-out atlas pairs (vs a gain control and an L1-generated null)? | laptop mode: ~15–25 min; needs the Notebook 04 fits |
| 20 E1g adaptation or under-fitted L1? | is Notebook 19's gain a better point inside the L1 family (rescaled L1 = slow current with a frozen gate), or genuine slow dynamics beyond it? | laptop mode: ~5–10 min (reuses Notebook 19's simulations); needs the Notebook 04 fits |
| 21 E1h consistent restart | Notebooks 19–20 redone with every model started from its exact resting state; how much do E1's predictions depend on the 30-s burn-in? | laptop mode: ~10–15 min; needs the Notebook 04 fits |
| 22 E1i fit, don't select | L1 refitted vs `L1s` (L1 + one global slow current) fitted with the same budget, from the selected slow current and from 'nearly off' | laptop mode: ~1–1.5 h (15 gradient fits of 300 steps); needs the Notebook 04 fits and reads Notebook 21's selections |

Long runs checkpoint after every fit to `results/`, so re-running resumes where a run stopped. Set `CLOSUREBENCH_RESULTS` to store results elsewhere.

## Package

```python
from closurebench import data, ladder, metrics, discrete
ds = data.load_atlas_dataset()                          # 300 neurons, 173 stimulated, WT + unc-31
preds, params, dl = metrics.run_crossvalidated_ladder(ds, protocol="pairs", checkpoint="results/ck")
point, boots = metrics.bootstrap_feve(preds, ds, list(preds))
print(metrics.format_decision(metrics.closure_decision(point, boots)))
```

## Data and citations

The atlas, connectomes, synapse-sign predictions and peptidergic connectome are redistributed by `wormneuroatlas` (Leifer lab). Cite the original papers: Randi et al. 2023 (*Nature*); White et al. 1986; Witvliet et al. 2021 (*Nature*); Ripoll-Sánchez et al. 2023 (*Neuron*); CeNGEN (Taylor et al. 2021); and see the package's `SEE_README_FOR_CITATIONS`. Theory: Rosas et al. 2024 (arXiv:2402.09090); Brette 2022, 2019.

## Known issues

* `NeuroAtlas.get_kernel` in `wormneuroatlas` 0.0.7.3 has a bug (`len(exp>0)`). Read kernels from `funatlas_h5` directly if you need them.
* The package's WormBase version check needs internet. `data.open_atlas()` disables it.
