"""Run modes and runtime estimation, so the same notebooks run on Colab GPUs,
a laptop CPU (e.g. Apple Silicon), or as a quick smoke test.

Mode is chosen by the environment variable CLOSUREBENCH_MODE, or set in a
notebook's config cell:

    smoke  : tiny problem, minutes; checks that the code runs (numbers meaningless)
    laptop : sized for a recent laptop CPU (e.g. M1, 16 GB); real but smaller runs,
             typically overnight; results are checkpointed and resumable
    full   : the proposal-scale run (all 300 neurons, 5 folds, 3 seeds); GPU or
             a long multi-day CPU run

Default ("auto"): full on a GPU, laptop on a CPU.
"""
from __future__ import annotations
import os
import time
import platform
import numpy as np

MODES = ("smoke", "laptop", "full")


def get_mode(override=None):
    import jax
    m = (override or os.environ.get("CLOSUREBENCH_MODE", "auto")).lower()
    if m == "auto":
        m = "full" if jax.devices()[0].platform != "cpu" else "laptop"
    if m not in MODES:
        raise ValueError(f"CLOSUREBENCH_MODE must be one of {MODES} or 'auto', got {m!r}")
    return m


def describe_machine():
    import jax
    d = jax.devices()[0]
    return (f"{platform.system()} {platform.machine()} | python {platform.python_version()} | "
            f"jax {jax.__version__} on {d.platform} ({d.device_kind}) | CPU cores: {os.cpu_count()}")


def _time_fit(level, ds, n_steps, st, cfg, protocol, seed=0):
    """Run a short fit and return (compile_seconds, seconds_per_step) measured
    step by step inside ONE fit (first step = compile + run; median of the rest)."""
    from . import ladder as lad
    from .data import Dataset
    if not np.any(ds.mean):   # e.g. a synthetic template: timing needs non-zero dummy targets
        rng = np.random.default_rng(0)
        ds = Dataset(**{**ds.__dict__, "mean": rng.normal(0, 0.1, ds.mean.shape),
                        "var_mean": np.full(ds.mean.shape, 1e-3)})
    kw = {}
    if protocol == "pairs":
        kw["train_mask"] = np.ones(ds.mask().shape, bool)
    else:
        kw["train_cols"] = np.arange(ds.S)[: max(1, int(ds.S * 0.8))]
    t0 = time.perf_counter()
    lad.fit(level, ds, steps=n_steps, verbose=0, st=st, cfg=cfg, seed=seed, **kw)
    t = np.asarray(lad.LAST_FIT_TIMES)
    dt = np.diff(t)
    per_step = float(np.median(dt[1:])) if len(dt) > 1 else float(dt[0])
    compile_s = max(float(t[1] - t0) - per_step, 0.0)   # includes init / gain calibration
    return compile_s, per_step


def estimate_runtime(ds, levels=("L0", "L1", "L2", "L3", "L4", "B"), k=5, steps=500, restarts=3,
                     equal_budget=True, protocol="pairs", cfg=None, tie_classes=True, n_runs=1,
                     probe=8, verbose=True):
    """Project the wall time of `metrics.run_crossvalidated_ladder` on THIS
    machine by timing short fits of every rung (compile time + per-step time),
    then counting the fits and steps of the equal-budget scheme.
    `n_runs` multiplies the total (e.g. number of synthetic truths x seeds).
    Returns (hours, per-level table)."""
    from . import ladder as lad
    cfg = lad.SimConfig() if cfg is None else cfg
    st = lad.Structure.from_dataset(ds, tie=tie_classes)
    chain = [L for L in ("L1", "L2", "L3", "L4") if L in levels]
    T = steps * max(len(chain), 1) if equal_budget else steps
    rows, total = [], 0.0
    for L in levels:
        compile_s, per_step = _time_fit(L, ds, probe, st, cfg, protocol)
        if L in chain:
            depth = chain.index(L) + 1
            n_fit_chain = restarts if (L == "L1" and restarts > 1) else 1
            remaining = max(T - depth * steps, 0)
            n_steps = n_fit_chain * steps + remaining
            n_fits = n_fit_chain + (remaining > 0)
        else:
            n_steps, n_fits = T, 1
        sec = k * (n_fits * compile_s + n_steps * per_step)
        rows.append((L, per_step * 1000, compile_s, n_steps * k, sec / 60))
        total += sec
    hours = total * n_runs / 3600
    if verbose:
        print(f"{'rung':5s} {'ms/step':>9s} {'compile s':>10s} {'steps (all folds)':>18s} {'minutes':>9s}")
        for L, ms, cs, ns, mins in rows:
            print(f"{L:5s} {ms:9.1f} {cs:10.1f} {ns:18d} {mins:9.1f}")
        print(f"estimated total: {hours:.1f} h" + (f"  ({n_runs} runs)" if n_runs > 1 else "")
              + "  -- rough (+/-50%); checkpoints make it resumable")
    return hours, rows
