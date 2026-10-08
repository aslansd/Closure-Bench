"""E3c - what KIND of change is the drift?

E3 (Notebook 10) found that a program fitted on the past goes stale, and that no
compact activity-driven meta-program absorbs the change.  E3b (Notebook 11)
found that the drift is neither concentrated in the neurons whose behavioral
encoding changed nor explained by slow behavioral state: it is diffuse.

This module asks which restricted change of the program best explains the
recent past, by re-estimating ONE family of parameters on the first half of a
forecast window (from the program fitted on the past) and scoring the second
half out of sample.  Out-of-sample scoring is what keeps richer families honest:
extra parameters help only if the change really has that form.

  family        what changes                                        new numbers
  bias          each neuron's drive (E3's "refit")                  N
  renorm        the MEASUREMENT: each trace's mean and scale         0 fitted to
                (bleaching, indicator or focus drift); the program    dynamics (2N
                is applied to traces mapped back to the past's        window moments)
                mean / sd
  rescale       measurement GAIN per neuron (traces rescaled to      N (+ N moments)
                the past's sd), then drives re-estimated
  gain_global   one gain on all recurrent input + drives            1 + N
  gain_neuron   a gain per neuron on its recurrent input + drives   2N
  lowrank1/2    a rank-1/2 change of the couplings + drives         ~2rN + N
  dense         an unstructured (ridge) change of all couplings     N^2 + N

Ridge strengths of the regularised families are chosen inside the first half
(fit on its first 70 %, validate on the rest), never on the scored data.

Surrogates (built from the frozen program fitted to the whole recording):
  null         stationary                                   (calibration)
  bias drift   drives do a random walk (drift.surrogate)    (positive control: bias)
  measurement  stationary dynamics, then a slow common      (what measurement drift
               bleach + per-neuron gain / offset random walks    looks like)
               applied to the TRACES

What the synthetic checks showed while building this (one recording):
  * multiplicative measurement drift (even a 60 % bleach) produces NO forecast
    drift: after z-scoring, a near-linear program is scale-equivariant;
  * additive offset drift does, and it is absorbed by `bias` just like a drive
    random walk: to first order an observation offset and a drive change are
    indistinguishable from the dynamics.  The measurement question is therefore
    answered from the raw traces (signal_trend), not from the families.
"""
from __future__ import annotations
import numpy as np
from . import drift as DR

FAMILIES = ("bias", "renorm", "rescale", "gain_global", "gain_neuron", "lowrank1", "lowrank2", "dense")
LAM_GRID = (1e-2, 1e-1, 1.0, 10.0)


# ----------------------------------------------------------- family fits ---
def _parts(p, X, U):
    """base = -a x + C u, H = W tanh(x) (recurrent input), T = tanh(x); all (len-1, N)."""
    base = -p["a"] * X[:-1] + DR._input_term(p, U)
    T = np.tanh(X[:-1])
    return base, T @ p["W"].T, T


def _split(n, frac=0.7):
    c = max(int(n * frac), 2)
    return slice(0, c), slice(c, n)


def _ridge_dW(Tt, R, lam):
    """Ridge solution of R ~ Tt dW^T (both demeaned): returns dW^T (N_in, N_out)."""
    n, k = Tt.shape
    return np.linalg.solve(Tt.T @ Tt + lam * n * np.eye(k), Tt.T @ R)


def _fit_lowrank(T, R, lam, rank):
    """Reduced-rank ridge: dW^T restricted to the top `rank` directions of the
    fitted values; biases are the mean remaining residual."""
    mT, mR = T.mean(0), R.mean(0)
    dWt = _ridge_dW(T - mT, R - mR, lam)
    if rank is not None and rank < dWt.shape[1]:
        Yh = (T - mT) @ dWt
        V = np.linalg.svd(Yh, full_matrices=False)[2][:rank].T          # (N_out, rank)
        dWt = dWt @ V @ V.T
    b = (R - T @ dWt).mean(0)
    return dWt, b


def _fit_gain_neuron(H, R, lam):
    """Per neuron: R_i ~ (g_i - 1) H_i + db_i, ridge on (g_i - 1)."""
    Hc, Rc = H - H.mean(0), R - R.mean(0)
    n = len(H)
    dg = (Hc * Rc).sum(0) / ((Hc ** 2).sum(0) + lam * n * (Hc ** 2).mean() + 1e-12)
    db = (R - dg * H).mean(0)
    return dg, db


def _choose_lam(fit_fn, score_fn, n):
    tr, va = _split(n)
    best, best_s = LAM_GRID[0], np.inf
    for lam in LAM_GRID:
        s = score_fn(fit_fn(tr, lam), va)
        if s < best_s:
            best, best_s = lam, s
    return best


def fit_family(fam, p, Xf, Uf, past_stats):
    """Re-estimate family `fam` of program p on segment Xf (with inputs Uf).
    Returns a function pred(Xs, Us) -> predicted increments (len(Xs) - 1, N)."""
    y = np.diff(Xf, axis=0)
    base, H, T = _parts(p, Xf, Uf)
    R = y - base - H                                     # what the frozen couplings leave (incl. drives)
    if fam == "bias":
        b = R.mean(0)
        return lambda Xs, Us: DR.predict_increments(p, Xs, b, Us)
    if fam == "renorm":
        mp, sp = past_stats
        mc, sc = Xf.mean(0), Xf.std(0) + 1e-9
        def pred(Xs, Us):
            Z = (Xs - mc) / sc * sp + mp                       # traces mapped back to the past's scale
            return DR.predict_increments(p, Z, U=Us) * (sc / sp)
        return pred
    if fam == "rescale":                                  # measurement gain per neuron, then drives
        mp, sp = past_stats
        mc, sc = Xf.mean(0), Xf.std(0) + 1e-9
        r = sp / sc
        Zf = (Xf - mc) * r + mc
        b = DR.fit_refit_biases(p, Zf, Uf)
        def pred(Xs, Us):
            Z = (Xs - mc) * r + mc
            return DR.predict_increments(p, Z, b, Us) / r
        return pred
    if fam == "gain_global":
        Hc, Rc = H - H.mean(0), R - R.mean(0)
        dg = float((Hc * Rc).sum() / ((Hc ** 2).sum() + 1e-12))
        db = (R - dg * H).mean(0)
        return lambda Xs, Us: _pred_with(p, Xs, Us, dg=dg, b=db)
    if fam == "gain_neuron":
        n = len(R)
        def err(par, sl):
            dg, db = par
            return float(((R[sl] - dg * H[sl] - db) ** 2).sum())
        lam = _choose_lam(lambda sl, l: _fit_gain_neuron(H[sl], R[sl], l), err, n)
        dg, db = _fit_gain_neuron(H, R, lam)
        return lambda Xs, Us: _pred_with(p, Xs, Us, dg=dg, b=db)
    if fam in ("lowrank1", "lowrank2", "dense"):
        rank = {"lowrank1": 1, "lowrank2": 2, "dense": None}[fam]
        n = len(R)
        def err(par, sl):
            dWt, db = par
            return float(((R[sl] - T[sl] @ dWt - db) ** 2).sum())
        lam = _choose_lam(lambda sl, l: _fit_lowrank(T[sl], R[sl], l, rank), err, n)
        dWt, db = _fit_lowrank(T, R, lam, rank)
        return lambda Xs, Us: _pred_with(p, Xs, Us, dWt=dWt, b=db)
    raise ValueError(fam)


def _pred_with(p, Xs, Us, dg=0.0, dWt=None, b=None):
    base, H, T = _parts(p, Xs, Us)
    out = base + (1.0 + dg) * H + b
    return out + T @ dWt if dWt is not None else out


# ------------------------------------------------------------ the analysis ---
def family_scores(X, S, U=None, n_windows=8, families=FAMILIES, ridge="auto"):
    """Forecast forward as in drift.analyse: for each window k >= n_windows/2 the
    program is fitted on windows < k, each family is re-estimated on the first half
    of window k and scored on its second half.  Pooled one-step R^2 per family,
    plus 'frozen'.  Also returns gap = bias - frozen (E3's drift gap) and
    absorbed[f] = (f - frozen) / gap."""
    S = DR._support(S)
    T, N = X.shape
    wins = DR.window_bounds(T, n_windows)
    Us = (lambda a, b: None) if U is None else (lambda a, b: U[a:b])
    sse = {f: 0.0 for f in ("frozen",) + tuple(families)}; sst = 0.0
    for k in range(n_windows // 2, n_windows):
        s0, e0 = wins[k]; mid = s0 + (e0 - s0) // 2
        past, Up = X[:s0], Us(0, s0)
        lam = DR.choose_ridge(past, S, Up) if ridge == "auto" else ridge
        p = DR.fit_frozen([past], S, lam, [Up] if Up is not None else None)
        stats = (past.mean(0), past.std(0) + 1e-9)
        Xs, Uss = X[mid:e0], Us(mid, e0)
        y = np.diff(Xs, axis=0)
        sst += ((y - y.mean(0)) ** 2).sum()
        sse["frozen"] += ((DR.predict_increments(p, Xs, U=Uss) - y) ** 2).sum()
        for f in families:
            pred = fit_family(f, p, X[s0:mid], Us(s0, mid), stats)
            sse[f] += ((pred(Xs, Uss) - y) ** 2).sum()
    score = {f: float(1 - v / sst) for f, v in sse.items()}
    gap = score["bias"] - score["frozen"] if "bias" in score else float("nan")
    absorbed = {f: (score[f] - score["frozen"]) / gap if gap > 1e-9 else float("nan") for f in families}
    return dict(score=score, gap=gap, absorbed=absorbed)


def bias_trajectory(X, S, U=None, n_windows=8, ridge="auto"):
    """Drives re-estimated per window under ONE program fitted to the whole
    recording; returns (B (n_windows, N), pc1_share): the share of the
    across-window drive variance on its first principal component.  1 = all
    neurons drift together along one direction (one slow global variable)."""
    S = DR._support(S)
    lam = DR.choose_ridge(X, S, U) if ridge == "auto" else ridge
    p = DR.fit_frozen([X], S, lam, [U] if U is not None else None)
    Us = (lambda a, b: None) if U is None else (lambda a, b: U[a:b])
    B = np.array([DR.fit_refit_biases(p, X[a:b], Us(a, b)) for a, b in DR.window_bounds(len(X), n_windows)])
    s = np.linalg.svd(B - B.mean(0), compute_uv=False)
    return B, float(s[0] ** 2 / (s ** 2).sum())


# ------------------------------------------------------------- surrogates ---
def measurement_surrogate(X, S, U=None, seed=0, bleach=0.3, gain_sd=0.004, offset_sd=0.03, ridge="auto"):
    """Stationary dynamics (drift.surrogate) seen through a drifting measurement:
    a common exponential bleach (amplitude falls by `bleach` over the recording,
    each neuron's rate scaled by U(0.5, 1.5)) times a per-neuron log-gain random walk
    (step gain_sd), plus a per-neuron offset random walk (step offset_sd, in units
    of the trace sd).  Returns the z-scored traces, or None if the generator is unstable."""
    Y = DR.surrogate(X, S, U, ridge=ridge, seed=seed)
    if Y is None:
        return None
    rng = np.random.default_rng(seed + 7919)
    T, N = Y.shape
    t = np.arange(T)[:, None] / T
    rate = -np.log(1 - bleach) * rng.uniform(0.5, 1.5, N)
    g = np.exp(-rate * t + np.cumsum(gain_sd * rng.normal(size=(T, N)), 0))
    o = np.cumsum(offset_sd * rng.normal(size=(T, N)), 0)
    Z = Y * g + o
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-9)


# ------------------------------------------------------ measurement trends ---
def signal_trend(path, n_windows=8):
    """From the UNnormalised traces (gcamp.trace_array_original, if present):
    median over neurons of the ratio last / first window of (a) the mean signal and
    (b) the within-window sd (amplitude).  Bleaching lowers both.
    Returns dict(mean_ratio, sd_ratio, mean_by_window, sd_by_window) or None."""
    import json
    d = json.load(open(str(path)))
    g = d.get("gcamp", {}) if isinstance(d.get("gcamp"), dict) else {}
    if "trace_array_original" not in g:
        return None
    r = np.asarray(g["trace_array_original"], float)
    if r.shape[0] > r.shape[1]:
        r = r.T
    r = r[~np.all(np.isnan(r), axis=1)]
    idx = np.array_split(np.arange(r.shape[1]), n_windows)
    m = np.array([np.nanmean(r[:, i], 1) for i in idx])            # (n_windows, N)
    s = np.array([np.nanstd(r[:, i], 1) for i in idx])
    with np.errstate(divide="ignore", invalid="ignore"):
        mr = float(np.nanmedian(m[-1] / m[0])); sr = float(np.nanmedian(s[-1] / s[0]))
    return dict(mean_ratio=mr, sd_ratio=sr, mean_by_window=np.nanmedian(m, 1).tolist(),
                sd_by_window=np.nanmedian(s, 1).tolist())
