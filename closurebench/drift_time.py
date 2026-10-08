"""E3d - the time structure of the drift: do the drives wander or return?

E3c (Notebook 12) found that the drift in spontaneous activity has the profile of
slow, per-neuron changes of each neuron's drive, with no detectable gain,
coupling, shared (low-dimensional) or measurement structure.  What imaging can
still tell about such drift is its TIME structure:

  wander   the drives do a random walk: no set point within the recording;
           nothing in the system pulls them back (no homeostasis at this scale)
  return   the drives fluctuate around set points with a finite time constant tau
           (an Ornstein-Uhlenbeck process): a slower, regulating layer (the
           ladder's L4, homeostasis) keeps the hardware near a fixed program

Tools
  drive_series(X, S, U, n_windows)  drives re-estimated in n_windows windows under
                                    one program fitted on the whole recording (K, N)
  variogram(B)                      semivariance of the drives vs lag (in windows),
                                    pooled over neurons; robust to estimation noise
                                    (which only adds a constant "nugget")
  fit_variogram(V, win_min)         nugget + sill * (1 - exp(-h / tau)) by weighted
                                    non-negative least squares over a tau grid;
                                    tau at the top of the grid = indistinguishable
                                    from a random walk
  ou_surrogate(...)                 frozen program + OU drives with time constant
                                    tau (positive control for "return")
  tracker_scores(...)               can an online, error-driven drive tracker
                                    (one time constant, chosen on the past) forecast
                                    as well as the per-window refit?
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import nnls
from . import drift as DR

TAU_GRID_MIN = np.geomspace(0.5, 240.0, 60)
TRACK_TAUS = (16, 32, 64, 128, 256, 512, 1024)          # tracker time constants, frames


# ------------------------------------------------------- drive time series ---
def drive_series(X, S, U=None, n_windows=16, ridge="auto"):
    """Drives (biases) of each of n_windows windows, estimated JOINTLY with one
    program (a, W, C) by the within-window (fixed-effects) estimator: features and
    increments are demeaned inside each window before the couplings are fitted, so
    the couplings are identified from fast fluctuations only and cannot absorb slow
    drift between windows; each window's drive is then its mean residual.
    (A program fitted on the whole recording WITHOUT window drives absorbs much of
    a slow drive drift through its self- and coupling terms -- each neuron's own
    level tracks its drive -- and the drift then disappears from the residuals.)
    Returns B (n_windows, N)."""
    S = DR._support(S)
    lam = DR.choose_ridge(X, S, U) if ridge == "auto" else ridge
    T, N = X.shape
    edges = DR.window_bounds(T, n_windows)
    lab = np.zeros(T - 1, int)
    for k, (a, b) in enumerate(edges):
        lab[a:max(b - 1, a)] = k
    lab[edges[-1][0]:] = n_windows - 1
    dX = np.diff(X, axis=0)
    B = np.zeros((n_windows, N))
    for i in range(N):
        F = DR.design_rows(X, i, S, U)[:, :-1]                     # drop the constant
        y = dX[:, i]
        Fm = np.zeros_like(F); ym = np.zeros_like(y)
        for k in range(n_windows):
            m = lab == k
            Fm[m] = F[m] - F[m].mean(0); ym[m] = y[m] - y[m].mean()
        w = np.linalg.solve(Fm.T @ Fm + lam * len(y) * np.eye(F.shape[1]), Fm.T @ ym)
        r = y - F @ w
        B[:, i] = [r[lab == k].mean() for k in range(n_windows)]
    return B


def variogram(B):
    """Semivariance gamma(h) = 0.5 * mean over neurons and window pairs of
    (b_{k+h} - b_k)^2, for h = 1..K-1.  Returns (V (K-1,), n_pairs (K-1,))."""
    K = len(B)
    V = np.array([0.5 * np.mean((B[h:] - B[:-h]) ** 2) for h in range(1, K)])
    return V, np.arange(K - 1, 0, -1)


def fit_variogram(V, win_min, n_pairs=None, taus=TAU_GRID_MIN, tau_fast=None):
    """Fit the drives' semivariance by weighted non-negative least squares
    (weights = number of window pairs per lag) over a grid of drift time constants
    tau (minutes):

        gamma(h) = nugget + s_fast * (1 - exp(-h / tau_fast)) + sill * (1 - exp(-h / tau))

    nugget: estimation noise; fast term: the program's own slow fluctuations, with
    tau_fast taken from the recording's STATIONARY surrogates (omitted if None);
    sill, tau: the drift (an OU process; tau at the top of the grid = a random walk).
    Returns dict(tau, nugget, s_fast, sill, sse, at_edge)."""
    h = np.arange(1, len(V) + 1) * win_min
    w = np.sqrt(np.ones_like(V) if n_pairs is None else np.asarray(n_pairs, float))
    cols = [np.ones_like(h)] + ([1 - np.exp(-h / tau_fast)] if tau_fast else [])
    best = None
    for tau in taus:
        A = np.column_stack(cols + [1 - np.exp(-h / tau)])
        coef, r = nnls(A * w[:, None], V * w)
        if best is None or r < best[0]:
            best = (r, tau, coef)
    r, tau, coef = best
    return dict(tau=float(tau), nugget=float(coef[0]), s_fast=float(coef[1]) if tau_fast else 0.0,
                sill=float(coef[-1]), sse=float(r ** 2), at_edge=bool(tau >= taus[-2]))


def growth_ratio(V, early=(2, 3, 4), late=None):
    """Model-free shape statistic: mean semivariance at the late lags (last third)
    divided by that at early lags (h = 2-4 windows).  ~1: drift saturates quickly
    (return); large: keeps growing (wander)."""
    late = range(2 * len(V) // 3, len(V)) if late is None else late
    return float(np.mean([V[h - 1] for h in late]) / np.mean([V[h - 1] for h in early]))


# -------------------------------------------------------------- surrogates ---
def ou_surrogate(X, S, U=None, dt=0.637, tau_min=3.0, drift_frac=0.07, ridge="auto", seed=0):
    """Like drift.surrogate(drift_frac > 0), but each neuron's drive is an
    Ornstein-Uhlenbeck process with time constant tau_min (minutes), scaled so its
    stationary sd equals the sd that the random walk of drift.surrogate reaches at
    half the recording (same drift size, different time structure).
    Returns the z-scored surrogate, or None if unstable."""
    rng = np.random.default_rng(seed)
    S = DR._support(S)
    lam = DR.choose_ridge(X, S, U) if ridge == "auto" else ridge
    p = DR.fit_frozen([X], S, lam, [U] if U is not None else None)
    sd = (np.diff(X, axis=0) - DR.predict_increments(p, X, U=U)).std(0)
    T = len(X); tau_f = tau_min * 60.0 / dt
    rho = np.exp(-1.0 / tau_f)
    stat_sd = drift_frac * sd * np.sqrt(T / 2.0)
    q = stat_sd * np.sqrt(1 - rho ** 2)
    Y = np.zeros_like(X); y = X[0].copy(); c = stat_sd * rng.normal(size=X.shape[1])
    for t in range(T):
        Y[t] = y
        u = 0.0 if U is None or p.get("C") is None else p["C"] @ U[t]
        y = y + (-p["a"] * y + p["W"] @ np.tanh(y) + p["b"] + c + u) + sd * rng.normal(size=y.shape)
        c = rho * c + q * rng.normal(size=c.shape)
        if not np.all(np.isfinite(y)) or np.abs(y).max() > 1e3:
            return None
    return (Y - Y.mean(0)) / (Y.std(0) + 1e-9)


# ----------------------------------------------------------------- tracker ---
def _ema_shifted(E, tau):
    """m_t = exponential moving average of E[:t] (strictly past), time constant tau frames."""
    a = 1.0 / tau; M = np.zeros_like(E); m = np.zeros(E.shape[1])
    for t in range(len(E)):
        M[t] = m; m = (1 - a) * m + a * E[t]
    return M


def tracker_scores(X, S, U=None, n_windows=8, taus=TRACK_TAUS, ridge="auto", val_frac=0.3):
    """Forecast forward as in drift.analyse (origins k >= n_windows/2; score on the
    second half of window k).  Models:
      frozen   program fitted on the past
      refit    + drives re-estimated on the first half of window k (E3's refit)
      tracker  + a per-neuron drive that follows the program's own past prediction
               errors with one time constant (exponential moving average), run
               continuously up to each step; its time constant is chosen on the
               past only: program fitted on the first (1 - val_frac) of the
               past, tracker scored on the rest.
    The tracker has no per-window refit: one rule, applied online.
    Returns dict(frozen, refit, tracker, gap, absorbed, taus_chosen)."""
    S = DR._support(S)
    T, N = X.shape
    wins = DR.window_bounds(T, n_windows)
    Us = (lambda a, b: None) if U is None else (lambda a, b: U[a:b])
    dX = np.diff(X, axis=0)
    sse = dict(frozen=0.0, refit=0.0, tracker=0.0); sst = 0.0; chosen = []
    for k in range(n_windows // 2, n_windows):
        s0, e0 = wins[k]; mid = s0 + (e0 - s0) // 2
        past, Up = X[:s0], Us(0, s0)
        lam = DR.choose_ridge(past, S, Up) if ridge == "auto" else ridge
        p = DR.fit_frozen([past], S, lam, [Up] if Up is not None else None)
        P = DR.predict_increments(p, X, U=U)                       # (T-1, N), whole recording
        E = dX - P
        # choose the time constant honestly: a program fitted on the EARLIER part of
        # the past, tracker scored on the later part (in-sample residuals have zero
        # mean, which would always favour the slowest tracker)
        cut = int(s0 * (1 - val_frac))
        pv = DR.fit_frozen([X[:cut]], S, lam, [Us(0, cut)] if U is not None else None)
        Ev = dX[:s0 - 1] - DR.predict_increments(pv, X[:s0], U=Us(0, s0))
        def val_err(tau):
            M = _ema_shifted(Ev, tau)
            return float(((Ev[cut:] - M[cut:]) ** 2).sum())
        tau = min(taus, key=val_err); chosen.append(int(tau))
        M = _ema_shifted(E[:e0 - 1], tau)
        y = dX[mid:e0 - 1]
        sst += ((y - y.mean(0)) ** 2).sum()
        sse["frozen"] += ((P[mid:e0 - 1] - y) ** 2).sum()
        b = DR.fit_refit_biases(p, X[s0:mid], Us(s0, mid))
        sse["refit"] += ((DR.predict_increments(p, X[mid:e0], b, Us(mid, e0)) - y) ** 2).sum()
        sse["tracker"] += ((P[mid:e0 - 1] + M[mid:e0 - 1] - y) ** 2).sum()
    sc = {m: float(1 - v / sst) for m, v in sse.items()}
    gap = sc["refit"] - sc["frozen"]
    return dict(**sc, gap=gap, absorbed=(sc["tracker"] - sc["frozen"]) / gap if gap > 1e-9 else float("nan"),
                taus_chosen=chosen)


# --------------------------------------- colored-residual control (NB 14) ---
def colored_surrogate(X, S, U=None, cut_frames=100, ridge="auto", seed=0):
    """Stationary surrogate whose residual noise has the real recording's FAST
    autocorrelation.  The residuals of the frozen program fitted to the whole
    recording are high-passed (components slower than `cut_frames` frames removed:
    no drift), phase-randomised per neuron (same power spectrum above the cutoff,
    new phases), and used to drive the frozen program instead of white noise.
    Tests how much the online tracker gains from autocorrelated residuals alone.
    Returns the z-scored surrogate, or None if unstable."""
    rng = np.random.default_rng(seed)
    S = DR._support(S)
    lam = DR.choose_ridge(X, S, U) if ridge == "auto" else ridge
    p = DR.fit_frozen([X], S, lam, [U] if U is not None else None)
    R = np.diff(X, axis=0) - DR.predict_increments(p, X, U=U)
    n = len(R); F = np.fft.rfft(R - R.mean(0), axis=0)
    f = np.fft.rfftfreq(n)                                          # cycles per frame
    F[f < 1.0 / cut_frames] = 0.0
    ph = np.exp(2j * np.pi * rng.uniform(size=F.shape)); ph[0] = 1.0
    E = np.fft.irfft(np.abs(F) * ph, n=n, axis=0)
    Y = np.zeros_like(X); y = X[0].copy()
    for t in range(len(X)):
        Y[t] = y
        if t == n:
            break
        u = 0.0 if U is None or p.get("C") is None else p["C"] @ U[t]
        y = y + (-p["a"] * y + p["W"] @ np.tanh(y) + p["b"] + u) + E[t]
        if not np.all(np.isfinite(y)) or np.abs(y).max() > 1e3:
            return None
    return (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
