"""E4a - where does the closed level live: the brain alone, or brain + body?

Proposal E4 asks whether the self-contained ("closed") description of the worm's
activity lives in the nervous system alone, or needs the body (and the
environment) as part of the system.  With whole-brain recordings of freely moving
worms (Atanas et al. 2023: neural activity + velocity, head angle, angular velocity,
pumping) the first half of the question can be asked:

  brain     neural program alone, no behavior        dx = -a x + W tanh(x) + b
  joint     brain + body as ONE closed system        dx = ... + C u
            (body state u predicted, not observed)   du = A_x tanh(x) + A_u u + c
  open      brain with the OBSERVED body as input    dx = ... + C u_observed
            (not closed: a reference for what perfect body prediction would give)
  body-AR   the body alone                           du = A_u u + c

All are rolled out autonomously for k steps from the observed state at time t, and
scored on the k-step change of neural activity (x_{t+k} - x_t), pooled R^2, with
interleaved-block cross-validation (training and test blocks spread over the whole
recording, so the slow drift of Notebooks 10-14 affects them equally).  If `joint` forecasts the brain
better than `brain` at horizons of seconds, beyond a control in which behavior is
circularly shifted by 0.37 of the recording (same statistics, no alignment), the body is part of what closes
the brain's dynamics: the closed level includes the body.

What cannot be asked here: the environment (no record of it), and causal direction
(the body helping forecasts does not prove sensory feedback rather than shared
drive) -- that needs closed-loop / perturbation experiments (E4b).
"""
from __future__ import annotations
import numpy as np
from . import drift as DR

HORIZONS = (1, 5, 20)
LAM_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _ridge(F, Y, lam):
    n = len(F)
    reg = lam * n * np.eye(F.shape[1]); reg[-1, -1] = 0.0           # last column = constant
    return np.linalg.solve(F.T @ F + reg, F.T @ Y)


def _body_feats(X, U, use_brain=True):
    cols = ([np.tanh(X)] if use_brain else []) + [U, np.ones((len(U), 1))]
    return np.column_stack(cols)


def fit_body(X_list, U_list, use_brain=True, val_frac=0.2):
    """du = [A_x tanh(x)] + A_u u + c by ridge over the segments given; strength
    chosen on the last val_frac of the stacked rows.  Returns coefficients (features, k)."""
    F = np.vstack([_body_feats(X[:-1], U[:-1], use_brain) for X, U in zip(X_list, U_list)])
    Y = np.vstack([np.diff(U, axis=0) for U in U_list])
    cut = int(len(F) * (1 - val_frac))
    best = min(LAM_GRID, key=lambda l: ((F[cut:] @ _ridge(F[:cut], Y[:cut], l) - Y[cut:]) ** 2).sum())
    return _ridge(F, Y, best)


def fit_models(X_list, U_list, S):
    """Fit brain / brain-with-inputs / body (from brain + body) / body-AR on the
    training segments.  Ridge strengths of the neural programs are chosen on the
    concatenated segments (past-only split inside them)."""
    S = DR._support(S)
    Xc, Uc = np.concatenate(X_list), np.concatenate(U_list)
    brain = DR.fit_frozen(X_list, S, DR.choose_ridge(Xc, S, None), None)
    withu = DR.fit_frozen(X_list, S, DR.choose_ridge(Xc, S, Uc), U_list)
    return dict(brain=brain, withu=withu, body=fit_body(X_list, U_list, True), body_ar=fit_body(X_list, U_list, False))


def rollout_scores(P, X, U, ks=HORIZONS):
    """Autonomous k-step rollouts from every time t of one test segment (X, U).
    Returns {model: {k: (sse, sst)}} for the neural k-step change (brain, joint,
    open) and the body's k-step change (body_joint, body_ar).  sst is taken about
    the mean of the k-step change within the segment."""
    T, N = X.shape; kmax = max(ks)
    starts = np.arange(0, T - kmax)
    out = {m: {} for m in ("brain", "joint", "open", "body_joint", "body_ar")}
    def neural_step(p, x, u):
        d = -p["a"] * x + np.tanh(x) @ p["W"].T + p["b"]
        return x + (d + u @ p["C"].T if u is not None else d)
    xb = X[starts].copy(); xj = X[starts].copy(); xo = X[starts].copy()
    uj = U[starts].copy(); ua = U[starts].copy()
    for step in range(1, kmax + 1):
        xb = neural_step(P["brain"], xb, None)
        xj, uj = neural_step(P["withu"], xj, uj), uj + _body_feats(xj, uj, True) @ P["body"]
        xo = neural_step(P["withu"], xo, U[starts + step - 1])
        ua = ua + _body_feats(xj, ua, False) @ P["body_ar"]
        for arr in (xb, xj, xo, uj, ua):
            np.clip(arr, -50, 50, out=arr)
        if step in ks:
            y = X[starts + step] - X[starts]
            sst = ((y - y.mean(0)) ** 2).sum()
            for m, x in (("brain", xb), ("joint", xj), ("open", xo)):
                out[m][step] = (float(((x - X[starts] - y) ** 2).sum()), float(sst))
            yu = U[starts + step] - U[starts]; sstu = ((yu - yu.mean(0)) ** 2).sum()
            out["body_joint"][step] = (float(((uj - U[starts] - yu) ** 2).sum()), float(sstu))
            out["body_ar"][step] = (float(((ua - U[starts] - yu) ** 2).sum()), float(sstu))
    return out


def embodiment_scores(X, U, S, ks=HORIZONS, shift_frac=None, n_blocks=10, n_folds=5):
    """Cross-validated k-step R^2 of every model with INTERLEAVED blocks: the
    recording is cut into n_blocks blocks and block i goes to fold i mod n_folds,
    so training and test data are spread over the whole recording and slow drift
    (Notebooks 10-14) affects them equally.  shift_frac: circularly shift the
    behavior by this fraction first (control; avoid 0.5, which maps blocks onto
    blocks of the same fold).  Returns {model: {k: R^2}} (pooled over folds)."""
    if shift_frac is not None:
        U = DR.circular_shift(U, shift_frac)
    blocks = DR.window_bounds(len(X), n_blocks)
    acc = {}
    for f in range(n_folds):
        tr = [b for i, b in enumerate(blocks) if i % n_folds != f]
        te = [b for i, b in enumerate(blocks) if i % n_folds == f]
        P = fit_models([X[a:b] for a, b in tr], [U[a:b] for a, b in tr], S)
        for a, b in te:
            if b - a <= max(ks) + 1:
                continue
            for m, d in rollout_scores(P, X[a:b], U[a:b], ks).items():
                for k, (e, s_) in d.items():
                    c = acc.setdefault(m, {}).setdefault(k, [0.0, 0.0]); c[0] += e; c[1] += s_
    return {m: {k: 1 - e / s_ for k, (e, s_) in d.items()} for m, d in acc.items()}
