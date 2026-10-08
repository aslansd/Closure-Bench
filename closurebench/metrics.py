"""E1 metrics: noise-ceiling-normalised accuracy on held-out stimulations,
bootstrap confidence intervals, and the pre-registered closure decision rule.

FEVE (fraction of explainable variance explained), pooled over observed pairs:

    FEVE = 1 - (SSE - sum noise) / (SS_resp - sum noise)

where SSE = sum (pred - mean)^2, SS_resp = sum mean^2 (null model: no
response), and noise = variance of each trial mean.  A model that matches the
true response exactly scores ~1 (the noise ceiling) up to sampling error.
"""
from __future__ import annotations
import numpy as np


def feve(pred, ds, cols=None, strain=None, min_trials=1, extra_mask=None):
    """Pooled FEVE over observed pairs of the given stimulus columns/strains.
    `extra_mask` (n_strain,N,S) or (N,S) restricts further (e.g. significant pairs)."""
    m = ds.mask(min_trials).copy()
    if extra_mask is not None:
        m &= np.broadcast_to(extra_mask, m.shape)
    m = np.broadcast_to(m.reshape(m.shape + (1,) * (ds.mean.ndim - 3)), ds.mean.shape).copy()
    if cols is not None:
        keep = np.zeros(ds.S, bool)
        keep[np.asarray(cols)] = True
        m &= keep.reshape((1, 1, ds.S) + (1,) * (m.ndim - 3))
    if strain is not None:
        k = ds.strains.index(strain)
        m[[i for i in range(len(ds.strains)) if i != k]] = False
    y, nv = ds.mean[m], ds.var_mean[m]
    sse = ((np.asarray(pred)[m] - y) ** 2).sum()
    ss = (y ** 2).sum()
    return 1.0 - (sse - nv.sum()) / max(ss - nv.sum(), 1e-12)


def per_column_sums(pred, ds, strain=None, min_trials=1, extra_mask=None):
    """Per-stimulus (SSE, SS_resp, noise) so FEVE can be bootstrapped over stimuli."""
    m = ds.mask(min_trials).copy()
    if extra_mask is not None:
        m &= np.broadcast_to(extra_mask, m.shape)
    if strain is not None:
        k = ds.strains.index(strain)
        m[[i for i in range(len(ds.strains)) if i != k]] = False
    pred = np.asarray(pred)
    m = m.reshape(m.shape + (1,) * (ds.mean.ndim - 3))           # broadcast over time bins
    ax = tuple(a for a in range(ds.mean.ndim) if a != 2)         # sum over everything but the stimulus axis
    sse = (((pred - ds.mean) ** 2) * m).sum(ax)
    ss = ((ds.mean ** 2) * m).sum(ax)
    nv = (ds.var_mean * m).sum(ax)
    return np.stack([sse, ss, nv], 1)  # (S, 3)


def feve_from_sums(sums):
    sse, ss, nv = sums.sum(0)
    return 1.0 - (sse - nv) / max(ss - nv, 1e-12)


def kfold_columns(S, k=5, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(S)
    return [np.sort(f) for f in np.array_split(perm, k)]


def kfold_pairs(ds, k=5, seed=0):
    """Assign every observed (responder, stimulus) pair -- the same pair in all
    strains together -- to one of k folds.  Returns list of (n_strain,N,S) bool masks."""
    rng = np.random.default_rng(seed)
    obs = ds.mask().any(0)
    fold_id = np.where(obs, rng.integers(0, k, obs.shape), -1)
    return [np.broadcast_to(fold_id == f, ds.mask().shape).copy() for f in range(k)]


def bootstrap_feve(heldout_preds, ds, levels, n_boot=2000, seed=0, strain=None, extra_mask=None):
    """heldout_preds[level] = (n_strain, N, S) cross-validated predictions
    (every column predicted by a model that never saw it).  Resamples
    stimulated neurons with replacement.  Returns dict level -> (n_boot,) and
    point estimates."""
    rng = np.random.default_rng(seed)
    sums = {L: per_column_sums(heldout_preds[L], ds, strain, extra_mask=extra_mask) for L in levels}
    S = ds.S
    boots = {L: np.empty(n_boot) for L in levels}
    for b in range(n_boot):
        idx = rng.integers(0, S, S)
        for L in levels:
            boots[L][b] = feve_from_sums(sums[L][idx])
    point = {L: feve_from_sums(sums[L]) for L in levels}
    return point, boots


def closure_decision(point, boots, ladder=("L0", "L1", "L2", "L3", "L4"), threshold=0.05,
                     blackbox="B"):
    """Pre-registered rule (proposal E1).

    L* = the lowest rung L such that (i) for every deeper rung L', the upper end
    of the bootstrap 95% CI of FEVE(L') - FEVE(L) is below `threshold` (5% of the
    noise ceiling), and (ii) FEVE(L) itself is significantly above 0 (lower CI
    bound > 0): a level that predicts nothing cannot be "the computation".
    If no rung qualifies, closure is not reached within the ladder.  Separately reports whether the black box beats L* by > threshold
    (simulable-but-not-closed signature, outcome C).
    """
    rows = []
    L_star = None
    for i, L in enumerate(ladder):
        deeper = ladder[i + 1:]
        if not deeper:
            ub = -np.inf
        else:
            ub = max(np.percentile(boots[D] - boots[L], 97.5) for D in deeper)
        lo = np.percentile(boots[L], 2.5)
        rows.append((L, point[L], ub))
        if L_star is None and ub < threshold and lo > 0:
            L_star = L
    out = {"L_star": L_star, "table": rows}
    if L_star is None:
        best = max(ladder, key=lambda L: point[L])
        out["note"] = ("no rung qualifies" if np.percentile(boots[best], 2.5) > 0
                       else "no rung predicts held-out responses better than zero (uninformative test)")
    if blackbox in boots and L_star is not None:
        d = boots[blackbox] - boots[L_star]
        out["blackbox_minus_Lstar"] = (float(np.mean(d)), float(np.percentile(d, 2.5)),
                                       float(np.percentile(d, 97.5)))
        out["blackbox_beats_Lstar"] = bool(np.percentile(d, 2.5) > threshold)
    return out


def necessity(point, boots, ladder=("L0", "L1", "L2", "L3", "L4")):
    """Rule v3 companion to closure_decision: is a rung *needed*?

    For each rung L (beyond the first), the bootstrap distribution of
    FEVE(L) - max(FEVE of every shallower rung), computed per bootstrap sample.
    L is needed if the 2.5th percentile of that gain is > 0, i.e. it beats every
    shallower rung significantly.  L_needed = the deepest needed rung (or the
    first rung if none is).  Together with the sufficiency rule
    (closure_decision's L*), it brackets the closed level:
    L_needed <= closed level <= L*.  A wide bracket means the data cannot decide.
    """
    rows, L_needed = [], ladder[0]
    for i, L in enumerate(ladder[1:], start=1):
        best_shallow = np.max(np.stack([boots[S] for S in ladder[:i]]), axis=0)
        d = boots[L] - best_shallow
        lo, hi = np.percentile(d, [2.5, 97.5])
        needed = bool(lo > 0)
        rows.append((L, float(np.mean(d)), float(lo), float(hi), needed))
        if needed:
            L_needed = L
    return {"L_needed": L_needed, "table": rows}


def format_necessity(nec):
    lines = ["rung   gain over best shallower rung [95% CI]   needed?"]
    for L, mu, lo, hi, nd in nec["table"]:
        lines.append(f"{L:5s}  {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]               {'yes' if nd else 'no'}")
    lines.append(f"=> deepest needed rung L_needed = {nec['L_needed']}")
    return "\n".join(lines)


def run_ensemble_ladder(ds, n_members=3, seed=0, checkpoint=None, log=print, **kw):
    """Ensemble estimator: fit the whole cross-validated ladder n_members times
    on IDENTICAL folds from different random initialisations (member m uses
    seed 10000*m, i.e. widely spread starting points), and average each rung's
    held-out predictions.  Equally good but non-identified fits err in
    different directions; their average estimates the rung's *function*
    rather than one arbitrary parameter set.  Returns (mean_preds, member_preds, dl)."""
    import os
    members, dl = [], None
    for mi in range(n_members):
        ck = os.path.join(checkpoint, f"member{mi}") if checkpoint else None
        preds, _, dl = run_crossvalidated_ladder(ds, seed=seed + 10000 * mi, fold_seed=seed, checkpoint=ck, log=log, **kw)
        members.append(preds)
    mean = {L: np.mean([mb[L] for mb in members], axis=0) for L in members[0]}
    return mean, members, dl


def format_decision(dec):
    lines = ["rung   FEVE(held-out)   max upper-CI gain of any deeper rung"]
    for L, f, ub in dec["table"]:
        ubs = "   —" if not np.isfinite(ub) else f"{ub:+.3f}"
        lines.append(f"{L:5s}  {f:8.3f}         {ubs}")
    lines.append(f"=> closed level L* = {dec['L_star']}" + (f"   ({dec['note']})" if dec.get("note") else ""))
    if "blackbox_minus_Lstar" in dec:
        m, lo, hi = dec["blackbox_minus_Lstar"]
        lines.append(f"   B - L*: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  -> black box beats L*: {dec['blackbox_beats_Lstar']}")
    return "\n".join(lines)


def run_crossvalidated_ladder(ds, levels=("L0", "L1", "L2", "L3", "L4", "B"), k=5, steps=1500,
                              lr=1e-2, cfg=None, seed=0, verbose=0, tie_classes=True, log=print, nested=True,
                              checkpoint=None, protocol="pairs", equal_budget=True, restarts=3, fold_seed=None):
    """Cross-validate the ladder and assemble full held-out prediction tensors.

    protocol="pairs"   : hold out random (responder, stimulus) pairs; every
                         stimulus is partly observed (primary E1 protocol).
    protocol="stimuli" : hold out whole stimulated neurons (stricter).

    Nested rungs (L1..L4) are first fitted as a chain, each warm-started from
    the one below with its new mechanism off (`steps` each).  With
    equal_budget=True every rung is then trained further so that ALL rungs,
    including L0 and B, receive the same total number of optimisation steps
    T = steps x (number of nested rungs).  Without this, deeper rungs inherit
    their parent's training and look better for reasons unrelated to mechanism
    (a bias toward the 'organismic' verdict, found on synthetic worms).

    restarts: the root rung L1 is fitted from `restarts` random initialisations
    and the best training fit is kept (the chain grows from it).  A stuck L1
    otherwise makes every deeper rung look useful merely by escaping its
    parent's local minimum.

    checkpoint: optional directory; every fit is saved there and a rerun
    resumes, skipping finished fits (Colab sessions disconnect).
    Returns (heldout_preds, fitted_params_per_fold, description_lengths)."""
    import os, pickle
    import jax
    from . import ladder as lad
    cfg = lad.SimConfig() if cfg is None else cfg
    st = lad.Structure.from_dataset(ds, tie=tie_classes)
    fs = seed if fold_seed is None else fold_seed           # fold assignment (independent of fitting seed)
    folds = kfold_pairs(ds, k, fs) if protocol == "pairs" else kfold_columns(ds.S, k, fs)
    chain_levels = [L for L in ("L1", "L2", "L3", "L4") if L in levels]
    if not nested:
        equal_budget = False
    T = steps * max(len(chain_levels), 1) if equal_budget else steps
    preds = {L: np.zeros(ds.mean.shape) for L in levels}
    params = {L: [] for L in levels}
    dl = {}

    def cached_fit(name, **kw):
        ck = os.path.join(checkpoint, f"{name}.pkl") if checkpoint else None
        if ck and os.path.exists(ck):
            with open(ck, "rb") as f:
                d = pickle.load(f)
            if log:
                log(f"{name}: loaded from checkpoint")
            return d["params"], d["losses"]
        p, losses = lad.fit(cfg=cfg, lr=lr, verbose=verbose, st=st, **kw)
        if ck:
            os.makedirs(checkpoint, exist_ok=True)
            with open(ck, "wb") as f:
                pickle.dump({"params": jax.device_get(p), "losses": losses}, f)
        if log:
            log(f"{name}: final train loss {losses[-1]:.4f}  ({kw.get('steps')} steps)")
        return p, losses

    for fi, test in enumerate(folds):
        if protocol == "pairs":
            fit_kw = {"train_mask": ~test}
            put = lambda P, Pnew, test=test: np.where(test.reshape(test.shape + (1,) * (P.ndim - 3)), Pnew, P)
        else:
            fit_kw = {"train_cols": np.setdiff1d(np.arange(ds.S), test)}
            def put(P, Pnew, test=test):
                P = P.copy(); P[:, :, test] = Pnew[:, :, test]; return P
        # phase 1: nested chain
        chain, prev = {}, None
        for depth, L in enumerate(chain_levels, start=1):
            init = None
            if nested and prev is not None and L != "L1":
                init = lad.extend_params(prev, L, st, jax.random.PRNGKey(seed + fi))
            if init is None and L == "L1" and restarts > 1:
                cands = [cached_fit(f"fold{fi}_{L}_chain_r{r}", level=L, ds=ds, steps=steps,
                                    seed=seed + fi + 1000 * r, **fit_kw) for r in range(restarts)]
                prev = min(cands, key=lambda c: c[1][-1])[0]
            else:
                prev, _ = cached_fit(f"fold{fi}_{L}_chain", level=L, ds=ds, steps=steps, seed=seed + fi, init=init, **fit_kw)
            chain[L] = (prev, depth)
        # phase 2: equalise the optimisation budget, predict held-out data
        for L in levels:
            if L in chain:
                p, depth = chain[L]
                remaining = T - depth * steps
                if remaining > 0:
                    p, _ = cached_fit(f"fold{fi}_{L}_final", level=L, ds=ds, steps=remaining, seed=seed + fi,
                                      init=p, **fit_kw)
            else:
                p, _ = cached_fit(f"fold{fi}_{L}_final", level=L, ds=ds, steps=T, seed=seed + fi, **fit_kw)
            full = np.asarray(lad.predict(L, p, st, ds.stim, ds.strains, cfg, np.arange(ds.S),
                                          proto=lad.proto_cols(ds, np.arange(ds.S))))
            preds[L] = put(preds[L], full)
            params[L].append(p)
            dl.setdefault(L, lad.description_length(L, ds, p, st))
    return preds, params, dl


# ---------------------------------------------- null-calibrated rule (v4) ---
def gain_lower_bounds(boots, ladder=("L0", "L1", "L2", "L3", "L4"), q=2.5):
    """For each rung beyond the first: the q-th percentile (default: lower end of
    the 95% CI) of the bootstrap distribution of FEVE(L) - max(FEVE of every
    shallower rung), computed per bootstrap sample."""
    out = {}
    for i, L in enumerate(ladder[1:], start=1):
        best = np.max(np.stack([boots[S] for S in ladder[:i]]), axis=0)
        out[L] = float(np.percentile(boots[L] - best, q))
    return out


def null_statistic(boots, truth, ladder=("L0", "L1", "L2", "L3", "L4")):
    """Spurious-depth statistic of a worm whose true rung is `truth`: the largest
    lower CI bound of the gain of any rung DEEPER than the truth.  Under a
    calibrated rule this must stay below the threshold delta."""
    lb = gain_lower_bounds(boots, ladder)
    deeper = [L for L in ladder if ladder.index(L) > ladder.index(truth)]
    return max(lb[L] for L in deeper) if deeper else -np.inf


def calibrate_delta(null_stats, floor=0.0):
    """Threshold delta from null worms (true rung known, deeper rungs absent):
    the largest spurious-depth statistic observed, never below `floor`.
    With n null worms this bounds the false-depth rate on new worms at about
    1/(n+1) (conservative small-sample choice; use more null worms for a quantile)."""
    return max(float(np.max(null_stats)) if len(null_stats) else floor, floor)


def calibrated_closure(boots, delta, ladder=("L0", "L1", "L2", "L3", "L4")):
    """Rule v4: a rung is NEEDED if the lower 95% CI bound of its gain over every
    shallower rung exceeds delta.  The closed level is the deepest needed rung
    (the first rung if none is).  One rule, one threshold, no bracket inversion."""
    lb = gain_lower_bounds(boots, ladder)
    L_closed = ladder[0]
    for L in ladder[1:]:
        if lb[L] > delta:
            L_closed = L
    return {"L_closed": L_closed, "lower_bounds": lb, "delta": float(delta)}
