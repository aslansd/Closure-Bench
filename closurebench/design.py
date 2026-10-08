"""E1d - optimal experimental design: which new interventions would constrain the
mechanisms that single-neuron stimulation cannot see?

Notebook 07 found that the atlas protocol constrains only a few dozen parameter
combinations (about 20 for the real fitted L1) and leaves the deeper mechanisms
essentially free.  This module scores candidate experiments by the Bayesian
information they would add, given the atlas data that already exist.

Candidate designs (each adds one new column per atlas-stimulated neuron, with
the same imaging coverage as the atlas and `trials` trials per pair):

  repeat   more trials of the atlas protocol            (control: "more of the same")
  strong   3 x stimulus amplitude                       (nonlinearity, thresholds)
  weak     0.3 x stimulus amplitude
  train    three 1-s pulses, 4 s apart                  (adaptation, slow accumulation)
  long     one 5-s pulse
  pairs    two neurons stimulated together              (interactions)
  gapless  atlas protocol in a gap-junction mutant      (electrical vs chemical routes)
  unc31    atlas protocol in unc-31 (no neuropeptides)  (the peptide channel)

Information gain (Gaussian approximation, prior N(0, prior_sd^2) on every
unconstrained parameter):
  IG = 0.5 * [log det Sigma_before - log det Sigma_after]   (bits),
for the marginal of any parameter group, with Sigma = (F + I/prior_sd^2)^-1 and
F = J^T W J the Fisher information (W = trials / noise variance per pair).
Per-stimulus efficacy offsets (stim_col) are nuisance parameters: they are kept
in F, so the gain is computed after accounting for their uncertainty.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from .data import Dataset

DESIGNS = ("repeat", "strong", "weak", "train", "long", "pairs", "gapless", "unc31")


def design_columns(template, kind, trials=3, seed=0):
    """A Dataset (no measurements yet) describing one candidate experiment on
    the template's neurons: same stimulated neurons, same responder coverage
    as the template's wild-type atlas data."""
    S = template.S
    proto = dict(amp=np.ones(S, np.float32), pair=-np.ones(S, np.int32), pattern=np.zeros(S, np.int32))
    strain = "wt"
    if kind == "strong":
        proto["amp"][:] = 3.0
    elif kind == "weak":
        proto["amp"][:] = 0.3
    elif kind == "train":
        proto["pattern"][:] = 1
    elif kind == "long":
        proto["pattern"][:] = 2
    elif kind == "pairs":
        rng = np.random.default_rng(seed)
        partner = rng.permutation(template.stim)
        clash = partner == template.stim
        partner[clash] = np.roll(partner, 1)[clash]
        proto["pair"] = partner.astype(np.int32)
    elif kind == "gapless":
        strain = "gapless"
    elif kind == "unc31":
        strain = "unc31"
    elif kind != "repeat":
        raise ValueError(kind)
    cover = template.mask()[0]                                   # same imaging coverage as the atlas (WT)
    n = (cover * trials).astype(int)[None]
    z = np.zeros(n.shape)
    ds = Dataset(ids=template.ids, chem=template.chem, gap=template.gap, sign=template.sign, pep=template.pep,
                 stim=template.stim.copy(), strains=(strain,), mean=z, var_mean=z.copy(), n=n,
                 meta={"design": kind, "trials": trials}, proto=proto)
    ds.n[:, ds.stim, np.arange(S)] = 0
    if kind == "pairs":
        ds.n[:, proto["pair"], np.arange(S)] = 0
    return ds


def combine(a, b):
    """Concatenate the columns of two Datasets on the same neurons; strains are
    the union (pairs never measured in a strain get n = 0)."""
    strains = tuple(dict.fromkeys(a.strains + b.strains))
    def lift(ds):
        shape = (len(strains),) + ds.n.shape[1:]
        n = np.zeros(shape, int); mean = np.zeros(shape + ds.mean.shape[3:]); var = np.zeros_like(mean)
        for k, s in enumerate(ds.strains):
            j = strains.index(s); n[j] = ds.n[k]; mean[j] = ds.mean[k]; var[j] = ds.var_mean[k]
        return n, mean, var
    def proto(ds):
        if ds.proto is not None:
            return {k: np.asarray(v) for k, v in ds.proto.items()}
        return dict(amp=np.ones(ds.S, np.float32), pair=-np.ones(ds.S, np.int32), pattern=np.zeros(ds.S, np.int32))
    (na, ma, va), (nb, mb, vb) = lift(a), lift(b)
    pa, pb = proto(a), proto(b)
    return Dataset(ids=a.ids, chem=a.chem, gap=a.gap, sign=a.sign, pep=a.pep,
                   stim=np.concatenate([a.stim, b.stim]), strains=strains,
                   mean=np.concatenate([ma, mb], 2), var_mean=np.concatenate([va, vb], 2), n=np.concatenate([na, nb], 2),
                   meta=dict(a.meta, combined_with=b.meta.get("design", "?")),
                   proto={k: np.concatenate([pa[k], pb[k]]) for k in pa})


def jacobian(level, p, ds, st, cfg, rows_mask, chunk=64):
    """Jacobian of predicted responses on the pairs in rows_mask (n_strain, N, S)
    with respect to the active parameters (see identifiability.active_index).
    Returns (J, groups, row_cols) where row_cols is each row's column index."""
    from . import ladder as lad
    from .identifiability import active_index, EXCLUDE
    p = {k: (jnp.asarray(v) if k not in EXCLUDE else v) for k, v in p.items()}
    idx, groups, flat0, unravel = active_index(level, p, st)
    sel = np.nonzero(np.asarray(rows_mask).ravel())[0]
    proto = lad.proto_cols(ds, np.arange(ds.S))
    flat0 = jnp.asarray(flat0); ij = jnp.asarray(idx)

    def f(theta):
        q = unravel(flat0.at[ij].set(theta))
        return lad.predict(level, q, st, ds.stim, ds.strains, cfg, np.arange(ds.S), proto=proto).ravel()[sel]
    jvp = jax.jit(lambda t: jax.jvp(f, (flat0[ij],), (t,))[1])
    P = len(idx); J = np.zeros((len(sel), P)); E = np.eye(P, dtype=np.float32)
    for s in range(0, P, chunk):
        J[:, s:s + chunk] = np.asarray(jax.vmap(jvp)(jnp.asarray(E[s:s + chunk]))).T
    row_cols = np.unravel_index(sel, np.asarray(rows_mask).shape)[2]
    return J, groups, row_cols


def information_gain(F_before, F_after, groups, prior_sd=1.0, exclude=("stim_col",)):
    """Information gain in bits, for all non-nuisance parameters together and per
    group, of the Gaussian posterior after vs before the new data."""
    P = F_before.shape[0]
    A = F_before + np.eye(P) / prior_sd ** 2
    B = F_after + np.eye(P) / prior_sd ** 2
    SA, SB = np.linalg.inv(A), np.linalg.inv(B)
    def ig(rows):
        la = np.linalg.slogdet(SA[np.ix_(rows, rows)])[1]; lb = np.linalg.slogdet(SB[np.ix_(rows, rows)])[1]
        return float(0.5 * (la - lb) / np.log(2))
    groups = np.asarray(groups)
    core = np.nonzero(~np.isin(groups, exclude))[0]
    out = {"total": ig(core)}
    for g in np.unique(groups[core]):
        out[str(g)] = ig(np.nonzero(groups == g)[0])
    return out


def score_designs(level, p_true, template, st, cfg, noise_sd, designs=DESIGNS, trials=3, prior_sd=1.0,
                  rel=1e-3, verbose=True):
    """For each design: information gain (bits; total and per mechanism) and the
    number of stiff directions before/after (eigenvalue > rel x the largest
    eigenvalue of the atlas-only Fisher information)."""
    out = {}
    for kind in designs:
        D = design_columns(template, kind, trials=trials)
        C = combine(template, D)
        p = dict(p_true); p["stim_col"] = jnp.zeros(C.S)
        rows = C.mask() & (C.n > 0)
        J, groups, row_cols = jacobian(level, p, C, st, cfg, rows)
        w = (np.asarray(C.n)[rows] / noise_sd ** 2)              # trials / per-trial noise variance
        atlas = row_cols < template.S
        Fa = (J[atlas] * w[atlas, None]).T @ J[atlas]
        Ft = (J * w[:, None]).T @ J
        ref = np.linalg.eigvalsh(Fa)[-1]
        out[kind] = dict(ig=information_gain(Fa, Ft, groups, prior_sd),
                         stiff_before=int((np.linalg.eigvalsh(Fa) > rel * ref).sum()),
                         stiff_after=int((np.linalg.eigvalsh(Ft) > rel * ref).sum()))
        if verbose:
            g = out[kind]
            print(f"  {kind:8s} IG total {g['ig']['total']:8.1f} bits | stiff directions {g['stiff_before']} -> {g['stiff_after']}", flush=True)
    return out
