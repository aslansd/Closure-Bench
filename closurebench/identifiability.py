"""E1c - identifiability of the ladder's rungs ("sloppiness").

Notebook 06's synthetic worms showed that the fitted true rung reaches the same
TRAINING loss as the true parameters but much lower held-out accuracy: many
parameter settings fit the data equally well.  This module quantifies that.

fisher_spectrum(level, p, ds, st, cfg, mask)
    Gauss-Newton Fisher information F = J^T W J of the (weighted) training loss
    at parameters p, where J is the Jacobian of the predicted responses on the
    training pairs with respect to the *active* parameters (synaptic weights on
    their support, junction conductances on theirs, per-neuron and global
    parameters).  Its eigenvalue spectrum shows how many parameter combinations
    the data constrain ("stiff") and how many they leave free ("sloppy");
    sloppy models have eigenvalues spread evenly over many decades
    (Gutenkunst et al. 2007; Machta et al. 2013).

group_stiffness(...)
    For each mechanism (W, G, tau, theta, bias, gK, ..., rec, ...), the share of
    its parameter directions that lie in the stiff subspace.

The Jacobian is computed with forward-mode differentiation in chunks, so memory
stays bounded; cost is about (number of active parameters / chunk) simulations.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

EXCLUDE = ("n_heads",)


def active_index(level, p, st):
    """Indices (into the flattened parameter vector) of parameters that affect
    the model, and a group label for each.  Masked-out synapse entries and
    parameters of other rungs are excluded."""
    flat, unravel = ravel_pytree({k: v for k, v in p.items() if k not in EXCLUDE})
    idx, groups, offset = [], [], 0
    for k in sorted(k for k in p if k not in EXCLUDE):       # ravel_pytree orders dict keys sorted
        v = np.asarray(p[k]); n = v.size
        if k == "W":
            keep = np.nonzero(np.asarray(st.chem_mask).ravel())[0]
        elif k == "G":   # symmetric use: keep upper triangle of the junction support
            gm = np.triu(np.asarray(st.gap_mask)); keep = np.nonzero(gm.ravel())[0]
        else:
            keep = np.arange(n)
        idx.extend(offset + keep); groups.extend([k] * len(keep)); offset += n
    assert offset == flat.size
    return np.asarray(idx), np.asarray(groups), flat, unravel


def fisher_spectrum(level, p, ds, st, cfg, mask, chunk=64, verbose=True):
    """Eigenvalues/eigenvectors of the Gauss-Newton Fisher information of the
    weighted training loss at p.  mask: (n_strain, N, S) training pairs.
    Returns dict(evals (descending), evecs, groups, n_data)."""
    from . import ladder as lad
    p = {k: (jnp.asarray(v) if k not in EXCLUDE else v) for k, v in p.items()}
    static = {k: p[k] for k in EXCLUDE if k in p}
    idx, groups, flat0, unravel = active_index(level, p, st)
    cols = np.arange(ds.S)
    m = np.broadcast_to(lad.bcast(mask & ds.mask(), ds.mean), ds.mean.shape)
    w = np.where(m, 1.0 / (ds.var_mean + 1e-3), 0.0)
    w = w / (w * ds.mean ** 2).sum()                       # same normalisation as the fit's loss
    sel = np.nonzero(m.ravel())[0]
    sqrt_w = jnp.asarray(np.sqrt(w.ravel()[sel]))
    flat0 = jnp.asarray(flat0); idx_j = jnp.asarray(idx)

    def f(theta_active):
        flat = flat0.at[idx_j].set(theta_active)
        q = dict(unravel(flat), **static)
        out = lad.predict(level, q, st, ds.stim, ds.strains, cfg, cols)
        return out.ravel()[sel] * sqrt_w

    theta0 = flat0[idx_j]
    jvp = jax.jit(lambda t: jax.jvp(f, (theta0,), (t,))[1])
    P = len(idx); J = np.zeros((len(sel), P))
    eye = np.eye(P, dtype=np.float32)
    for s in range(0, P, chunk):
        J[:, s:s + chunk] = np.asarray(jax.vmap(jvp)(jnp.asarray(eye[s:s + chunk]))).T
        if verbose and (s // chunk) % 5 == 0:
            print(f"  Jacobian columns {min(s + chunk, P)}/{P}", flush=True)
    F = J.T @ J
    evals, evecs = np.linalg.eigh(F)
    order = np.argsort(evals)[::-1]
    return dict(evals=np.clip(evals[order], 0, None), evecs=evecs[:, order], groups=groups, n_data=len(sel))


def summarize(spec, rel=1e-3):
    """Number of stiff directions (eigenvalue > rel x largest), the decades
    spanned by the non-zero spectrum, and the effective dimension
    (participation ratio of the eigenvalues)."""
    ev = spec["evals"]; top = ev[0]
    nz = ev[ev > top * 1e-12]
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())
    return dict(n_params=len(ev), n_stiff=int((ev > rel * top).sum()),
                decades=float(np.log10(nz[0] / nz[-1])) if len(nz) > 1 else 0.0, participation=pr)


def group_stiffness(spec, rel=1e-3):
    """Share of each parameter group's directions lying in the stiff subspace:
    sum over stiff eigenvectors of the squared loadings on that group, divided
    by the group's size.  1 = fully constrained, 0 = fully free."""
    ev = spec["evals"]; stiff = ev > rel * ev[0]
    V = spec["evecs"][:, stiff]
    out = {}
    for g in np.unique(spec["groups"]):
        rows = spec["groups"] == g
        out[str(g)] = float((V[rows] ** 2).sum() / rows.sum())
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
