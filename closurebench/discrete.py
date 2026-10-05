"""Exact closure measures for coarse-grainings of finite Markov chains.

Setting (Rosas et al. 2024, "Software in the natural world"): a micro process
X_t and a coarse-graining Z_t = f(X_t) given by a partition of micro states into
macro blocks.  We compute, exactly:

* informational closure gap  I(Z_{t+1}; X_t | Z_{t-k+1:t})  at stationarity
  (does the micro state add predictive information beyond the macro past?),
* causal closure gap: under interventions do(X_t = x), does the distribution of
  Z_{t+1} depend only on the block of x?  (an interventional, not
  observational, quantity: it also probes micro states the system never visits
  on its own),
* a computational-closure proxy: for each macro history, do all compatible micro
  states predict the macro *future* (L steps) identically?  This compares the
  macro epsilon-machine's predictions with those of the coarse-grained micro
  machine (Rosas et al.'s upsilon-machine).  Check the paper for the exact
  theorem statements linking the three notions; this module computes the
  quantities, it does not prove equivalences.

and a finite-data estimator of the informational gap by held-out predictive
gain, which is the same logic the model ladder uses at scale.
"""
from __future__ import annotations
import itertools
import numpy as np


# ---------------------------------------------------------------- chains ---
def random_chain(n, rng, concentration=1.0):
    """Random row-stochastic n x n transition matrix (Dirichlet rows)."""
    return rng.dirichlet(np.full(n, concentration), size=n)


def stationary(P, tol=1e-12):
    """Stationary distribution of P (left eigenvector for eigenvalue 1)."""
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    pi = np.abs(pi) / np.abs(pi).sum()
    pi[pi < tol] = 0.0
    return pi / pi.sum()


def partition_to_labels(partition, n):
    lab = np.full(n, -1)
    for b, block in enumerate(partition):
        lab[list(block)] = b
    assert (lab >= 0).all(), "partition must cover all micro states"
    return lab


def macro_kernel(P, partition):
    """K[x, B] = P(Z_{t+1}=B | X_t=x): micro-to-macro transition kernel."""
    return np.stack([P[:, list(b)].sum(1) for b in partition], axis=1)


def make_strongly_lumpable(partition, rng, concentration=1.0):
    """Chain whose every micro state in block A has the same macro kernel Q[A,:]
    (Kemeny-Snell strong lumpability), with random detail inside blocks."""
    n = sum(len(b) for b in partition)
    K = len(partition)
    Q = rng.dirichlet(np.full(K, concentration), size=K)
    P = np.zeros((n, n))
    for a, A in enumerate(partition):
        for x in A:
            for b, B in enumerate(partition):
                P[x, list(B)] = Q[a, b] * rng.dirichlet(np.ones(len(B)))
    return P


def add_hidden_state(P, partition, block, rng, concentration=1.0):
    """Append a micro state x* to `block` that is never entered (column of
    zeros) but, if forced into it, has a *different* macro kernel.  The
    result is observationally closed but not causally closed."""
    n = P.shape[0]
    P2 = np.zeros((n + 1, n + 1))
    P2[:n, :n] = P
    P2[n, :n] = rng.dirichlet(np.full(n, concentration))
    parts = [list(b) for b in partition]
    parts[block] = parts[block] + [n]
    return P2, parts


# -------------------------------------------------------------- measures ---
def _entropy(p):
    p = p[p > 0]
    return -(p * np.log2(p)).sum()


def informational_closure_gap(P, partition, k=1, pi=None):
    """Exact I(Z_{t+1}; X_t | Z_{t-k+1..t}) in bits at stationarity.

    Enumerates all micro paths of length k (cost n^k), so keep n^k <~ 1e5.
    Zero iff the micro present adds nothing to the k-step macro past.
    """
    n = P.shape[0]
    pi = stationary(P) if pi is None else pi
    lab = partition_to_labels(partition, n)
    Kmat = macro_kernel(P, partition)
    nZ = len(partition)
    joint = {}  # (zhist, x_t) -> prob
    for path in itertools.product(range(n), repeat=k):
        p = pi[path[0]]
        for a, b in zip(path[:-1], path[1:]):
            p *= P[a, b]
            if p == 0:
                break
        if p == 0:
            continue
        key = (tuple(lab[list(path)]), path[-1])
        joint[key] = joint.get(key, 0.0) + p
    # I(Z';X|H) = H(Z'|H) - H(Z'|X,H)
    by_h = {}
    h_zx = 0.0
    for (h, x), p in joint.items():
        by_h.setdefault(h, np.zeros(nZ))
        by_h[h] += p * Kmat[x]
        h_zx += p * _entropy(Kmat[x])
    h_z = sum(v.sum() * _entropy(v / v.sum()) for v in by_h.values())
    return max(h_z - h_zx, 0.0)


def causal_closure_gap(P, partition):
    """Interventional gap: average over blocks A and micro states x in A of
    KL( P(Z'|do(X=x)) || P(Z'|do(X~Uniform(A))) ) in bits.  Uses every micro
    state, visited or not, which is the point of intervening."""
    Kmat = macro_kernel(P, partition)
    gaps = []
    for A in partition:
        ref = Kmat[list(A)].mean(0)
        for x in A:
            p = Kmat[x]
            m = p > 0
            gaps.append((p[m] * np.log2(p[m] / ref[m])).sum())
    return max(float(np.mean(gaps)), 0.0)


def _future_dists(P, partition, L):
    """F[x] = distribution over macro futures (Z_{t+1..t+L}) given X_t = x."""
    n = P.shape[0]
    lab = partition_to_labels(partition, n)
    nZ = len(partition)
    F = np.zeros((n, nZ ** L))
    # propagate distribution over (x_{t+s}, future-so-far index)
    for x0 in range(n):
        state = {(x0, 0): 1.0}
        for _ in range(L):
            new = {}
            for (x, idx), p in state.items():
                for y in np.nonzero(P[x])[0]:
                    key = (y, idx * nZ + lab[y])
                    new[key] = new.get(key, 0.0) + p * P[x, y]
            state = new
        for (_, idx), p in state.items():
            F[x0, idx] += p
    return F


def computational_closure_gap(P, partition, k=2, L=2, pi=None):
    """E_h sum_x p(x|h) TV( P(future|x), P(future|h) ).

    For every macro history h (length k), compares the macro-level prediction
    of the next L macro symbols with the prediction made from each micro state
    compatible with h.  Zero iff macro causal states are not refined by micro
    detail for predicting the macro future, i.e. the macro 'software' runs on
    its own.  Returns (gap, n_macro_causal_states, n_upsilon_states).
    """
    n = P.shape[0]
    pi = stationary(P) if pi is None else pi
    lab = partition_to_labels(partition, n)
    F = _future_dists(P, partition, L)
    post = {}
    for path in itertools.product(range(n), repeat=k):
        p = pi[path[0]]
        for a, b in zip(path[:-1], path[1:]):
            p *= P[a, b]
        if p == 0:
            continue
        h = tuple(lab[list(path)])
        post.setdefault(h, np.zeros(n))
        post[h][path[-1]] += p
    gap = 0.0
    macro_preds = []
    for h, w in post.items():
        ph = w.sum()
        px = w / ph
        Fh = px @ F
        macro_preds.append(Fh)
        tv = 0.5 * np.abs(F - Fh).sum(1)
        gap += ph * (px * tv).sum()
    n_eps = len(_unique_rows(np.array(macro_preds)))
    reachable = pi > 0
    n_ups = len(_unique_rows(F[reachable]))
    return gap, n_eps, n_ups


def _unique_rows(A, decimals=8):
    return np.unique(np.round(A, decimals), axis=0)


# ------------------------------------------------- finite-data estimator ---
def sample_chain(P, T, rng, x0=None):
    n = P.shape[0]
    x = np.empty(T, dtype=int)
    x[0] = rng.choice(n, p=stationary(P)) if x0 is None else x0
    cum = np.cumsum(P, axis=1)
    u = rng.random(T)
    for t in range(1, T):
        x[t] = min(np.searchsorted(cum[x[t - 1]], u[t]), n - 1)
    return x


def heldout_predictive_gain(x, partition, k=1, alpha=0.5, train_frac=0.5):
    """Estimate I(Z'; X | Z-history) as the held-out log-loss improvement (bits
    per step) of a count model that also sees X_t over one that sees only the
    macro history.  Positive => micro detail adds predictive power on new data.
    This is the discrete analogue of comparing adjacent rungs of the ladder.
    """
    n = max(max(b) for b in partition) + 1
    lab = partition_to_labels(partition, n)
    z = lab[x]
    nZ = len(partition)
    T = len(x)
    split = int(T * train_frac)

    def keys(t):
        return tuple(z[t - k + 1:t + 1]), x[t]

    c_macro, c_micro = {}, {}
    for t in range(k - 1, split - 1):
        h, xt = keys(t)
        c_macro.setdefault(h, np.zeros(nZ))[z[t + 1]] += 1
        c_micro.setdefault((h, xt), np.zeros(nZ))[z[t + 1]] += 1
    ll_macro = ll_micro = 0.0
    m = 0
    for t in range(max(split, k - 1), T - 1):
        h, xt = keys(t)
        cm = c_macro.get(h, np.zeros(nZ)) + alpha
        cu = c_micro.get((h, xt), np.zeros(nZ)) + alpha
        ll_macro += np.log2(cm[z[t + 1]] / cm.sum())
        ll_micro += np.log2(cu[z[t + 1]] / cu.sum())
        m += 1
    return (ll_micro - ll_macro) / max(m, 1)
