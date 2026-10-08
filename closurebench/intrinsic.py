"""E2d - slow intrinsic currents: how slow may the HARDWARE's own dynamics be
before the fitted program changes?

E2-E2c realised the fitted L1 on substrates whose extra dynamics were either
absent (numerical substrates), fast (spikes, tau_m = 20 ms) or exactly
compensated at rest (conductance synapses).  Real neurons also carry SLOW
voltage-gated currents, e.g. K+ conductances that activate over hundreds of ms
to seconds.  Such a current is part of the hardware, not of the fitted program.

Realization.  A fraction rho of each neuron's leak is carried by a slow,
voltage-gated K+-like conductance (linearised around rest):

    tau_i dV_i/dt = -(1 - rho) V_i - rho w_i + ... (L1 otherwise unchanged)
    tau_n dw_i/dt = -w_i + V_i

At steady state w = V, so the fixed point and the DC gain are exactly L1's: the
compiler matches the program at rest and in the limit tau_n -> 0.  For slow
tau_n the membrane first responds with less leak (a larger transient) and then
adapts -- the slow channel adds an adaptation the fitted program does not have
(the ladder's L2 mechanism, but in the substrate).

Base synapses: `current` (L1's own) or `conductance` (E2c's compensated
conductance synapses at the physiological voltage scale).
"""
from __future__ import annotations
import numpy as np
from .substrates import _sig, _stim_matrix


def simulate_slow_k(e, stim_idx, cfg, V_rest, rho=0.3, tau_n=1.0, syn=None, h=0.02):
    """Response matrix (N, S), RK4 dt = h, same protocol and read-out as
    substrates.simulate_numpy('rk4').  syn: None for current-based synapses, or a
    dict from conductance.compile_synapses (compensated conductance synapses).
    Starts from L1's resting state V_rest (from conductance.rest_state)."""
    N = len(e["bias"]); B = len(stim_idx) + 1
    Iw = _stim_matrix(e, stim_idx, N)
    if syn is not None:
        g, kap = syn["g"], syn["kappa"]; gE = g * syn["E"]
    def f(V, S, Wk, I):
        a = _sig(e["k"] * (V - e["theta"]))
        if syn is None:
            Isyn = S @ e["W"].T
        else:
            v = syn["v_mid"] + kap * (V - syn["V_mid"])
            Isyn = S @ gE.T - v * (S @ g.T) + syn["comp"] * (V - syn["V_rest"])
        lap = V @ e["G"].T - e["G"].sum(1) * V
        dV = (-(1 - rho) * V - rho * Wk + e["bias"] + Isyn + lap + I) / e["tau"]
        return dV, (-S + a) / e["tau_s"], (-Wk + V) / tau_n
    def rk4(X, I):
        k1 = f(*X, I); k2 = f(*[x + h / 2 * k for x, k in zip(X, k1)], I)
        k3 = f(*[x + h / 2 * k for x, k in zip(X, k2)], I); k4 = f(*[x + h * k for x, k in zip(X, k3)], I)
        return tuple(x + h / 6 * (a + 2 * b + 2 * c + d) for x, a, b, c, d in zip(X, k1, k2, k3, k4))
    V = np.tile(V_rest, (B, 1))
    X = (V, np.tile(_sig(e["k"] * (V_rest - e["theta"])), (B, 1)), V.copy())
    with np.errstate(all="ignore"):
        for _ in range(int(round(cfg.t_burn / h))):
            X = rk4(X, 0.0)
        n = int(round(cfg.t_win / h)); n_on = int(round(cfg.t_stim / h))
        a_prev = _sig(e["k"] * (X[0] - e["theta"])); acc = np.zeros((B, N))
        for t in range(n):
            X = rk4(X, Iw if t < n_on else 0.0)
            if not np.all(np.isfinite(X[0])) or np.abs(X[0]).max() > 1e3:
                return np.full((N, B - 1), np.nan)
            a = _sig(e["k"] * (X[0] - e["theta"])); acc += 0.5 * (a + a_prev); a_prev = a
    mean = acc / n
    return e["gain"][:, None] * (mean[1:] - mean[0]).T


# ------------------------------------------------------------- E1g (NB 20) ---
def rescaled_l1(e, V_rest, rho):
    """The tau_n -> infinity limit of simulate_slow_k, written as an ordinary L1.

    With the slow gate frozen at rest (w = V_rest), simulate_slow_k's membrane is
        tau dV/dt = -(1 - rho) V - rho V_rest + bias + W s + lap(G) + I,
    which is exactly L1 with
        tau' = tau / (1-rho), bias' = (bias - rho V_rest) / (1-rho),
        W' = W / (1-rho), G' = G / (1-rho), amp' = amp / (1-rho):
    same resting state, weaker effective leak for transients, i.e. stronger
    recurrent amplification and slower responses.  It is a point INSIDE the L1
    family, so a held-out gain from it means the fitted L1 was not at the best
    point of its own family."""
    s = 1.0 / (1.0 - rho)
    out = dict(e)
    out.update(tau=e["tau"] * s, bias=(e["bias"] - rho * V_rest) * s, W=e["W"] * s, G=e["G"] * s, amp=e["amp"] * s)
    return out


def simulate_l1_from(e, stim_idx, cfg, V0, h=0.02):
    """substrates.simulate_numpy('rk4') for L1 parameters e, but starting from the
    state V0 (with s at its steady value for V0) instead of zero -- the same start
    simulate_slow_k uses.  Needed whenever the network may have more than one
    resting state: from zero, a rescaled L1 can settle into a different attractor
    than the fitted L1 (Notebook 20's failed EXPRESSIBLE check)."""
    return simulate_slow_k(e, stim_idx, cfg, V0, rho=0.0, tau_n=1.0, h=h)


def fixed_point_residual(e, V):
    """max |dV/dt| * tau of L1 at state V (with s = its steady value): ~0 at a fixed point."""
    s = _sig(e["k"] * (V - e["theta"]))
    lap = V @ e["G"].T - e["G"].sum(1) * V
    return float(np.abs(-V + e["bias"] + s @ e["W"].T + lap).max())


def true_rest(e, V_init, tol=1e-10):
    """L1's resting state solved exactly (Newton / hybrid root finding on
    -V + bias + W s(V) + lap(V) = 0, starting from V_init), with its stability:
    the largest real part of the eigenvalues of the full (V, s) Jacobian.
    Returns dict(V, residual, max_real_eig, ok).  ok = solver converged, residual
    < 1e-8 and the fixed point is stable (max_real_eig < 0)."""
    from scipy.optimize import root
    N = len(e["bias"]); G = e["G"]; gsum = G.sum(1)
    def F(V):
        s = _sig(e["k"] * (V - e["theta"]))
        return -V + e["bias"] + e["W"] @ s + G @ V - gsum * V
    sol = root(F, np.asarray(V_init, float).ravel(), tol=tol)
    V = sol.x
    s = _sig(e["k"] * (V - e["theta"])); ds = e["k"] * s * (1 - s)
    tau = e["tau"][:, None]; ts = e["tau_s"]
    J = np.block([[(-np.eye(N) + G - np.diag(gsum)) / tau, e["W"] / tau],
                  [np.diag(ds) / ts, -np.eye(N) / ts]])
    lam = float(np.linalg.eigvals(J).real.max())
    res = float(np.abs(F(V)).max())
    return dict(V=V, residual=res, max_real_eig=lam, ok=bool(sol.success and res < 1e-8 and lam < 0))
