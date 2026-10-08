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
