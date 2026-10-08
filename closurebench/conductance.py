"""E2c - a CONDUCTANCE-based realization of the fitted L1.

L1's synapses are current-based: neuron j adds W_ij * s_j to neuron i whatever
i's voltage.  Real chemical synapses open conductances with reversal potentials:
their current is g_ij s_j (E_ij - v_i), so their effect shrinks as v_i approaches
E_ij and every open synapse also lowers the input resistance (shunting).  This
is the first change of equations that biophysics actually imposes, and it is
multiplicative where L1 is additive.

Compilation.  The model's dimensionless V is mapped to millivolts,
v = v_mid + kappa (V - V_mid), so that the neurons' resting potentials span a
physiological range (default -70 to -20 mV).  Each synapse gets a reversal
potential from its sign (excitatory W > 0 -> E_exc = 0 mV; inhibitory W < 0 ->
E_inh = -80 mV, as for GABA_A / glutamate-gated chloride channels) and a
conductance chosen so that, AT REST, it injects exactly L1's current:

    g_ij = W_ij / (E_ij - v_rest_i)             (current matched at rest)

  naive        that alone: identical fixed point, but the extra synaptic
               conductance changes each neuron's effective leak (gain and time
               constant) around rest
  compensated  plus a leak correction that cancels the first-order conductance
               change at rest, so the LINEARISATION around rest matches L1 too;
               the remaining differences are second order (shunting proper)

Gap junctions are already conductances in L1 and are unchanged.  The driving
force is clipped to at least 2 mV on the physical side of the reversal potential
(relevant only when the voltage scale is stretched in the sweep).
"""
from __future__ import annotations
import numpy as np
from .substrates import _sig, _stim_matrix


def rest_state(e, cfg, h=0.02):
    """L1's resting state (V, s), as reached by substrates.simulate_numpy('rk4'):
    RK4 from zero for cfg.t_burn without stimulus."""
    N = len(e["bias"])
    V = np.zeros((1, N)); S = np.zeros((1, N))
    def f(V, S):
        a = _sig(e["k"] * (V - e["theta"]))
        lap = V @ e["G"].T - e["G"].sum(1) * V
        return (-V + e["bias"] + S @ e["W"].T + lap) / e["tau"], (-S + a) / e["tau_s"]
    for _ in range(int(round(cfg.t_burn / h))):
        k1 = f(V, S); k2 = f(V + h / 2 * k1[0], S + h / 2 * k1[1])
        k3 = f(V + h / 2 * k2[0], S + h / 2 * k2[1]); k4 = f(V + h * k3[0], S + h * k3[1])
        V = V + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); S = S + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return V[0], S[0]


def voltage_map(V_rest, lo=-70.0, hi=-20.0):
    """kappa (mV per model unit) and the centre (V_mid, v_mid) mapping the range of
    resting V across neurons onto [lo, hi] mV."""
    span = max(V_rest.max() - V_rest.min(), 1e-6)
    return dict(kappa=(hi - lo) / span, V_mid=0.5 * (V_rest.max() + V_rest.min()), v_mid=0.5 * (lo + hi))


def compile_synapses(e, V_rest, S_rest, vmap, scale=1.0, E_exc=0.0, E_inh=-80.0, min_drive=2.0):
    """Conductances, reversal potentials and the leak correction.  `scale`
    stretches the voltage map (driving-force sweep)."""
    kap = vmap["kappa"] * scale
    vr = vmap["v_mid"] + kap * (V_rest - vmap["V_mid"])
    W = e["W"]
    E = np.where(W > 0, E_exc, E_inh) * (W != 0)
    drive = E - vr[:, None]
    drive = np.where(W > 0, np.maximum(drive, min_drive), np.minimum(drive, -min_drive))
    g = np.where(W != 0, W / drive, 0.0)                      # model-current units per mV of driving force
    comp = kap * (g @ S_rest)                                 # first-order conductance change at rest, per neuron
    return dict(g=g, E=E, kappa=kap, v_mid=vmap["v_mid"], V_mid=vmap["V_mid"], comp=comp, V_rest=V_rest,
                v_rest=vr, frac_clipped=float(((np.abs(E - vr[:, None]) < min_drive) & (W != 0)).sum() / max((W != 0).sum(), 1)))


def simulate_conductance(e, stim_idx, cfg, syn, compensate=True, h=0.02):
    """Response matrix (N, S), RK4 dt = h, same protocol and read-out as
    substrates.simulate_numpy('rk4')."""
    N = len(e["bias"]); B = len(stim_idx) + 1
    Iw = _stim_matrix(e, stim_idx, N)
    g, E, kap = syn["g"], syn["E"], syn["kappa"]
    gE = (g * E).astype(float)
    def f(V, S, I):
        a = _sig(e["k"] * (V - e["theta"]))
        v = syn["v_mid"] + kap * (V - syn["V_mid"])
        Isyn = S @ gE.T - v * (S @ g.T)                       # sum_j g_ij s_j (E_ij - v_i)
        if compensate:
            Isyn = Isyn + syn["comp"] * (V - syn["V_rest"])          # cancels d Isyn / dV at rest
        lap = V @ e["G"].T - e["G"].sum(1) * V
        return (-V + e["bias"] + Isyn + lap + I) / e["tau"], (-S + a) / e["tau_s"]
    def rk4(V, S, I):
        k1 = f(V, S, I); k2 = f(V + h / 2 * k1[0], S + h / 2 * k1[1], I)
        k3 = f(V + h / 2 * k2[0], S + h / 2 * k2[1], I); k4 = f(V + h * k3[0], S + h * k3[1], I)
        return V + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), S + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    V = np.tile(syn["V_rest"], (B, 1)); S = np.tile(_rest_s(e, syn["V_rest"]), (B, 1))
    with np.errstate(all="ignore"):
        for _ in range(int(round(cfg.t_burn / h))):
            V, S = rk4(V, S, 0.0)
        n = int(round(cfg.t_win / h)); n_on = int(round(cfg.t_stim / h))
        a_prev = _sig(e["k"] * (V - e["theta"])); acc = np.zeros((B, N))
        for t in range(n):
            V, S = rk4(V, S, Iw if t < n_on else 0.0)
            if not np.all(np.isfinite(V)) or np.abs(V).max() > 1e3:
                return np.full((N, B - 1), np.nan)                 # unstable realisation
            a = _sig(e["k"] * (V - e["theta"])); acc += 0.5 * (a + a_prev); a_prev = a
    mean = acc / n
    return e["gain"][:, None] * (mean[1:] - mean[0]).T


def leak_feasibility(syn):
    """The compensated compiler lowers each neuron's effective leak by `comp`
    (in units of L1's leak, which is 1).  comp > 1 would need a NEGATIVE leak,
    i.e. an active inward conductance (e.g. a persistent Ca2+ current, which C.
    elegans neurons do have).  Returns (median comp, fraction of neurons > 1)."""
    c = syn["comp"]
    return float(np.median(c)), float((c > 1).mean())


def _rest_s(e, V_rest):
    return _sig(e["k"] * (V_rest - e["theta"]))
