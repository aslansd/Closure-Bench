"""E2 - substrate transfer: is the closed level independent of its realisation?

The fitted closed rung L* (from E1) is a set of parameters plus equations.  If
it is "software", the same parameters run on a different substrate should
produce the same interventional responses.  Each substrate here is a separate
implementation with different numerics:

  ref        JAX, explicit Euler, dt = 0.2 s, float32   (the substrate it was fitted on)
  euler64    NumPy, explicit Euler, dt = 0.2 s, float64  (sanity check: must match ref)
  ode        NumPy, RK4, dt = 0.02 s, float64            (continuous-time limit)
  brian2     Brian2 (numpy code generation), RK4, dt = 0.02 s
  fixed      neuromorphic constraints: integer weights (w_bits), fixed-point state
             (frac_bits), sigmoid look-up table (lut_bits), Euler dt = 0.2 s

Plus two further E2 quantities:

  degeneracy  do independently fitted parameter sets (CV folds) differ in
              parameters while agreeing in macro behaviour? (Prinz et al. 2004)
  lyapunov    largest Lyapunov exponent of the fitted network, at rest and
              under random pulse drive: if positive, macro behaviour depends on
              numerical precision and the "software" is not separable from its
              physics.

Currently implemented for rung L1 (the E1 laptop result).  L2+ need their
extra currents added to `_deriv`.
"""
from __future__ import annotations
import numpy as np


def import_brian2():
    """Import Brian2 with a clear message for the known NumPy incompatibility:
    Brian2 <= 2.9 (the newest for Python 3.11) needs ndarray.ptp, removed in NumPy 2.4."""
    try:
        import brian2
        return brian2
    except AttributeError as e:
        if "ptp" in str(e):
            raise ImportError(
                f"Brian2 cannot load with NumPy {np.__version__} (no ndarray.ptp). Fix: "
                "`.venv/bin/pip install 'numpy==2.3.5'` and restart the kernel (Python 3.11), "
                "or use Python >= 3.12 with brian2 >= 2.10.") from None
        raise


# ------------------------------------------------------------------ params ---
def effective_params(level, p, st, cols):
    """Constrained, tied parameters of a fitted rung as NumPy float64 arrays.
    `cols` are the stimulus columns (indices into ds.stim) to simulate."""
    import jax
    from . import ladder as lad
    if level != "L1":
        raise NotImplementedError("substrate transfer is implemented for L1; extend _deriv for L2+")
    p = jax.tree_util.tree_map(jax.numpy.asarray, p)
    q = lad.tie(st, p)
    sp = jax.nn.softplus
    f = lambda x: np.asarray(x, dtype=np.float64)
    cols = np.asarray(cols)
    amp = sp(q["stim"] + q["stim_col"][cols]) if "stim_col" in q else sp(q["stim"]) * np.ones(len(cols))
    return dict(
        W=f(q["W"] * st.chem_mask),
        G=f(sp(0.5 * (q["G"] + q["G"].T)) * st.gap_norm),
        tau=f(lad.TAU_MIN + sp(q["tau"])), tau_s=float(lad.TAU_MIN + sp(q["tau_s"])),
        theta=f(q["theta"]), bias=f(q["bias"]), gain=f(np.exp(p["log_gain"])),
        amp=f(amp), k=float(lad.SimConfig().k))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _deriv(e, V, Sv, I):
    """Batched L1 vector field. V, Sv, I: (B, N)."""
    a = _sig(e["k"] * (V - e["theta"]))
    lap = V @ e["G"].T - e["G"].sum(1) * V
    dV = (-V + e["bias"] + Sv @ e["W"].T + lap + I) / e["tau"]
    dS = (-Sv + a) / e["tau_s"]
    return dV, dS, a


def _stim_matrix(e, stim_idx, N):
    """(S+1, N): row 0 = no-stimulus control, row s+1 = pulse to stim_idx[s]."""
    I = np.zeros((len(stim_idx) + 1, N))
    I[np.arange(1, len(stim_idx) + 1), stim_idx] = e["amp"]
    return I


# --------------------------------------------------------------- substrates ---
def simulate_numpy(e, stim_idx, cfg, method="euler", dt=None, dtype=np.float64):
    """Response matrix (N, S) on a NumPy substrate.

    method="euler": the exact discrete protocol of the JAX ladder (dt = cfg.dt).
    method="rk4"  : continuous-time version; dt is the RK4 step (default 0.02 s);
                    the window mean is the time average of a(t) over the window.
    """
    N = len(e["bias"])
    e = {k: (v.astype(dtype) if isinstance(v, np.ndarray) else v) for k, v in e.items()}
    B = len(stim_idx) + 1
    Iw = _stim_matrix(e, stim_idx, N).astype(dtype)
    V = np.zeros((B, N), dtype); Sv = np.zeros((B, N), dtype)
    if method == "euler":
        h = cfg.dt
        for _ in range(cfg.n_burn):
            dV, dS, _ = _deriv(e, V, Sv, 0.0)
            V, Sv = V + h * dV, Sv + h * dS
        acc = np.zeros((B, N), dtype)
        for t in range(cfg.n_win):
            I = Iw if t < cfg.n_stim else 0.0
            dV, dS, _ = _deriv(e, V, Sv, I)
            V, Sv = V + h * dV, Sv + h * dS
            acc += _sig(e["k"] * (V - e["theta"]))
        mean = acc / cfg.n_win
    elif method == "rk4":
        h = 0.02 if dt is None else dt

        def rk4(V, Sv, I):
            k1 = _deriv(e, V, Sv, I)[:2]
            k2 = _deriv(e, V + 0.5 * h * k1[0], Sv + 0.5 * h * k1[1], I)[:2]
            k3 = _deriv(e, V + 0.5 * h * k2[0], Sv + 0.5 * h * k2[1], I)[:2]
            k4 = _deriv(e, V + h * k3[0], Sv + h * k3[1], I)[:2]
            return (V + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                    Sv + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))
        for _ in range(int(round(cfg.t_burn / h))):
            V, Sv = rk4(V, Sv, 0.0)
        n = int(round(cfg.t_win / h)); n_on = int(round(cfg.t_stim / h))
        a_prev = _sig(e["k"] * (V - e["theta"])); acc = np.zeros((B, N), dtype)
        for t in range(n):
            V, Sv = rk4(V, Sv, Iw if t < n_on else 0.0)
            a = _sig(e["k"] * (V - e["theta"]))
            acc += 0.5 * (a + a_prev); a_prev = a          # trapezoid rule
        mean = acc / n
    else:
        raise ValueError(method)
    return (e["gain"][:, None] * (mean[1:] - mean[0]).T).astype(np.float64)


def _quantize(x, bits, max_abs=None):
    """Symmetric uniform quantisation to `bits` (sign included); returns float values on the grid."""
    max_abs = np.max(np.abs(x)) if max_abs is None else max_abs
    if max_abs == 0:
        return x.copy()
    q = 2 ** (bits - 1) - 1
    return np.round(x / max_abs * q) / q * max_abs


def simulate_fixed_point(e, stim_idx, cfg, w_bits=8, frac_bits=10, int_bits=5, lut_bits=8):
    """Neuromorphic-style substrate: Euler dt = cfg.dt with
    * synaptic and junction weights quantised to `w_bits`-bit integers (per-matrix scale),
    * all neuron state (v, s) and parameters on a fixed-point grid with `frac_bits`
      fractional bits, saturating at +/- 2**int_bits,
    * the activation sigmoid read from a 2**lut_bits-entry look-up table on [-8, 8].
    """
    N = len(e["bias"])
    grid = 2.0 ** -frac_bits; lim = 2.0 ** int_bits
    fx = lambda x: np.clip(np.round(x / grid) * grid, -lim, lim)
    lut_x = np.linspace(-8, 8, 2 ** lut_bits); lut_y = fx(_sig(lut_x))
    def act(V):
        z = np.clip(e["k"] * (V - th), -8, 8)
        return lut_y[np.round((z + 8) / 16 * (2 ** lut_bits - 1)).astype(int)]
    W = _quantize(e["W"], w_bits); G = _quantize(e["G"], w_bits)
    th, b = fx(e["theta"]), fx(e["bias"])
    cv, cs = fx(cfg.dt / e["tau"]), fx(cfg.dt / e["tau_s"])     # integration constants
    Iw = fx(_stim_matrix(e, stim_idx, N))
    B = len(stim_idx) + 1
    V = np.zeros((B, N)); Sv = np.zeros((B, N)); gsum = G.sum(1)

    def step(V, Sv, I):
        a = act(V)
        drive = fx(fx(Sv @ W.T) + fx(V @ G.T - gsum * V))
        V2 = fx(V + fx(cv * (-V + b + drive + I)))
        S2 = fx(Sv + fx(cs * (-Sv + a)))
        return V2, S2
    for _ in range(cfg.n_burn):
        V, Sv = step(V, Sv, 0.0)
    acc = np.zeros((B, N))
    for t in range(cfg.n_win):
        V, Sv = step(V, Sv, Iw if t < cfg.n_stim else 0.0)
        acc += act(V)
    mean = acc / cfg.n_win
    return e["gain"][:, None] * (mean[1:] - mean[0]).T


def simulate_brian2(e, stim_idx, cfg, dt=0.02, method="rk4"):
    """Brian2 substrate (numpy code generation, no compiler needed).  The S
    stimulations plus one control run as S+1 independent copies of the network
    in one NeuronGroup; synapses are replicated block-diagonally.  The window
    integral of activity is accumulated by an ODE variable gated by a TimedArray."""
    b2 = import_brian2()
    b2.prefs.codegen.target = "numpy"
    b2.start_scope()
    b2.defaultclock.dt = dt * b2.second
    N = len(e["bias"]); C = len(stim_idx) + 1
    T_total = cfg.t_burn + cfg.t_win
    nt = int(round(T_total / dt)) + 2
    t_grid = np.arange(nt) * dt
    stim_on = b2.TimedArray(((t_grid >= cfg.t_burn) & (t_grid < cfg.t_burn + cfg.t_stim)).astype(float),
                            dt=dt * b2.second)
    win_on = b2.TimedArray(((t_grid >= cfg.t_burn) & (t_grid < cfg.t_burn + cfg.t_win)).astype(float),
                           dt=dt * b2.second)
    eqs = """
    dv/dt = (-v + bias + Isyn + Igap + amp_i * stim_on(t)) / tau : 1
    ds/dt = (-s + 1 / (1 + exp(-kk * (v - theta)))) / tau_s : 1
    dacc/dt = win_on(t) / (1 + exp(-kk * (v - theta))) / second : 1
    Isyn : 1
    Igap : 1
    bias : 1 (constant)
    theta : 1 (constant)
    amp_i : 1 (constant)
    tau : second (constant)
    tau_s : second (constant)
    kk : 1 (constant)
    """
    G = b2.NeuronGroup(N * C, eqs, method=method, namespace={"stim_on": stim_on, "win_on": win_on})
    G.bias = np.tile(e["bias"], C); G.theta = np.tile(e["theta"], C)
    G.tau = np.tile(e["tau"], C) * b2.second; G.tau_s = e["tau_s"] * b2.second; G.kk = e["k"]
    amp = np.zeros(N * C)
    amp[np.arange(1, C) * N + np.asarray(stim_idx)] = e["amp"]
    G.amp_i = amp
    post, pre = np.nonzero(e["W"])
    off = (np.arange(C) * N)[:, None]
    syn = b2.Synapses(G, G, "w : 1 (constant)\nIsyn_post = w * s_pre : 1 (summed)")
    syn.connect(i=(pre[None] + off).ravel(), j=(post[None] + off).ravel())
    syn.w = np.tile(e["W"][post, pre], C)
    gpost, gpre = np.nonzero(e["G"])
    gap = b2.Synapses(G, G, "g : 1 (constant)\nIgap_post = g * (v_pre - v_post) : 1 (summed)")
    gap.connect(i=(gpre[None] + off).ravel(), j=(gpost[None] + off).ravel())
    gap.g = np.tile(e["G"][gpost, gpre], C)
    net = b2.Network(G, syn, gap)
    net.run(T_total * b2.second)
    mean = np.asarray(G.acc[:]).reshape(C, N) / cfg.t_win
    return e["gain"][:, None] * (mean[1:] - mean[0]).T


# ------------------------------------------------------------- comparisons ---
def substrate_gap(R_sub, R_ref, ds, cols):
    """Squared substrate difference on observed pairs, relative to the
    measurement noise of those pairs (sum of trial-mean variances).
    < 1: the substrates differ by less than the experiment could detect."""
    m = ds.mask()[0][:, cols]
    return float(((R_sub - R_ref)[m] ** 2).sum() / ds.var_mean[0][:, cols][m].sum())


def feve_on(R, ds, cols, test_mask=None):
    """FEVE of an (N, len(cols)) response matrix against WT data, optionally
    restricted to held-out pairs."""
    m = ds.mask()[0][:, cols].copy()
    if test_mask is not None:
        m &= test_mask[0][:, cols]
    y, nv = ds.mean[0][:, cols][m], ds.var_mean[0][:, cols][m]
    return 1.0 - (((R[m] - y) ** 2).sum() - nv.sum()) / ((y ** 2).sum() - nv.sum())


def degeneracy(eff_list, R_list):
    """Parameter vs behaviour similarity across independently fitted models.
    Returns dict of mean pairwise correlations (parameters on shared support;
    responses on all pairs)."""
    def corr(a, b):
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    supp = eff_list[0]["W"] != 0
    out = {k: [] for k in ("W", "G", "bias", "theta", "tau", "responses")}
    n = len(eff_list)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = eff_list[i], eff_list[j]
            out["W"].append(corr(a["W"][supp], b["W"][supp]))
            gs = a["G"] != 0
            out["G"].append(corr(a["G"][gs], b["G"][gs]))
            for k in ("bias", "theta", "tau"):
                out[k].append(corr(a[k], b[k]))
            out["responses"].append(corr(R_list[i], R_list[j]))
    return {k: (float(np.mean(v)), float(np.min(v)), float(np.max(v))) for k, v in out.items()}


def lyapunov(e, T=300.0, dt=0.02, renorm=1.0, drive="none", pulse_every=2.0, seed=0, d0=1e-8,
             transient=50.0):
    """Largest Lyapunov exponent (1/s) of the fitted network (Benettin method,
    RK4 float64).  drive="pulses": both copies receive the same random sequence
    of 1-s pulses to random neurons (amplitudes from the fitted efficacies).
    Negative: perturbations die out, precision does not matter."""
    rng = np.random.default_rng(seed)
    N = len(e["bias"]); h = dt
    n_steps = int(round((transient + T) / h)); n_tr = int(round(transient / h)); n_ren = int(round(renorm / h))
    n_pulse = int(round(pulse_every / h)); n_on = int(round(1.0 / h))
    V = np.zeros((2, N)); Sv = np.zeros((2, N))
    I = np.zeros((2, N)); log_sum = 0.0; t_meas = 0.0

    def rk4(V, Sv, I):
        k1 = _deriv(e, V, Sv, I)[:2]
        k2 = _deriv(e, V + 0.5 * h * k1[0], Sv + 0.5 * h * k1[1], I)[:2]
        k3 = _deriv(e, V + 0.5 * h * k2[0], Sv + 0.5 * h * k2[1], I)[:2]
        k4 = _deriv(e, V + h * k3[0], Sv + h * k3[1], I)[:2]
        return (V + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                Sv + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))
    for t in range(n_steps):
        if drive == "pulses":
            if t % n_pulse == 0:
                I = np.zeros((2, N)); j = rng.integers(N); I[:, j] = rng.choice(e["amp"])
            elif t % n_pulse == n_on:
                I = np.zeros((2, N))
        V, Sv = rk4(V, Sv, I)
        if t == n_tr:                                        # start measuring: perturb copy 1
            dvec = rng.normal(size=2 * N); dvec *= d0 / np.linalg.norm(dvec)
            V[1] = V[0] + dvec[:N]; Sv[1] = Sv[0] + dvec[N:]
        elif t > n_tr and (t - n_tr) % n_ren == 0:
            diff = np.concatenate([V[1] - V[0], Sv[1] - Sv[0]]); d = np.linalg.norm(diff)
            log_sum += np.log(max(d, 1e-300) / d0); t_meas += renorm
            V[1] = V[0] + (V[1] - V[0]) * d0 / d; Sv[1] = Sv[0] + (Sv[1] - Sv[0]) * d0 / d
    return log_sum / t_meas
