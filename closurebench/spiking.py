"""E2b - equation-level substrate transfer: a SPIKING realization of the fitted L1.

E2 (Notebook 05) ran the fitted L1 on numerical substrates that share its
equations (RK4, Brian2, fixed point) and found it invariant.  E2b changes the
EQUATIONS.  Each rate unit i of L1,

    tau_i dV_i/dt = -V_i + bias_i + sum_j W_ij s_j + gap_i(V) + I_i
    tau_s ds_i/dt = -s_i + a_i,        a_i = sigmoid(k (V_i - theta_i)),

keeps its graded input variable V_i, but its output nonlinearity and synaptic
transmission are carried by a population of M spiking neurons.  Three
substrates separate the two ways this can fail:

  lif_rate  M leaky integrate-and-fire tuning curves (analytic rates, no spikes):
            a_i is replaced by a DECODED sum of LIF rates (Neural Engineering
            Framework: decoders solved by regularised least squares so that the
            population's rates approximate the sigmoid).  Error: decoding only.
  poisson   M Poisson neurons firing at r_max * sigmoid(.): the mean is exact,
            the synapse sees shot noise.  Error: spike noise only.
  lif       M spiking LIF neurons (tau_m = 20 ms, refractory 2 ms, heterogeneous
            intercepts and maximal rates, on/off encoders), decoded output
            spikes drive s.  Both errors.

The measured activity is the decoded population output averaged over the
response window (the spiking analogue of the window mean of a).

Note: C. elegans neurons are mostly graded, so this is a test of the fitted
program's SUBSTRATE independence, not a claim about worm biology.  The question
is how many spiking neurons a graded unit of the program needs.
"""
from __future__ import annotations
import numpy as np
from .substrates import _sig, _stim_matrix

TAU_M, T_REF = 0.02, 0.002
X_RANGE = 8.0                 # the sigmoid's argument x = k (V - theta) is represented on [-8, 8];
                              # inputs beyond it are clipped (saturating input stage; sigmoid(8) = 1 - 3e-4)


def lif_rate(J, tau_m=TAU_M, t_ref=T_REF):
    """Steady-state LIF firing rate (Hz) for normalised input current J (threshold 1)."""
    J = np.asarray(J, float)
    out = np.zeros_like(J)
    m = J > 1.0
    out[m] = 1.0 / (t_ref - tau_m * np.log1p(-1.0 / J[m]))
    return out


def make_population(M, seed=0, rmax=(60.0, 150.0), reg=0.01):
    """NEF population representing x on [-X_RANGE, X_RANGE] and decoding sigmoid(x).
    Half the neurons are 'on' (fire for x above their intercept), half 'off'.
    Returns dict(enc, alpha, beta, dec, rmse) with decoders solved by ridge
    regression with regularisation reg * max rate (NEF convention; 0.01 = a
    precise compiler; spike noise is then handled by the population size)."""
    rng = np.random.default_rng(seed)
    enc = np.where(np.arange(M) % 2 == 0, 1.0, -1.0)
    c = rng.uniform(-0.95 * X_RANGE, 0.95 * X_RANGE, M)          # intercepts in e*x units
    r = rng.uniform(*rmax, M)
    Jmax = 1.0 / (1.0 - np.exp((T_REF - 1.0 / r) / TAU_M))         # current giving rate r
    alpha = (Jmax - 1.0) / (X_RANGE - c)
    beta = 1.0 - alpha * c
    x = np.linspace(-X_RANGE, X_RANGE, 401)
    A = lif_rate(alpha * (enc * x[:, None]) + beta)                 # (401, M)
    lam = (reg * A.max()) ** 2 * len(x)
    dec = np.linalg.solve(A.T @ A + lam * np.eye(M), A.T @ _sig(x))
    rmse = float(np.sqrt(np.mean((A @ dec - _sig(x)) ** 2)))
    return dict(enc=enc, alpha=alpha, beta=beta, dec=dec, rmse=rmse, M=M)


def _act_rate(e, V, pop):
    """Rate-level activation of each unit: exact sigmoid (pop None) or the decoded
    LIF tuning curves.  V: (B, N)."""
    x = e["k"] * (V - e["theta"])
    if pop is None:
        return _sig(x)
    x = np.clip(x, -X_RANGE, X_RANGE)
    J = pop["alpha"] * (pop["enc"] * x[..., None]) + pop["beta"]
    return lif_rate(J) @ pop["dec"]


def _drive(e, V, Sv, I):
    lap = V @ e["G"].T - e["G"].sum(1) * V
    return (-V + e["bias"] + Sv @ e["W"].T + lap + I) / e["tau"]


def simulate_spiking(e, stim_idx, cfg, kind="lif", M=30, dt=1e-3, dt_rate=0.02, t_settle=2.0,
                     r_poisson=100.0, seed=0, pop=None):
    """Response matrix (N, S) of the fitted L1 (effective params e, from
    substrates.effective_params) realised on substrate `kind`
    ('lif_rate', 'poisson' or 'lif') with M neurons per unit.

    Protocol (as in E2): burn-in to rest (cfg.t_burn, rate-level dynamics of the
    same substrate, Euler dt_rate), then for spiking substrates t_settle s of
    spiking without stimulus, then the response window (cfg.t_win, stimulus on
    for cfg.t_stim at its start).  Each stimulation and the control run as
    independent copies (independent spike noise)."""
    rng = np.random.default_rng(seed)
    N = len(e["bias"]); B = len(stim_idx) + 1
    Iw = _stim_matrix(e, stim_idx, N)
    if kind in ("lif", "lif_rate") and pop is None:
        pop = make_population(M, seed=seed)
    rate_pop = pop if kind in ("lif", "lif_rate") else None
    V = np.zeros((B, N)); Sv = np.zeros((B, N))
    for _ in range(int(round(cfg.t_burn / dt_rate))):                 # rest, rate level
        a = _act_rate(e, V, rate_pop)
        V, Sv = V + dt_rate * _drive(e, V, Sv, 0.0), Sv + dt_rate * (-Sv + a) / e["tau_s"]
    if kind == "lif_rate":
        n = int(round(cfg.t_win / dt_rate)); n_on = int(round(cfg.t_stim / dt_rate)); acc = np.zeros((B, N))
        for t in range(n):
            a = _act_rate(e, V, rate_pop)
            V, Sv = V + dt_rate * _drive(e, V, Sv, Iw if t < n_on else 0.0), Sv + dt_rate * (-Sv + a) / e["tau_s"]
            acc += _act_rate(e, V, rate_pop)
        mean = acc / n
        return e["gain"][:, None] * (mean[1:] - mean[0]).T
    if kind == "lif":
        v = rng.uniform(0, 1, (B, N, M)); ref = np.zeros((B, N, M))
    n_settle = int(round(t_settle / dt)); n = int(round(cfg.t_win / dt)); n_on = int(round(cfg.t_stim / dt))
    acc = np.zeros((B, N))
    for t in range(n_settle + n):
        x = e["k"] * (V - e["theta"])
        if kind == "poisson":
            spikes = rng.binomial(M, np.clip(r_poisson * _sig(x) * dt, 0, 1))
            out = spikes / (M * r_poisson)                                  # decoded impulse (area = activity x dt)
        else:
            J = pop["alpha"] * (pop["enc"] * np.clip(x, -X_RANGE, X_RANGE)[..., None]) + pop["beta"]
            active = ref <= 0
            v = np.where(active, v + dt / TAU_M * (J - v), v)
            ref -= dt
            spk = v >= 1.0
            v[spk] = 0.0; ref[spk] = T_REF
            out = spk.astype(float) @ pop["dec"]
        I = Iw if n_settle <= t < n_settle + n_on else 0.0
        V = V + dt * _drive(e, V, Sv, I)
        Sv = Sv + dt * (-Sv / e["tau_s"]) + out / e["tau_s"]
        if t >= n_settle:
            acc += out
    mean = acc / cfg.t_win
    return e["gain"][:, None] * (mean[1:] - mean[0]).T


def estimate_seconds(e, n_stim, cfg, kind, M, dt=1e-3, t_settle=2.0, probe_steps=200):
    """Rough wall-clock estimate of simulate_spiking, from timing a few steps."""
    import time
    N = len(e["bias"]); B = n_stim + 1
    if kind == "lif_rate":
        return 0.0
    pop = make_population(M) if kind == "lif" else None
    V = np.zeros((B, N)); Sv = np.zeros((B, N)); v = np.zeros((B, N, M)); ref = np.zeros((B, N, M))
    rng = np.random.default_rng(0)
    t0 = time.time()
    for _ in range(probe_steps):
        x = e["k"] * (V - e["theta"])
        if kind == "poisson":
            out = rng.binomial(M, np.clip(100 * _sig(x) * dt, 0, 1)) / (M * 100.0)
        else:
            J = pop["alpha"] * (pop["enc"] * x[..., None]) + pop["beta"]
            v = np.where(ref <= 0, v + dt / TAU_M * (J - v), v); ref -= dt
            spk = v >= 1.0; v[spk] = 0.0; ref[spk] = T_REF; out = spk.astype(float) @ pop["dec"]
        V = V + dt * _drive(e, V, Sv, 0.0); Sv = Sv + dt * (-Sv / e["tau_s"]) + out / e["tau_s"]
    per = (time.time() - t0) / probe_steps
    return per * (t_settle + cfg.t_win) / dt
