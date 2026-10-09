"""The ladder of formal descriptions of one worm, L0-L4, plus a black box B.

Every rung simulates the same experiment as the signal-propagation atlas:
from rest, inject a brief current pulse into one neuron, and read out each
neuron's mean activity change over a post-stimulus window.  Rungs are nested:
each adds one class of mechanism and nothing else.

    L0  linear rate network on the connectome; wiring + signs + 4 global scales
    L1  nonlinear graded network: per-synapse chemical weights with synaptic
        dynamics, per-junction gap conductances, per-neuron tau / threshold / bias
    L2  L1 + intrinsic excitability: slow K+ adaptation and a regenerative
        inward current per neuron (conductance-style terms)
    L3  L2 + extrasynaptic neuropeptide channel: slow release, diffuse action
        through the peptidergic connectome (low-rank: release gain x receptor gain);
        switched off in the unc-31 strain
    L4  L3 + activity-dependent slow state: homeostatic integral feedback
    B   black box: a transformer over neuron tokens, no mechanism

This is a fast, differentiable *surrogate* of the proposal's NeuroML ->
Brian2/Jaxley ladder (dimensionless units, Euler steps).  It is what makes the
E1 pipeline testable now; Notebook 05 ports L1-L2 to Jaxley/Brian2 for E2.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import partial
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

LEVELS = ("L0", "L1", "L2", "L3", "L4")
LAST_FIT_TIMES = []
N_STATE = {"L0": 1, "L1": 2, "L2": 3, "L3": 4, "L4": 5, "L1s": 3, "L0s": 2}  # state variables per neuron
# "L1s" (Notebook 22) / "L0s" (Notebook 24): L1 / L0 + ONE global slow current (2 extra numbers: rho, tau_n), not part of LEVELS
sp = jax.nn.softplus
TAU_MIN = 0.4   # s; lower bound on fast time constants keeps Euler (dt=0.2) stable
sig = jax.nn.sigmoid


def inv_sp(x):
    return np.log(np.expm1(x))


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.2          # s
    t_burn: float = 30.0     # s, relaxation to rest (no stimulus)
    t_stim: float = 1.0      # s, optogenetic pulse
    t_win: float = 20.0      # s, response window after pulse onset
    k: float = 4.0           # slope of activation nonlinearities
    n_bins: int = 1          # time bins in the response window (1 = window mean, as in E1)

    @property
    def n_burn(self):
        return int(round(self.t_burn / self.dt))

    @property
    def n_win(self):
        return int(round(self.t_win / self.dt))

    @property
    def bin_len(self):
        assert self.n_win % self.n_bins == 0, "n_win must be divisible by n_bins"
        return self.n_win // self.n_bins

    @property
    def n_stim(self):
        return int(round(self.t_stim / self.dt))


# ------------------------------------------------------------ structure ---
def neuron_classes(ids):
    """Group bilateral homologs (XXXL / XXXR) into one class when both exist.
    Returns an int label per neuron."""
    ids = [str(i) for i in ids]
    present = set(ids)
    names = []
    for n in ids:
        if len(n) > 2 and n[-1] in "LR" and (n[:-1] + ("R" if n[-1] == "L" else "L")) in present:
            names.append(n[:-1])
        else:
            names.append(n)
    uniq = {c: k for k, c in enumerate(dict.fromkeys(names))}
    return np.array([uniq[c] for c in names])


def tie(st, p):
    """Apply bilateral tying: per-neuron vectors -> class means; synaptic and
    junction matrices -> class-pair means over existing contacts."""
    out = dict(p)
    N = st.N
    for k, v in p.items():
        if k in ("W", "G"):
            mask = st.chem_mask if k == "W" else st.gap_mask
            num = st.Csum @ (v * mask) @ st.Csum.T
            den = st.Csum @ mask @ st.Csum.T
            out[k] = jnp.where(den > 0, num / jnp.maximum(den, 1e-9), v)
        elif hasattr(v, "shape") and v.shape == (N,) and k != "log_gain":
            out[k] = st.C @ v
    return out



@dataclass
class Structure:
    """Fixed (non-trainable) wiring information derived from a Dataset."""
    N: int
    chem_mask: jnp.ndarray   # (N,N) 1 where a chemical synapse exists
    chem_init: jnp.ndarray   # (N,N) sign * normalised log-count (init for L1+ weights)
    S_exc: jnp.ndarray       # (N,N) row-normalised excitatory / inhibitory / unknown (L0)
    S_inh: jnp.ndarray
    S_unk: jnp.ndarray
    gap_mask: jnp.ndarray    # (N,N) symmetric
    gap_norm: jnp.ndarray    # (N,N) normalised gap weights (L0)
    pep: jnp.ndarray         # (N,N) row-normalised peptidergic connectome
    C: jnp.ndarray           # (N,N) class-averaging operator (identity if untied)
    Csum: jnp.ndarray        # (N,N) class-sum operator (block of ones within a class)

    @staticmethod
    def from_dataset(ds, tie=True):
        N = ds.N
        chem = np.asarray(ds.chem, float)
        np.fill_diagonal(chem, 0)
        lc = np.log1p(chem)
        indeg = lc.sum(1, keepdims=True) + 1e-9

        def rown(M):
            return M / indeg
        sign = np.asarray(ds.sign) * (chem > 0)
        gap = np.log1p(np.asarray(ds.gap, float))
        np.fill_diagonal(gap, 0)
        gap = 0.5 * (gap + gap.T)
        gdeg = gap.sum(1, keepdims=True) + 1e-9
        pep = np.log1p(np.asarray(ds.pep, float))
        np.fill_diagonal(pep, 0)
        pep = pep / (pep.sum(1, keepdims=True) + 1e-9)
        init = np.where(sign == 0, 0.3, sign) * lc / indeg * 4.0
        cls = neuron_classes(ds.ids) if tie else np.arange(N)
        Csum = (cls[:, None] == cls[None, :]).astype(float)
        C = Csum / Csum.sum(1, keepdims=True)
        f = lambda a: jnp.asarray(a, jnp.float32)
        return Structure(N=N, C=f(C), Csum=f(Csum), chem_mask=f(chem > 0), chem_init=f(init),
                         S_exc=f(rown(lc * (sign > 0))), S_inh=f(rown(lc * (sign < 0))),
                         S_unk=f(rown(lc * (sign == 0))), gap_mask=f(gap > 0),
                         gap_norm=f(gap / gdeg.max()), pep=f(pep))


_ST_FIELDS = ("chem_mask", "chem_init", "S_exc", "S_inh", "S_unk", "gap_mask", "gap_norm", "pep", "C", "Csum")
jax.tree_util.register_pytree_node(
    Structure,
    lambda s: (tuple(getattr(s, f) for f in _ST_FIELDS), s.N),
    lambda N, xs: Structure(N, *xs))


# ---------------------------------------------------------------- params ---
def init_params(level, st: Structure, key, jitter=0.1):
    """Unconstrained parameters for a rung.  `jitter` scales random
    perturbations around a sensible default (larger -> more varied dynamics,
    used to generate synthetic ground truths)."""
    N = st.N
    ks = iter(jax.random.split(key, 32))
    r = lambda shape, s=1.0: jitter * s * jax.random.normal(next(ks), shape)
    p = {"log_gain": jnp.zeros(N) + r((N,), 0.5), "stim": jnp.array(inv_sp(2.0))}
    if level in ("L0", "L0s"):
        p.update(a_exc=jnp.array(0.5), a_inh=jnp.array(-0.5), a_unk=jnp.array(0.0),
                 g_gap=jnp.array(inv_sp(0.3)), tau=jnp.array(inv_sp(0.6)))
        if level == "L0s":
            p.update(rho_s=jnp.array(float(np.log(0.05 / 0.85))), tau_ns=jnp.array(inv_sp(10.0 - TAU_MIN)))
        return p
    bias = jnp.full(N, -0.2) + r((N,), 0.5)
    p.update(W=st.chem_init * (1 + r((N, N))) + r((N, N), 0.2) * st.chem_mask,
             G=jnp.full((N, N), inv_sp(1.0)) + r((N, N)),
             tau=jnp.full(N, inv_sp(0.2)) + r((N,)),
             theta=bias + r((N,), 0.3),      # neurons start near their operating point
             bias=bias,
             tau_s=jnp.array(inv_sp(0.2)))
    if level == "L1s":                   # global slow K+-like current (Notebooks 18-21): rho = 0.9 sig(rho_s)
        p.update(rho_s=jnp.array(float(np.log(0.05 / 0.85))), tau_ns=jnp.array(inv_sp(10.0 - TAU_MIN)))
    if level in ("L2", "L3", "L4"):
        p.update(gK=jnp.full(N, inv_sp(0.3)) + r((N,), 2.0),
                 th_w=jnp.full(N, 0.0) + r((N,), 0.5),
                 gCa=jnp.full(N, inv_sp(0.2)) + r((N,), 2.0),
                 th_ca=jnp.full(N, 0.3) + r((N,), 0.5),
                 tau_w=jnp.array(inv_sp(3.0)))
    if level in ("L3", "L4"):
        p.update(rel=jnp.full(N, inv_sp(1.0)) + r((N,), 3.0),
                 rec=r((N,), 2.0),
                 tau_p=jnp.array(inv_sp(8.0)))
    if level == "L4":
        p.update(eta=jnp.full(N, inv_sp(0.3)) + r((N,), 2.0),
                 rho=jnp.full(N, 0.2) + r((N,)),
                 tau_h=jnp.array(inv_sp(20.0)))
    return p


NEW_MECHANISM_OFF = {"gK": inv_sp(0.02), "gCa": inv_sp(0.02), "rec": 0.0, "eta": inv_sp(0.02),
                     "rho_s": float(np.log(0.01 / 0.89))}


def extend_params(p_lower, level, st: Structure, key):
    """Warm start for a nested rung: copy everything the lower rung fitted and
    switch the newly added mechanism (nearly) off, so the deeper rung starts
    exactly where its parent ended."""
    p = init_params(level, st, key, jitter=0.0)
    for k, v in p_lower.items():
        if k in p or k == "stim_col":
            p[k] = v
    for k in p:
        if k not in p_lower and k in NEW_MECHANISM_OFF:
            p[k] = jnp.full_like(p[k], NEW_MECHANISM_OFF[k])
    return p


def trainable_count(level, st: Structure):
    """Effective free parameters after tying (masked weights counted on their
    support; tied neurons / class pairs counted once)."""
    N = st.N
    Csum = np.asarray(st.Csum)
    _, first = np.unique(Csum, axis=0, return_index=True)
    O = Csum[:, np.sort(first)]          # (N, n_classes) one-hot class membership
    n_cls = O.shape[1]
    pair = lambda M: int(((O.T @ np.asarray(M) @ O) > 0).sum())
    n_chem = pair(st.chem_mask)
    n_gap = pair(st.gap_mask) // 2
    c = N + 1  # readout gains (per neuron: measurement, never tied) + stimulus amplitude
    if level in ("L0", "L0s"):
        return c + 5 + (2 if level == "L0s" else 0)
    c += n_chem + n_gap + 3 * n_cls + 1
    if level == "L1s":
        c += 2
    if level in ("L2", "L3", "L4"):
        c += 4 * n_cls + 1
    if level in ("L3", "L4"):
        c += 2 * n_cls + 1
    if level == "L4":
        c += 2 * n_cls + 1
    return c


# ------------------------------------------------------------- dynamics ---
def _step(level, p, st, cfg, state, I, pep_on, gap_on=1.0):
    """One Euler step.  state: dict of (N,) arrays.  pep_on / gap_on switch
    neuropeptide release (unc-31) and gap junctions (e.g. unc-7/unc-9) off."""
    dt, k = cfg.dt, cfg.k
    v = state["v"]
    if level in ("L0", "L0s"):
        W = p["a_exc"] * st.S_exc + p["a_inh"] * st.S_inh + p["a_unk"] * st.S_unk
        lap = gap_on * (st.gap_norm @ v - st.gap_norm.sum(1) * v)
        if level == "L0s":               # a share rho of the leak acts through a slow gate u (as in L1s)
            rho = 0.9 * sig(p["rho_s"]); u = state["u"]
            dv = (-(1 - rho) * v - rho * u + W @ v + sp(p["g_gap"]) * lap + I) / (TAU_MIN + sp(p["tau"]))
            return {"v": v + dt * dv, "u": u + dt * (-u + v) / (TAU_MIN + sp(p["tau_ns"]))}
        dv = (-v + W @ v + sp(p["g_gap"]) * lap + I) / (TAU_MIN + sp(p["tau"]))
        return {"v": v + dt * dv}
    a = sig(k * (v - p["theta"]))
    W = p["W"] * st.chem_mask
    G = sp(0.5 * (p["G"] + p["G"].T)) * st.gap_norm   # row sums <= max conductance
    lap = gap_on * (G @ v - G.sum(1) * v)
    cur = p["bias"] + W @ state["s"] + lap + I
    new = {}
    if level in ("L2", "L3", "L4"):
        w = state["w"]
        cur = cur - sp(p["gK"]) * w * (v + 1.0) + sp(p["gCa"]) * sig(k * (v - p["th_ca"]))
        new["w"] = w + dt * (-w + sig(k * (v - p["th_w"]))) / sp(p["tau_w"])
    if level in ("L3", "L4"):
        pp = state["p"]
        cur = cur + p["rec"] * (st.pep @ (sp(p["rel"]) * pp))
        new["p"] = pp + dt * (-pp + pep_on * a) / sp(p["tau_p"])
    if level == "L4":
        h = state["h"]
        cur = cur - sp(p["eta"]) * h
        new["h"] = h + dt * (a - p["rho"]) / sp(p["tau_h"])
    if level == "L1s":                   # a share rho of the leak acts through a slow gate u
        rho = 0.9 * sig(p["rho_s"]); u = state["u"]
        new["u"] = u + dt * (-u + v) / (TAU_MIN + sp(p["tau_ns"]))
        new["v"] = v + dt * (-(1 - rho) * v - rho * u + cur) / (TAU_MIN + sp(p["tau"]))
    else:
        new["v"] = v + dt * (-v + cur) / (TAU_MIN + sp(p["tau"]))
    new["s"] = state["s"] + dt * (-state["s"] + a) / (TAU_MIN + sp(p["tau_s"]))
    return new


def _activity(level, p, cfg, state):
    if level in ("L0", "L0s"):
        return state["v"]
    return sig(cfg.k * (state["v"] - p["theta"]))


def _zero_state(level, N):
    names = {"L0": ["v"], "L0s": ["v", "u"], "L1": ["v", "s"], "L1s": ["v", "s", "u"], "L2": ["v", "s", "w"],
             "L3": ["v", "s", "w", "p"], "L4": ["v", "s", "w", "p", "h"]}[level]
    return {n: jnp.zeros(N) for n in names}


PATTERNS = ("single", "train", "long")   # temporal stimulus patterns (see pulse_patterns)


def pulse_patterns(cfg):
    """(len(PATTERNS), n_win) on/off masks of the stimulus within the response window:
    single = one pulse of t_stim (the atlas protocol); train = three t_stim pulses
    4 s apart; long = one 5-s pulse."""
    t = np.arange(cfg.n_win) * cfg.dt
    single = t < cfg.t_stim
    train = np.zeros_like(single)
    for k in range(3):
        train |= (t >= 4.0 * k) & (t < 4.0 * k + cfg.t_stim)
    long = t < 5.0
    return np.stack([single, train, long]).astype(np.float32)


@partial(jax.jit, static_argnames=("level", "cfg"))
def simulate_responses(p, st, stim_idx, level, cfg, pep_on=1.0, amp_off=None, gap_on=1.0, proto=None):
    """Predicted response matrix (N, S): mean activity change of each neuron
    over the window following a pulse to neuron stim_idx[s], times readout gain.

    proto (optional, per stimulus column): dict with "amp" (amplitude scale),
    "pair" (index of a second neuron stimulated simultaneously, -1 = none) and
    "pattern" (index into PATTERNS).  None = the atlas protocol."""
    N = st.N
    p = tie(st, p)
    zero_I = jnp.zeros(N)

    @jax.checkpoint   # rematerialise during backprop: store states only (memory)
    def burn(state, _):
        ns = _step(level, p, st, cfg, state, zero_I, pep_on, gap_on)
        if level in ("L1s", "L0s"):      # gate tracks V during burn-in: at stimulus onset it sits at rest
            ns = dict(ns, u=ns["v"])     # (Notebooks 20-21: a gate frozen before rest acts as a hidden input)
        return ns, None
    rest, _ = jax.lax.scan(burn, _zero_state(level, N), None, length=cfg.n_burn)

    pulses = jnp.asarray(pulse_patterns(cfg))

    def window_mean(I_vec, pulse):
        """(N,) window mean, or (n_bins, N) bin means when cfg.n_bins > 1."""
        @jax.checkpoint
        def body(carry, t):
            state, acc = carry
            ns = _step(level, p, st, cfg, state, pulse[t] * I_vec, pep_on, gap_on)
            return (ns, acc.at[t // cfg.bin_len].add(_activity(level, p, cfg, ns))), None
        (_, acc), _ = jax.lax.scan(body, (rest, jnp.zeros((cfg.n_bins, N))), jnp.arange(cfg.n_win))
        acc = acc / cfg.bin_len
        return acc[0] if cfg.n_bins == 1 else acc
    # matched no-stimulus control removes any slow drift left after burn-in
    control = window_mean(jnp.zeros(N), pulses[0])
    S = stim_idx.shape[0]
    off = jnp.zeros(S) if amp_off is None else amp_off
    if proto is None:
        amp_s, pair, pat = jnp.ones(S), -jnp.ones(S, dtype=jnp.int32), jnp.zeros(S, dtype=jnp.int32)
    else:
        amp_s, pair, pat = jnp.asarray(proto["amp"]), jnp.asarray(proto["pair"]), jnp.asarray(proto["pattern"])
    def one(j, o, a, q, k):     # one_hot(-1) is all zeros, so q = -1 means "no second neuron"
        return window_mean(a * sp(p["stim"] + o) * (jax.nn.one_hot(j, N) + jax.nn.one_hot(q, N)), pulses[k]) - control
    R = jax.vmap(one)(stim_idx, off, amp_s, pair, pat)
    if cfg.n_bins == 1:
        return jnp.exp(p["log_gain"])[:, None] * R.T                       # (N, S)
    return jnp.exp(p["log_gain"])[:, None, None] * jnp.transpose(R, (2, 0, 1))   # (N, S, n_bins)


def predict(level, p, st, stim_idx, strains, cfg, cols=None, proto=None):
    """(n_strain, N, S).  Strain 'unc31' switches peptide release off; strain
    'gapless' switches gap junctions off (an unc-7/unc-9-like mutant).
    proto: optional per-column stimulus protocol (see simulate_responses).
    `cols` (indices into ds.stim) selects per-stimulus efficacy offsets when
    the parameters contain them (pair-holdout protocol)."""
    out = []
    off = None
    if "stim_col" in p and cols is not None:
        off = p["stim_col"][jnp.asarray(cols)]
    for s in strains:
        pep_on = 0.0 if s == "unc31" else 1.0
        gap_on = 0.0 if s == "gapless" else 1.0
        if level == "B":
            r = blackbox_forward(p, stim_idx, pep_on)
            if off is not None:
                r = r * (jnp.exp(off)[None, :] if r.ndim == 2 else jnp.exp(off)[None, :, None])
            out.append(r)
        else:
            out.append(simulate_responses(p, st, jnp.asarray(stim_idx), level, cfg, pep_on, off, gap_on, proto))
    return jnp.stack(out)


# ------------------------------------------------------------ black box ---
def init_blackbox(N, key, d=32, n_layers=2, n_heads=4, n_bins=1):
    ks = iter(jax.random.split(key, 8 + 6 * n_layers))
    nrm = lambda shape, s: s * jax.random.normal(next(ks), shape)
    p = {"emb": nrm((N, d), 0.3), "e_stim": nrm((d,), 0.3), "e_pep": nrm((d,), 0.3),
         "out_w": nrm((d,) if n_bins == 1 else (d, n_bins), 0.1),
         "out_b": jnp.zeros(() if n_bins == 1 else (n_bins,)), "n_heads": n_heads, "layers": []}
    for _ in range(n_layers):
        p["layers"].append({"qkv": nrm((d, 3 * d), d ** -0.5), "o": nrm((d, d), d ** -0.5),
                            "f1": nrm((d, 2 * d), d ** -0.5), "f2": nrm((2 * d, d), (2 * d) ** -0.5)})
    return p


def _ln(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-5)


def blackbox_forward(p, stim_idx, pep_on):
    """Neuron tokens = learned identity embedding (+ stim flag) (+ strain flag);
    transformer layers; per-token linear readout of the response."""
    N, d = p["emb"].shape
    H = p["n_heads"]

    def one(j):
        x = p["emb"] + jax.nn.one_hot(j, N)[:, None] * p["e_stim"] + pep_on * p["e_pep"]
        for L in p["layers"]:
            q, k_, v = jnp.split(_ln(x) @ L["qkv"], 3, -1)
            sh = lambda t: t.reshape(N, H, d // H).transpose(1, 0, 2)
            att = jax.nn.softmax(sh(q) @ sh(k_).transpose(0, 2, 1) / np.sqrt(d // H), -1)
            x = x + (att @ sh(v)).transpose(1, 0, 2).reshape(N, d) @ L["o"]
            x = x + jax.nn.gelu(_ln(x) @ L["f1"]) @ L["f2"]
        return _ln(x) @ p["out_w"] + p["out_b"]
    out = jax.vmap(one)(jnp.asarray(stim_idx))          # (S, N) or (S, N, n_bins)
    return jnp.swapaxes(out, 0, 1)


def blackbox_count(p):
    leaves = [l for l in jax.tree_util.tree_leaves(p) if hasattr(l, "shape")]
    return int(sum(np.prod(l.shape) for l in leaves))


# --------------------------------------------------------------- fitting ---
def proto_cols(ds, cols):
    """The dataset's per-column stimulus protocol restricted to `cols`, or None."""
    pr = getattr(ds, "proto", None)
    if pr is None:
        return None
    cols = np.asarray(cols)
    return {k: jnp.asarray(np.asarray(v)[cols]) for k, v in pr.items()}


def bcast(mask, like):
    """Append singleton axes to a (strain, N, S) mask so it broadcasts against
    arrays with trailing time-bin axes."""
    mask = np.asarray(mask)
    return mask.reshape(mask.shape + (1,) * (np.ndim(like) - mask.ndim))


def fit(level, ds, train_cols=None, cfg=SimConfig(), steps=1500, lr=1e-2, seed=0,
        l2=1e-3, init=None, verbose=200, st=None, tie_classes=True, train_mask=None,
        per_stim_gain=None):
    """Fit one rung, all strains jointly.  Returns params, losses.

    Two protocols:
      * train_cols  (indices into ds.stim): fit those stimuli completely
        (held-out-stimulus protocol);
      * train_mask  (n_strain, N, S) bool: fit only these (responder, stimulus)
        pairs of every stimulus (held-out-pair protocol).  Each stimulus then
        gets an efficacy offset (actuator expression differs across animals).
    """
    expected = ds.n.shape + ((cfg.n_bins,) if cfg.n_bins > 1 else ())
    if ds.mean.shape != expected:
        raise ValueError(f"data shape {ds.mean.shape} does not match SimConfig(n_bins={cfg.n_bins}) -> {expected}; "
                         "use window-mean data with n_bins=1, or time-binned data with the same n_bins")
    if train_mask is not None:
        train_cols = np.arange(ds.S)
    per_stim_gain = (train_mask is not None) if per_stim_gain is None else per_stim_gain
    st = Structure.from_dataset(ds, tie=tie_classes) if st is None else st
    key = jax.random.PRNGKey(seed)
    if init is not None:
        p0 = init
    elif level == "B":
        p0 = init_blackbox(ds.N, key, n_bins=cfg.n_bins)
    else:
        p0 = init_params(level, st, key, jitter=0.05 if seed < 1000 else 0.3)  # restarts explore more
    static = {}
    if level == "B":
        p0 = dict(p0)                                    # never mutate a caller's init
        static["n_heads"] = int(p0.pop("n_heads"))
    train_cols = np.asarray(train_cols)
    if per_stim_gain and "stim_col" not in p0:
        p0["stim_col"] = jnp.zeros(ds.S)
    stim_idx = jnp.asarray(ds.stim[train_cols])
    proto = proto_cols(ds, train_cols)
    y = jnp.asarray(ds.mean[:, :, train_cols], jnp.float32)
    mk = ds.mask() if train_mask is None else (ds.mask() & train_mask)
    m = jnp.asarray(bcast(mk[:, :, train_cols], y), jnp.float32)   # mask, broadcast over time bins
    w = m / (jnp.asarray(ds.var_mean[:, :, train_cols], jnp.float32) + 1e-3)
    w = w / jnp.maximum(jnp.sum(w * y ** 2), 1e-12)   # data loss = relative squared error, O(1)
    strains = ds.strains
    if level == "B" and init is None:
        pred0 = np.asarray(predict("B", dict(p0, **static), st, stim_idx, strains, cfg, train_cols, proto))
        mm = np.broadcast_to(np.asarray(m) > 0, pred0.shape)
        scale = np.asarray(y)[mm].std() / (pred0[mm].std() + 1e-12)
        p0["out_w"] = p0["out_w"] * scale * 0.05   # start near the zero-response predictor
        p0["out_b"] = jnp.zeros(())
    if level == "B":
        lr = lr * 0.1                              # (0.21: a warm-started B keeps its readout)
    if level != "B" and init is None:
        # start at the data's scale: least-squares global readout gain
        pred0 = np.asarray(predict(level, p0, st, stim_idx, strains, cfg, train_cols, proto))
        mm = np.broadcast_to(np.asarray(m) > 0, pred0.shape)
        num, den = (pred0[mm] * np.asarray(y)[mm]).sum(), (pred0[mm] ** 2).sum()
        if den > 0 and num > 0:
            p0["log_gain"] = p0["log_gain"] + np.log(num / den)
    p_ref = jax.tree_util.tree_map(jnp.array, p0)

    # data and wiring are passed as ARGUMENTS (not closed-over constants), so XLA
    # does not constant-fold large matrices at compile time
    def loss_fn(p, st_, y_, w_, p_ref_):
        q = dict(p, **static)
        pred = predict(level, q, st_, stim_idx, strains, cfg, train_cols, proto)
        reg = sum(jnp.sum((a - b) ** 2) for a, b in
                  zip(jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(p_ref_)))
        return jnp.sum(w_ * (pred - y_) ** 2) + l2 * reg

    sched = optax.warmup_cosine_decay_schedule(0.0, lr, max(1, min(100, steps // 10)), max(steps, 2), lr * 0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    state = opt.init(p0)

    @jax.jit
    def update(p, s, st_, y_, w_, p_ref_):
        l, g = jax.value_and_grad(loss_fn)(p, st_, y_, w_, p_ref_)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, l
    p, losses = p0, []
    global LAST_FIT_TIMES
    LAST_FIT_TIMES = [time.perf_counter()]      # per-step wall clock (used by config.estimate_runtime)
    for it in range(steps):
        p, state, l = update(p, state, st, y, w, p_ref)
        losses.append(float(l))                 # float() synchronises, so timings are real
        LAST_FIT_TIMES.append(time.perf_counter())
        if not np.isfinite(losses[-1]):
            print(f"  [{level}] non-finite loss at step {it}; stopping")
            break
        if verbose and (it % verbose == 0 or it == steps - 1):
            print(f"  [{level}] step {it:5d}  loss {losses[-1]:.5f}")
    return dict(p, **static), np.array(losses)


def description_length(level, ds, params=None, st=None):
    """(free parameters, state dimensions) for the compression-cost curve."""
    extra = int(params["stim_col"].shape[0]) if params is not None and "stim_col" in params else 0
    if level == "B":
        n = blackbox_count({k: v for k, v in params.items() if k not in ("n_heads", "stim_col")})
        return n + extra, ds.N * params["emb"].shape[1]
    st = Structure.from_dataset(ds) if st is None else st
    return trainable_count(level, st) + extra, N_STATE[level] * ds.N


# -------------------------------------------------------- synthetic worms ---
def make_synthetic(ds_template, true_level="L3", seed=0, jitter=0.6, noise_sd=None,
                   trials=None, cfg=SimConfig(), coverage=None, peptide_scale=0.5, tie_classes=True):
    """Generate a synthetic worm whose true closed level is known.

    With cfg.n_bins > 1 the responses are time courses (N, S, n_bins) and the
    per-bin trial noise is sqrt(n_bins) x the window-mean noise, so that
    data.average_bins(ds) yields exactly window-mean data with the usual noise.

    Wiring is taken from `ds_template` (real or random).  Ground-truth
    parameters are a randomly perturbed rung `true_level`; responses to every
    stimulus in both strains are simulated, then trial noise is added with the
    template's trial counts (or `trials`) and coverage pattern.
    Returns (Dataset, true_params).
    """
    from .data import Dataset
    rng = np.random.default_rng(seed)
    st = Structure.from_dataset(ds_template, tie=tie_classes)
    p_true = init_params(true_level, st, jax.random.PRNGKey(seed + 1000), jitter=jitter)
    if "rec" in p_true:
        p_true["rec"] = p_true["rec"] * peptide_scale
    R = np.asarray(predict(true_level, p_true, st, ds_template.stim, ds_template.strains, cfg,
                           proto=proto_cols(ds_template, np.arange(ds_template.S))))
    if noise_sd is None:
        R_win = R if R.ndim == 3 else R.mean(-1)      # calibrate noise on the window mean
        noise_sd = 0.45 * R_win[ds_template.mask()].std() * np.sqrt(cfg.n_bins)
    n = ds_template.n.copy() if trials is None else np.full_like(ds_template.n, trials)
    if coverage is not None:
        n = n * (rng.random(n.shape) < coverage)
    n[:, ds_template.stim, np.arange(ds_template.S)] = 0
    nb = bcast(n, R)                                   # trial counts, broadcast over time bins
    noise = rng.normal(size=R.shape) * noise_sd / np.sqrt(np.maximum(nb, 1))
    mean = np.where(nb > 0, R + noise, 0.0)
    var_mean = np.broadcast_to(np.where(nb > 0, noise_sd ** 2 / np.maximum(nb, 1), 0.0), R.shape).copy()
    ds = Dataset(ids=ds_template.ids, chem=ds_template.chem, gap=ds_template.gap,
                 sign=ds_template.sign, pep=ds_template.pep, stim=ds_template.stim,
                 strains=ds_template.strains, mean=mean, var_mean=var_mean, n=n, proto=ds_template.proto,
                 meta={"synthetic": True, "true_level": true_level, "seed": seed,
                       "noise_sd": float(noise_sd)})
    ds.meta["truth"] = None
    return ds, p_true, R


def random_wiring(N=60, S=None, seed=0, p_chem=0.08, p_gap=0.03, strains=("wt", "unc31"), trials=4):
    """A random connectome in Dataset format (for quick tests without the atlas)."""
    from .data import Dataset
    rng = np.random.default_rng(seed)
    chem = (rng.random((N, N)) < p_chem) * rng.integers(1, 20, (N, N))
    np.fill_diagonal(chem, 0)
    gap = np.triu((rng.random((N, N)) < p_gap) * rng.integers(1, 10, (N, N)), 1)
    gap = gap + gap.T
    sign = np.where(chem > 0, rng.choice([-1.0, 0.0, 1.0], (N, N), p=[0.3, 0.3, 0.4]), 0.0)
    pep = (rng.random((N, N)) < 0.5) * rng.integers(1, 10, (N, N)).astype(float)
    np.fill_diagonal(pep, 0)
    S = N if S is None else S
    stim = np.sort(rng.choice(N, S, replace=False))
    n = np.full((len(strains), N, S), trials)
    n[:, stim, np.arange(S)] = 0
    z = np.zeros(n.shape)
    ids = np.array([f"c{i // 2}{'LR'[i % 2]}" for i in range(N)])  # bilateral pairs
    return Dataset(ids=ids, chem=chem.astype(float),
                   gap=gap.astype(float), sign=sign, pep=pep, stim=stim, strains=tuple(strains),
                   mean=z, var_mean=z.copy(), n=n, meta={"random_wiring": True})
