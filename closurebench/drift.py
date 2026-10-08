"""E3 - program versus hardware.

In a computer, the transfer function is fixed on the timescale of the
computation.  E3 asks whether the brain's closed level behaves like that: fit a
fixed "program" to spontaneous whole-brain activity and see whether it keeps
predicting later activity; if not, ask what it costs to absorb the drift.

Model of activity dynamics (one step dt, per neuron i):
    dx_i(t) = -a_i x_i(t) + sum_j W_ij tanh(x_j(t)) + b_i(t) + noise
W lives on the connectome support (chemical + electrical, both directions);
a_i > 0 is a leak, b_i(t) a drive.  For fixed b this is linear in (a, W, b),
so each neuron is a small ridge regression.

Three ways to handle b(t):
  frozen    b fixed; one program for the whole recording
  refit     b re-estimated in every window (the "hardware" is rewritten:
            N new numbers per window, so its state grows with recording length)
  meta(d)   b(t) = b + B z(t), with d slow states driven by activity,
            z(t+1) = (1 - lam) z(t) + lam * A tanh(x(t))
            -- a fixed "meta-program" of size d (self-programming)

Synthetic worms with known answers:
  stationary      b constant                        -> frozen suffices
  self_program    b driven by d_true slow states     -> meta(d small) absorbs drift
  rewired         b does an activity-independent random walk
                                                     -> no compact meta-program
"""
from __future__ import annotations
import numpy as np


# ------------------------------------------------------------ synthetic ---
def connectome_matrix(ds, rng, rho=0.8):
    """Signed coupling on the template's chemical+gap support, scaled so that
    the spectral radius is `rho` (stable but active).  [post, pre]."""
    chem, gap = np.asarray(ds.chem, float), np.asarray(ds.gap, float)
    sign = np.asarray(ds.sign, float)
    sign = np.where(sign == 0, rng.choice([-1.0, 1.0], sign.shape), sign)
    W = sign * np.log1p(chem) + 0.5 * np.log1p(gap)
    np.fill_diagonal(W, 0)
    r = np.max(np.abs(np.linalg.eigvals(W)))
    return W * (rho / r if r > 0 else 1.0)


def simulate(W, kind, T=1600, dt=0.6, tau=2.0, noise=0.3, d_true=2, tau_h=300.0, drift_sd=0.02,
             strength=1.0, seed=0, burn=200):
    """Spontaneous activity (T, N) of a synthetic worm.  kind in
    {"stationary", "self_program", "rewired"}.  `strength` scales the drift."""
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    a = dt / tau
    b0 = rng.normal(0, 0.3, N)
    A = rng.normal(0, 1, (d_true, N)) / np.sqrt(N)
    B = rng.normal(0, 0.5, (N, d_true)) * strength
    z = rng.normal(0, 1.0, d_true)          # slow states start far from equilibrium
    lam = dt / tau_h
    c = np.zeros(N)
    x = np.zeros(N)
    out = np.zeros((T + burn, N))
    for t in range(T + burn):
        if kind == "stationary":
            b = b0
        elif kind == "self_program":
            b = b0 + B @ z
            z = (1 - lam) * z + lam * 3.0 * (A @ np.tanh(x)) if t >= burn else z
        elif kind == "rewired":
            if t >= burn:
                c = c + drift_sd * strength * np.sqrt(dt) * rng.normal(size=N)
            b = b0 + c
        else:
            raise ValueError(kind)
        x = x + a * (-x + W @ np.tanh(x) + b) + noise * np.sqrt(dt) * rng.normal(size=N)
        out[t] = x
    return out[burn:]


# ---------------------------------------------------------------- fitting ---
def _support(mask_W):
    S = np.asarray(mask_W, bool).copy()
    np.fill_diagonal(S, False)
    return S


def design_rows(X, i, S, U=None):
    """Features for neuron i: [-x_i, tanh(x_j) for j in support, (behavior inputs u), 1]."""
    cols = [-X[:-1, i], np.tanh(X[:-1, S[i]])]
    if U is not None:
        cols.append(U[:-1])
    return np.column_stack(cols + [np.ones(len(X) - 1)])


def fit_frozen(X_list, S, ridge=1e-2, U_list=None):
    """One program for all segments in X_list (each (T_k, N)): per-neuron ridge.
    U_list: optional matching exogenous inputs (T_k, k), e.g. behavior.
    Returns dict(a, W, b, C) with C (N, k) the input weights (None without inputs)."""
    N = X_list[0].shape[1]
    U_list = [None] * len(X_list) if U_list is None else U_list
    nu = 0 if U_list[0] is None else U_list[0].shape[1]
    a, b, W, C = np.zeros(N), np.zeros(N), np.zeros((N, N)), np.zeros((N, nu))
    for i in range(N):
        F = np.vstack([design_rows(X, i, S, U) for X, U in zip(X_list, U_list)])
        y = np.concatenate([np.diff(X[:, i]) for X in X_list])
        P = F.shape[1]
        reg = ridge * np.eye(P); reg[-1, -1] = 0
        w = np.linalg.solve(F.T @ F + reg * len(y), F.T @ y)
        ns = int(S[i].sum())
        a[i], W[i, S[i]], b[i] = w[0], w[1:1 + ns], w[-1]
        if nu:
            C[i] = w[1 + ns:1 + ns + nu]
    return dict(a=a, W=W, b=b, C=C if nu else None)


RIDGE_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def choose_ridge(X, S, U=None, grid=RIDGE_GRID, val_frac=0.2):
    """Ridge strength for fit_frozen, chosen ONLY from the data given (the past):
    fit on its first (1 - val_frac), forecast its last val_frac, keep the best."""
    n = len(X); cut = int(n * (1 - val_frac))
    Ua, Ub = (None, None) if U is None else (U[:cut], U[cut:])
    best, best_r2 = grid[0], -np.inf
    for lam in grid:
        p = fit_frozen([X[:cut]], S, lam, [Ua] if Ua is not None else None)
        r = _r2_on(predict_increments(p, X[cut:], U=Ub), np.diff(X[cut:], axis=0))
        if r > best_r2:
            best, best_r2 = lam, r
    return best


def _input_term(p, U):
    if U is None or p.get("C") is None:
        return 0.0
    return U[:-1] @ p["C"].T


def predict_increments(p, X, b=None, U=None):
    """One-step predicted increments (T-1, N); b may be (N,) or (T-1, N); U are
    the exogenous inputs aligned with X (T, k), if the program uses any."""
    b = p["b"] if b is None else b
    return -p["a"] * X[:-1] + np.tanh(X[:-1]) @ p["W"].T + b + _input_term(p, U)


def r2(pred, X):
    """R^2 of increments, pooled over neurons (1 - SSE / SS of increments about their mean)."""
    y = np.diff(X, axis=0)
    return 1.0 - ((pred - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()


def fit_refit_biases(p, X, U=None):
    """Keep (a, W, C) of program p; re-estimate only the biases on segment X
    (closed form: mean residual)."""
    resid = np.diff(X, axis=0) - (-p["a"] * X[:-1] + np.tanh(X[:-1]) @ p["W"].T + _input_term(p, U))
    return resid.mean(0)


def fit_meta(X_full, train_mask, p0, d, steps=1500, lr=1e-2, seed=0, l2=1e-3, U_full=None):
    """Meta-program with d slow activity-driven states, fitted by gradient descent
    on the training time steps (train_mask over the T-1 increments).  The program's
    couplings (a, W, and input weights C) are taken from the frozen program p0 and
    held FIXED; only the drives b and the slow-state rules (A, B, lam, z0) are
    learned -- the meta-program can absorb drift only through its d states.
    Returns params (with 'pred': predicted increments over the whole recording)."""
    import jax
    import jax.numpy as jnp
    import optax
    N = X_full.shape[1]
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    params = {"a": jnp.asarray(p0["a"]), "W": jnp.asarray(p0["W"]), "b": jnp.asarray(p0["b"]),
              "A": 0.1 * jax.random.normal(k1, (d, N)) / np.sqrt(N), "B": 0.01 * jax.random.normal(k2, (N, d)),
              "llam": jnp.linspace(np.log(1e-3), np.log(1e-1), d), "z0": jnp.zeros(d)}
    use_u = U_full is not None and p0.get("C") is not None
    if use_u:
        params["C"] = jnp.asarray(p0["C"]); Uj = jnp.asarray(U_full[:-1])
    S = jnp.asarray(_support(p0["W"] != 0) | (np.asarray(p0["W"]) != 0))
    Xj = jnp.asarray(X_full); y = jnp.diff(Xj, axis=0); m = jnp.asarray(train_mask, jnp.float32)[:, None]
    th = jnp.tanh(Xj[:-1])

    def zs(q):
        lam = jax.nn.sigmoid(q["llam"])
        def step(z, h):
            return (1 - lam) * z + lam * (q["A"] @ h), z
        _, Z = jax.lax.scan(step, q["z0"], th)
        return Z                                                  # (T-1, d): state before each step

    def pred(q):
        out = -q["a"] * Xj[:-1] + th @ (q["W"] * S).T + q["b"] + zs(q) @ q["B"].T
        return out + Uj @ q["C"].T if use_u else out

    fixed = {k: params.pop(k) for k in ("a", "W", "C") if k in params}

    def loss(q):
        q = dict(q, **fixed)
        e = (pred(q) - y) ** 2 * m
        return e.sum() / m.sum() / N + l2 * (jnp.sum(q["B"] ** 2) + jnp.sum(q["A"] ** 2))

    opt = optax.adam(lr); st = opt.init(params)

    @jax.jit
    def upd(q, s):
        l, g = jax.value_and_grad(loss)(q)
        u, s = opt.update(g, s, q)
        return optax.apply_updates(q, u), s, l
    for _ in range(steps):
        params, st, l = upd(params, st)
    params = dict(params, **fixed)
    params["pred"] = np.asarray(pred(params))
    return params


# ------------------------------------------------------------- the test ---
def window_bounds(T, n_windows):
    edges = np.linspace(0, T, n_windows + 1).astype(int)
    return list(zip(edges[:-1], edges[1:]))


def _r2_on(pred, y):
    return 1.0 - ((pred - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()


def analyse(X, S, n_windows=8, d_grid=(1, 2, 4), meta_steps=3000, ridge="auto", seed=0, first_origin=None, U=None):
    """E3 analysis of one recording X (T, N) with coupling support S (N, N), by
    forecasting forward in time.  For each forecast window k (k >= first_origin,
    0-based), every model is fitted on windows < k only and scored on the SECOND
    half of window k (one-step increments, R^2):

      frozen   one program (a, W, b) fitted on the past
      refit    frozen's (a, W) with N biases re-estimated from the FIRST half of
               window k: the hardware is rewritten, N new numbers per window
      meta(d)  fixed program + d slow states driven by observed activity, fitted
               on the past; its state runs through window k's first half, but no
               parameter is re-estimated.  Cost: d fixed rules, not N numbers.

    Also returns `decay`: the program fitted on window 0 alone, scored on each
    later window's second half (how fast a frozen program goes stale).
    absorbed[d] = mean over forecast windows of (meta - frozen) / (refit - frozen)
    computed on pooled sums (so tiny per-window gaps do not dominate).
    U (T, k): optional exogenous inputs (e.g. behavior) available to every model.
    ridge="auto": the ridge strength is chosen anew at every forecast origin from
    the past only (choose_ridge); a number fixes it."""
    def lam_for(Xp, Up):
        return choose_ridge(Xp, S, Up) if ridge == "auto" else ridge
    T, N = X.shape
    S = _support(S)
    if first_origin is None:            # forecast only once at least half the recording is past:
        first_origin = n_windows // 2   # a meta-program cannot be learned from less than its own timescale
    wins = window_bounds(T, n_windows)
    first = [(s, s + (e - s) // 2) for s, e in wins]
    second = [(s + (e - s) // 2, e) for s, e in wins]
    dX = np.diff(X, axis=0)
    def y_of(r):
        return dX[r[0]:r[1] - 1]
    def x_of(r):
        return X[r[0]:r[1] - 1]

    Us = (lambda a, b: None) if U is None else (lambda a, b: U[a:b])
    p0 = fit_frozen([X[wins[0][0]:wins[0][1]]], S, lam_for(X[wins[0][0]:wins[0][1]], Us(*wins[0])), [Us(*wins[0])])
    decay = [float(_r2_on(predict_increments(p0, X[r[0]:r[1]], U=Us(*r)), y_of(r))) for r in second]

    sse = {"frozen": 0.0, "refit": 0.0, **{d: 0.0 for d in d_grid}}
    sst = 0.0
    per_window = []
    for k in range(first_origin, n_windows):
        past = X[:wins[k][0]]
        lam = lam_for(past, Us(0, wins[k][0]))
        glob = fit_frozen([past], S, lam, [Us(0, wins[k][0])])
        r = second[k]; y = y_of(r); xk = X[r[0]:r[1]]; uk = Us(*r)
        pf = predict_increments(glob, xk, U=uk)
        pr = predict_increments(glob, xk, fit_refit_biases(glob, X[first[k][0]:first[k][1]], Us(*first[k])), U=uk)
        row = {"k": k, "ridge": lam, "frozen": float(_r2_on(pf, y)), "refit": float(_r2_on(pr, y))}
        sse["frozen"] += ((pf - y) ** 2).sum(); sse["refit"] += ((pr - y) ** 2).sum()
        sst_k = ((y - y.mean(0)) ** 2).sum(); sst += sst_k
        tm = np.zeros(T - 1, bool); tm[:wins[k][0] - 1] = True       # train on the past only
        for d in d_grid:
            q = fit_meta(X, tm, glob, d, steps=meta_steps, seed=seed, U_full=U)
            pm = q["pred"][r[0]:r[1] - 1]
            row[d] = float(_r2_on(pm, y)); sse[d] += ((pm - y) ** 2).sum()
        per_window.append(row)
    score = {m: float(1 - v / sst) for m, v in sse.items()}
    gap = score["refit"] - score["frozen"]
    absorbed = {d: float((score[d] - score["frozen"]) / gap) if gap > 1e-9 else float("nan") for d in d_grid}
    return dict(decay=decay, frozen=score["frozen"], refit=score["refit"], meta={d: score[d] for d in d_grid},
                absorbed=absorbed, gap=gap, per_window=per_window, n_windows=n_windows, T=T, N=N)


def temporal_generalization(X, S, U=None, n_windows=8, ridge="auto"):
    """Fit the program on each window separately (ridge chosen within that window)
    and forecast every other window.  Returns (G, slope): G[j, k] = one-step R^2 on
    window k of the program fitted on window j, and `slope` = least-squares slope
    of R^2 against |j - k| (off-diagonal pairs; R^2 per unit of window distance).

    For a stationary system G does not depend on |j - k| even if the model is
    misspecified (misfit lowers all entries equally), so slope ~ 0.  Drift makes
    programs go stale with distance: slope < 0.  A drift detector that needs no
    per-window refit."""
    S = _support(S)
    wins = window_bounds(len(X), n_windows)
    Us = (lambda a, b: None) if U is None else (lambda a, b: U[a:b])
    G = np.full((n_windows, n_windows), np.nan)
    for j, (a0, a1) in enumerate(wins):
        Xa, Ua = X[a0:a1], Us(a0, a1)
        lam = choose_ridge(Xa, S, Ua) if ridge == "auto" else ridge
        p = fit_frozen([Xa], S, lam, [Ua] if Ua is not None else None)
        for k, (b0, b1) in enumerate(wins):
            if k != j:
                G[j, k] = _r2_on(predict_increments(p, X[b0:b1], U=Us(b0, b1)), np.diff(X[b0:b1], axis=0))
    jj, kk = np.nonzero(~np.isnan(G))
    lag = np.abs(jj - kk).astype(float); val = G[jj, kk]
    slope = float(np.polyfit(lag, val, 1)[0])
    return G, slope


def staleness_absorption(X, S, U=None, d_grid=(1, 2, 4), n_windows=8, meta_steps=3000, ridge="auto", seed=0):
    """Does a meta-program go stale more slowly than a frozen program?
    Both are fitted on the first half of the recording (windows < n_windows/2) and
    forecast each window of the second half (one-step R^2).  For each model, the
    staleness is the least-squares slope of R^2 against forecast window (negative =
    goes stale).  absorbed[d] = 1 - slope_meta(d) / slope_frozen: 1 = the
    meta-program stays as good as at the start, 0 = it goes stale like the frozen
    program.  Defined only when the frozen program goes stale (slope_frozen < 0);
    needs no per-window refit, so it keeps its power when refits are noisy."""
    S = _support(S)
    T = len(X); wins = window_bounds(T, n_windows); h = n_windows // 2
    Us = (lambda a, b: None) if U is None else (lambda a, b: U[a:b])
    cut = wins[h][0]
    lam = choose_ridge(X[:cut], S, Us(0, cut)) if ridge == "auto" else ridge
    p = fit_frozen([X[:cut]], S, lam, [Us(0, cut)])
    dX = np.diff(X, axis=0)
    def r2w(pred_full, k):
        a, b = wins[k]; return _r2_on(pred_full[a:b - 1], dX[a:b - 1])
    pf = predict_increments(p, X, U=U)
    ks = list(range(h, n_windows))
    fro = [r2w(pf, k) for k in ks]
    slope_f = float(np.polyfit(ks, fro, 1)[0])
    tm = np.zeros(T - 1, bool); tm[:cut - 1] = True
    out = {"frozen": fro, "slope_frozen": slope_f, "meta": {}, "slope_meta": {}, "absorbed": {}}
    for d in d_grid:
        q = fit_meta(X, tm, p, d, steps=meta_steps, seed=seed, U_full=U)
        mr = [r2w(q["pred"], k) for k in ks]
        sm = float(np.polyfit(ks, mr, 1)[0])
        out["meta"][d] = mr; out["slope_meta"][d] = sm
        out["absorbed"][d] = float(1 - sm / slope_f) if slope_f < 0 else float("nan")
    return out


def calibrate_gap(null_gaps, floor=0.005):
    """Drift threshold from stationary null worms: the largest refit-minus-frozen gap
    observed among them, never below `floor` (the smallest drift we treat as relevant)."""
    return max(float(np.max(null_gaps)) if len(null_gaps) else floor, floor)


def absorption_by_window(res, d):
    """Per forecast window: (k, fraction of that window's refit-frozen gap absorbed by meta(d))."""
    out = []
    for w in res["per_window"]:
        g = w["refit"] - w["frozen"]
        out.append((w["k"], (w[d] - w["frozen"]) / g if abs(g) > 1e-9 else float("nan")))
    return out


def verdict(res, gap_min=0.005, absorb_min=0.5, d_max=4):
    """Pre-registered E3 classification of one recording.
      stationary      refit - frozen < gap_min (calibrated on stationary null worms):
                      a fixed program forecasts as well as a rewritten one
      self_program    otherwise, some meta(d <= d_max) absorbs >= absorb_min of the gap
      rewired         otherwise: drift not captured by a compact activity-driven meta-program"""
    if res["gap"] < gap_min:
        return "stationary"
    best = max((v for d, v in res["absorbed"].items() if d <= d_max), default=0.0)
    return "self_program" if best >= absorb_min else "rewired"


# ------------------------------------------------------------- real data ---
def _find_labels(d, N):
    """Neuron labels if the file has any ('labeled' / 'labels' / 'neuron_labels',
    at top level or under 'gcamp'); otherwise generic names n0..n{N-1}."""
    labels = np.array([f"n{i}" for i in range(N)], dtype=object)
    for src in (d, d.get("gcamp", {}) if isinstance(d.get("gcamp"), dict) else {}):
        lab = next((src[k] for k in ("labeled", "labels", "neuron_labels") if k in src), None)
        if isinstance(lab, dict):          # e.g. {index: {"label": "AVAL", ...}}, 1-based (Julia)
            for k, v in lab.items():
                name = v.get("label", v) if isinstance(v, dict) else v
                if str(k).isdigit() and 0 <= int(k) - 1 < N:
                    labels[int(k) - 1] = str(name)
            break
        if isinstance(lab, list) and len(lab) == N:
            labels = np.asarray(lab, dtype=object).astype(str); break
    return labels.astype(str)


def load_recording(path, behavior=False):
    """Load a whole-brain recording.  Returns (X (T, N), labels (N,), dt), or
    (X, labels, dt, U) with behavior=True, where U (T, k) holds z-scored behavior
    (velocity, head_angle, angular_velocity, pumping when present; None if absent).

    Accepts
      * WormWideWeb JSON (e.g. Atanas et al. 2023): gcamp.trace_array (N, T),
        timing.mean_timestep (s), behavior.{velocity, head_angle, ...}
      * flat JSON with a trace array under 'trace_array' / 'traces' / 'activity'
      * .npz with traces (T, N) [or (N, T)], labels (N,), dt [or time], optional behavior (T, k)
    NaN-only neurons are dropped; remaining NaNs are linearly interpolated;
    traces are z-scored per neuron."""
    import json
    p = str(path)
    U = None
    if p.endswith(".npz"):
        z = np.load(p, allow_pickle=True)
        X = np.asarray(z["traces"], float)
        N_guess = min(X.shape)
        labels = np.asarray(z["labels"]).astype(str) if "labels" in z.files else np.array([f"n{i}" for i in range(N_guess)])
        dt = float(z["dt"]) if "dt" in z.files else float(np.median(np.diff(z["time"]))) if "time" in z.files else 0.6
        if "behavior" in z.files:
            U = np.asarray(z["behavior"], float)
    elif p.endswith(".json"):
        d = json.load(open(p))
        g = d.get("gcamp") if isinstance(d.get("gcamp"), dict) else d
        key = next((k for k in ("trace_array", "traces", "activity", "trace_original") if k in g), None)
        if key is None:
            raise KeyError(f"no trace array found; top-level keys are: {sorted(d.keys())}")
        X = np.asarray(g[key], float)
        N_guess = min(X.shape)
        labels = _find_labels(d, N_guess)
        tm = d.get("timing", {}) if isinstance(d.get("timing"), dict) else {}
        dt = float(tm.get("mean_timestep", d.get("avg_timestep", 0.6)))
        beh = d.get("behavior") if isinstance(d.get("behavior"), dict) else None
        if beh:
            cols = [np.asarray(beh[k], float) for k in ("velocity", "head_angle", "angular_velocity", "pumping")
                    if k in beh and np.ndim(beh[k]) == 1 and len(beh[k]) == max(X.shape)]
            U = np.column_stack(cols) if cols else None
    else:
        raise ValueError("expected .npz or .json")
    if X.shape[0] < X.shape[1]:          # (N, T) -> (T, N)
        X = X.T
    keep = ~np.all(np.isnan(X), axis=0)
    X, labels = X[:, keep], np.asarray(labels)[keep]
    for M in [X] + ([U] if U is not None else []):
        for j in range(M.shape[1]):
            v = M[:, j]; bad = np.isnan(v)
            if bad.any() and (~bad).any():
                M[bad, j] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), v[~bad])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)            # z-score per neuron
    if U is not None:
        U = (U - U.mean(0)) / (U.std(0) + 1e-9)
    return (X, labels, dt, U) if behavior else (X, labels, dt)


def surrogate(X, S, U=None, ridge="auto", seed=0, drift_frac=0.0):
    """A recording of the same length generated by the frozen program fitted to
    the WHOLE recording X, driven by Gaussian noise with each neuron's residual
    variance and by the real behavior U.

    drift_frac = 0: stationary by construction -> a recording-specific NULL for
      the drift threshold.
    drift_frac > 0: each neuron's drive does an activity-independent random walk
      whose per-step size is drift_frac x that neuron's residual sd ("rewired",
      as in the synthetic validation, where the ratio is about 0.07) -> a
      POSITIVE CONTROL: can drift of this size be detected in this recording?
    Returns the z-scored surrogate (T, N), or None if the fitted program is
    unstable as a generator."""
    rng = np.random.default_rng(seed)
    S = _support(S)
    lam = choose_ridge(X, S, U) if ridge == "auto" else ridge
    p = fit_frozen([X], S, lam, [U] if U is not None else None)
    sd = (np.diff(X, axis=0) - predict_increments(p, X, U=U)).std(0)
    Y = np.zeros_like(X); y = X[0].copy(); c = np.zeros(X.shape[1])
    for t in range(len(X)):
        Y[t] = y
        u = 0.0 if U is None or p.get("C") is None else p["C"] @ U[t]
        y = y + (-p["a"] * y + p["W"] @ np.tanh(y) + p["b"] + c + u) + sd * rng.normal(size=y.shape)
        c = c + drift_frac * sd * rng.normal(size=c.shape)
        if not np.all(np.isfinite(y)) or np.abs(y).max() > 1e3:
            return None
    return (Y - Y.mean(0)) / (Y.std(0) + 1e-9)


def stationary_surrogate(X, S, U=None, ridge="auto", seed=0):
    return surrogate(X, S, U, ridge, seed, drift_frac=0.0)


def support_from_labels(labels, atlas_ds, min_identified=10):
    """Coupling support among identified neurons (chemical or electrical contact in
    either direction), from the atlas Dataset.  Unidentified neurons get no
    coupling.  If fewer than `min_identified` neurons are identified (e.g. a
    recording without NeuroPAL labels), the support is ALL-TO-ALL instead:
    a data-driven, ridge-regularised coupling with no connectome constraint.
    Returns (S, n_identified)."""
    ids = [str(i) for i in atlas_ds.ids]
    idx = [ids.index(l) if l in ids else -1 for l in labels]
    con = (np.asarray(atlas_ds.chem) > 0) | (np.asarray(atlas_ds.gap) > 0)
    con = con | con.T
    N = len(labels)
    S = np.zeros((N, N), bool)
    for a, ia in enumerate(idx):
        for b, ib in enumerate(idx):
            if ia >= 0 and ib >= 0 and a != b:
                S[a, b] = con[ia, ib]
    n_id = int(sum(i >= 0 for i in idx))
    if n_id < min_identified:
        S = ~np.eye(N, dtype=bool)
    return S, n_id


# ------------------------------------------------------- E3b: what drifts? ---
def load_atanas(path):
    """Load a WormWideWeb / Atanas et al. 2023 JSON with its annotations.
    Returns dict(X, labels, dt, U, keep, changing, ranges):
      keep      original (0-based) indices of the neurons kept by load_recording
      changing  bool (N_kept,) -- neurons whose behavioral encoding changed between
                the two halves according to the authors (encoding.encoding_changing_neurons,
                1-based in the file); None if the file has no such annotation
      ranges    the authors' time ranges (1-based frames) for that comparison"""
    import json
    X, labels, dt, U = load_recording(path, behavior=True)
    d = json.load(open(str(path)))
    raw = np.asarray(d["gcamp"]["trace_array"], float)
    if raw.shape[0] > raw.shape[1]:
        raw = raw.T
    keep = np.flatnonzero(~np.all(np.isnan(raw), axis=1))
    enc = d.get("encoding", {}) if isinstance(d.get("encoding"), dict) else {}
    changing = None
    if "encoding_changing_neurons" in enc:
        ch = set(int(i) - 1 for i in enc["encoding_changing_neurons"])
        changing = np.array([k in ch for k in keep])
    return dict(X=X, labels=labels, dt=dt, U=U, keep=keep, changing=changing, ranges=enc.get("ranges"))


def slow_inputs(U, dt, taus=(60.0, 240.0)):
    """Causal exponential low-pass filters of the behavior inputs at timescales
    `taus` (s), appended to U: slow behavioral state (e.g. roaming/dwelling)."""
    if U is None:
        return None
    cols = [U]
    for tau in taus:
        a = min(dt / tau, 1.0); Y = np.zeros_like(U); y = U[0].copy()
        for t in range(len(U)):
            y = (1 - a) * y + a * U[t]; Y[t] = y
        cols.append((Y - Y.mean(0)) / (Y.std(0) + 1e-9))
    return np.column_stack(cols)


def per_neuron_drift(X, S, U=None, ridge="auto"):
    """Per-neuron drift between the two halves of a recording.
    Program A is fitted on the first half, program B on the first half of the
    second half; both forecast the final quarter.  drift_i = R2_B,i - R2_A,i:
    how much better a program fitted *recently* forecasts neuron i than one
    fitted on the older half.  B uses half as much data as A, so drift is
    underestimated equally for every neuron (comparisons between neurons are fair).
    Returns (drift (N,), r2_A (N,), r2_B (N,))."""
    S = _support(S)
    T = len(X); h, q = T // 2, (3 * T) // 4
    Us = (lambda a, b: None) if U is None else (lambda a, b: U[a:b])
    def fit(a, b):
        Xa, Ua = X[a:b], Us(a, b)
        lam = choose_ridge(Xa, S, Ua) if ridge == "auto" else ridge
        return fit_frozen([Xa], S, lam, [Ua] if Ua is not None else None)
    pA, pB = fit(0, h), fit(h, q)
    Xt, Ut = X[q:], Us(q, T)
    y = np.diff(Xt, axis=0); sst = ((y - y.mean(0)) ** 2).sum(0)
    r2 = lambda p: 1 - ((predict_increments(p, Xt, U=Ut) - y) ** 2).sum(0) / sst
    rA, rB = r2(pA), r2(pB)
    return rB - rA, rA, rB


def circular_shift(U, frac=0.5):
    """Behavior circularly shifted by `frac` of the recording: same statistics
    and slow structure, but no longer aligned with the neural activity."""
    return None if U is None else np.roll(U, int(len(U) * frac), axis=0)
