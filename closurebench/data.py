"""Common data format + loaders for the C. elegans signal-propagation atlas.

Conventions (match wormneuroatlas): every matrix is indexed [post, pre], i.e.
M[i, j] describes j -> i.  Responses are R[strain, i, s] = response of neuron i
when neuron stim[s] is optogenetically stimulated (Randi et al. 2023).
Strain 'unc31' lacks dense-core-vesicle (neuropeptide) release, so WT - unc31
isolates the extrasynaptic peptidergic contribution: a direct test of rung L3.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class Dataset:
    ids: np.ndarray            # (N,) neuron names
    chem: np.ndarray           # (N,N) chemical synapse counts [post, pre]
    gap: np.ndarray            # (N,N) gap-junction counts (symmetric)
    sign: np.ndarray           # (N,N) predicted chemical sign in {-1,0,+1}; 0 = unknown/mixed
    pep: np.ndarray            # (N,N) peptidergic connectome weights [post, pre]
    stim: np.ndarray           # (S,) indices of stimulated neurons
    strains: tuple             # e.g. ('wt', 'unc31')
    mean: np.ndarray           # (n_strain, N, S) trial-mean response
    var_mean: np.ndarray       # (n_strain, N, S) variance of the trial mean (noise)
    n: np.ndarray              # (n_strain, N, S) number of trials (0 = unobserved)
    meta: dict = field(default_factory=dict)
    proto: dict = None         # optional per-column stimulus protocol: amp (S,), pair (S,), pattern (S,)

    @property
    def N(self):
        return len(self.ids)

    @property
    def S(self):
        return len(self.stim)

    def mask(self, min_trials=1):
        """Observed (responder, stimulus) pairs, excluding the stimulated neuron itself."""
        m = self.n >= min_trials
        m[:, self.stim, np.arange(self.S)] = False
        if self.proto is not None and "pair" in self.proto:      # second stimulated neuron, too
            q = np.asarray(self.proto["pair"]); c = np.nonzero(q >= 0)[0]
            m[:, q[c], c] = False
        return m

    def save(self, path):
        d = asdict(self)
        d["strains"] = np.array(self.strains)
        d["meta"] = np.array([repr(self.meta)])
        if self.proto is None:
            d.pop("proto")
        else:
            d["proto"] = np.array([{k: np.asarray(v) for k, v in self.proto.items()}], dtype=object)
        np.savez_compressed(path, **d)

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        d["strains"] = tuple(str(s) for s in d["strains"])
        d["meta"] = eval(str(d["meta"][0]), {"array": np.array, "nan": np.nan})
        if "proto" in d:
            d["proto"] = d["proto"][0]
        return Dataset(**d)

    def summary(self):
        m = self.mask()
        lines = [f"Dataset: N={self.N} neurons, S={self.S} stimulated neurons, strains={self.strains}",
                 f"  chemical synapses: {(self.chem > 0).sum()}  gap-junction pairs: {(self.gap > 0).sum() // 2}"
                 f"  peptidergic pairs: {(self.pep > 0).sum()}",
                 f"  signs: +{(self.sign > 0).sum()} / -{(self.sign < 0).sum()} / unknown {((self.sign == 0) & (self.chem > 0)).sum()} (on existing synapses)"]
        for k, s in enumerate(self.strains):
            lines.append(f"  {s}: {m[k].sum()} observed pairs, median trials {np.median(self.n[k][m[k]]):.0f}")
        return "\n".join(lines)


def trials_to_arrays(trial_lists, pooled_noise_var=None, lower=-1.0):
    """Object array of per-trial lists -> (mean, var_mean, n).  Pairs with a
    single trial get the pooled (mean) single-trial noise variance.
    Trials below `lower` are rejected: dF/F < -1 is physically impossible
    (fluorescence cannot go negative), so such values are artifacts."""
    shape = trial_lists.shape
    mean = np.zeros(shape)
    var = np.full(shape, np.nan)
    n = np.zeros(shape, dtype=int)
    for idx in np.ndindex(shape):
        t = np.asarray(trial_lists[idx], dtype=float)
        t = t[np.isfinite(t) & (t >= lower)]
        n[idx] = len(t)
        if len(t):
            mean[idx] = t.mean()
        if len(t) > 1:
            var[idx] = t.var(ddof=1)
    if pooled_noise_var is None:
        pooled_noise_var = np.nanmean(var[n > 1]) if (n > 1).any() else 0.0
    var = np.where(np.isfinite(var), var, pooled_noise_var)
    var_mean = np.where(n > 0, var / np.maximum(n, 1), 0.0)
    return mean, var_mean, n


def combine_signs(sign3):
    """(n_transmitters, N, N) predicted signs -> (N, N) in {-1, 0, +1}.
    Conflicting or missing predictions become 0 (unknown)."""
    pos = np.any(sign3 == 1, axis=0)
    neg = np.any(sign3 == -1, axis=0)
    return np.where(pos & ~neg, 1.0, np.where(neg & ~pos, -1.0, 0.0))


def open_atlas():
    """Instantiate wormneuroatlas.NeuroAtlas without the online WormBase version
    check (which fails offline or when WormBase is down)."""
    import warnings
    import wormneuroatlas as wa
    warnings.filterwarnings("ignore")
    wa.WormBase.assert_db_version_consistency = lambda self: None
    return wa.NeuroAtlas(verbose=False)


def load_atlas_dataset(strains=("wt", "unc31"), min_stim_trials=1, atlas=None):
    """Build a Dataset from the public atlas bundled in `wormneuroatlas`.

    Wiring: Witvliet/White anatomical connectome as compiled by the package,
    CeNGEN-based synapse signs, Ripoll-Sanchez peptidergic connectome.
    Responses: per-trial dF/F amplitudes (signal_propagation_map_all).
    All 300 neurons are kept in the wiring, observed or not.
    """
    a = open_atlas() if atlas is None else atlas
    chem = np.nan_to_num(a.get_chemical_synapses())
    gap = np.nan_to_num(a.get_gap_junctions())
    gap = 0.5 * (gap + gap.T)
    sign = combine_signs(a.get_chemical_synapse_sign()) * (chem > 0)
    pep = np.nan_to_num(a.get_peptidergic_connectome())
    np.fill_diagonal(pep, 0)
    occ = a.get_signal_propagation_occurrence_matrix(strains[0])
    stim = np.nonzero(occ.sum(0) >= min_stim_trials)[0]
    means, vms, ns = [], [], []
    for s in strains:
        allt = a.get_signal_propagation_map_all(s)[:, stim]
        m, v, n = trials_to_arrays(allt)
        means.append(m), vms.append(v), ns.append(n)
    return Dataset(ids=np.array(a.neuron_ids), chem=chem, gap=gap, sign=sign, pep=pep,
                   stim=stim, strains=tuple(strains), mean=np.stack(means),
                   var_mean=np.stack(vms), n=np.stack(ns),
                   meta={"source": "wormneuroatlas (Randi et al. 2023 + compiled connectomes)",
                         "sigprop_version": getattr(a, "sigprop_v", "?")})


def kernel_timecourses(strain="wt", t=None, atlas=None):
    """Evaluate the atlas's fitted response kernels (Randi et al. 2023) on a time
    grid.  Each kernel is a sum of terms factor * t**power * exp(-g * t), stored
    as rows (g, factor, power, branch).  Returns (K, t) with K of shape
    (300, 300, len(t)) indexed [post, pre, time]; NaN where no kernel was fitted.
    Evaluated in float64 because some fits contain large cancelling terms."""
    a = open_atlas() if atlas is None else atlas
    t = np.arange(0.0, 30.0, 0.5) if t is None else np.asarray(t, float)
    raw = a.funatlas_h5[strain]["kernels"]
    N = len(a.neuron_ids)
    K = np.full((N, N, len(t)), np.nan)
    for i in range(N):
        for j in range(N):
            k = raw[i, j]
            if k is None or len(k) == 0:
                continue
            k = np.asarray(k, float).reshape(-1, 4)
            g, fac, pw = k[:, 0:1], k[:, 1:2], k[:, 2:3]
            K[i, j] = (fac * np.where(pw == 0, 1.0, t[None] ** pw) * np.exp(-g * t[None])).sum(0)
    return K, t


def kernel_descriptors(K, t):
    """Per-kernel shape descriptors on (..., T) arrays: signed peak value, time of
    the absolute peak, and the 'late fraction' = share of |kernel| area after 10 s."""
    absK = np.abs(K)
    ipk = np.nanargmax(np.where(np.isfinite(absK), absK, -1), axis=-1)
    peak = np.take_along_axis(K, ipk[..., None], -1)[..., 0]
    area = np.nansum(absK, -1)
    late = np.nansum(absK[..., t >= 10], -1) / np.where(area > 0, area, np.nan)
    ok = np.isfinite(K).all(-1)
    return (np.where(ok, peak, np.nan), np.where(ok, t[ipk], np.nan), np.where(ok, late, np.nan))


def average_bins(ds: Dataset):
    """Average a time-binned Dataset (trailing bin axis) over its bins: exactly
    the window-mean data an E1-style experiment would record from the same trials."""
    if ds.mean.ndim == 3:
        return ds
    B = ds.mean.shape[-1]
    return Dataset(**{**ds.__dict__, "mean": ds.mean.mean(-1), "var_mean": ds.var_mean.mean(-1) / B,
                      "meta": dict(ds.meta, averaged_from_bins=B)})


def subset(ds: Dataset, keep):
    """Restrict a Dataset to neurons `keep` (indices); stimuli outside are dropped."""
    keep = np.asarray(keep)
    pos = {k: i for i, k in enumerate(keep)}
    cols = [s for s, j in enumerate(ds.stim) if j in pos]
    sub = lambda M: M[np.ix_(keep, keep)]
    return Dataset(ids=ds.ids[keep], chem=sub(ds.chem), gap=sub(ds.gap), sign=sub(ds.sign),
                   pep=sub(ds.pep), stim=np.array([pos[ds.stim[s]] for s in cols], dtype=int),
                   strains=ds.strains, mean=ds.mean[:, keep][:, :, cols],
                   var_mean=ds.var_mean[:, keep][:, :, cols], n=ds.n[:, keep][:, :, cols],
                   meta=dict(ds.meta, subset=len(keep)),
                   proto=_subset_proto(ds.proto, cols, pos))


def _subset_proto(proto, cols, pos):
    """Restrict a protocol to `cols`; remap paired-stimulation partners to the new
    neuron indices (-1 if the partner was dropped)."""
    if proto is None:
        return None
    out = {k: np.asarray(v)[cols] for k, v in proto.items()}
    if "pair" in out:
        out["pair"] = np.array([pos.get(int(q), -1) if q >= 0 else -1 for q in out["pair"]], dtype=np.int32)
    return out
