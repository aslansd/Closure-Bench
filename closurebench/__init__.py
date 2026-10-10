"""closurebench: decisive tests of computational closure in a whole nervous system.

Modules
-------
discrete : exact closure measures (informational / causal / computational) on
           finite Markov chains, plus a finite-data estimator (Notebook 01).
ladder   : the L0-L4 + B model ladder in JAX, simulation of single-neuron
           stimulation experiments, and fitting (Notebooks 02, 04); slow-current
           rungs L0s / L1s (Notebooks 22-24); warm-started B fits (0.21, Notebook 25);
           the 2 x 2 rungs L0w / L1c, l1c_from_l1 and early stopping in fit (0.22, Notebook 26).
data     : loaders for the C. elegans wiring + signal-propagation atlas, and the
           common Dataset format shared by synthetic and real data (Notebook 03).
metrics  : noise-ceiling-normalised accuracy (FEVE), description length,
           bootstrap CIs and the pre-registered closure decision rule.
drift    : E3/E3b program-vs-hardware forecasting tests on spontaneous activity.
drift_kinds : E3c - which restricted change of the program explains the drift.
drift_time  : E3d/E3e - time structure of the drift (wander vs return), online drift tracker, colored-residual nulls.
embodiment  : E4a - brain alone vs brain + body as a closed system (autonomous rollouts).
spiking     : E2b - spiking (LIF / Poisson, NEF-compiled) realizations of the fitted L1.
conductance : E2c - conductance-based (shunting) synapse realization of the fitted L1.
intrinsic   : E2d - slow intrinsic (K+-like) currents in the substrate; timescale separation;
              E1g/E1h - rescaled_l1 (the frozen-gate limit as an L1), simulate_l1_from, true_rest.
"""
__version__ = "0.22.0"
