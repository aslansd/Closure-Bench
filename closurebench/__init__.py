"""closurebench: decisive tests of computational closure in a whole nervous system.

Modules
-------
discrete : exact closure measures (informational / causal / computational) on
           finite Markov chains, plus a finite-data estimator (Notebook 01).
ladder   : the L0-L4 + B model ladder in JAX, simulation of single-neuron
           stimulation experiments, and fitting (Notebooks 02, 04).
data     : loaders for the C. elegans wiring + signal-propagation atlas, and the
           common Dataset format shared by synthetic and real data (Notebook 03).
metrics  : noise-ceiling-normalised accuracy (FEVE), description length,
           bootstrap CIs and the pre-registered closure decision rule.
drift    : E3/E3b program-vs-hardware forecasting tests on spontaneous activity.
drift_kinds : E3c - which restricted change of the program explains the drift.
drift_time  : E3d - time structure of the drift (wander vs return) and an online drift tracker.
"""
__version__ = "0.12.0"
