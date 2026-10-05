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
"""
__version__ = "0.3.4"
