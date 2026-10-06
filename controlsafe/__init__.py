"""controlsafe: control-theoretic safety instruments for the attention-free parent (CPU-testable).

Modules: ``adapters`` (RWKV-7 / GDN as one recurrence), ``trust_gate`` (G15 structural trust-gated
reachability), ``persistence`` (injection persistence curves + per-layer decay bounds), ``gramian``
(untrusted-channel controllability), ``snapshot`` (G4 erase-and-check with exact prefix-state reuse),
``monitor`` (G14 hidden-state poisoning telemetry). See ``controlsafe/CONTROLSAFE.md`` for the exact
guarantee statements and what remains empirical.
"""
