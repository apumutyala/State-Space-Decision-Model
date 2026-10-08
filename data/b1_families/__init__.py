"""B1 add-on pool: ssdm-train-v0 records for Index families absent from the 150k training mix.

Families built here (eval presentation copied from the Index rows; see each module's docstring):

* ``API-Bank-tool-selection`` (catalog 3): official API-Bank TRAINING split (liminghao1630/API-Bank training-data/).
* ``BFCL-tool-selection`` (catalog 1): synthetic generator (BFCL has no train split).
* ``CRUXEval-output-choice`` (catalog 43, heldout-draw-v1 GENERATABLE): synthetic generator, outputs by execution.

Refused (``harness.heldout`` heldout-draw-v1 FROZEN: no training data or generator may target them, by rule):
MuSR (32), SATA-Bench (33), CLadder (44). ``build.py`` asserts this at start-up instead of hard-coding it.

Build: ``python -X utf8 -m data.b1_families.build`` from the repo root (CPU only). Records are written outside git
under ``ssdm-data/m2/build/b1-families/``; the manifest is committed in ``data/manifests/b1-families/``.
"""
